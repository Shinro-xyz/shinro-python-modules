---
name: wire-linalg-kernel
description: After writing a kernel in src/shinro/runtime/linalg.zig, wire a new VM op end-to-end — register it (ops.py), add the TraceBackend surface, lower_zig enum/tables/aux mapping, the lower.zig VM arm, tests (zig unit + .so oracle + op-shape matrix), regenerate graph_data.zig, lint and test. Use when the user asks to integrate, wire up, finish, or expose a new kernel as a graph op.
---
# Wire a Linalg Kernel Into the Graph VM

A new op touches **six surfaces**. Miss one and either the build breaks or a
silent shape/stride bug survives. This skill is the checklist plus the traps.

## When to Use

Use when a kernel has been (or will be) written in
`src/shinro/runtime/linalg.zig` and needs to become a graph op. Covers the six
surfaces: `ops.py` numpy handler, `TraceBackend` method, `lower_zig.py`
serialization, the `lower.zig` comptime VM arm, the tests, and the ONNX
importer when the op is an ONNX op. Also use when adding an ONNX op that needs
a **new** VM op (as opposed to composing one from existing ops).

## Procedure

1. **Finish the kernel first** in `src/shinro/runtime/linalg.zig`. House
   conventions: comptime dimensions, `[]const f64` flat row-major buffers,
   return a fixed array **by value** (`[m*n]f64`), runtime `for` loops over
   comptime-known bounds (never `inline for` for element loops), and a `///`
   doc comment stating the exact arithmetic contract. It must compile before
   you wire it: `zig build test --build-file src/shinro/runtime/build.zig`.

2. **Pin down the contract**: number of inputs, output shape, and whether it
   has f64 scalar parameters. f64 scalars can **not** ride in the node's
   `aux` (a single `usize`) — see Pitfalls.

3. **`src/shinro/codegen/ops.py`** — add an `@register_op("<op>")` handler with
   signature `(node, values, inputs) -> np.ndarray`. This *is* the correctness
   oracle: `interpret()`, the ONNX importer's eager evaluation, and the Zig
   `.so` are all checked against it. Read scalar attrs from `node.attrs`.

4. **`src/shinro/codegen/trace_backend.py`** — add a `TraceBackend.<op>`
   method: `_lift` each operand, compute the output shape from input shapes
   (numpy rank conventions; 1-D stays 1-D), then
   `self._emit("<op>", [ids...], out_shape, **attrs)`.

5. **`src/shinro/codegen/lower_zig.py`** —
   (a) add the op name to the emitted `Op` enum list;
   (b) if it has f64 attrs, add a pre-pass collecting them into parallel lists
   plus an aux-packing dict (the `clip_offsets` / `gemm_aux` pattern), emit
   `pub const <op>_<attr> = [_]f64{...};` after `clip_hi`, and pack
   `aux = (table_index << nflag_bits) | flags`;
   (c) map it in `_node_vm_info`: `if node.op == "<op>": return "<vm_op>", packed_aux`.
   Thread any new dict through `_node_line`, `_graph_manifest`, and their
   `_node_vm_info` call sites.

6. **`src/shinro/runtime/lower.zig`** — add the switch arm next to the related
   op. Pull operands with `node_input` / `node_input_at`, derive comptime dims
   from `node.rows`/`node.cols` and the input nodes' shapes, read baked tables
   via `g.<op>_<attr>[node.aux ...]`, call the kernel, copy the result into
   `out` with a runtime `for`. Normalize 1-D inputs: the VM stores a 1-D shape
   as `rows=n, cols=1, vec=true` — if the kernel wants `m=1`, derive it from
   `node.vec`. Add a `comptime { if (...) @compileError(...) }` shape gate if
   the op has a baked-size contract.

7. **Regenerate the shipped graph**: `make zig-gen`, then `make zig-build`.
   `lower.zig` references the new tables/enum variant unconditionally, so the
   build fails until `graph_data.zig` defines them.

8. **Tests**:
   (a) `src/shinro/runtime/tests/linalg.zig` — hand-computed cases
   (`expectEqual` for f64-exact values, `expectApproxEqAbs(..., 1e-12)`
   otherwise), one per code path.
   (b) `tests/test_zig_lowering.py` — hand-build a `Graph` with
   `g.emit("<op>", ...)`, wrap in `ComposedGraph`, `_build_so(...)`, and
   compare the compiled `.so` against `interpret()` per named output at
   ~1e-13. Cover every shape/branch class.
   (c) `tests/test_op_shape_matrix.py` — add cells to the matching graph
   builder so the suite's "every op in the vocabulary" claim stays true, and
   make the declared output shape equal numpy's.
   (d) If it is an ONNX op, extend `_SUPPORTED_OPS` in `onnx_import.py` and
   add an `emit_*` method (or route an existing decomposition at the new fused
   op).

9. **Verify**: `make lint` → `zig build test --build-file src/shinro/runtime/build.zig`
   → `python3 -m pytest tests/unit/test_onnx_import.py tests/test_codegen.py
   tests/test_op_shape_matrix.py tests/test_zig_lowering.py -q
   --override-ini="addopts="` → full `make test`.

10. **Restore generated churn** before reporting: tests overwrite the shared
    `src/shinro/runtime/graph_data.zig` and often `runtime/codegen/emosqp/`.
    Re-run `make zig-gen`, and `git checkout --` unrelated drift
    (`graph_data_manifest.json`, `runtime/codegen/emosqp/`,
    `runtime/tests/emosqp_data.zig`).

11. **Write the semantic lab note** in `lab-notes/daily/<date>.md`. Do **not**
    commit or push unprompted — the user reviews each step and approves the
    commit.

12. **Report** the changed files plus the key node-count/parity numbers and
    the observed verification output.

## Pitfalls

- **`aux` is one `usize` with no float channel.** f64 attributes (alpha/beta,
  eps) must be baked into parallel f64 tables emitted by `lower_zig` (the
  `clip_lo`/`clip_hi` pattern) or passed as const-node scalar inputs. Pack a
  table index and small flags: `aux = (index << nbits) | flags`
  (gemm: `(idx << 1) | transB`).
- **Regenerate before building.** `lower.zig` always references the generated
  tables and enum variants; adding one breaks the build until `make zig-gen`
  runs. Do it first or you will chase a phantom compile error.
- **`_build_so` converts a `zig build` failure into `pytest.skip`.** A green
  oracle run can hide a real compile error. Verify the build returncode
  explicitly (run `zig build ... -Dgraph=<tmp>/graph_data.zig` by hand) and/or
  run the oracle with `-rs` and confirm there are no skips.
- **Inner element loops must stay runtime `for`.** `inline for` unrolls one
  statement per element and blows up compile time and `.so` size.
- **Comptime parameters**: pass `node.rows` / `g.nodes[i].cols` directly — they
  are comptime in the unrolled `inline for`. If Zig rejects a derived value,
  bind it with `comptime` or as a `const` first.
- **1-D shapes are `rows=n, cols=1, vec=true`.** Treat a vec input as `m=1` in
  the arm; fix strides there, never with a reshape node.
- **`_node_vm_info` is the single source of truth for `(vm_op, aux)`** — it
  feeds both the emitted table and the manifest, and
  `test_drift_guard_nodes_match_emitted_table` compares them. Never compute aux
  in two places.
- **Lint coverage**: `make lint` runs ruff repo-wide (line length 140) but
  pyrefly only over `utils/ components.py controllers/ estimators/
  trajectories/ plants/` — `codegen/` is ruff-only.
- **Tests clobber shared generated paths** (`graph_data.zig`, the emosqp bake).
  Restore after.

## Verification

1. `zig build test --build-file src/shinro/runtime/build.zig` passes.
2. The op-bearing graph actually **compiles** in Zig — check the `zig build`
   returncode directly (not just the oracle test).
3. `.so` oracle test passes vs `interpret()` with no skips (`-rs`).
4. `tests/test_op_shape_matrix.py` includes the new op and passes (values and
   manifest-declared shapes).
5. `make lint` clean.
6. Full `make test` green.
7. `tests/test_zig_lowering.py -k test_drift_guard_nodes_match_emitted_table`
   passes.
8. Generated files restored; the diff is scoped to the change.

## Worked Example: the fused `gemm` op (2026-09-22)

A `Gemm` used to decompose into `transpose + matmul + mul + mul + add`, which
materialized a second full copy of every weight matrix in the workspace. The
fused op is one node. Snippets are the actual edits.

Kernel (`linalg.zig`) — comptime params, runtime inner loop:

```zig
pub fn gemm(
    comptime m: usize, comptime k: usize, comptime n: usize,
    comptime alpha: f64, comptime beta: f64, comptime transB: bool,
    a: []const f64, b: []const f64, c: []const f64,
    comptime c_len: usize,
) [m * n]f64 {
    var out: [m * n]f64 = undefined;
    for (0..m) |i| {
        for (0..n) |j| {
            var s: f64 = 0.0;
            for (0..k) |p| {
                const bv = if (transB) b[j * k + p] else b[p * n + j];
                s += a[i * k + p] * bv;
            }
            const cj = if (c_len == 1) c[0] else if (c_len == n) c[j] else c[i * n + j];
            out[i * n + j] = alpha * s + beta * cj;
        }
    }
    return out;
}
```

Oracle (`ops.py`):

```python
@register_op("gemm")
def _gemm(node, values, inputs):
    a = values[node.inputs[0]]
    b = values[node.inputs[1]]
    c = values[node.inputs[2]]
    if node.attrs.get("transB", False):
        b = b.T
    return node.attrs.get("alpha", 1.0) * (a @ b) + node.attrs.get("beta", 1.0) * c
```

Trace surface (`trace_backend.py`):

```python
def gemm(self, a, b, c, *, alpha=1.0, beta=1.0, transB=False):
    a = _lift(self.g, a); b = _lift(self.g, b); c = _lift(self.g, c)
    n = b.shape[0] if transB else b.shape[1]
    out_shape = (n,) if len(a.shape) == 1 else (a.shape[0], n)
    return self._emit("gemm", [a, b, c], out_shape, alpha=alpha, beta=beta, transB=transB)
```

Serialization (`lower_zig.py`) — four spots: enum append, bake + aux pack,
emitted tables, `_node_vm_info` branch:

```python
lines.append("    abs, sign, pow, lt, min, gemm,")

gemm_alpha, gemm_beta, gemm_aux = [], [], {}
for i, node in enumerate(g.nodes):
    if node.op == "gemm":
        gemm_aux[i] = 2 * len(gemm_alpha) + (1 if bool(node.attrs.get("transB", False)) else 0)
        gemm_alpha.append(float(node.attrs.get("alpha", 1.0)))
        gemm_beta.append(float(node.attrs.get("beta", 1.0)))

lines.append("pub const gemm_alpha = [_]f64{" + _zig_floats(gemm_alpha) + "};")
lines.append("pub const gemm_beta = [_]f64{" + _zig_floats(gemm_beta) + "};")

if node.op == "gemm":
    return "gemm", gemm_aux[i]
```

VM arm (`lower.zig`):

```zig
.gemm => {
    const a = node_input(g.nodes[0..], node, &workspace);
    const b = node_input_at(g.nodes[0..], node.inputs[1], &workspace);
    const c = node_input_at(g.nodes[0..], node.inputs[2], &workspace);
    const a_n = g.nodes[node.inputs[0]];
    const c_n = g.nodes[node.inputs[2]];
    const r = la.gemm(
        if (node.vec) 1 else node.rows,
        if (a_n.vec) a_n.rows else a_n.cols,
        if (node.vec) node.rows else node.cols,
        g.gemm_alpha[node.aux / 2],
        g.gemm_beta[node.aux / 2],
        node.aux % 2 == 1,
        a, b, c,
        c_n.rows * c_n.cols,
    );
    for (0..node.rows * node.cols) |j| out[j] = r[j];
},
```

`.so` oracle (`tests/test_zig_lowering.py`) — then compare `step_so` vs
`interpret` at ~1e-13:

```python
g = Graph()
x = g.input("x", (3,))
w = g.const(np.array([[1., 2., 3.], [4., 5., 6.]]))
bias = g.const(np.array([0.5, -0.5]))
out = g.emit("gemm", [x, w, bias], (2,), transB=True)
g.output("tb", out)
cg = ComposedGraph(graph=g, inputs=["x"], outputs=["tb"], state_inputs=[], state_outputs=[])
lib, cg = _build_so(cg, d, graph_path=d / "graph_data.zig")
```

ONNX importer — one node per `Gemm`, missing bias as a `const(0.0)` scalar:

```python
def emit_gemm(self, inputs, attrs):
    a = self.tensor(inputs[0])
    b = self.tensor(inputs[1])
    c = self.tensor(inputs[2]) if len(inputs) == 3 else self.const(0.0)
    return self.emit("gemm", [a, b, c],
                     alpha=float(attrs.get("alpha", 1.0)),
                     beta=float(attrs.get("beta", 1.0)),
                     transB=bool(int(attrs.get("transB", 0))))
```
