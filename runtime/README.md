# runtime/ — the Zig lowering VM

This directory holds the Zig half of the codegen pipeline: a comptime-unrolled
virtual machine that executes a generated control-loop graph as native code.
The graph is serialized from Python by `shinro.codegen.lower_zig`; the VM that
runs it is handwritten here once and never regenerated.

See [`../docs/codegen.md`](../docs/codegen.md) for the full pipeline narrative
and the XLA-fidelity model.

## Files

| File | Role |
|------|------|
| `build.zig` | Build script. Produces `libbase.so` from `lower.zig` + `graph_data.zig`. |
| `build.zig.zon` | Package/dependency manifest for the Zig build. |
| `lower.zig` | The comptime VM. Exports the `shinro_step` C-ABI function: one `inline for` over the node table, dispatching each node's op with `rows`/`cols` as comptime constants. |
| `linalg.zig` | Shared linear-algebra kernels (matmul, elementwise ops, `inv`, ...) used by the VM. |
| `graph_data.zig` | **Generated** — the graph as Zig constants (op enum, node table, offsets, `const_blob`). Produced by `scripts/gen_base.py` / `shinro.codegen.lower_zig`. Not hand-edited. |
| `tests/linalg.zig` | Zig unit tests for the linear-algebra kernels. |

## The "fixed at compile time" guarantee

The VM is XLA-flavored, mirroring the Python tracer:

- **Comptime specialization** — `inline for (g.nodes, 0..)` unrolls the whole
  node table at compile time, so every node's `rows`/`cols` are comptime loop
  bounds.
- **Static buffers** — one stack array `var buf: [g.buf_len]f64 align(16)`,
  sliced per-node via `g.offsets`. No heap, no per-op allocation.
- **Pure dataflow** — `inp` is the Infeed, `out` is the Outfeed; no side effects.
- **Closed op set** — an exhaustive `switch (node.op)` over the generated enum.

The no-heap property means "no per-op allocation and no op dispatch at
runtime", **not** "no numeric iteration inside an op". `.inv` already performs
runtime LU iteration inside its comptime-shaped buffer — the same way XLA
lowers `tf.linalg.inv` or `Select` to runtime loops. Any future
convergence-iterative op (e.g. a `solve_qp`/OSQP op) must keep the same shape:
a comptime-bounded static workspace sized from `node.rows`/`node.cols`, never
a per-tick heap allocator (that would break the no-heap intent).

## Building and testing

Requires `zig` on `PATH`.

```bash
make test-zig    # zig-gen → zig-build → zig test → pytest tests/test_zig_lowering.py
```

Individual steps:

```bash
make zig-gen     # python3 scripts/gen_base.py → rewrites runtime/graph_data.zig
make zig-build   # zig build --build-file runtime/build.zig --prefix build/
```

The resulting `libbase.so` lands in `build/` (gitignored); the cross-check
against the Python interpreter lives in `tests/test_zig_lowering.py`.

`runtime/graph_data.zig` is a generated artifact. To regenerate after a
codegen change, run `make zig-gen`. Treat it like a checked-in build output:
it exists so the VM can be compiled without a Python environment, but it is
regenerated from `scripts/gen_base.py`.
