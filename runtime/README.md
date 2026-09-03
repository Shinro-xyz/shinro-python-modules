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
| `build.zig` | Build script. Produces `libbase.so` from `lower.zig` + `graph_data.zig`, compiling the generated OSQP codegen solver into it. |
| `build.zig.zon` | Package/dependency manifest for the Zig build. |
| `lower.zig` | The comptime VM. Exports the `shinro_step` C-ABI function: one `inline for` over the node table, dispatching each node's op with `rows`/`cols` as comptime constants. |
| `linalg.zig` | Shared linear-algebra kernels (matmul, elementwise ops, `inv`, ...) used by the VM. |
| `qp.zig` | The `.solve_qp` op wrapper: drives the generated static OSQP solver (update q → solve → copy solution out). |
| `graph_data.zig` | **Generated** — the graph as Zig constants (op enum, node table, offsets, `const_blob`). Produced by `scripts/gen_base.py` / `shinro.codegen.lower_zig`. Not hand-edited. |
| `codegen/emosqp/` | **Generated** — the statically-allocated OSQP solver for the base MPC problem (no malloc, no libosqp). Emitted by `scripts/gen_emosqp_test.py`. |
| `tests/linalg.zig` | Zig unit tests for the linear-algebra kernels. |
| `tests/emosqp.zig` | Handwritten Zig test driving the codegen static solver, compared against the Python oracle. |
| `tests/emosqp_data.zig` | **Generated** — the oracle test vectors (sample `q` + expected solution, hex floats). Emitted by `scripts/gen_emosqp_test.py`. |

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
lowers `tf.linalg.inv` or `Select` to runtime loops. The `.solve_qp` op follows
the same shape: it calls `qp.solve_qp`, which drives the **statically-allocated**
codegen OSQP solver (`codegen/emosqp/workspace.c`'s `solver` global). The
problem data (P, A, l, u) and the pre-factorized KKT matrix are baked in at
generation time — only the linear cost `q` is updated per tick, so there is no
per-tick heap allocator and no `libosqp.so` dependency. The op's output size
must match the baked problem's `n_vars`.

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
against the Python interpreter lives in `tests/test_zig_lowering.py` (the
`.solve_qp` op is exercised by the MPC graph fixture, which traces
`MPC_LTI` and compares `shinro_step` against `interpret()`).

## Generated artifacts — shared paths, last build wins

Three paths feed `zig build`, and all of them are **generated, single-instance,
and mutually exclusive** — every generator overwrites the same file, so the
shipped content is always "whichever generator ran last":

| Artifact | Written by | Restore shipped default |
|---|---|---|
| `runtime/graph_data.zig` | `scripts/gen_base.py` (KF+LQR), `scripts/gen_mpc.py` (KF+MPC_LTI / KF+MPC_DeltaU), and the pytest fixtures in `tests/test_zig_lowering.py` | `make zig-gen` |
| `runtime/codegen/emosqp/` | `scripts/gen_emosqp_test.py` — bakes the whole static solver tree for one MPC problem | re-run the script (default: `mpc_lti_base.toml`) |
| `runtime/tests/emosqp_data.zig` | `scripts/gen_emosqp_test.py` (oracle vectors for the same bake) | re-run the script |

The committed default is the shipped bring-up pair: **KF + LQR graph +
`mpc_lti_base.toml` bake (n_vars=30)**.

### Building a different graph/solver pair without clobbering

`runtime/build.zig` takes two build options that select which generated graph
and which baked solver a build compiles in, without touching the shared paths:

```bash
zig build --build-file runtime/build.zig --prefix build/ \
    -Dgraph=<path-to-graph_data.zig> -Dsolver_dir=<path-to-bake-dir>
```

- `-Dgraph` — the generated graph (default `graph_data.zig`). `lower.zig`
  imports it as an anonymous module, so a build can consume a graph from any
  path (e.g. a pytest tmp dir).
- `-Dsolver_dir` — the baked OSQP codegen solver tree (default
  `codegen/emosqp`). The C sources and include paths are rooted there, and the
  bake's `solver_meta.zig` (`pub const n_vars`) is imported for the comptime
  check below.

This is how the MPC_DeltaU `.so` coexists with the shipped MPC_LTI one: bake
DeltaU into a second directory (`scripts/gen_emosqp_test.py --config
configs/controllers/mpc_base.toml --out-dir <dir>`), lower the KF+DeltaU
composed graph (`scripts/gen_mpc.py` with the DeltaU config), and build with
`-Dgraph`/`-Dsolver_dir` pointing at the pair. The shipped default is never
touched.

### The comptime graph↔bake check

Coupling rule: **a graph containing a `.solve_qp` node must be built against a
bake from the same MPC problem** — the solver's n_vars is baked into
fixed-size C arrays. This is now enforced at **compile time**: `lower.zig`
comptime-asserts that every `.solve_qp` node's output size equals the baked
`solver_meta.n_vars`, so a cross-config build fails with a `@compileError`
naming both sizes instead of silently linking a shape-mismatched solver.

## Build manifest — the audit trail

Every build writes a **deterministic report** next to the artifact plus a
**timestamped archive copy**, so teams can see what a binary contains and
browse which controller combinations were built and when:

- `<prefix>/lib/libbase.manifest.json` — the report: build facts (resolved
  target triple, optimize mode, zig version, libc, `float_type`), provenance
  (`-Dgraph`/`-Dsolver_dir` paths + sha256 of `graph_data.zig` and the bake's
  `workspace.c`), solver facts (baked `n_vars`/`n_cons`/`eps`/`config` from
  `solver_meta.zig`), and the graph content (op histogram, port layout,
  `buf_len`, the `.solve_qp` n_vars the graph expects).
- `<prefix>/manifests/<UTC>-<graphsha8>.json` — the archive copy. The
  timestamp lives in the **filename only**, never in the report, so the report
  is a pure function of its inputs: identical inputs ⇒ byte-identical report.
  Diffing two reports answers "did the binary content change, op-wise?"; the
  archive answers "when was this combination built?".

The graph content is emitted by `shinro.codegen.lower_zig` as
`<graph>_manifest.json` next to `graph_data.zig` (same node table the VM
compiles, so the report describes what is actually inside the `.so`). It
carries the **ordered node list** — `{i, op (Python), vm_op (Zig), inputs
(wiring), rows, cols, offset (buffer start), aux}` — one entry per node in
execution order, so the whole computation is reproducible from the report.

To reproduce "the LQR we had two weeks ago": `git checkout` the old graph (or
re-run `gen_base.py` at that commit), rebuild, and the new report must
byte-match the archived one — the sha256 provenance pins the exact inputs.

Two deliberate notes:

- The pytest fixtures lower each graph into per-fixture paths and build into
  per-fixture prefixes (`tmp_path`), so compiled `.so`s never collide — and
  with `-Dgraph`/`-Dsolver_dir` the *source* table and bake are no longer
  last-fixture-wins either. Re-run `make zig-gen` after the test suite to
  restore the shipped base graph.
- One robot = one controller = one binary, so a single shared path per
  artifact is acceptable: per-scenario bundle directories (and/or multiple
  baked solvers in one `.so`) were considered and deliberately deferred. The
  `-Dgraph`/`-Dsolver_dir` options are the minimal plumbing that makes a
  second bake a first-class operation; named scenario bundles can layer on
  top of them later if ever needed.
