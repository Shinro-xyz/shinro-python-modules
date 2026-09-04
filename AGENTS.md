# shinro-python-modules — AGENTS.md

This is a pip-installable Python package. Source lives under `src/shinro/`.
Import code as `shinro.*` (never by top-level module path).

## Codebase Index

**No longer maintained.** The repo was previously indexed in a SQLite
`.codebase/` database, but the project has outgrown it — the index is stale
(module paths changed when the package moved under `src/shinro/`) and is not
kept current. **Do not query or re-index it.** Read the source directly with
your normal tools.

The package is small (~40 source files) and well-structured: the five abstract
base classes live in `src/shinro/components.py`, and each subpackage
(`controllers/`, `plants/`, `estimators/`, `trajectories/`, `utils/`,
`factories/`) has a public API exported from its `__init__.py`. Read those
instead of an index.

For lab notes, read `lab-notes/daily/` directly (see below).

## Overview

A modular control suite built on five ABCs — **Controller**, **Plant**,
**StateEstimator**, **TrajectoryGenerator**, **PhysicsEngine** — composed via a
registry/factory pattern from TOML configs (`src/shinro/configs/`), with
numpy/torch support through the `ArrayBackend` abstraction. The full component
catalog (registered names, config files) lives in
[`docs/components.md`](./docs/components.md); architecture narrative in
[`docs/how-it-works.md`](./docs/how-it-works.md).

Key design: the arm's `step()` takes a Cartesian velocity twist
`[dx, dy, dz, droll, dpitch, dyaw]`, integrates it into a target pose, runs IK
internally, and sends joint angles to servos. The controller **never touches
joint space**.

## Commands

- `make test` — full suite, **skips** `test_very_large_horizon_mpc_times_out`.
  Per-file targets (`make test-controllers`, etc.) do **not** skip it.
- Single test: `python3 -m pytest tests/test_x.py -v -k "name"`.
- **Marker gotcha:** pytest's default `addopts` excludes the `integration` and
  `mcp` markers, so those tests silently don't run under plain pytest. Use the
  Make targets (`make test-integration`, `make test-mcp-functional`), which
  pass `--override-ini="addopts="`.
- `make lint` = `ruff check .` + pyright **only on** `utils/`, `components.py`,
  `controllers/`, `estimators/`, `trajectories/`, `plants/`. `codegen/`,
  `mcp/`, `simulation/` are not typechecked — code added there won't be caught
  by lint.
- Optional extras (`[mujoco]`, `[torch]`, `[lerobot]`, `[onnx-rl]`) are not
  installed by default; tests `importorskip` and silently pass over. If a
  backend test "didn't run", that's why. Requires Python ≥3.12; CI matrix is
  3.12–3.14.
- Demos: `python -m demos.demo_*`.

## Zig lowering (codegen → `.so`)

- Requires `zig` on PATH. Artifacts land in `build/` (gitignored).
  `make test-zig` chains: gen → build → `zig build test` → pytest
  `tests/test_zig_lowering.py`.
- **Shared generated paths, last build wins:** `runtime/graph_data.zig` is
  overwritten by `scripts/gen_base.py`, `scripts/gen_mpc.py`, *and* the pytest
  fixtures in `tests/test_zig_lowering.py`; `runtime/codegen/emosqp/` likewise
  holds one MPC bake. After running the test suite, re-run `make zig-gen` to
  restore the shipped KF+LQR graph.
- Build an alternate graph/solver pair without clobbering the shared paths via
  `zig build -Dgraph=<path> -Dsolver_dir=<dir>`. A `.solve_qp` node whose
  output size doesn't match the bake's `n_vars` is rejected at compile time.
- `runtime/graph_data.zig` and the solver bake are **generated** — never
  hand-edit. `runtime/lower.zig` (the comptime VM) is handwritten and never
  regenerated. See `runtime/README.md` for the build-manifest audit trail.

## Key Files

| File | Purpose |
|------|---------|
| `src/shinro/components.py` | The five ABCs |
| `src/shinro/codegen/` | Trace → compose → interpret → lower pipeline; `lower_zig.py` serializes a graph to `runtime/graph_data.zig` |
| `src/shinro/factories/registry.py` | Component registry + config-driven factory |
| `src/shinro/utils/array_backend.py` | NumpyBackend / TorchBackend abstraction |
| `src/shinro/mcp/server.py` | MCP server (`shinro-mcp` console command, wired in `.mcp.json`) |
| `src/shinro/simulation/robotsim.py` | Config-driven robot simulation factory |
| `tests/integration/` | Full-loop physics-backed tests (MuJoCo, opt-in) |
| `runtime/` | Zig comptime VM + linalg kernels + generated graph |

## Lab Notebooks

Experiment logs live in `lab-notes/daily/` in the repo. Read the files directly.

Each session's lab note MUST contain a semantic summary of the changes made —
what was built, why, key design decisions, and test results. This is not a git
log; it's a narrative record of intent and outcomes. (A CI bot
(`.github/workflows/update-labnotes.yml`) also appends commit logs to today's
file on push to main.)

## Workflow

0. **Check tickets** — Review open GitHub issues (`gh issue list`) before starting; pick up or close anything relevant
1. **Understand** — Read the source directly (`src/shinro/**`) and `lab-notes/daily/` for relevant context
2. **Plan** — Describe the change and which files to modify
3. **Implement** — Use OpenCode or direct editing
4. **Verify** — Run tests or check the output
5. **Document** — Write a semantic summary in `lab-notes/daily/<date>.md` covering what changed, why, and key results

> **Releases:** `make release-patch/minor/major` auto-regenerates `CHANGELOG.md`
> via `git cliff` (Conventional Commits, see `cliff.toml`) and stages it. Use
> Conventional Commit prefixes (`feat:`, `fix:`, `chore:`, ...) so the
> changelog stays clean — non-conforming commits land under "Other". Version is
> the git tag via setuptools-scm.

## Porting Numpy Scripts to the Framework

See the `port-numpy-component` skill (`.opencode/skills/port-numpy-component/SKILL.md`) — it loads on-demand via the `skill` tool when porting a raw numpy implementation into the framework.
