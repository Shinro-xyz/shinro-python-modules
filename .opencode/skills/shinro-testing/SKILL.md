---
name: shinro-testing
description: Run and debug the shinro test suite — pick the right target (unit / integration / mcp / zig), avoid the marker and optional-extra gotchas that silently skip tests, and verify changes with lint + tests + lab note. Use when the user asks to run the tests, debug a test failure, or verify that a change is green.
license: MIT
compatibility: opencode
metadata:
  author: shinro
  version: "1.0.0"
  domain: robotics
  triggers: run tests, make test, why is my test not running, test not running, integration tests, zig tests, mcp tests, test failure, verify changes, is the suite green, lint
  role: specialist
  scope: verification
  output-format: none
  related-skills: port-numpy-component, test-repo
---

# Shinro Testing

Run and debug the `shinro-python-modules` test suite. The suite is pytest-based
with several **silent-skip** traps: marker-gated suites, zig-gated oracles, and
optional-extra tests that `importorskip`. Follow the workflow below so a green
run actually means the tests ran.

## When to Use This Skill

- The user asks to run the tests, verify a change, or debug a test failure
- You changed a component and need to know which suite to run
- A test "passes" but you suspect it never actually ran (silent skip)
- You are adding a new test group or a new component that needs test coverage

## Repo Reference

- **Pytest config (ground truth):** `pyproject.toml` → `[tool.pytest.ini_options]`
  (`testpaths`, `python_files`, `pythonpath = ["src", "."]`, `addopts`,
  `markers`). If docs disagree with this file, trust the TOML.
- **Make targets:** `Makefile` — the recommended entry points (see below).
- **Wrapper:** `run_tests.py` — pytest forwarding for environments without make.
- **Fixtures:** `tests/conftest.py` — `bk` (parametrized numpy+torch), `rng`
  (seeded), `numpy_backend`, `torch_backend`.
- **Human reference:** `docs/testing.md` (may drift; trust the TOML/Makefile).
- **Lab notes:** `lab-notes/daily/<date>.md` — every session's changes get a
  semantic summary with test results.

## Command Cheat-Sheet

| Situation | Command |
|-----------|---------|
| Quick loop while developing | `make test-quick` (unit files only) |
| Single test | `python3 -m pytest tests/test_x.py -v -k "name"` |
| Done-check (default suite) | `make test` |
| Lint | `make lint` |
| Full-loop / physics (MuJoCo) | `make test-integration` |
| MCP server protocol | `make test-functional` |
| Codegen / Zig VM | `ZIG_GLOBAL_CACHE_DIR=/tmp/shinro-zig-cache make test-zig` |
| Restore shipped graph after zig | `make zig-gen` |
| E2E scenario → verified `.so` | `make compile SCENARIO=<scenario.toml>` |
| Per-file targets | `make test-controllers`, `test-estimators`, `test-plants`, `test-trajectories`, `test-armrobot`, `test-components`, `test-array-backend`, `test-batched-adapter`, `test-controllability`, `test-factories`, `test-linearization`, `test-adversarial`, `test-mcp-server` |

## Gotchas (read before running anything)

1. **Marker exclusion is silent.** `addopts = "-m 'not integration and not mcp'"`
   means `tests/integration/` and `tests/test_mcp_server_functional.py` do
   **not run** under plain pytest or `make test`. Use `make test-integration` /
   `make test-functional` — they pass `--override-ini="addopts="` to clear it.
2. **Per-file targets don't skip the slow test.** `make test` excludes
   `test_very_large_horizon_mpc_times_out` via `-k`; `make test-controllers`
   and `make test-all` include it (it's intentionally slow).
3. **Zig-gated tests skip the whole file.** `tests/test_zig_lowering.py` calls
   `pytest.skip` when `zig` is not on PATH — a "21 passed" run can be 0 tests.
   Check `zig version` first; use `ZIG_GLOBAL_CACHE_DIR=/tmp/shinro-zig-cache`
   to avoid cache contention.
4. **Optional extras silently skip.** Tests for `[mujoco]`, `[torch]`,
   `[lerobot]`, `[onnx-rl]` use `importorskip` — if the extra isn't installed
   they report as skipped, not failed. A backend test that "didn't run" is why
   an optional-extra change can look green.
5. **`bk` runs every test twice.** The `bk` fixture is parametrized over numpy
   and torch, so a test written against `bk` executes once per backend. Counts
   are ×2; a torch-only failure is a real failure.
6. **`runtime/graph_data.zig` is generated, last-build-wins.** `scripts/gen_base.py`,
   `scripts/gen_mpc.py`, and the zig test fixtures all overwrite it. After any
   zig test run, re-run `make zig-gen` to restore the shipped KF+LQR graph.
   Never hand-edit it or the `runtime/codegen/emosqp/` bake.
7. **Lint scope is partial.** `make lint` runs pyright only on `utils/`,
   `components.py`, `controllers/`, `estimators/`, `trajectories/`, `plants/`.
   Code in `codegen/`, `mcp/`, `simulation/` is not typechecked — lint won't
   catch errors there.

## Interpreting Results

- **"No output" / all-skip ≠ passing.** Run with `-v` (or `-rs`) and look for
  `SKIPPED` lines. If a file you expected to run shows nothing, check the
  gotchas above.
- **A green `make test` does not cover integration, mcp, or zig.** Those are
  separate opt-in suites; run them when your change touches their areas.
- **Counts are ×2 for `bk`-parametrized tests** (numpy + torch).

## Workflow

1. **Identify the change's surface.** Unit component → `make test-quick` or the
   per-file target. Full-loop/physics → `make test-integration`. Codegen/VM →
   `make test-zig`. MCP → `make test-functional`. E2E compile → `make compile`.
2. **Run the targeted suite first** for fast feedback, then the full default
   suite (`make test`) before finishing.
3. **Lint** — `make lint` (ruff + pyright). Fix all findings.
4. **If you touched zig/codegen**, run `make test-zig` then `make zig-gen` to
   restore the shipped graph.
5. **Document** — write the semantic summary in `lab-notes/daily/<date>.md`
   with test results. This step is mandatory.

## Constraints

### MUST DO
- Use the Make targets for marker-gated suites (`test-integration`,
  `test-functional`) — plain pytest silently skips them
- Run `make lint` and `make test` before finishing a change
- Re-run `make zig-gen` after any zig test run
- Write the lab note with test results
- Trust `pyproject.toml` / `Makefile` over `docs/testing.md` when they disagree

### MUST NOT DO
- Treat silent skips (markers, zig absent, extras absent) as passes
- Hand-edit generated `runtime/graph_data.zig` or the `runtime/codegen/emosqp/`
  solver bake
- Add a new test group without wiring the `Makefile` target and `.PHONY` entry
- Query or re-index the stale `.codebase/` SQLite index
