# Testing Protocols

This document describes how the `shinro-python-modules` test suite is organized and how to run it.

## Overview

The suite is built on **pytest**. Configuration lives in [`pyproject.toml`](../pyproject.toml) under `[tool.pytest.ini_options]`:

- `testpaths = ["tests"]` — pytest runs the `tests/` directory by default.
- `python_files = ["test_*.py"]` — only files named `test_*.py` are collected.
- `pythonpath = ["src"]` — the `src/` layout is on the import path, so components are imported as `shinro.*` (e.g. `from shinro.plants.armrobot import ArmRobot`).
- `addopts = "-m 'not integration'"` — integration tests (marked `@pytest.mark.integration`) are excluded by default; run them with `make test-integration` (see below).

All test files live in `tests/`, with one file per component group. Integration tests live in
`tests/integration/`. There is also a standalone demo script, `demo_codegen.py`, at the repo root.

## Entry Points

There are three ways to run the tests. All of them funnel into `pytest`; none reinvent the runner.

### 1. Direct pytest

```bash
python3 -m pytest tests/ -v --tb=short           # full suite
python3 -m pytest tests/test_controllers.py -v   # single file
python3 -m pytest tests/ -k "arm"                # keyword-filter tests
```

### 2. Make targets (recommended)

The [`Makefile`](../Makefile) provides short named targets. `make test-<name>` runs exactly one test file.

| Target | Runs |
|--------|------|
| `make test` | Full suite, **skips** `test_very_large_horizon_mpc_times_out` and the opt-in markers (`integration`, `mcp`) |
| `make test-all` | Full suite, including the slow horizon test (still excludes the opt-in markers) |
| `make test-quick` | Unit tests only (controllers, estimators, trajectories, plants, factories, components, array backend, batched adapter, controllability, mcp server) |
| `make test-functional` | Functional MCP server tests (spawns a real server; opt-in, clears the `mcp` marker exclusion) |
| `make test-integration` | Full-loop physics-backed suite in `tests/integration/` (requires MuJoCo; opt-in, not in CI) |
| `make test-controllers` | `tests/test_controllers.py` |
| `make test-estimators` | `tests/test_estimators.py` |
| `make test-plants` | `tests/test_plants.py` |
| `make test-trajectories` | `tests/test_trajectories.py` |
| `make test-armrobot` | `tests/test_armrobot.py` |
| `make test-components` | `tests/test_components.py` |
| `make test-array-backend` | `tests/test_array_backend.py` |
| `make test-batched-adapter` | `tests/test_batched_adapter.py` |
| `make test-controllability` | `tests/test_controllability_checker.py` |
| `make test-factories` | `tests/test_factories.py` |
| `make test-linearization` | `tests/test_linearization.py` |
| `make test-adversarial` | `tests/test_adversarial.py` |
| `make test-mcp-server` | `tests/test_mcp_server.py` (unit-level, runs by default) |
| `make test-mcp-functional` | `tests/test_mcp_server_functional.py` (subprocess protocol tests; opt-in via the `mcp` marker) |
| `make test-zig` | Generate `runtime/graph_data.zig`, build the Zig VM, run `tests/test_zig_lowering.py` (requires `zig` on PATH) |
| `make zig-gen` | Serialize the `base_tracking` composed graph to `runtime/graph_data.zig` only |
| `make zig-build` | Compile the Zig VM to `build/lib/libbase.so` (implies `zig-gen`) |
| `make lint` | `ruff check .` + `pyright` on source dirs |

The per-file targets run their file unconditionally — `make test-controllers` includes the slow horizon test, unlike `make test` which excludes it.

### 3. `run_tests.py` wrapper

A thin Python wrapper that forwards flags to pytest. Useful for environments without `make`.

```bash
python3 run_tests.py             # full suite, skips slow horizon test
python3 run_tests.py --all       # full suite, including slow tests
python3 run_tests.py --quick     # unit tests only
python3 run_tests.py --func      # functional tests only
python3 run_tests.py --int       # integration suite (MuJoCo required, opt-in)
```

## Slow Horizon Test

`test_very_large_horizon_mpc_times_out` exercises the MPC timeout guard with a very large horizon and is intentionally slow. It lives in `tests/test_controllers.py`.

- `make test` and bare `python3 run_tests.py` skip it with `-k "not test_very_large_horizon_mpc_times_out"`.
- `make test-all` and `make test-controllers` include it.
- CI (`.github/workflows/test.yml`) runs the full `tests/` directory with no filter, so the slow test runs there.

## Integration Tests

Tests under `tests/integration/` run full closed-loop simulations against MuJoCo and are marked
`@pytest.mark.integration`. Because pytest is configured with `addopts = "-m 'not integration'"`,
they are **excluded by default** — run them explicitly:

```bash
make test-integration   # equivalent to:
python3 -m pytest tests/integration/ -v --tb=short --override-ini="addopts="
```

These are opt-in for a reason: they require a working MuJoCo install and are heavier than the unit
suite, and they are intentionally not part of CI.

## CI

`.github/workflows/test.yml` runs on push/PR to `main`:

1. Checkout + set up Python 3.12.
2. `pip install -r requirements.txt` and `requirements-dev.txt`.
3. `pip install torch` (CPU index) so torch-backend tests run.
4. `pytest tests/ -v --tb=short` — no keyword filter.

To run the CI test job locally with [act](https://github.com/nektos/act):

```bash
act -j test --container-architecture linux/amd64
```

(the `test` job name is `test` in `test.yml`).

## Fixtures

Shared fixtures are defined in `tests/conftest.py`:

| Fixture | Purpose |
|---------|---------|
| `numpy_backend` | A `NumpyBackend` instance |
| `torch_backend` | A `TorchBackend` on CPU; skips if `torch` is not installed |
| `bk` | Parameterized over `numpy` and `torch`, so any test using `bk` runs twice (once per backend) |
| `rng` | `numpy.random.default_rng(42)` — seeded for reproducible randomness |

The parametrized `bk` fixture is how the suite guarantees backend-agnostic behavior: tests written against `bk` are automatically exercised on both numpy and torch.

## Test Categories

The tests fall into four groups:

- **Unit tests** — construction validation, shape checking, mathematical accuracy (governing equations verified analytically), convergence, error handling, and `from_config` factory loading. Most files fall here.
- **Functional tests** — `tests/test_mcp_server_functional.py` spawns a real MCP server and talks to it end to end; slower and heavier. Gated behind the `mcp` marker (opt-in, like `integration`) so the default suite doesn't spawn subprocesses.
- **Integration tests** — `tests/integration/` full-loop physics-backed simulations (MuJoCo), excluded by default (see above).
- **Adversarial / robustness** — `tests/test_adversarial.py` probes edge cases and misuse of the public API.

## Adding a New Test Group

1. Create `tests/test_<name>.py` following the existing class/method style.
2. Add a `make test-<name>` target to the `Makefile` and list it in `.PHONY`.
3. Update `test-quick` in the Makefile and `--quick` in `run_tests.py` if the new group is unit-level.
4. If the tests are full-loop physics-backed, put them in `tests/integration/` and mark them
   `@pytest.mark.integration` instead.
