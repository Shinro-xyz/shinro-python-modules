# Quickstart

This guide takes you from a fresh checkout to a running, verified control loop
in four steps. It assumes Python 3.12+ and a working pip.

If you'd rather read than run: [`how-it-works.md`](./how-it-works.md) explains
the architecture, [`components.md`](./components.md) is the full component
catalog, and [`testing.md`](./testing.md) covers the test suite.

## 1. Install

From a source checkout (recommended for development):

```bash
pip install -e .             # core: numpy, scipy, osqp, mcp
pip install -e ".[mujoco]"   # optional: MuJoCo physics backend
pip install -e ".[torch]"    # optional: torch backend
```

Core install gives you numpy-backed controllers, estimators, trajectories, and
the MCP server. MuJoCo is needed for the physics-backed scenarios and the
`demos/` that render a viewer.

## 2. Build a controller from TOML

Everything in shinro is built from a TOML config via a factory. A config says
`type = "LQR"`, and the factory looks that name up in the registry and calls
the class's `from_config`. Example, `src/shinro/configs/controllers/lqr_base.toml`:

```toml
type = "LQR"
name = "base_controller"
dt = 0.02
state_cost = [100.0, 100.0, 50.0]
control_cost = [0.1, 0.1, 0.1]
```

Create it in code:

```python
from shinro.factories import ControllerFactory, EstimatorFactory

lqr = ControllerFactory("src/shinro/configs/controllers/lqr_base.toml").create()
kf = EstimatorFactory("src/shinro/configs/estimators/kalman_base.toml").create()
print(lqr.K.shape)   # (3, 3) — the DARE-optimal gain, baked at construction
```

No wiring, no manual `A`/`B`/`Q`/`R` — `from_config` does the linearization
and solve for you. All config files shipped with the package live under
`src/shinro/configs/` (packaged as `shinro.configs`); the catalog in
[`components.md`](./components.md) lists every registered name and its config.

> The `type` string is the **registered name** — what factories, scenarios,
> and the MCP server all dispatch on. It is not always the class name
> (e.g. `MPC_LTI` vs the `MPC_LTI_Base` class behind it).

## 3. Run a simulation scenario

A *scenario* TOML declares the whole closed loop — plant, controller,
estimator, trajectory, input limits, physics backend — as data. The
`base_tracking` scenario (LQR + Kalman on the holonomic base) lives in
`tests/integration/scenarios/base_tracking.toml`:

```toml
[scenario]
name = "base_tracking"
duration = 16.0
dt = 0.02
input_limits = { min = [-0.5, -0.5, -1.0], max = [0.5, 0.5, 1.0] }

[plant]
name = "base"

[controller]
type = "LQR"
config = "configs/controllers/lqr_base.toml"

[estimator]
type = "KalmanFilter"
config = "configs/estimators/kalman_base.toml"

[trajectory]
type = "waypoints"
config = "configs/trajectories/base_straight.toml"
```

Assemble it and run one closed-loop step by hand — the loop is just the four
ABCs wired in the fixed dataflow (see [`how-it-works.md`](./how-it-works.md)):

```python
import numpy as np
from shinro.factories import ScenarioFactory

scenario = ScenarioFactory("tests/integration/scenarios/base_tracking.toml").build()
sim, plant = scenario.sim, scenario.plant
ctrl, est, traj = scenario.controller, scenario.estimator, scenario.trajectory

dt = scenario.config["scenario"]["dt"]
lo = np.array(scenario.config["scenario"]["input_limits"]["min"])
hi = np.array(scenario.config["scenario"]["input_limits"]["max"])

u_prev = np.zeros(3)
for step in range(800):                      # 16 s at 50 Hz
    true_state = np.asarray(plant.get_state()).flatten()
    ref = np.asarray(traj[step]).flatten()   # waypoint schedule
    estimate = est.estimate(true_state.reshape(-1, 1), u_prev.reshape(-1, 1)).flatten()
    control = ctrl.compute(estimate, ref)    # LQR: (current, target)
    plant.step(np.clip(control, lo, hi))     # apply and advance the plant
    sim.step()                               # advance the physics engine
    u_prev = control

print(sim.get_state())
```

> LQR/PID take `compute(current, target)`; `MPC_LTI` takes `compute(error)`;
> `MPC_DeltaU` takes `compute(error, u_prev=...)`. The integration runners in
> `tests/integration/helpers/scenario_runner.py` dispatch on controller type —
> reuse them rather than reimplementing this switch.

To swap LQR for MPC, change one line in the TOML:

```toml
[controller]
type = "MPC_LTI"
config = "configs/controllers/mpc_lti_base.toml"
```

No code changes. This is the point of the design: *a scenario is data, not
code*.

To watch it, run the packaged demo instead:

```bash
python -m demos.demo_simple              # terminal output, no viewer
python -m demos.demo_base_tracking --controller mpc   # base tracking, live viewer
```

MuJoCo must be installed (`pip install -e ".[mujoco]"`) for the viewer demos.

## 4. Trace, verify, and lower to native code

The codegen pipeline compiles a closed-loop step into a static graph that is
verified to float-exactness against numpy, then lowers it to a Zig
comptime-unrolled VM exposed as a `.so`. The full walkthrough is
[`codegen.md`](./codegen.md); here's the 30-second version:

```bash
python demo_codegen.py     # trace → compose → interpret, PASS/FAIL per stage
make test-zig              # + serialize to runtime/graph_data.zig, build .so,
                           #   cross-check against the interpreter (needs zig)
```

`demo_codegen.py` traces a Kalman filter, composes it with an LQR controller
into one closed-loop tick, verifies the composed step matches a live numpy
loop, then re-composes with a Luenberger observer — reusing the controller's
graph without re-tracing (the modularity proof). Each stage prints
`PASS`/`FAIL` based on max abs error vs numpy.

## What next

- **[`how-it-works.md`](./how-it-works.md)** — the five ABCs and how data flows
  through the loop.
- **[`components.md`](./components.md)** — every registered component, class,
  file, and bundled config.
- **[`codegen.md`](./codegen.md)** — tracing, composition, and Zig lowering in
  depth.
- **[`testing.md`](./testing.md)** — running the suite; `make test` for the
  fast path, `make test-integration` for MuJoCo-backed full-loop tests.
- **[`mcp_server.md`](./mcp_server.md)** — drive controllers/estimators over
  Model Context Protocol.
- **[`../AGENTS.md`](../AGENTS.md)** — how to contribute (agent workflow,
  lab-notes, Conventional Commits).
