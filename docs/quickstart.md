# Quickstart

This guide takes you from `pip install -e .` to a running, verified control
loop in three steps. No MuJoCo, no viewer, no optional dependencies — just the
core package.

If you'd rather read than run: [`how-it-works.md`](./how-it-works.md) explains
the architecture, [`components.md`](./components.md) is the full component
catalog, and [`testing.md`](./testing.md) covers the test suite.

## 1. Install

```bash
pip install -e .             # core: numpy, scipy, osqp, mcp
```

Optional extras:

```bash
pip install -e ".[mujoco]"   # MuJoCo physics backend (viewer demos)
pip install -e ".[torch]"    # torch backend (same controllers, torch arrays)
```

## 2. Build a controller from TOML

Everything in shinro is built from a TOML config via a factory. A config says
`type = "LQR"`, and the factory looks that name up in the registry and calls
the class's `from_config`.

```python
from shinro import ControllerFactory, EstimatorFactory

lqr = ControllerFactory("configs/controllers/lqr_base.toml").create()
kf  = EstimatorFactory("configs/estimators/kalman_base.toml").create()
print(lqr.K.shape)   # (3, 3) — the DARE-optimal gain, baked at construction
```

No wiring, no manual `A`/`B`/`Q`/`R` — `from_config` does the linearization
and solve for you. All shipped config files live under `src/shinro/configs/`;
the catalog in [`components.md`](./components.md) lists every registered name
and its config.

## 3. Run a simulation — one scenario TOML, three lines of Python

A *scenario* TOML declares the whole closed loop — plant, controller,
estimator, trajectory — as data. `configs/scenarios/cartpole_balance.toml`
ships with the package as a complete MuJoCo-free example:

```python
from shinro import ScenarioFactory

scenario = ScenarioFactory("configs/scenarios/cartpole_balance.toml").build()
result = scenario.run(steps=2000)   # 20 s at 100 Hz

print(f"final pole angle: {result[-1].state[2]:.6f} rad")
```

That's it. The scenario factory builds the plant from the plant TOML, derives
the LQR/KF model from the plant's dynamics, wires the loop, and validates the
dimensions. To iterate on your robot, edit the TOML — no code changes.

<details>
<summary>The TOML (for reference)</summary>

```toml
[scenario]
name = "cartpole_balance"
dt = 0.01
duration = 20.0
input_limits = { min = [-10.0], max = [10.0] }

[plant]
type = "CartPole"
config = "configs/plants/cartpole.toml"
initial_state = [0.0, 0.0, 0.2, 0.0]

[controller]
type = "LQR"
config = "configs/controllers/lqr_cartpole.toml"

[estimator]
type = "KalmanFilter"
config = "configs/estimators/kalman_cartpole.toml"

[trajectory]
type = "waypoints"
config = "configs/trajectories/cartpole_upright.toml"
```

</details>

To swap LQR for MPC, change one line in the TOML:

```toml
[controller]
type = "MPC_LTI"
config = "configs/controllers/mpc_lti_base.toml"
```

No code changes. This is the point of the design: *a scenario is data, not
code*.

## 4. Trace, verify, and lower to native code

Once the loop is verified in Python, compile it to a dependency-free native
`.so`:

```bash
shinro-compile configs/scenarios/cartpole_balance.toml --out build/scenario
```

This chains trace → compose → oracle-verify → lower to a Zig comptime VM →
`.so`. The compiled library exposes `shinro_step(inputs, outputs, state_out)`
— a C-ABI entry point ready for deployment (e.g. to a Cake module on a
Raspberry Pi). See [`codegen.md`](./codegen.md) for the full walkthrough.

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
