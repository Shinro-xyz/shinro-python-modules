# How the System Works

This document explains how `shinro` is put together and how data flows through
it at each layer: from a TOML config, to a running component, to a simulated
closed loop, and finally to a compiled graph. It complements
[`codegen.md`](./codegen.md) (the tracing/compilation pipeline) and
[`testing.md`](./testing.md) (how it's verified).

## The core idea: five contracts, everything plugs in

`shinro` is built around five abstract base classes in `src/shinro/components.py`.
Everything else in the library is a concrete implementation of one of them:

| ABC | What it is | What it does |
|-----|------------|--------------|
| `Controller` | The "brain" | Turns a desired target and the current state into a command (`PID`, `LQR`, `MPC_LTI`, `MPPI`, ...) |
| `Plant` | The "body" | The thing being controlled — a robot, cart, pendulum. Reports its state, exposes a math model, steps one timestep |
| `StateEstimator` | The "eyes" | Reconstructs the true state from noisy sensor readings (`KalmanFilter`, `LuenbergerObserver`) |
| `TrajectoryGenerator` | The "route planner" | Produces a smooth reference path (`CubicPolynomial`, `QuinticPolynomial`, waypoints) |
| `PhysicsEngine` | The "world" | Runs a real simulation backend (MuJoCo) behind a plant |

The power of the design is that each component only knows the *contract* — the
abstract methods — not the other components' concrete types. So you can mix and
match any controller with any plant, estimator, trajectory, or physics engine
without rewriting anything.

## The dataflow: how a control loop actually runs

At each timestep, values flow through the components in a fixed order:

```
TrajectoryGenerator ──x_ref──▶ Controller ──u──▶ Plant (physics)
                                    ▲               │
                    state_estimate │               │ state
                                    │               ▼
                              StateEstimator ◀── sensors / measurements
```

1. The **trajectory generator** asks: *where should we be right now?* It
   returns a reference `x_ref`.
2. The **state estimator** asks: *where are we actually?* It fuses the noisy
   sensor measurements into a best-guess state `x_hat`.
3. The **controller** compares `x_hat` to `x_ref` and computes a command `u`
   (torque, force, wheel speed).
4. The **plant** applies `u` and advances one timestep.
5. The new state produces new measurements → back to step 1.

This loop is the whole system. Everything else — the codegen pipeline, the MCP
server, the simulation factory — is a different way to *build* or *run* this
same loop.

## Layer 1: Config-driven construction

You almost never construct components by hand. Instead, you write a small TOML
file and let a **factory** build the component from it. For example
`src/shinro/configs/controllers/lqr_base.toml`:

```toml
type = "LQR"
name = "base_controller"
dt = 0.02
state_cost = [100.0, 100.0, 50.0]
control_cost = [0.1, 0.1, 0.1]
```

And in code:

```python
from shinro.factories.controller_factory import ControllerFactory

lqr = ControllerFactory("src/shinro/configs/controllers/lqr_base.toml").create()
```

How the factory works:

1. `ControllerFactory` loads the TOML and reads the `type` field (`"LQR"`).
2. It looks up that name in the **registry** (`src/shinro/factories/registry.py`) —
   a dict mapping registered names to classes.
3. It calls the class's `from_config(config, backend=...)`, which builds the
   component, computing derived quantities (e.g. LQR solves a Riccati equation
   for the gain `K` at construction time).

To add a new component type, you register it once:

```python
@register_controller("MyController")
class MyController(Controller):
    ...
```

Then it's immediately available to every factory, scenario, the MCP server,
and the tracer — no wiring code anywhere.

There is a factory per category (`ControllerFactory`, `EstimatorFactory`,
`TrajectoryFactory`, `PlantFactory`), all with the same shape. Configs live in
`src/shinro/configs/`, packaged with the library.

## Layer 2: Simulation — putting the loop together

`src/shinro/simulation/robotsim.py` assembles a whole scenario from a single
scenario TOML (e.g. `tests/integration/scenarios/base_tracking.toml`). A
scenario file declares:

- which plant to control,
- which controller and estimator to use (by config path),
- the trajectory to follow,
- input limits (clipping),
- the physics backend to use.

The sim factory builds each component via the factories above, wires them into
the closed-loop dataflow, and runs it — either against a real physics engine
(MuJoCo) for hardware-fidelity simulation, or against the plant's own
analytical `step()` for a fast in-memory simulation.

The key design point: a *scenario* is data, not code. You can swap the
controller from `LQR` to `MPC` by editing one line of a TOML file.

## Layer 3: Backend-agnostic math (`ArrayBackend`)

Controllers and plants never import `numpy` or `torch` directly — they go
through `ArrayBackend` (`src/shinro/utils/array_backend.py`). This is a thin
adapter exposing `array`, `zeros`, `eye`, `inv`, `clip`, etc. Both
`NumpyBackend` and `TorchBackend` implement it, and tests run the entire
suite against both (`conftest.py`). The same controller code runs on either
backend with zero changes — and this same seam is what makes tracing possible
(see the codec pipeline below).

## Layer 4: The codegen pipeline (compiling the loop)

The most distinctive piece is `src/shinro/codegen/`. Its goal: turn a
closed-loop step into a **static computation graph** that is compiled to a
native `.so` (via a Zig comptime-unrolled VM in `runtime/`) that provably
matches the Python math.

The mechanism is **tracing** (JAX/XLA style). See [`codegen.md`](./codegen.md)
for the full walkthrough. The short version:

1. **Trace** — run the component once with `Tracer` placeholder values instead
   of real arrays. Every `@`, `+`, `-`, `*`, `bk.inv(...)`, etc. is recorded
   as a node in a graph. Constants (gains, matrices) are baked in.
2. **Compose** — merge the per-component graphs into one graph for a single
   closed-loop tick, auto-inserting `reshape` nodes where shapes differ.
3. **Interpret** — replay the graph on real numpy inputs. If the replay
   matches the live numpy loop to `1e-12`, the trace is proven correct.
4. **Lower** — emit the graph as Zig compile-time constants
   (`runtime/graph_data.zig`) and run it through the comptime VM
   (`runtime/lower.zig`) to produce `base.so`. The VM is fixed at compile time:
   shapes and constants are baked, no heap allocation, no runtime dispatch.
   See [`codegen.md`](./codegen.md) and [`runtime/README.md`](../runtime/README.md).

Because the graph *is* a transcription of the Python execution, the only thing
that can go wrong at deployment is the small set of primitive operations —
each verified once against `NumpyBackend` and reused for every component.

This is the deployment path: run and verify the control law in Python, then
compile the exact same math into a small, fast, dependency-free native library.

## Layer 5: The MCP server (remote control)

`src/shinro/mcp/` exposes the entire library as a Model Context Protocol
server (`shinro-mcp`). An MCP client (Claude Desktop, Cline, custom hosts) can
create controllers, run them, build estimators, generate trajectories, and
analyze system properties — all over stdio, without importing the package.
See [`mcp_server.md`](./mcp_server.md) for the full tool reference.

The server is just another consumer of the same factories and components — it
holds a name → instance store and forwards tool calls to the objects.

## Dataflow summary

```
          TOML config                    runtime arrays
               │                              │
               ▼                              ▼
          Factories ──▶ Components ──▶ closed-loop step (numpy/torch)
               │                              │
               │                      ┌───────┴────────┐
               ▼                      ▼                ▼
   MCP server (remote)      codec trace        physics sim
               │            (graph)            (MuJoCo)
               │              │
               │              ▼
               │        interpret / lower to .so
```

Every entry point above reuses the same component classes, the same registry,
and the same config system. There is one set of math to reason about; the rest
is orchestration.
