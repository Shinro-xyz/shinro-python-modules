# Component Catalog

A reference of every shinro component, its **registered name** (the `type`
string used in TOML configs and by the factories/registry), its source file,
and its bundled config. Registered names are the contract — config files and
the factories dispatch on them.

How to read: the **registered name** is what you put in a `type = "..."` TOML
field or pass to a factory. The **class** is the Python class behind it.
Several registered names may map to the same class (e.g. `MPC_LTI` and
`MPC_DeltaU` are two registration names of the same `MPC_LTI_DeltaU` class).

## Controllers

Registered via `register_controller`; created with `ControllerFactory`. All
implement `compute(current, target)` and `reset()`.

| Registered name | Class | File | Config |
|-----------------|-------|------|--------|
| `LQR` | `LQR` | `controllers/lqr.py` | `configs/controllers/lqr_base.toml` |
| `PID` | `PIDController` | `controllers/pid.py` | `configs/controllers/pid_arm.toml` |
| `MPC_LTI` | `MPC_LTI_Base` | `controllers/mpc_lti.py` | `configs/controllers/mpc_lti_base.toml` |
| `MPC_DeltaU` | `MPC_LTI_DeltaU` | `controllers/mpc_lti.py` | `configs/controllers/mpc_base.toml` |
| `MPPI` | `MPPIController` | `controllers/mppi.py` | `configs/controllers/mppi_base.toml`, `mppi.toml` |
| `SMC` | `SlidingModeController` | `controllers/smc.py` | `configs/controllers/smc.toml` |
| `onnx_rl` | `OnnxRLAdapter` | `controllers/onnx_rl_adapter.py` | `configs/controllers/onnx_rl.toml` |
| `lerobot_diffusion` | `LeRobotDiffusionAdapter` | `controllers/lerobot_adapter.py` | `configs/controllers/lerobot_diffusion.toml` |

> `MPC_LTI` and `MPC_DeltaU` are both built on `MPC_LTI` in `mpc_lti.py`:
> `MPC_DeltaU` adds Δu (control-rate) regularization. The two names are
> distinct registrations, not aliases.

## Plants

Registered via `register_plant`; created with `PlantFactory`. All implement
`get_state()`, `get_model()`, and `step(u)`. Plants that support
linearization set `input_dim` and expose `dynamics(x, u)` (see
`docs/how-it-works.md` and `utils/linearization.py`).

| Registered name | Class | File | Config |
|-----------------|-------|------|--------|
| `ArmRobot` | `ArmRobot` | `plants/armrobot.py` | `configs/plants/armrobot.toml` |
| `HolonomicMobileRobot` | `HolonomicMobileRobot` | `plants/holonomicmobilerobot.py` | `configs/plants/holonomic_base.toml` |
| `InvertedPendulum` | `InvertedPendulum` | `plants/inverted_pendulum.py` | `configs/plants/inverted_pendulum.toml` |
| `CartPole` | `CartPole` | `plants/cartpole.py` | `configs/plants/cartpole.toml` |
| `DoublePendulum` | `DoublePendulum` | `plants/double_pendulum.py` | `configs/plants/double_pendulum.toml` |
| `Quadrotor` | `Quadrotor` | `plants/quadrotor.py` | `configs/plants/quadrotor.toml` (placeholder) |

> `cartpole.toml` and `inverted_pendulum.toml` are parameter-only configs
> (they carry physical constants like mass/length/gravity and no `type`
> field); the other plant configs are full factory configs.

## Estimators

Registered via `register_estimator`; created with `EstimatorFactory`. All
implement `estimate(measurement, control_input)` and `reset()`.

| Registered name | Class | File | Config |
|-----------------|-------|------|--------|
| `KalmanFilter` | `KalmanFilter` | `estimators/kalman_filter.py` | `configs/estimators/kalman_base.toml`, `kalman_arm.toml` |
| `LuenbergerObserver` | `LuenbergerObserver` | `estimators/luenberger_observer.py` | `configs/estimators/luenberger_base.toml` |

## Trajectories

Registered via `register_trajectory`; created with `TrajectoryFactory`. All
implement `generate(...)` and `position_at(t)`.

| Registered name | Class | File | Config |
|-----------------|-------|------|--------|
| `cubic_segments` | `CubicPolynomial` | `trajectories/cubic_polynomial.py` | `configs/trajectories/arm_extension.toml` |
| `quintic_segments` | `QuinticPolynomial` / `QuinticPolynomialConfigAdapter` | `trajectories/quintic_polynomial.py` | `configs/trajectories/arm_quintic.toml` |
| `waypoints` | `WaypointSchedule` | `trajectories/quintic_polynomial.py` | `configs/trajectories/arm_lift.toml`, `base_straight.toml`, `base_triangle.toml` |
| `phase_list` | `PhaseSchedule` | `trajectories/quintic_polynomial.py` | `configs/trajectories/pick_and_place.toml` |

## Physics engines

| Registered name | Class | File |
|-----------------|-------|------|
| MuJoCo | `MuJoCoEngine` | `physics_engine/mujoco.py` |

The engine ABC is `PhysicsEngine` in `components.py`; engines attach to plants
via `plant.physics_engine(engine)`. MuJoCo requires the optional
`pip install -e ".[mujoco]"`.

## Utilities

Not registry-based; import directly from `shinro.utils`.

| Symbol | Module | Purpose |
|--------|--------|---------|
| `ArrayBackend`, `NumpyBackend`, `TorchBackend` | `utils/array_backend.py` | Backend-agnostic array abstraction; `parse_matrix` converts TOML lists to matrices |
| `BatchedDynamicsAdapter` | `utils/batched_adapter.py` | Batches N parallel trajectory rollouts for sampling-based controllers (MPPI) |
| `linearize`, `linearize_plant` | `utils/linearization.py` | Numeric linearization of plant dynamics around an operating point |
| `LTISystemsAnalyzer` | `utils/controllability_checker.py` | Controllability/observability, Gramians, balanced truncation |
| `resolve_config_path`, `get_config_path` | `utils/config_resolver.py` | Resolve config paths inside or outside the installed package |

## Codegen

Tracing/composition/lowering pipeline. See `docs/codegen.md` for the full
walkthrough.

| Symbol | Module | Purpose |
|--------|--------|---------|
| `Tracer`, `Graph`, `Node` | `codegen/tracing.py` | Abstract values + graph records; operator overloads emit nodes |
| `TraceBackend` | `codegen/trace_backend.py` | Recording `ArrayBackend` used during tracing |
| `trace_node`, `trace_node_with_state` | `codegen/trace_node.py` | Trace one component call |
| `compose` | `codegen/compose.py` | Merge per-component graphs into one closed-loop tick |
| `interpret`, `interpret_step` | `codegen/interpreter.py` | Replay a graph on real numpy inputs (correctness oracle) |
| `register_op`, `available_ops` | `codegen/ops.py` | Op-handler registry |
| `lower_zig` | `codegen/lower_zig.py` | Serialize a composed graph to `runtime/graph_data.zig` |

## Simulation & MCP

| Symbol | Module | Purpose |
|--------|--------|---------|
| `RobotSim`, `ScenarioFactory`, `Scenario` | `simulation/robotsim.py`, `factories/scenario_factory.py` | Config-driven closed-loop simulation; scenarios live in `tests/integration/scenarios/*.toml` |
| `shinro-mcp` (console script) | `mcp/server.py` | MCP server exposing factories + analysis as tools (see `docs/mcp_server.md`) |
