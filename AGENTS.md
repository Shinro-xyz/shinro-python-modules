# shinro-python-modules — AGENTS.md

This is now a pip-installable Python package. Source lives under `src/shinro/`.
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


## Project Overview

**Comprehensive control suite** for robotics — a modular library of controllers, plants, state estimators, and trajectory generators. Built on five abstract base classes: **Controller**, **Plant**, **StateEstimator**, **TrajectoryGenerator**, and **PhysicsEngine**.

### Design Goals

- **Modular**: Mix and match any controller with any plant, estimator, trajectory, or physics engine
- **Backend-agnostic**: Full numpy/torch support via `ArrayBackend` abstraction
- **Engine-agnostic**: Physics engine abstraction (MuJoCo, Isaac Lab, PyBullet, Drake, etc.)
- **Linear + Nonlinear**: LTI components for fast prototyping, nonlinear variants for real-world fidelity
- **Config-driven**: TOML-based component configuration via factory/registry pattern

### Current Components

| Category | Shipped | Planned / experimental |
|----------|---------|------------------------|
| **Controllers** | LQR, PID, MPC_LTI, MPC_DeltaU, MPPI, SMC, OnnxRL, LeRobotDiffusion | NMPC, iLQR, computed torque, adaptive |
| **Plants** | HolonomicMobileRobot, ArmRobot (6-DOF), InvertedPendulum, CartPole, DoublePendulum, Quadrotor (placeholder) | Unicycle, freeflyer |
| **Estimators** | KalmanFilter, LuenbergerObserver | EKF, UKF, particle filter |
| **Trajectories** | CubicPolynomial, QuinticPolynomial, waypoints, phase_list | B-spline, Lissajous, minimum-snap |
| **Physics Engines** | MuJoCo | Isaac Lab, PyBullet, Drake, null (no sim) |

> For the full per-component catalog (registered names, config files), see
> [`docs/components.md`](./docs/components.md).

### Key Files

| File | Purpose |
|------|---------|
| `src/shinro/components.py` | ABCs: Controller, Plant, StateEstimator, TrajectoryGenerator, PhysicsEngine |
| `src/shinro/controllers/mpc_lti.py` | MPC with OSQP QP solver — trajectory optimization |
| `src/shinro/controllers/lqr.py` | LQR with DARE solve — regulation/stabilization |
| `src/shinro/controllers/pid.py` | PID with anti-windup — joint-space position servo |
| `src/shinro/controllers/mppi.py` | Sampling-based Model Predictive Path Integral |
| `src/shinro/controllers/smc.py` | Sliding Mode Control — robust nonlinear |
| `src/shinro/controllers/onnx_rl_adapter.py` | ONNX neural-net policy adapter |
| `src/shinro/controllers/lerobot_adapter.py` | Learned policy adapter (diffusion, ACT, pi0) |
| `src/shinro/plants/holonomicmobilerobot.py` | 3-DOF base with omni-wheel kinematics |
| `src/shinro/plants/armrobot.py` | 6-DOF arm: FK, Jacobian, IK, Cartesian step |
| `src/shinro/plants/inverted_pendulum.py` | 2D inverted pendulum with analytical dynamics |
| `src/shinro/plants/cartpole.py` | 4D cart-pole with coupled dynamics |
| `src/shinro/plants/double_pendulum.py` | 4D planar double pendulum |
| `src/shinro/plants/quadrotor.py` | 12D quadrotor placeholder |
| `src/shinro/estimators/kalman_filter.py` | Discrete Kalman filter — predict-update cycle |
| `src/shinro/estimators/luenberger_observer.py` | Observer dynamics — x̂ = Ax̂ + Bu + L(y − Cx̂) |
| `src/shinro/trajectories/cubic_polynomial.py` | 3rd-order, position + velocity continuity |
| `src/shinro/trajectories/quintic_polynomial.py` | 5th-order, position + velocity + acceleration continuity |
| `src/shinro/codegen/` | Trace → compose → interpret → lower (tracing pipeline) |
| `src/shinro/codegen/lower_zig.py` | Serialize a composed graph to `runtime/graph_data.zig` |
| `runtime/` | Zig comptime VM + linalg kernels + generated graph (see `runtime/README.md`) |
| `scripts/gen_base.py` | Regenerate the `base_tracking` graph in `runtime/graph_data.zig` |
| `src/shinro/physics_engine/mujoco.py` | MuJoCo engine adapter |
| `src/shinro/simulation/robotsim.py` | Generic robot simulation factory (config-driven) |
| `lekiwi_sim.py` | Legacy LeKiwi simulation wrapper (not packaged) |
| `src/shinro/factories/registry.py` | Component registry + factory classes |
| `src/shinro/utils/array_backend.py` | NumpyBackend / TorchBackend abstraction |
| `src/shinro/utils/controllability_checker.py` | LTI system analysis (Gramians, balanced truncation) |
| `tests/integration/` | Full-loop physics-backed integration tests |

### Architecture

```
TrajectoryGenerator → Controller → Plant
StateEstimator → Controller (state feedback)
PhysicsEngine → Plant (simulation backend)
ArrayBackend → All components (numpy/torch)
```

Key design: the arm's `step()` takes a Cartesian velocity twist `[dx, dy, dz, droll, dpitch, dyaw]`, integrates it into a target pose, runs IK internally, and sends joint angles to servos. The controller **never touches joint space**.

### Key Parameters

**MPC_LTI:**
- Horizon: configurable (default ~10 steps)
- QP solver: OSQP
- State dimension: 3 (base) or 6 (arm Cartesian)
- Input dimension: 3 (base) or 6 (arm twist)

**LQR:**
- DARE solve for infinite-horizon gain
- Configurable Q, R weight matrices

**PID:**
- Joint-space position control with anti-windup
- Configurable Kp, Ki, Kd per joint

**ArmRobot:**
- 6-DOF: shoulder roll, shoulder pitch, elbow roll, elbow pitch, wrist roll, wrist pitch
- FK via homogeneous transforms, Jacobian via geometric method
- IK via damped pseudoinverse + step clamp

**HolonomicMobileRobot:**
- 3-DOF: x, y, θ
- Omni-wheel kinematics: velocity → individual wheel speeds

## Lab Notebooks

Experiment logs live in `lab-notes/daily/` in the repo. Read the files directly.

Each session's lab note MUST contain a semantic summary of the changes made — what was built, why, key design decisions, and test results. This is not a git log; it's a narrative record of intent and outcomes.

## Workflow

1. **Understand** — Read the source directly (`src/shinro/**`) and `lab-notes/daily/` for relevant context
2. **Plan** — Describe the change and which files to modify
3. **Implement** — Use OpenCode or direct editing
4. **Verify** — Run tests or check the output
5. **Document** — Write a semantic summary in `lab-notes/daily/<date>.md` covering what changed, why, and key results

> **Releases:** `make release-patch/minor/major` auto-regenerates `CHANGELOG.md`
> via `git cliff` (Conventional Commits, see `cliff.toml`) and stages it. Use
> Conventional Commit prefixes (`feat:`, `fix:`, `chore:`, ...) so the
> changelog stays clean — non-conforming commits land under "Other".

## Porting Numpy Scripts to the Framework

See the `port-numpy-component` skill (`.opencode/skills/port-numpy-component/SKILL.md`) — it loads on-demand via the `skill` tool when porting a raw numpy implementation into the framework.
