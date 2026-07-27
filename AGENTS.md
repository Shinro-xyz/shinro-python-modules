# shinro-python-modules — AGENTS.md

This project uses a **SQLite codebase index** stored in the repo itself. Query the index to understand the codebase before making changes — no need to keep the full codebase in context.

## Codebase Index

**Location:** `.codebase/codebase_index.db` (in-repo, versionable)
**Last indexed:** 2026-07-05

### How to query

```bash
# List all files with summaries
python3 ~/.hermes/scripts/codebase_indexer.py --list --repo lerobot-mpc-lekiwi

# Search for specific files or concepts
python3 ~/.hermes/scripts/codebase_indexer.py --query "MPC" --repo lerobot-mpc-lekiwi
python3 ~/.hermes/scripts/codebase_indexer.py --query "arm" --repo lerobot-mpc-lekiwi
python3 ~/.hermes/scripts/codebase_indexer.py --query "kalman" --repo lerobot-mpc-lekiwi

# List lab notebook entries
python3 ~/.hermes/scripts/codebase_indexer.py --vault --repo lerobot-mpc-lekiwi

# Direct SQLite queries for deeper analysis
python3 -c "import sqlite3; c=sqlite3.connect('.codebase/codebase_index.db'); [print(r) for r in c.execute('SELECT path, purpose FROM files WHERE ext=\".py\" ORDER BY path')]"
```

### Re-index after changes

```bash
python3 ~/.hermes/scripts/codebase_indexer.py .
```

## Project Overview

**Comprehensive control suite** for robotics — a modular library of controllers, plants, state estimators, and trajectory generators. Built on five abstract base classes: **Controller**, **Plant**, **StateEstimator**, **TrajectoryGenerator**, and **PhysicsEngine**.

### Design Goals

- **Modular**: Mix and match any controller with any plant, estimator, trajectory, or physics engine
- **Backend-agnostic**: Full numpy/torch support via `ArrayBackend` abstraction
- **Engine-agnostic**: Physics engine abstraction (MuJoCo, Isaac Lab, PyBullet, Drake, etc.)
- **Linear + Nonlinear**: LTI components for fast prototyping, nonlinear variants for real-world fidelity
- **Config-driven**: TOML-based component configuration via factory/registry pattern

### Current Components

| Category | Linear | Nonlinear (planned) |
|----------|--------|---------------------|
| **Controllers** | LQR, PID, MPC (LTI), LeRobot diffusion | NMPC, iLQR, MPPI, computed torque, sliding mode, adaptive |
| **Plants** | HolonomicMobileRobot, ArmRobot (double integrator), InvertedPendulum, CartPole | Quadrotor, unicycle, double pendulum |
| **Estimators** | Kalman filter, Luenberger observer | EKF, UKF, particle filter |
| **Trajectories** | Cubic, Quintic, waypoint/phase schedules | B-spline, Lissajous, minimum-snap |
| **Physics Engines** | MuJoCo | Isaac Lab, PyBullet, Drake, null (no sim) |

### Key Files

| File | Purpose |
|------|---------|
| `components.py` | ABCs: Controller, Plant, StateEstimator, TrajectoryGenerator, PhysicsEngine |
| `controllers/mpc_lti.py` | MPC with OSQP QP solver — trajectory optimization |
| `controllers/lqr.py` | LQR with DARE solve — regulation/stabilization |
| `controllers/pid.py` | PID with anti-windup — joint-space position servo |
| `controllers/lerobot_adapter.py` | Learned policy adapter (diffusion, ACT, pi0) |
| `plants/holonomicmobilerobot.py` | 3-DOF base with omni-wheel kinematics |
| `plants/armrobot.py` | 6-DOF arm: FK, Jacobian, IK, Cartesian step |
| `plants/inverted_pendulum.py` | 2D inverted pendulum with analytical dynamics |
| `plants/cartpole.py` | 4D cart-pole with coupled dynamics |
| `plants/quadrotor.py` | 12D quadrotor placeholder |
| `estimators/kalman_filter.py` | Discrete Kalman filter — predict-update cycle |
| `estimators/luenberger_observer.py` | Observer dynamics — x̂ = Ax̂ + Bu + L(y − Cx̂) |
| `trajectories/cubic_polynomial.py` | 3rd-order, position + velocity continuity |
| `trajectories/quintic_polynomial.py` | 5th-order, position + velocity + acceleration continuity |
| `physics_engine/mujoco.py` | MuJoCo engine adapter |
| `simulation/robotsim.py` | Generic robot simulation factory (config-driven) |
| `lekiwi_sim.py` | Legacy LeKiwi simulation wrapper |
| `factories/registry.py` | Component registry + factory classes |
| `utils/array_backend.py` | NumpyBackend / TorchBackend abstraction |
| `utils/controllability_checker.py` | LTI system analysis (Gramians, balanced truncation) |
| `test_pick_and_place.py` | Integration tests |

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

Experiment logs live in `lab-notes/daily/` in the repo. Query them with `--vault` flag.

Each session's lab note MUST contain a semantic summary of the changes made — what was built, why, key design decisions, and test results. This is not a git log; it's a narrative record of intent and outcomes.

## Workflow

1. **Understand** — Query the index for relevant files + lab notes
2. **Plan** — Describe the change and which files to modify
3. **Implement** — Use OpenCode or direct editing
4. **Verify** — Run tests or check the output
5. **Document** — Write a semantic summary in `lab-notes/daily/<date>.md` covering what changed, why, and key results
6. **Re-index** — `python3 ~/.hermes/scripts/codebase_indexer.py .`
