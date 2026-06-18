# Lekiwi MPC — Whole-Body Control Framework


A clean, modular control framework built on three abstract base classes — **Controller**, **Plant**, and **StateEstimator** — with concrete implementations for the lekiwi robot's holonomic base and 6-DOF arm. Designed for the Shinro robotics IDE integration.

## Architecture

```mermaid
flowchart TB
    subgraph ABCs["Abstract Base Classes"]
        C[Controller ABC] -->|compute, reset| PID[PIDController]
        C --> MPC[MPC_LTI]
        C --> LQR[LQR]
        P[Plant ABC] -->|get_state, get_model, step| HMR[HolonomicMobileRobot]
        P --> AR[ArmRobot]
        SE[StateEstimator ABC]
    end

    subgraph Plants["Concrete Plants"]
        HMR -->|3-DOF [x, y, θ]| HMR_desc["Omni-wheel kinematics<br/>step: velocity → wheel speeds"]
        AR -->|6-DOF Cartesian| AR_desc["FK · Jacobian · IK<br/>step: integrate pose → IK → joints"]
        AR --> SO[SO-ARM100]
        SO -->|position-controlled servos| SO_desc["Inherits ArmRobot<br/>just configures params"]
    end

    subgraph Controllers["Concrete Controllers"]
        PID --> PID_desc["Joint-space position<br/>Anti-windup"]
        MPC --> MPC_desc["Trajectory optimization<br/>OSQP QP solver"]
        LQR --> LQR_desc["Regulation / stabilization<br/>DARE solve"]
    end

    Controllers -->|u = control input| Plants
    Plants -->|state feedback| Controllers

    subgraph Sim["Simulation"]
        MJ[MuJoCo] -->|joint angles| Plants
    end

    subgraph Integration["Planned"]
        SH[Shinro IDE] --> Controllers
    end
```

## Key Design Decisions

### Cartesian Arm Abstraction
The arm's `step()` method takes a Cartesian velocity twist `[dx, dy, dz, droll, dpitch, dyaw]`, integrates it into a target pose, runs inverse kinematics internally, and sends joint angles to the servos. The controller **never touches joint space** — it thinks it's controlling a 6-DOF Cartesian plant.

```
Controller: [dx, dy, dz, droll, dpitch, dyaw]
    ↓
step(u): state += dt·u → _pose_to_transform() → IK → joint angles
    ↓
Servos
```

### Clean Separation of Concerns
- **Controller** = algorithm (MPC, PID, LQR)
- **Plant** = what you're controlling (base, arm)
- **StateEstimator** = what you measure (extensible)

Swap any controller onto any plant — same interface.

### Verified Kinematics
- Forward kinematics via homogeneous transforms
- Geometric Jacobian verified against numerical differentiation (max error: 0.00018)
- Inverse kinematics with damped pseudoinverse + step clamp (converges from zero to any reachable pose)
- Full round-trip test: Cartesian state → IK → FK → state matches

## Project Structure

```
lerobot-mpc-lekiwi/
├── components.py              # ABCs: Controller, Plant, StateEstimator
├── holonomicmobilerobot.py    # 3-DOF base with omni-wheel kinematics
├── armrobot.py                # 6-DOF arm: FK, Jacobian, IK, Cartesian step
├── pid.py                     # PID controller with anti-windup
├── mpc_lti.py                 # MPC with OSQP QP solver
├── lqr.py                     # LQR with DARE solve
├── ARCHITECTURE.html          # Interactive architecture diagram
├── lekiwi-sim/                # MuJoCo simulation files
│   ├── mjcf_lcmm_robot.xml    # Full robot model
│   ├── so_arm100.xml          # SO-ARM100 arm model
│   └── meshes/                # STL meshes for all parts
└── README.md
```

## Quick Start

```bash
# Clone
git clone https://github.com/adilfaisal01/lerobot-mpc-lekiwi
cd lerobot-mpc-lekiwi

# Dependencies
pip install numpy scipy osqp

# Run a test
python3 -c "
import numpy as np
from armrobot import ArmRobot

robot = ArmRobot(
    num_dof=6, dt=0.01,
    joint_limits=np.array([[-np.pi, np.pi]]*6),
    joint_offsets=np.array([[0,0,0.2],[0,0,0],[0.3,0,0],[0,0,0],[0.25,0,0],[0,0,0]]),
    rot_axes=['z','y','y','x','y','x']
)

# Move in X
for _ in range(20):
    robot.step(np.array([0.05, 0, 0, 0, 0, 0]))

T, _, _ = robot.forward_kinematics(robot._last_joints)
print(f'End effector at: {T[:3, 3]}')
"
```

## Controllers

| Controller | Plant | Use Case |
|-----------|-------|----------|
| **PID** | Arm (joint space) | Position servo — send joint angles directly |
| **MPC_LTI** | Base (3D) | Trajectory optimization for holonomic drive |
| **MPC_LTI** | Arm (6D Cartesian) | End-effector trajectory — IK handles joint math |
| **LQR** | Base (3D) | Regulation / stabilization |

## Status

- ✅ Base kinematics (3-DOF holonomic)
- ✅ Arm kinematics (6-DOF: FK, Jacobian, IK)
- ✅ Cartesian state + IK-in-step pipeline
- ✅ PID, MPC_LTI, LQR controllers
- 🔄 Combined state space (base + arm coupling) — *in progress*
- 🔄 MuJoCo closed-loop simulation — *in progress*
- 🔄 Shinro IDE integration — *planned*
