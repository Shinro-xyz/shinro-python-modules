# FILE: lekiwi_sim.py (MuJoCo physics engine)
"""
MuJoCo physics engine for the LeKiwi robot.

This is NOT a Plant subclass. It's a physics backend that the existing
ArmRobot and HolonomicMobileRobot use internally. Controllers talk to
those robot classes — they never touch MuJoCo directly.

Usage:
    from lekiwi_sim import MuJoCoEngine
    from armrobot import ArmRobot

    engine = MuJoCoEngine()
    arm = ArmRobot(num_dof=6, dt=0.02, ...)
    arm.attach_engine(engine)  # step() now uses MuJoCo physics

    # Or use the convenience wrapper:
    from lekiwi_sim import LeKiwiSim
    sim = LeKiwiSim()
    sim.arm.step(target_joints)   # MuJoCo-backed
    sim.base.step(target_velocity)  # MuJoCo-backed
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from typing import Optional, Callable

from components import Controller

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
MJCF_PATH = str(HERE / "lekiwi-sim" / "mjcf_lcmm_robot.xml")

# ── Joint / Actuator Index Constants ───────────────────────────────────────
# No free joint — base is welded to world, arm and drive joints only.
# Drive joints (3 omni wheels) are at qpos[0:3], ctrl[0:3].
# Arm joints (6-DOF + gripper) are at qpos[3:9], ctrl[3:9].
DRIVE_QPOS_SLICE = slice(0, 3)    # qpos[0:3]
ARM_QPOS_SLICE   = slice(3, 9)    # qpos[3:9]
DRIVE_CTRL_SLICE = slice(0, 3)    # ctrl[0:3]
ARM_CTRL_SLICE   = slice(3, 9)    # ctrl[3:9]

ARM_JOINT_NAMES   = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
DRIVE_JOINT_NAMES = [
    "ST3215_Servo_Motor-v1-2_Hub---Servo",
    "ST3215_Servo_Motor-v1-1_Hub-2---Servo",
    "ST3215_Servo_Motor-v1_Revolute-40",
]


# ── MuJoCoEngine ──────────────────────────────────────────────────────────
class MuJoCoEngine:
    """
    Low-level MuJoCo wrapper. Holds the model + data, provides read/write
    access to joint positions, velocities, and control signals.

    ArmRobot and HolonomicMobileRobot each hold a reference to one engine
    and read/write their respective slices of the shared state.
    """

    def __init__(self, model_path: str = MJCF_PATH, dt: float = 0.02):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        self.dt = dt

        # Cache arm joint limits (indices 3-8, after 3 drive joints)
        self.arm_limits = np.zeros((6, 2))
        for i in range(6):
            self.arm_limits[i] = self.model.jnt_range[3 + i, :]

        mujoco.mj_forward(self.model, self.data)

    # ── Read ───────────────────────────────────────────────────────────────

    def get_arm_qpos(self) -> np.ndarray:
        """6-dim arm joint positions."""
        return self.data.qpos[ARM_QPOS_SLICE].copy()

    def get_drive_qpos(self) -> np.ndarray:
        """3-dim drive joint positions."""
        return self.data.qpos[DRIVE_QPOS_SLICE].copy()

    def get_base_pose(self) -> np.ndarray:
        """3-dim base pose [x, y, yaw].
        
        Base is welded to world (no free joint), so pose is always [0, 0, 0].
        The kinematic base state is tracked in HolonomicMobileRobot.state.
        """
        return np.array([0.0, 0.0, 0.0])

    def get_full_qpos(self) -> np.ndarray:
        """Full joint position vector (9-dim: 3 drive + 6 arm)."""
        return self.data.qpos.copy()

    def get_sensor_data(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "base_pose": self.get_base_pose(),
            "arm_joints": self.get_arm_qpos(),
            "time": self.data.time,
        }

    # ── Write ──────────────────────────────────────────────────────────────

    def set_arm_ctrl(self, targets: np.ndarray):
        """Set arm actuator targets (6-dim, clipped to limits)."""
        targets = np.clip(targets, self.arm_limits[:, 0], self.arm_limits[:, 1])
        self.data.ctrl[ARM_CTRL_SLICE] = targets

    def set_drive_ctrl(self, targets: np.ndarray):
        """Set drive actuator targets (3-dim)."""
        self.data.ctrl[DRIVE_CTRL_SLICE] = targets

    def set_full_ctrl(self, ctrl: np.ndarray):
        """Set all 9 actuator targets at once."""
        self.data.ctrl[:] = ctrl

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self):
        """Advance physics by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def reset(self, qpos: Optional[np.ndarray] = None):
        """Reset simulation state."""
        if qpos is not None:
            self.data.qpos[:] = qpos
        else:
            self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def print_info(self):
        print(f"MuJoCoEngine: mjcf_lcmm_robot")
        print(f"  nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
        print(f"  nbody={self.model.nbody}, njnt={self.model.njnt}")
        print(f"  dt={self.dt}")
        print(f"  Arm joints: {ARM_JOINT_NAMES}")
        print(f"  Drive joints: {DRIVE_JOINT_NAMES}")
        print(f"  Arm limits:\n{self.arm_limits}")


# ── LeKiwiSim (convenience wrapper) ───────────────────────────────────────
class LeKiwiSim:
    """
    High-level wrapper that connects ArmRobot + HolonomicMobileRobot
    to a shared MuJoCoEngine.

    Usage:
        sim = LeKiwiSim()
        sim.arm.step(target_joints)       # MuJoCo-backed arm
        sim.base.step(target_velocity)     # MuJoCo-backed base
        sim.engine.step()                 # advance physics
        arm_state = sim.arm.get_state()   # read from MuJoCo
    """

    def __init__(self, dt: float = 0.02):
        from armrobot import ArmRobot
        from holonomicmobilerobot import HolonomicMobileRobot

        self.engine = MuJoCoEngine(dt=dt)

        # Arm: 6-DOF, limits from MuJoCo, offsets from the MJCF model
        # (these match the SO-ARM100 kinematics in arm_plant.py)
        link_offsets = np.array([
            [0.0388,  0.0,     0.0624],
            [-0.0304, -0.0183, -0.0542],
            [-0.1126, -0.028,   0.0],
            [-0.1349,  0.0052,  0.0],
            [0.0,    -0.0611,  0.0181],
            [0.0,     0.0,     0.05],
        ])
        # All arm joints in the MJCF rotate about Z in the local frame
        rot_axes = ["z"] * 6

        self.arm = ArmRobot(
            num_dof=6,
            dt=dt,
            joint_limits=self.engine.arm_limits,
            joint_offsets=link_offsets,
            rot_axes=rot_axes,
        )
        self.arm._engine = self.engine  # inject MuJoCo backend

        # Base: 3 omni wheels, 120° apart, ~0.09m wheel radius
        self.base = HolonomicMobileRobot(
            num_wheels=3,
            radius_robots=0.12,   # approximate distance center→wheel
            gamma=-np.pi / 2,     # first wheel angle from MJCF
            radius_wheels=0.09,
            dt=dt,
        )
        self.base._engine = self.engine  # inject MuJoCo backend

    def reset(self):
        self.engine.reset()
        self.base.state = np.zeros(3, dtype=np.float64)

    def step(self):
        """Advance physics. Call after setting arm/base targets."""
        self.engine.step()
        # Roll the wheels visually to match the kinematic base motion
        if hasattr(self.base, '_target_wheel_delta') and self.base._target_wheel_delta is not None:
            self.engine.data.qpos[0:3] += self.base._target_wheel_delta
            self.base._target_wheel_delta = None

    def get_state(self) -> dict:
        return self.engine.get_sensor_data()


# ── Simulation Runner ─────────────────────────────────────────────────────
def run_simulation(
    sim: LeKiwiSim,
    controller: Controller,
    max_steps: int = 1000,
    render: bool = True,
    callback: Optional[Callable] = None,
):
    """
    Run a closed-loop simulation with optional viewer.

    The controller receives the full 9-dim state and returns a 9-dim action.
    """
    sim.reset()

    if render:
        with mujoco.viewer.launch_passive(sim.engine.model, sim.engine.data) as viewer:
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -30

            for step in range(max_steps):
                state = sim.engine.get_full_qpos()
                action = controller.compute(state)
                sim.engine.set_full_ctrl(action)
                sim.engine.step()

                if callback:
                    callback(step, sim)

                viewer.sync()
                import time
                time.sleep(sim.engine.dt / 4)

                if not viewer.is_running():
                    break
    else:
        for step in range(max_steps):
            state = sim.engine.get_full_qpos()
            action = controller.compute(state)
            sim.engine.set_full_ctrl(action)
            sim.engine.step()

            if callback:
                callback(step, sim)


# ── Interactive Viewer ────────────────────────────────────────────────────
def interactive_viewer(model_path: str = MJCF_PATH):
    """Open an interactive MuJoCo viewer for manual inspection."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Interactive viewer opened. Close window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--viewer" in sys.argv:
        interactive_viewer()
    else:
        # Smoke test
        engine = MuJoCoEngine()
        engine.print_info()

        # Send a small arm movement (ctrl[3:9] = arm actuators)
        ctrl = np.zeros(9)
        ctrl[3] = 0.5   # Rotation
        ctrl[4] = -0.3  # Pitch
        ctrl[5] = 0.8   # Elbow

        for _ in range(50):
            engine.set_full_ctrl(ctrl)
            engine.step()

        print(f"\nAfter 50 steps with arm movement:")
        print(f"  Arm joints: {engine.get_arm_qpos()}")
        print(f"  Base pose:  {engine.get_base_pose()}")
        print("✅ MuJoCoEngine smoke test passed")
