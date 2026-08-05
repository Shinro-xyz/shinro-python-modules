# FILE: lekiwi_sim.py (LeKiwi-specific simulation)
"""
LeKiwi-specific convenience wrapper. Kept for backward compatibility.

The generic RobotSim now lives in simulation/robotsim.py.
"""

from pathlib import Path

import mujoco
import numpy as np

from physics_engine.mujoco import MuJoCoEngine
from simulation.robotsim import RobotSim  # noqa: F401 — re-export

HERE = Path(__file__).parent
MJCF_PATH = str(HERE / "lekiwi-sim" / "mjcf_lcmm_robot.xml")


class LeKiwiSim:
    """
    High-level wrapper that connects ArmRobot + HolonomicMobileRobot
    to a shared MuJoCoEngine.

    Usage:
        sim = LeKiwiSim()
        sim.arm.step(target_joints)
        sim.base.step(target_velocity)
        sim.step()
        arm_state = sim.arm.get_state()
    """

    ARM_JOINT_NAMES   = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    DRIVE_JOINT_NAMES = [
        "ST3215_Servo_Motor-v1-2_Hub---Servo",
        "ST3215_Servo_Motor-v1-1_Hub-2---Servo",
        "ST3215_Servo_Motor-v1_Revolute-40",
    ]

    def __init__(self, dt: float = 0.02, xml_string: str = None, assets: dict = None):
        from plants.armrobot import ArmRobot
        from plants.holonomicmobilerobot import HolonomicMobileRobot

        self.engine = MuJoCoEngine(dt=dt, xml_string=xml_string, assets=assets)

        rot_axes = ["y", "z", "z", "x", "z", "z"]

        link_offsets = np.array([
            [0.018300,  0.030600,  0.052200],
            [-0.001500, -0.114582,  0.018082],
            [-0.001500,  0.132932,  0.028720],
            [-0.020100,  0.025822, -0.055375],
            [0.019800,  0.026631, -0.013098],
            [0.0,        0.0,       0.0],
        ])

        arm_limits = np.array([self.engine.get_joint_limits(n) for n in self.ARM_JOINT_NAMES])

        self.arm = ArmRobot(
            num_dof=6, dt=dt, joint_limits=arm_limits, joint_offsets=link_offsets,
            rot_axes=rot_axes, joint_names=self.ARM_JOINT_NAMES,
            ee_body_name="Moving_Jaw_08d-v1",
        )
        self.arm.physics_engine(self.engine)

        self.base = HolonomicMobileRobot(
            num_wheels=3, radius_robots=0.12, gamma=-np.pi / 2, radius_wheels=0.09, dt=dt,
        )
        self.base.physics_engine(self.engine)

    def reset(self):
        self.engine.reset()
        self.base.state = np.zeros(3, dtype=np.float64)

    def step(self):
        self.engine.step()
        base_state = self.base.state
        self.engine.data.qpos[0] = base_state[0]
        self.engine.data.qpos[1] = base_state[1]
        self.engine.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.engine.data.qvel[:6] = 0.0
        if hasattr(self.base, '_target_wheel_delta') and self.base._target_wheel_delta is not None:
            for name, delta in zip(self.DRIVE_JOINT_NAMES, self.base._target_wheel_delta):
                jid = self.engine._joint_name_to_id[name]
                qpos_adr = self.engine.model.jnt_qposadr[jid]
                self.engine.data.qpos[qpos_adr] += delta
            self.base._target_wheel_delta = None

    def get_state(self) -> dict:
        return self.engine.get_sensor_data()


if __name__ == "__main__":
    import sys

    if "--viewer" in sys.argv:
        engine = MuJoCoEngine()
        model = engine.model
        data = engine.data
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Interactive viewer opened. Close window to exit.")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    else:
        engine = MuJoCoEngine()
        engine.print_info()

        ctrl = np.zeros(engine.model.nu)
        ctrl[3] = 0.5
        ctrl[4] = -0.3
        ctrl[5] = 0.8

        for _ in range(50):
            engine.set_full_ctrl(ctrl)
            engine.step()

        print("\nAfter 50 steps with arm movement:")
        print(f"  Arm joints: {engine.get_arm_qpos(LeKiwiSim.ARM_JOINT_NAMES)}")
        print(f"  Base pose:  {engine.get_base_pose()}")
        print("✅ MuJoCoEngine smoke test passed")
