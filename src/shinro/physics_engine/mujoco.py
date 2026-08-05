from pathlib import Path

import mujoco
import numpy as np

from shinro.components import PhysicsEngine

HERE = Path(__file__).parent.parent
MJCF_PATH = str(HERE / "lekiwi-sim" / "mjcf_lcmm_robot.xml")


class MuJoCoEngine(PhysicsEngine):
    """
    Low-level MuJoCo wrapper implementing the PhysicsEngine protocol.

    Holds the model + data, provides name-based read/write access to
    joint positions, velocities, and control signals. No hardcoded
    knowledge of any specific robot — all access is by name.

    ArmRobot and HolonomicMobileRobot each hold a reference to one engine
    and read/write their respective joints by name.
    """

    def __init__(self, model_path: str = MJCF_PATH, dt: float = 0.02, xml_string: str = None, assets: dict = None):
        if xml_string is not None:
            self.model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        self._dt = dt

        self._joint_name_to_id: dict[str, int] = {}
        self._joint_id_to_name: dict[int, str] = {}
        for jid in range(self.model.njnt):
            name = self.model.joint(jid).name
            self._joint_name_to_id[name] = jid
            self._joint_id_to_name[jid] = name

        self._actuator_name_to_id: dict[str, int] = {}
        for aid in range(self.model.nu):
            name = self.model.actuator(aid).name
            self._actuator_name_to_id[name] = aid

        self._body_name_to_id: dict[str, int] = {}
        for bid in range(self.model.nbody):
            name = self.model.body(bid).name
            self._body_name_to_id[name] = bid

        self.has_free_joint = (self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE.value)
        if self.has_free_joint:
            self.free_qvel_slice = slice(0, 6)
        else:
            self.free_qvel_slice = slice(0, 0)

        mujoco.mj_forward(self.model, self.data)

    def get_joint_qpos(self, name: str) -> float:
        jid = self._joint_name_to_id[name]
        qpos_adr = self.model.jnt_qposadr[jid]
        return float(self.data.qpos[qpos_adr])

    def set_joint_qpos(self, name: str, value: float):
        jid = self._joint_name_to_id[name]
        qpos_adr = self.model.jnt_qposadr[jid]
        self.data.qpos[qpos_adr] = value

    def get_joint_vel(self, name: str) -> float:
        jid = self._joint_name_to_id[name]
        dof_adr = self.model.jnt_dofadr[jid]
        return float(self.data.qvel[dof_adr])

    def set_joint_ctrl(self, name: str, value: float):
        aid = self._actuator_name_to_id[name]
        self.data.ctrl[aid] = value

    def get_joint_limits(self, name: str) -> tuple[float, float]:
        jid = self._joint_name_to_id[name]
        return (float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1]))

    def get_body_xpos(self, name: str) -> np.ndarray:
        bid = self._body_name_to_id[name]
        return self.data.xpos[bid].copy()

    def get_body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
        for bid in range(self.model.nbody):
            if name in self.model.body(bid).name:
                return bid
        return -1

    def compute_jacobian(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self._body_name_to_id[body_name]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.xpos[bid], bid)
        return jacp, jacr

    def compute_jacobian_for_joints(self, body_name: str, joint_names: list[str]) -> np.ndarray:
        bid = self._body_name_to_id[body_name]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.xpos[bid], bid)
        cols = []
        for name in joint_names:
            jid = self._joint_name_to_id[name]
            dof_adr = self.model.jnt_dofadr[jid]
            cols.append(dof_adr)
        return np.vstack([jacp[:, cols], jacr[:, cols]])

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def reset(self, qpos: np.ndarray | None = None):
        if qpos is not None:
            self.data.qpos[:] = qpos
        else:
            self.data.qpos[:] = 0.0
            if self.has_free_joint:
                self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_sensor_data(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": self.data.time,
        }

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_name_to_id.keys())

    @property
    def actuator_names(self) -> list[str]:
        return list(self._actuator_name_to_id.keys())

    @property
    def body_names(self) -> list[str]:
        return list(self._body_name_to_id.keys())

    def get_arm_qpos(self, arm_joint_names: list[str]) -> np.ndarray:
        return np.array([self.get_joint_qpos(n) for n in arm_joint_names])

    def get_drive_qpos(self, drive_joint_names: list[str]) -> np.ndarray:
        return np.array([self.get_joint_qpos(n) for n in drive_joint_names])

    def get_base_pose(self) -> np.ndarray:
        if self.has_free_joint:
            x = self.data.qpos[0]
            y = self.data.qpos[1]
            qw, qx, qy, qz = self.data.qpos[3:7]
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            return np.array([x, y, yaw])
        return np.array([0.0, 0.0, 0.0])

    def get_full_qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def set_arm_ctrl(self, targets: np.ndarray, arm_joint_names: list[str]):
        for name, val in zip(arm_joint_names, targets):
            limits = self.get_joint_limits(name)
            val = np.clip(val, limits[0], limits[1])
            self.set_joint_ctrl(name, val)

    def set_drive_ctrl(self, targets: np.ndarray, drive_joint_names: list[str]):
        for name, val in zip(drive_joint_names, targets):
            self.set_joint_ctrl(name, val)

    def set_full_ctrl(self, ctrl: np.ndarray):
        self.data.ctrl[:] = ctrl

    def print_info(self):
        print(f"MuJoCoEngine: {self.model.nq} qpos, {self.model.nv} vel, {self.model.nu} actuators")
        print(f"  nbody={self.model.nbody}, njnt={self.model.njnt}")
        print(f"  dt={self._dt}")
        print(f"  Joints: {self.joint_names}")
        print(f"  Actuators: {self.actuator_names}")
