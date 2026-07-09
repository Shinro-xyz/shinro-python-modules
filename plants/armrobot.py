from typing import Optional, List
from components import Plant, PhysicsEngine
from factories.registry import register_plant
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_plant("ArmRobot")
class ArmRobot(Plant):
    """6-DOF robotic arm plant with forward kinematics, Jacobian, and IK.

    Models a serial-link manipulator with configurable joint offsets and
    rotation axes. Supports two modes:

    1. **Standalone** — uses simplified FK/IK for quick testing.
    2. **Physics engine** — attaches a PhysicsEngine for mesh-accurate
       Jacobian IK via MuJoCo.

    The arm operates in **Cartesian space**: ``step()`` takes a 6D velocity
    twist :math:`[dx, dy, dz, d\\text{roll}, d\\text{pitch}, d\\text{yaw}]`,
    integrates to a target pose, and uses inverse kinematics to compute
    joint angles. The controller **never touches joint space**.

    Forward kinematics uses homogeneous transforms (4x4) chained per joint.
    The Jacobian is computed via the geometric method (cross product of
    joint axes with position vectors). Inverse kinematics uses damped
    pseudoinverse with step clamping and joint limit enforcement.

    Args:
        num_dof: Number of degrees of freedom.
        dt: Time step in seconds.
        joint_limits: Array of shape (num_dof, 2) with [min, max] per joint.
        joint_offsets: Array of shape (num_dof, 3) with link offset vectors.
        rot_axes: List of rotation axes per joint (``"x"``, ``"y"``, or ``"z"``).
        joint_names: Optional list of joint name strings.
        ee_body_name: Optional name of the end-effector body in the physics engine.
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        num_dof: int,
        dt: float,
        joint_limits,
        joint_offsets,
        rot_axes: List[str],
        joint_names: Optional[List[str]] = None,
        ee_body_name: Optional[str] = None,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.num_dof = num_dof
        self.dt = dt
        self.state = self.bk.zeros(6)
        self.joint_offsets = joint_offsets
        self.joint_limits = joint_limits
        self.axes = rot_axes
        self._last_joints = self.bk.zeros(num_dof)
        self._engine = None
        self._ee_body_name = ee_body_name
        self._joint_names = joint_names if joint_names is not None else [f"joint_{i}" for i in range(num_dof)]

    def _get_ee_pos(self):
        """Get the end-effector position from the physics engine.

        Returns:
            Position vector (3,) from the engine.
        """
        return self._engine.get_body_xpos(self._ee_body_name)

    def _get_ee_jacobian(self):
        """Get the end-effector Jacobian from the physics engine.

        Returns:
            Jacobian matrix of shape (6, num_dof) from the engine.
        """
        return self._engine.compute_jacobian_for_joints(self._ee_body_name, self._joint_names)

    def physics_engine(self, engine: Optional[PhysicsEngine]):
        """Attach a physics engine.

        When attached, the backend is inherited from the engine, the
        end-effector body name is auto-detected, and the initial state
        is read from the engine.

        When detached, the standalone FK is used for state initialization.

        Args:
            engine: PhysicsEngine instance or None to detach.
        """
        self._engine = engine
        if engine is not None:
            self.bk = engine.backend
            if self._ee_body_name is None:
                self._ee_body_name = self._find_ee_body_name(engine)
            self._engine.forward()
            ee = self._get_ee_pos()
            self.state = self.bk.array([ee[0], ee[1], ee[2], 0.0, 0.0, 0.0])
        else:
            T_home, _, _ = self.forward_kinematics(self.bk.zeros(self.num_dof))
            self.state = self.bk.array([T_home[0, 3], T_home[1, 3], T_home[2, 3], 0.0, 0.0, 0.0])

    def _find_ee_body_name(self, engine: PhysicsEngine) -> str:
        """Auto-detect the end-effector body name from the engine.

        Checks a list of common names and returns the first match.

        Args:
            engine: PhysicsEngine instance.

        Returns:
            Body name string.
        """
        candidates = ["Moving_Jaw_08d-v1", "Moving_Jaw", "end_effector", "ee", "gripper"]
        for name in candidates:
            bid = engine.get_body_id(name)
            if bid >= 0:
                return name
        return engine.body_names[-1]

    def get_state(self):
        """Get the current end-effector pose.

        When a physics engine is attached, reads the EE position from the
        engine. Otherwise returns the internally tracked state.

        Returns:
            State vector (6,) — [x, y, z, 0, 0, 0] (orientation always 0).
        """
        if self._engine is not None:
            ee = self._get_ee_pos()
            return self.bk.array([ee[0], ee[1], ee[2], 0.0, 0.0, 0.0])
        return self.bk.copy(self.state)

    def get_model(self):
        """Get the discrete-time state-space model.

        Returns:
            Tuple of (A, B) where A = I_6 and B = dt * I_6.
        """
        A = self.bk.eye(6)
        B = self.dt * self.bk.eye(6)
        return A, B

    def _pose_to_transform(self, pose):
        """Convert a 6D pose vector to a 4x4 homogeneous transform.

        Uses ZYX Euler angle convention (yaw-pitch-roll).

        Args:
            pose: Pose vector (6,) — [x, y, z, roll, pitch, yaw].

        Returns:
            Homogeneous transform matrix (4, 4).
        """
        x, y, z, roll, pitch, yaw = pose
        Rx = self.bk.array([[1, 0, 0], [0, self.bk.cos(roll), -self.bk.sin(roll)], [0, self.bk.sin(roll), self.bk.cos(roll)]])
        Ry = self.bk.array([[self.bk.cos(pitch), 0, self.bk.sin(pitch)], [0, 1, 0], [-self.bk.sin(pitch), 0, self.bk.cos(pitch)]])
        Rz = self.bk.array([[self.bk.cos(yaw), -self.bk.sin(yaw), 0], [self.bk.sin(yaw), self.bk.cos(yaw), 0], [0, 0, 1]])
        T = self.bk.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = self.bk.array([x, y, z])
        return T

    def step(self, u):
        """Execute one control step.

        When a physics engine is attached:
        - Integrates the Cartesian velocity twist to get a target EE position.
        - Runs ``engine_ik()`` to compute joint angles.
        - Sends joint angles to the engine via ``set_joint_ctrl``.

        When standalone:
        - Integrates the state directly.
        - Runs the standalone ``inverse_kinematics()``.
        - Clips joint angles to limits.

        Args:
            u: Control input (6,) — [dx, dy, dz, droll, dpitch, dyaw].

        Returns:
            Joint angle vector (num_dof,).
        """
        if self._engine is not None:
            current_ee = self._get_ee_pos()
            target_ee = current_ee + u[:3] * self.dt
            joint_targets = self.engine_ik(target_ee)
            for name, val in zip(self._joint_names, joint_targets):
                self._engine.set_joint_ctrl(name, val)
            self._last_joints = self.bk.array([self._engine.get_joint_qpos(n) for n in self._joint_names])
            ee = self._get_ee_pos()
            self.state = self.bk.array([ee[0], ee[1], ee[2], 0.0, 0.0, 0.0])
            return self._last_joints

        self.state = self.state + self.dt * u
        target = self._pose_to_transform(self.state)
        q = self.inverse_kinematics(target)
        self._last_joints = self.bk.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return self._last_joints

    def _homogenous_transform(self, joint_angles):
        """Compute per-joint homogeneous transforms.

        Each joint's transform encodes a rotation about its axis and a
        translation by its link offset.

        Args:
            joint_angles: Joint angle vector (num_dof,).

        Returns:
            Array of shape (num_dof, 4, 4) — one transform per joint.
        """
        sines = self.bk.sin(joint_angles)
        cosines = self.bk.cos(joint_angles)
        T = self.bk.zeros((self.num_dof, 4, 4))
        for i in range(self.num_dof):
            axis = self.axes[i]
            s, c = sines[i], cosines[i]
            if axis == 'x':
                R = self.bk.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis == 'y':
                R = self.bk.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            elif axis == 'z':
                R = self.bk.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            else:
                raise ValueError(f"Invalid axis '{axis}' at joint {i}. Choose 'x', 'y', or 'z'.")
            offset_vector = self.joint_offsets[i, :3]
            T[i, :3, :3] = R
            T[i, :3, 3] = offset_vector
            T[i, 3, 3] = 1.0
        return T

    def forward_kinematics(self, joint_angles):
        """Compute forward kinematics for a given joint configuration.

        Chains the per-joint homogeneous transforms and returns the
        end-effector transform, joint positions, and joint axes.

        Args:
            joint_angles: Joint angle vector (num_dof,).

        Returns:
            Tuple of (T_ee, positions, axes):
            - T_ee: End-effector homogeneous transform (4, 4).
            - positions: List of joint position vectors (num_dof,) each (3,).
            - axes: List of joint axis direction vectors (num_dof,) each (3,).
        """
        T_joints = self._homogenous_transform(joint_angles)
        T_cumulative = self.bk.eye(4)
        positions = []
        axes = []
        for i in range(self.num_dof):
            T_cumulative = T_cumulative @ T_joints[i]
            axis_local = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[self.axes[i]]
            z_i = T_cumulative[:3, :3] @ self.bk.array(axis_local)
            positions.append(T_cumulative[:3, 3])
            axes.append(z_i)
        return T_cumulative, positions, axes

    def _jacobian(self, joint_angles):
        """Compute the geometric Jacobian.

        The position part of each column is the cross product of the joint
        axis with the vector from the joint to the end-effector. The
        orientation part is the joint axis itself.

        Args:
            joint_angles: Joint angle vector (num_dof,).

        Returns:
            Jacobian matrix of shape (6, num_dof).
        """
        T_endeffector, pos, axes = self.forward_kinematics(joint_angles)
        p_endeffector = pos[-1]
        J = self.bk.zeros((6, self.num_dof))
        for i in range(self.num_dof):
            J[:3, i] = self.bk.cross(axes[i], p_endeffector - pos[i])
            J[3:, i] = axes[i]
        return J

    def inverse_kinematics(
        self,
        target_pose,
        max_iters: int = 100,
        q_init=None,
        tol: float = 1e-4,
        max_step: float = 0.2,
    ):
        """Compute inverse kinematics via damped pseudoinverse.

        Iteratively minimizes the pose error (position + orientation) using
        the geometric Jacobian. The orientation error is computed from the
        rotation matrix trace.

        Args:
            target_pose: Target homogeneous transform (4, 4).
            max_iters: Maximum number of IK iterations.
            q_init: Initial joint angle guess. Defaults to last joint angles.
            tol: Convergence tolerance on position and orientation error.
            max_step: Maximum joint angle change per iteration.

        Returns:
            Joint angle vector (num_dof,) that achieves the target pose.
        """
        q = q_init if q_init is not None else self.bk.copy(self._last_joints)
        for j in range(max_iters):
            T_cur, positions, axes = self.forward_kinematics(q)
            pos_err = target_pose[:3, 3] - T_cur[:3, 3]
            R_err = target_pose[:3, :3] @ T_cur[:3, :3].T
            angle = self.bk.arccos(self.bk.clip((self.bk.trace(R_err) - 1) / 2, -1, 1))

            if angle < tol and self.bk.norm(pos_err) < tol:
                break

            axis = self.bk.array([R_err[2, 1] - R_err[1, 2],
                                  R_err[0, 2] - R_err[2, 0],
                                  R_err[1, 0] - R_err[0, 1]])

            if self.bk.norm(axis) > 1e-6:
                ori_err = (axis / self.bk.norm(axis)) * angle
            else:
                ori_err = self.bk.zeros(3)

            v = self.bk.hstack([pos_err, ori_err])

            J = self._jacobian(q)
            dq = self.bk.pinv(J) @ v
            dq = self.bk.clip(dq, -max_step, max_step)
            q = q + dq
            q = self.bk.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return q

    def engine_ik(
        self,
        target_ee,
        max_iters: int = 20,
        lam: float = 0.01,
        max_dq: float = 0.5,
    ):
        """Compute inverse kinematics using the physics engine's Jacobian.

        Uses the engine's mesh-accurate Jacobian (position rows only) with
        damped least squares. Sets joint positions in the engine and calls
        ``forward()`` each iteration.

        Args:
            target_ee: Target end-effector position (3,).
            max_iters: Maximum IK iterations.
            lam: Damping factor for the least squares solve.
            max_dq: Maximum joint angle change per iteration.

        Returns:
            Joint angle vector (num_dof,) that achieves the target position.

        Raises:
            RuntimeError: If no physics engine is attached.
        """
        if self._engine is None:
            raise RuntimeError("engine_ik requires a physics engine (call physics_engine first)")

        current_ee = self._get_ee_pos()
        error = target_ee - current_ee

        if self.bk.norm(error) < 0.001:
            return self.bk.array([self._engine.get_joint_qpos(n) for n in self._joint_names])

        current_joints = self.bk.array([self._engine.get_joint_qpos(n) for n in self._joint_names])

        for _ in range(max_iters):
            J = self._get_ee_jacobian()[:3, :]

            JJT = J @ J.T
            dq = J.T @ self.bk.solve(JJT + lam**2 * self.bk.eye(3), error)
            dq = self.bk.clip(dq, -max_dq, max_dq)
            current_joints = current_joints + dq
            current_joints = self.bk.clip(current_joints, self.joint_limits[:, 0], self.joint_limits[:, 1])

            for name, val in zip(self._joint_names, current_joints):
                self._engine.set_joint_qpos(name, val)
            self._engine.forward()

            current_ee = self._get_ee_pos()
            error = target_ee - current_ee
            if self.bk.norm(error) < 0.001:
                break

        for name, val in zip(self._joint_names, current_joints):
            self._engine.set_joint_qpos(name, val)
        self._engine.forward()

        return current_joints

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create an ArmRobot from a TOML config dict.

        Config fields:
            joint_group: Name of the joint group in ``joint_groups``.
            num_dof: Number of degrees of freedom.
            dt: Time step.
            joint_offsets: List of link offset vectors.
            rot_axes: List of rotation axes.
            ee_body_name: Optional end-effector body name.
            engine: PhysicsEngine instance to attach.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            ArmRobot instance.
        """
        bk = backend or NumpyBackend()
        joint_names = config["joint_groups"][config["joint_group"]]
        engine = config["engine"]
        limits = np.array([engine.get_joint_limits(n) for n in joint_names])
        num_dof = config["num_dof"]
        plant = cls(
            num_dof=num_dof,
            dt=config["dt"],
            joint_limits=bk.from_numpy(limits),
            joint_offsets=bk.from_numpy(np.array(config["joint_offsets"])),
            rot_axes=config["rot_axes"],
            joint_names=joint_names,
            ee_body_name=config.get("ee_body_name"),
            backend=bk,
        )
        plant.physics_engine(engine)
        return plant
