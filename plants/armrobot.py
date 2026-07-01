import numpy as np
from typing import Optional, List
from components import Plant
import mujoco


class ArmRobot(Plant):
    """6-DOF robotic arm plant with forward kinematics, Jacobian, and IK.

    Models a serial-link manipulator with configurable joint offsets and
    rotation axes. Supports two modes:
    1. **Standalone** — uses simplified FK/IK for quick testing
    2. **MuJoCo** — attaches a physics engine for mesh-accurate Jacobian IK

    The arm operates in Cartesian space: step() takes a 6D velocity twist
    [dx, dy, dz, droll, dpitch, dyaw], integrates to a target pose, and
    uses inverse kinematics to compute joint angles.

    Usage:
        arm = ArmRobot(
            num_dof=6, dt=0.02,
            joint_limits=np.array([[-np.pi, np.pi]] * 6),
            joint_offsets=np.array([[...], [...], ...]),
            rot_axes=['y', 'z', 'z', 'x', 'z', 'z'],
        )
        arm.physics_engine(mujoco_engine)
        joints = arm.step(np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]))
    """

    def __init__(
        self,
        num_dof: int,
        dt: float,
        joint_limits: np.ndarray,
        joint_offsets: np.ndarray,
        rot_axes: List[str],
    ):
        """Initialize the ArmRobot.

        Args:
            num_dof: Number of degrees of freedom (joints).
            dt: Time step for state integration (s).
            joint_limits: Shape (num_dof, 2) array of [min, max] limits per joint.
            joint_offsets: Shape (num_dof, 3) translation offsets for each joint.
            rot_axes: List of rotation axes ('x', 'y', or 'z') for each joint.
        """
        self.num_dof = num_dof
        self.dt = dt
        self.state = np.zeros(6)
        self.joint_offsets = joint_offsets
        self.joint_limits = joint_limits
        self.axes = rot_axes
        self._last_joints = np.zeros(num_dof)
        self._engine = None
        self._arm_jac_start = None

    def _get_ee_pos(self):
        """Return [x, y, z] of end-effector from MuJoCo data."""
        return self._engine.data.xpos[self._ee_body_id].copy()

    def _compute_arm_jac_start(self):
        """Return the column offset in the full Jacobian for arm joints."""
        return 9 if self._engine.has_free_joint else 3

    def _get_ee_jacobian(self):
        """Return the 6×6 full Jacobian (position + orientation) for the arm.

        Uses MuJoCo's mj_jac to compute the exact mesh-based Jacobian at the
        end-effector body. Only the arm joint columns are extracted.

        Returns:
            Jacobian matrix (6, 6) — position and orientation rows.
        """
        jacp = np.zeros((3, self._engine.model.nv))
        jacr = np.zeros((3, self._engine.model.nv))
        mujoco.mj_jac(self._engine.model, self._engine.data, jacp, jacr,
                      self._engine.data.xpos[self._ee_body_id], self._ee_body_id)
        cols = slice(self._arm_jac_start, self._arm_jac_start + 6)
        return np.vstack([jacp[:, cols], jacr[:, cols]])

    def physics_engine(self, engine):
        """Attach a MuJoCo physics engine.

        After attachment, step() and get_state() use MuJoCo's exact FK
        and Jacobian for IK. The EE home position is initialized from the
        MuJoCo model.

        Args:
            engine: MuJoCoEngine instance or None to detach.
        """
        self._engine = engine
        if engine is not None:
            self._ee_body_id = self._find_ee_body_id(engine)
            self._arm_jac_start = self._compute_arm_jac_start()
            if self._ee_body_id >= 0:
                mujoco.mj_forward(engine.model, engine.data)
                ee = self._get_ee_pos()
                self.state = np.array([ee[0], ee[1], ee[2], 0.0, 0.0, 0.0])
        else:
            self._ee_body_id = -1
            self._arm_jac_start = None
            T_home, _, _ = self.forward_kinematics(np.zeros(self.num_dof))
            self.state = np.array([T_home[0, 3], T_home[1, 3], T_home[2, 3], 0.0, 0.0, 0.0])

    def _find_ee_body_id(self, engine):
        """Find the end-effector body ID in the MuJoCo model.

        Searches by name first ("Moving_Jaw_08d-v1"), then falls back to
        a substring search on all body names.

        Args:
            engine: MuJoCoEngine instance.

        Returns:
            Body ID of the end-effector, or last body if not found.
        """
        ee_body_id = mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw_08d-v1")
        if ee_body_id < 0:
            for bid in range(engine.model.nbody):
                if "Moving_Jaw" in engine.model.body(bid).name:
                    ee_body_id = bid
                    break
        return ee_body_id

    def get_state(self):
        """Return the current 6D end-effector pose [x, y, z, 0, 0, 0].

        When a physics engine is attached, reads the actual EE position
        from MuJoCo's xpos (reflects the latest IK/physics step).

        Returns:
            Pose vector (6,) — [x, y, z, roll, pitch, yaw] (orientation
            tracking is currently position-only, orientation set to 0).
        """
        if self._engine is not None:
            ee = self._get_ee_pos()
            return np.array([ee[0], ee[1], ee[2], 0.0, 0.0, 0.0])
        return self.state.copy()

    def get_model(self):
        """Get the discrete-time state-space model.

        Returns a simple integrator model: A = I₆, B = dt * I₆.

        Returns:
            Tuple of (A, B) where A = I₆ and B = dt * I₆.
        """
        A = np.eye(6)
        B = self.dt * np.eye(6)
        return A, B

    def _pose_to_transform(self, pose: np.ndarray):
        """Convert a 6D pose to a 4×4 homogeneous transformation matrix.

        Uses ZYX Euler angle convention: R = Rz @ Ry @ Rx.

        Args:
            pose: 6D pose [x, y, z, roll, pitch, yaw].

        Returns:
            4×4 transformation matrix.
        """
        x, y, z, roll, pitch, yaw = pose
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [x, y, z]
        return T

    def step(self, u: np.ndarray):
        """Update the system state based on control input.

        When a MuJoCo engine is attached:
        1. Integrates the velocity twist to a target EE position
        2. Uses damped least-squares IK on MuJoCo's exact Jacobian
        3. Sets joint positions in the physics engine

        Without an engine:
        1. Integrates state directly
        2. Uses simplified FK-based IK

        Args:
            u: 6D control input [dx, dy, dz, droll, dpitch, dyaw].

        Returns:
            Joint angles (num_dof,) clipped to limits.
        """
        if self._engine is not None:
            current_ee = self._get_ee_pos()
            target_ee = current_ee + u[:3] * self.dt
            joint_targets = self.mujoco_ik(target_ee)
            self._engine.set_arm_ctrl(joint_targets)
            self._last_joints = self._engine.get_arm_qpos()
            ee = self._get_ee_pos()
            self.state = np.array([ee[0], ee[1], ee[2], 0.0, 0.0, 0.0])
            return self._last_joints

        self.state += self.dt * u
        target = self._pose_to_transform(self.state)
        q = self.inverse_kinematics(target)
        self._last_joints = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return self._last_joints

    def _homogenous_transform(self, joint_angles: np.ndarray):
        """Compute individual joint transformation matrices.

        Each joint's transform is a rotation about its axis followed by a
        translation by its offset.

        Args:
            joint_angles: Current joint angles (num_dof,).

        Returns:
            Array of 4×4 transformation matrices, shape (num_dof, 4, 4).
        """
        sines, cosines = np.sin(joint_angles), np.cos(joint_angles)
        T = np.zeros((self.num_dof, 4, 4))
        for i in range(self.num_dof):
            axis = self.axes[i]
            s, c = sines[i], cosines[i]
            if axis == 'x':
                R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif axis == 'y':
                R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            elif axis == 'z':
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            else:
                raise ValueError(f"Invalid axis '{axis}' at joint {i}. Choose 'x', 'y', or 'z'.")
            offset_vector = self.joint_offsets[i, :3]
            T[i, :3, :3] = R
            T[i, :3, 3] = offset_vector
            T[i, 3, 3] = 1.0
        return T

    def forward_kinematics(self, joint_angles: np.ndarray):
        """Compute the end-effector pose and intermediate joint positions/axes.

        Chains homogeneous transforms from base to end-effector.

        Args:
            joint_angles: Input joint angles (num_dof,).

        Returns:
            Tuple of:
                T_cumulative: Final 4×4 end-effector transformation matrix.
                positions: List of 3D positions for each joint.
                axes: List of 3D rotation axes for each joint in world frame.
        """
        T_joints = self._homogenous_transform(joint_angles)
        T_cumulative = np.eye(4)
        positions = []
        axes = []
        for i in range(self.num_dof):
            T_cumulative = T_cumulative @ T_joints[i]
            axis_local = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[self.axes[i]]
            z_i = T_cumulative[:3, :3] @ axis_local
            positions.append(T_cumulative[:3, 3])
            axes.append(z_i)

        return T_cumulative, positions, axes

    def _jacobian(self, joint_angles: np.ndarray):
        """Compute the 6×N geometric Jacobian matrix.

        Maps joint velocities to end-effector spatial velocity:
            [v; ω] = J @ q̇

        Args:
            joint_angles: Current joint angles (num_dof,).

        Returns:
            Jacobian matrix (6, num_dof).
        """
        T_endeffector, pos, axes = self.forward_kinematics(joint_angles)
        p_endeffector = pos[-1]
        J = np.zeros((6, self.num_dof))
        for i in range(self.num_dof):
            J[:3, i] = np.cross(axes[i], p_endeffector - pos[i])
            J[3:, i] = axes[i]
        return J

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        max_iters: int = 100,
        q_init: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        max_step: float = 0.2,
    ):
        """Solve for joint angles using Jacobian pseudoinverse IK.

        Iterative Newton-Raphson method with damped pseudoinverse:
            q_{k+1} = q_k + J^† v
        where v is the 6D twist (position + orientation error).

        Args:
            target_pose: 4×4 target transformation matrix.
            max_iters: Maximum iterations for convergence.
            q_init: Initial joint angle guess. Defaults to last joint state.
            tol: Convergence tolerance for position and orientation error.
            max_step: Maximum joint step per iteration (rad).

        Returns:
            Joint angles (num_dof,) clipped to joint limits.
        """
        q = q_init if q_init is not None else self._last_joints.copy()
        for j in range(max_iters):
            T_cur, positions, axes = self.forward_kinematics(q)
            pos_err = target_pose[:3, 3] - T_cur[:3, 3]
            R_err = target_pose[:3, :3] @ T_cur[:3, :3].T
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))

            if angle < tol and np.linalg.norm(pos_err) < tol:
                break

            axis = np.array([R_err[2, 1] - R_err[1, 2],
                             R_err[0, 2] - R_err[2, 0],
                             R_err[1, 0] - R_err[0, 1]])

            if np.linalg.norm(axis) > 1e-6:
                ori_err = (axis / np.linalg.norm(axis)) * angle
            else:
                ori_err = np.zeros(3)

            v = np.concatenate([pos_err, ori_err])

            J = self._jacobian(q)
            dq = np.linalg.pinv(J) @ v
            dq = np.clip(dq, -max_step, max_step)
            q = q + dq
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return q

    def mujoco_ik(
        self,
        target_ee: np.ndarray,
        max_iters: int = 20,
        lam: float = 0.01,
        max_dq: float = 0.5,
    ):
        """Iterative damped least-squares IK using MuJoCo's exact Jacobian.

        Uses the real mesh Jacobian (mj_jac) rather than the simplified FK
        model. Sets qpos to the computed joints so position servos start
        close to target.

        The damped pseudoinverse is computed as:
            dq = J^T (J J^T + λ²I)^{-1} e

        Args:
            target_ee: Target end-effector position [x, y, z].
            max_iters: Maximum IK iterations.
            lam: Damping factor for pseudoinverse regularization.
            max_dq: Maximum joint change per iteration (rad).

        Returns:
            Joint targets (6-dim) clipped to limits.

        Raises:
            RuntimeError: If no MuJoCo engine is attached.
        """
        if self._engine is None:
            raise RuntimeError("mujoco_ik requires a MuJoCo engine (call physics_engine first)")

        engine = self._engine
        arm_qpos_slice = engine.arm_qpos_slice
        arm_limits = self.joint_limits

        current_ee = self._get_ee_pos()
        error = target_ee - current_ee

        if np.linalg.norm(error) < 0.001:
            return engine.get_arm_qpos()

        current_joints = engine.get_arm_qpos().copy()

        for _ in range(max_iters):
            J = self._get_ee_jacobian()[:3, :]

            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), error)
            dq = np.clip(dq, -max_dq, max_dq)
            current_joints = current_joints + dq
            current_joints = np.clip(current_joints, arm_limits[:, 0], arm_limits[:, 1])

            engine.data.qpos[arm_qpos_slice] = current_joints
            mujoco.mj_forward(engine.model, engine.data)

            current_ee = self._get_ee_pos()
            error = target_ee - current_ee
            if np.linalg.norm(error) < 0.001:
                break

        engine.data.qpos[arm_qpos_slice] = current_joints
        mujoco.mj_forward(engine.model, engine.data)

        return current_joints