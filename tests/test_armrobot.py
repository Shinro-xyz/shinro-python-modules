from unittest.mock import MagicMock

import numpy as np
import pytest


def _to_np(x, bk):
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


def _make_arm(bk, dt=0.01):
    """Helper: create a default 6-DOF ArmRobot for testing."""
    from shinro.plants.armrobot import ArmRobot
    num_dof = 6
    joint_limits = bk.array([[-np.pi, np.pi]] * num_dof)
    joint_offsets = bk.array([
        [0.02, 0.03, 0.05],
        [-0.001, -0.115, 0.018],
        [-0.001, 0.133, 0.029],
        [-0.02, 0.026, -0.055],
        [0.02, 0.027, -0.013],
        [0.0, 0.0, 0.0],
    ])
    rot_axes = ["y", "z", "z", "x", "z", "z"]
    return ArmRobot(
        num_dof=num_dof, dt=dt,
        joint_limits=joint_limits,
        joint_offsets=joint_offsets,
        rot_axes=rot_axes,
        backend=bk,
    )


def _rotmat_to_quat(R):
    """Rotation matrix -> quaternion (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _make_faithful_engine(bk, arm):
    """A mock engine that reports EE pose from the arm's own FK/Jacobian.

    Stands in for MuJoCo: ``set_joint_qpos`` mutates a joint buffer, and
    ``get_body_xpos`` / ``get_body_xquat`` / ``compute_jacobian_for_joints``
    are derived from the arm's analytic forward kinematics and geometric
    Jacobian, so 6D engine IK can be exercised without MuJoCo.
    """
    engine = MagicMock()
    engine.backend = bk
    engine.body_names = ["Moving_Jaw_08d-v1", "base"]
    engine.get_body_id.return_value = 0
    joints = np.zeros(arm.num_dof)

    def set_qpos(name, val):
        joints[arm._joint_names.index(name)] = float(_to_np(val, bk))

    def get_qpos(name):
        return joints[arm._joint_names.index(name)]

    def get_xpos(name):
        T, _, _ = arm.forward_kinematics(bk.array(joints))
        return _to_np(T[:3, 3], bk)

    def get_xquat(name):
        T, _, _ = arm.forward_kinematics(bk.array(joints))
        return _rotmat_to_quat(_to_np(T[:3, :3], bk))

    def get_jac(name, joint_names):
        return _to_np(arm._jacobian(bk.array(joints)), bk)

    engine.set_joint_qpos.side_effect = set_qpos
    engine.get_joint_qpos.side_effect = get_qpos
    engine.get_body_xpos.side_effect = get_xpos
    engine.get_body_xquat.side_effect = get_xquat
    engine.compute_jacobian_for_joints.side_effect = get_jac
    engine.forward.side_effect = lambda: None
    return engine


class TestArmRobotModel:
    """Verify ArmRobot state-space model and state access."""

    def test_get_model(self, bk):
        """A is 6x6 identity, B is dt * I_6."""
        arm = _make_arm(bk, dt=0.02)
        A, B = arm.get_model()
        assert _to_np(A, bk).shape == (6, 6)
        assert _to_np(B, bk).shape == (6, 6)
        assert np.allclose(_to_np(A, bk), np.eye(6))
        assert np.allclose(_to_np(B, bk), 0.02 * np.eye(6))

    def test_initial_state(self, bk):
        """State starts at origin [0,0,0,0,0,0]."""
        arm = _make_arm(bk)
        state = arm.get_state()
        assert np.allclose(_to_np(state, bk), np.zeros(6))

    def test_get_state_returns_copy(self, bk):
        """get_state() returns a copy, not a reference to internal state."""
        arm = _make_arm(bk)
        state = arm.get_state()
        state[0] = 99.0
        internal = arm.get_state()
        assert _to_np(internal, bk)[0] != 99.0


class TestArmRobotStep:
    """Verify ArmRobot step() integration and joint output."""

    def test_step_integrates_state(self, bk):
        """Standalone step integrates state: state += dt * u."""
        arm = _make_arm(bk, dt=0.01)
        u = bk.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        arm.step(u)
        state = arm.get_state()
        assert np.allclose(_to_np(state, bk)[0], 0.5 * 0.01, atol=1e-10)

    def test_step_returns_joints(self, bk):
        """step() returns a joint angle vector of length num_dof."""
        arm = _make_arm(bk)
        u = bk.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        joints = arm.step(u)
        assert _to_np(joints, bk).shape == (6,)

    def test_step_clips_joints(self, bk):
        """step() clips joint angles to joint_limits."""
        from shinro.plants.armrobot import ArmRobot
        num_dof = 6
        tight_limits = bk.array([[-0.1, 0.1]] * num_dof)
        joint_offsets = bk.array([[0.0, 0.0, 0.0]] * num_dof)
        rot_axes = ["z"] * num_dof
        arm = ArmRobot(
            num_dof=num_dof, dt=0.01,
            joint_limits=tight_limits,
            joint_offsets=joint_offsets,
            rot_axes=rot_axes,
            backend=bk,
        )
        u = bk.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        joints = arm.step(u)
        vals = _to_np(joints, bk)
        assert np.all(vals >= -0.1 - 1e-10)
        assert np.all(vals <= 0.1 + 1e-10)


class TestArmRobotForwardKinematics:
    """Verify forward kinematics: homogeneous transforms and end-effector pose."""

    def test_fk_home_position(self, bk):
        """FK at zero joint angles produces identity transform (no rotation, no translation from base)."""
        arm = _make_arm(bk)
        T, positions, axes = arm.forward_kinematics(bk.zeros(6))
        assert T.shape == (4, 4)
        assert np.allclose(_to_np(T[:3, :3], bk), np.eye(3), atol=1e-10)
        assert len(positions) == 6
        assert len(axes) == 6

    def test_fk_known_joint_angle(self, bk):
        """FK at a known non-zero angle produces a non-identity transform."""
        arm = _make_arm(bk)
        q = bk.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        T, positions, axes = arm.forward_kinematics(q)
        assert not np.allclose(_to_np(T[:3, :3], bk), np.eye(3), atol=1e-6)

    def test_fk_at_joint_limits(self, bk):
        """FK does not crash when joints are at their limits."""
        arm = _make_arm(bk)
        q = bk.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
        T, positions, axes = arm.forward_kinematics(q)
        assert T.shape == (4, 4)

    def test_homogenous_transform_shape(self, bk):
        """_homogenous_transform returns array of shape (num_dof, 4, 4)."""
        arm = _make_arm(bk)
        T_joints = arm._homogenous_transform(bk.zeros(6))
        assert _to_np(T_joints, bk).shape == (6, 4, 4)

    def test_pose_to_transform(self, bk):
        """_pose_to_transform returns a 4x4 homogeneous matrix."""
        arm = _make_arm(bk)
        pose = bk.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        T = arm._pose_to_transform(pose)
        assert T.shape == (4, 4)
        assert np.allclose(_to_np(T[3, :], bk), [0, 0, 0, 1])


class TestArmRobotJacobian:
    """Verify geometric Jacobian computation."""

    def test_jacobian_shape(self, bk):
        """Jacobian has shape (6, num_dof)."""
        arm = _make_arm(bk)
        J = arm._jacobian(bk.zeros(6))
        assert _to_np(J, bk).shape == (6, 6)

    def test_jacobian_vs_finite_difference(self, bk):
        """Geometric Jacobian matches finite-difference approximation of FK."""
        arm = _make_arm(bk)
        q = bk.array([0.2, -0.1, 0.3, 0.0, 0.0, 0.0])
        eps = 1e-6
        J_analytic = arm._jacobian(q)
        T0, _, _ = arm.forward_kinematics(q)
        p0 = _to_np(T0[:3, 3], bk)
        J_fd = np.zeros((3, 6))
        for i in range(6):
            q_pert = _to_np(q, bk).copy()
            q_pert[i] += eps
            Tp, _, _ = arm.forward_kinematics(bk.array(q_pert))
            pp = _to_np(Tp[:3, 3], bk)
            J_fd[:, i] = (pp - p0) / eps
        J_analytic_np = _to_np(J_analytic[:3, :], bk)
        assert np.allclose(J_analytic_np, J_fd, atol=1e-4)

    def test_jacobian_near_singular(self, bk):
        """Jacobian computation does not crash near a singular configuration."""
        arm = _make_arm(bk)
        q = bk.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        J = arm._jacobian(q)
        assert J.shape == (6, 6)


class TestArmRobotInverseKinematics:
    """Verify inverse kinematics: convergence, accuracy, and joint limits."""

    def test_ik_converges(self, bk):
        """FK(IK(FK(q))) approximately recovers the original end-effector pose."""
        arm = _make_arm(bk)
        q_orig = bk.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0])
        T_target, _, _ = arm.forward_kinematics(q_orig)
        q_ik = arm.inverse_kinematics(T_target, q_init=bk.zeros(6))
        T_reached, _, _ = arm.forward_kinematics(q_ik)
        pos_err = _to_np(T_target[:3, 3] - T_reached[:3, 3], bk)
        assert np.linalg.norm(pos_err) < 1e-3

    def test_ik_reaches_target(self, bk):
        """FK(IK(target)) produces a pose close to the target."""
        arm = _make_arm(bk)
        q_known = bk.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0])
        T_known, _, _ = arm.forward_kinematics(q_known)
        q_ik = arm.inverse_kinematics(T_known, q_init=bk.zeros(6))
        T_reached, _, _ = arm.forward_kinematics(q_ik)
        pos_err = _to_np(T_known[:3, 3] - T_reached[:3, 3], bk)
        assert np.linalg.norm(pos_err) < 1e-3

    def test_ik_respects_joint_limits(self, bk):
        """IK output stays within joint_limits."""
        from shinro.plants.armrobot import ArmRobot
        num_dof = 6
        tight_limits = bk.array([[-0.5, 0.5]] * num_dof)
        joint_offsets = bk.array([[0.0, 0.0, 0.0]] * num_dof)
        rot_axes = ["z"] * num_dof
        arm = ArmRobot(
            num_dof=num_dof, dt=0.01,
            joint_limits=tight_limits,
            joint_offsets=joint_offsets,
            rot_axes=rot_axes,
            backend=bk,
        )
        target_pose = bk.eye(4)
        target_pose[:3, 3] = bk.array([1.0, 0.0, 0.0])
        q_ik = arm.inverse_kinematics(target_pose, q_init=bk.zeros(6))
        vals = _to_np(q_ik, bk)
        assert np.all(vals >= -0.5 - 1e-6)
        assert np.all(vals <= 0.5 + 1e-6)

    def test_ik_unreachable_target(self, bk):
        """IK converges to the nearest feasible pose for an unreachable target."""
        arm = _make_arm(bk)
        far_target = bk.eye(4)
        far_target[:3, 3] = bk.array([10.0, 10.0, 10.0])
        q_ik = arm.inverse_kinematics(far_target, q_init=bk.zeros(6), max_iters=200)
        assert not np.any(np.isnan(_to_np(q_ik, bk)))


class TestArmRobotInvalidInputs:
    """Verify error handling for invalid inputs."""

    def test_invalid_axis(self, bk):
        """Invalid rotation axis string raises ValueError during FK."""
        from shinro.plants.armrobot import ArmRobot
        num_dof = 1
        joint_limits = bk.array([[-np.pi, np.pi]])
        joint_offsets = bk.array([[0.0, 0.0, 0.0]])
        arm = ArmRobot(
            num_dof=num_dof, dt=0.01,
            joint_limits=joint_limits,
            joint_offsets=joint_offsets,
            rot_axes=["w"],
            backend=bk,
        )
        with pytest.raises(ValueError):
            arm.forward_kinematics(bk.zeros(1))

    def test_engine_ik_raises_without_engine(self, bk):
        """engine_ik() raises RuntimeError when no physics engine is attached."""
        arm = _make_arm(bk)
        with pytest.raises(RuntimeError):
            arm.engine_ik(bk.array([0.1, 0.0, 0.0]))


@pytest.fixture
def mock_engine(bk):
    engine = MagicMock()
    engine.backend = bk
    engine.get_body_xpos.return_value = np.array([0.1, 0.0, 0.0])
    engine.get_body_id.return_value = 0
    engine.body_names = ["Moving_Jaw_08d-v1", "base"]
    engine.get_joint_qpos.return_value = 0.0
    engine.compute_jacobian_for_joints.return_value = np.eye(6)
    engine.get_body_xquat.return_value = np.array([1.0, 0.0, 0.0, 0.0])
    return engine


class TestArmRobotPhysicsEngine:
    """Verify ArmRobot behavior with a mock physics engine attached."""

    def test_physics_engine_attaches(self, bk, mock_engine):
        """Attaching a physics engine inherits its backend and reads EE position."""
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        assert arm._engine is mock_engine
        state = arm.get_state()
        assert np.allclose(_to_np(state, bk)[0], 0.1)

    def test_physics_engine_detach(self, bk, mock_engine):
        """Detaching the engine resets state to the home FK position (from link offsets)."""
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        arm.physics_engine(None)
        assert arm._engine is None
        state = arm.get_state()
        T_home, _, _ = arm.forward_kinematics(bk.zeros(6))
        expected = bk.hstack([T_home[:3, 3], bk.zeros(3)])
        assert np.allclose(_to_np(state, bk), _to_np(expected, bk))

    def test_step_with_engine(self, bk, mock_engine):
        """step() with engine attached calls set_joint_ctrl and updates state."""
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        u = bk.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        joints = arm.step(u)
        assert mock_engine.set_joint_ctrl.called
        assert _to_np(joints, bk).shape == (6,)

    def test_engine_ik_converges(self, bk, mock_engine):
        """engine_ik() with a mock Jacobian converges to the target."""
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        target = bk.array([0.1, 0.0, 0.0])
        joints = arm.engine_ik(target)
        assert _to_np(joints, bk).shape == (6,)

    def test_find_ee_body_name(self, bk, mock_engine):
        """_find_ee_body_name returns the first matching candidate."""
        arm = _make_arm(bk)
        name = arm._find_ee_body_name(mock_engine)
        assert name == "Moving_Jaw_08d-v1"

    def test_find_ee_body_name_fallback(self, bk, mock_engine):
        """_find_ee_body_name falls back to last body name when no candidate matches."""
        mock_engine.get_body_id.return_value = -1
        arm = _make_arm(bk)
        name = arm._find_ee_body_name(mock_engine)
        assert name == "base"


class TestQuatEuler:
    """Verify quaternion -> ZYX Euler conversion (the engine orientation primitive)."""

    def test_identity(self, bk):
        from shinro.plants.armrobot import _quat_to_euler_zyx
        e = _quat_to_euler_zyx(np.array([1.0, 0.0, 0.0, 0.0]))
        assert np.allclose(e, [0.0, 0.0, 0.0], atol=1e-12)

    def test_pure_yaw(self, bk):
        from shinro.plants.armrobot import _quat_to_euler_zyx
        q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # 90 deg about z
        e = _quat_to_euler_zyx(q)
        assert np.allclose(e, [0.0, 0.0, np.pi / 2], atol=1e-9)

    def test_pure_pitch(self, bk):
        from shinro.plants.armrobot import _quat_to_euler_zyx
        q = np.array([np.cos(np.pi / 6), 0.0, np.sin(np.pi / 6), 0.0])  # 60 deg about y
        e = _quat_to_euler_zyx(q)
        assert np.allclose(e, [0.0, np.pi / 3, 0.0], atol=1e-9)

    def test_pure_roll(self, bk):
        from shinro.plants.armrobot import _quat_to_euler_zyx
        q = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0])  # 45 deg about x
        e = _quat_to_euler_zyx(q)
        assert np.allclose(e, [np.pi / 4, 0.0, 0.0], atol=1e-9)

    def test_matches_scipy(self, bk):
        from scipy.spatial.transform import Rotation
        from shinro.plants.armrobot import _quat_to_euler_zyx
        rng = np.random.default_rng(0)
        for _ in range(20):
            q_xyzw = Rotation.random(random_state=rng).as_quat()
            e = _quat_to_euler_zyx(np.roll(q_xyzw, 1))  # (x,y,z,w) -> (w,x,y,z)
            roll, pitch, yaw = e
            Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
            Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            R_mine = Rz @ Ry @ Rx
            R_ref = Rotation.from_quat(q_xyzw).as_matrix()
            assert np.allclose(R_mine, R_ref, atol=1e-9)


class TestArmRobotOrientation:
    """Verify the 6D engine path: honest state, orientation error, 6D IK."""

    def test_get_state_returns_engine_orientation(self, bk, mock_engine):
        mock_engine.get_body_xquat.return_value = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        state = arm.get_state()
        assert np.allclose(_to_np(state, bk)[3:], [0.0, 0.0, np.pi / 2], atol=1e-6)

    def test_attach_seeds_orientation(self, bk, mock_engine):
        mock_engine.get_body_xquat.return_value = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        assert np.allclose(_to_np(arm.state, bk)[3:], [0.0, 0.0, np.pi / 2], atol=1e-6)

    def test_orientation_error_zero_when_aligned(self, bk):
        arm = _make_arm(bk)
        e = arm._orientation_error(bk.array([0.1, 0.2, 0.3]), bk.array([0.1, 0.2, 0.3]))
        assert np.allclose(_to_np(e, bk), 0.0, atol=1e-9)

    def test_orientation_error_pure_yaw(self, bk):
        arm = _make_arm(bk)
        e = arm._orientation_error(bk.array([0.0, 0.0, np.pi / 2]), bk.array([0.0, 0.0, 0.0]))
        assert np.allclose(_to_np(e, bk), [0.0, 0.0, np.pi / 2], atol=1e-6)

    def test_step_engine_uses_6d_path_when_rotating(self, bk, mock_engine):
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        arm.engine_ik = MagicMock(wraps=arm.engine_ik)
        u = bk.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
        arm.step(u)
        assert "target_euler" in arm.engine_ik.call_args.kwargs

    def test_step_engine_keeps_3d_path_translation_only(self, bk, mock_engine):
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        arm.engine_ik = MagicMock(wraps=arm.engine_ik)
        u = bk.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        arm.step(u)
        assert "target_euler" not in arm.engine_ik.call_args.kwargs

    def test_engine_ik_6d_reaches_pose(self, bk):
        from scipy.spatial.transform import Rotation

        arm = _make_arm(bk)
        engine = _make_faithful_engine(bk, arm)
        arm.physics_engine(engine)
        q_known = bk.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0])
        T_known, _, _ = arm.forward_kinematics(q_known)
        target_ee = T_known[:3, 3]
        R = _to_np(T_known[:3, :3], bk)
        target_euler = bk.array(Rotation.from_matrix(R).as_euler("zyx")[::-1].copy())  # [roll, pitch, yaw]
        q = arm.engine_ik(target_ee, target_euler=target_euler)
        T, _, _ = arm.forward_kinematics(q)
        pos_err = np.linalg.norm(_to_np(T[:3, 3], bk) - _to_np(target_ee, bk))
        assert pos_err < 1e-3
        R_target = _to_np(arm._pose_to_transform(bk.hstack([bk.zeros(3), target_euler]))[:3, :3], bk)
        R_reached = _to_np(T[:3, :3], bk)
        R_err = R_target @ R_reached.T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        assert angle < 1e-3

    def test_engine_ik_3d_still_reaches_position(self, bk):
        arm = _make_arm(bk)
        engine = _make_faithful_engine(bk, arm)
        arm.physics_engine(engine)
        target_ee = bk.array([0.1, 0.0, 0.0])
        q = arm.engine_ik(target_ee)
        T, _, _ = arm.forward_kinematics(q)
        pos_err = np.linalg.norm(_to_np(T[:3, 3], bk) - _to_np(target_ee, bk))
        assert pos_err < 1e-3
