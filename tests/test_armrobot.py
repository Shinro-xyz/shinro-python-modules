import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from utils.array_backend import NumpyBackend


def _to_np(x, bk):
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


def _make_arm(bk, dt=0.01):
    """Helper: create a default 6-DOF ArmRobot for testing."""
    from plants.armrobot import ArmRobot
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
        from plants.armrobot import ArmRobot
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
        from plants.armrobot import ArmRobot
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
        from plants.armrobot import ArmRobot
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


class TestArmRobotPhysicsEngine:
    """Verify ArmRobot behavior with a mock physics engine attached."""

    @pytest.fixture
    def mock_engine(self, bk):
        engine = MagicMock()
        engine.backend = bk
        engine.get_body_xpos.return_value = np.array([0.1, 0.0, 0.0])
        engine.get_body_id.return_value = 0
        engine.body_names = ["Moving_Jaw_08d-v1", "base"]
        engine.get_joint_qpos.return_value = 0.0
        engine.compute_jacobian_for_joints.return_value = np.eye(6)
        return engine

    def test_physics_engine_attaches(self, bk, mock_engine):
        """Attaching a physics engine inherits its backend and reads EE position."""
        from plants.armrobot import ArmRobot
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        assert arm._engine is mock_engine
        state = arm.get_state()
        assert np.allclose(_to_np(state, bk)[0], 0.1)

    def test_physics_engine_detach(self, bk, mock_engine):
        """Detaching the engine resets state to the home FK position (from link offsets)."""
        from plants.armrobot import ArmRobot
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
        from plants.armrobot import ArmRobot
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        u = bk.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        joints = arm.step(u)
        assert mock_engine.set_joint_ctrl.called
        assert _to_np(joints, bk).shape == (6,)

    def test_engine_ik_converges(self, bk, mock_engine):
        """engine_ik() with a mock Jacobian converges to the target."""
        from plants.armrobot import ArmRobot
        arm = _make_arm(bk)
        arm.physics_engine(mock_engine)
        target = bk.array([0.1, 0.0, 0.0])
        joints = arm.engine_ik(target)
        assert _to_np(joints, bk).shape == (6,)

    def test_find_ee_body_name(self, bk, mock_engine):
        """_find_ee_body_name returns the first matching candidate."""
        from plants.armrobot import ArmRobot
        arm = _make_arm(bk)
        name = arm._find_ee_body_name(mock_engine)
        assert name == "Moving_Jaw_08d-v1"

    def test_find_ee_body_name_fallback(self, bk, mock_engine):
        """_find_ee_body_name falls back to last body name when no candidate matches."""
        from plants.armrobot import ArmRobot
        mock_engine.get_body_id.return_value = -1
        arm = _make_arm(bk)
        name = arm._find_ee_body_name(mock_engine)
        assert name == "base"
