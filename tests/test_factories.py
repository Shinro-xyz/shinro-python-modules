
import pytest


def _to_np(x, bk):
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestRegistry:
    """Verify all four registries contain the expected entries."""

    def test_controller_registry(self):
        """Controller registry contains LQR, PID, MPC_LTI, MPC_DeltaU, lerobot_diffusion."""
        from shinro.factories.registry import _CONTROLLER_REGISTRY
        expected = {"LQR", "PID", "MPC_LTI", "MPC_DeltaU", "lerobot_diffusion", "onnx_rl"}
        assert expected.issubset(_CONTROLLER_REGISTRY.keys())

    def test_estimator_registry(self):
        """Estimator registry contains KalmanFilter and LuenbergerObserver."""
        from shinro.factories.registry import _ESTIMATOR_REGISTRY
        assert "KalmanFilter" in _ESTIMATOR_REGISTRY
        assert "LuenbergerObserver" in _ESTIMATOR_REGISTRY

    def test_trajectory_registry(self):
        """Trajectory registry contains cubic_segments, quintic_segments, waypoints, phase_list."""
        from shinro.factories.registry import _TRAJECTORY_REGISTRY
        expected = {"cubic_segments", "quintic_segments", "waypoints", "phase_list"}
        assert expected.issubset(_TRAJECTORY_REGISTRY.keys())

    def test_plant_registry(self):
        """Plant registry contains ArmRobot and HolonomicMobileRobot."""
        from shinro.factories.registry import _PLANT_REGISTRY
        assert "ArmRobot" in _PLANT_REGISTRY
        assert "HolonomicMobileRobot" in _PLANT_REGISTRY


class TestControllerFactory:
    """Verify ControllerFactory creates controllers from TOML configs."""

    def test_create_lqr(self, bk, tmp_path):
        """ControllerFactory creates an LQR controller from a valid config."""
        from shinro.factories.controller_factory import ControllerFactory
        config = tmp_path / "lqr.toml"
        config.write_text("""type = "LQR"\ndt = 0.02\nstate_cost = [1.0, 1.0]\ncontrol_cost = [0.1, 0.1]\n""")
        factory = ControllerFactory(str(config))
        ctrl = factory.create(backend=bk)
        from shinro.controllers.lqr import LQR
        assert isinstance(ctrl, LQR)
        assert ctrl.K is not None

    def test_create_lqr_full_matrix(self, bk, tmp_path):
        """ControllerFactory creates an LQR with full Q/R matrices and custom A/B."""
        from shinro.factories.controller_factory import ControllerFactory
        config = tmp_path / "lqr_full.toml"
        config.write_text("""type = "LQR"\ndt = 0.1\nstate_cost = [[2.0, 0.5], [0.5, 1.0]]\ncontrol_cost = [[0.1, 0.0], [0.0, 0.2]]\nA_dynamics = [[0.9, 0.1], [0.0, 0.9]]\nB_dynamics = [[0.0, 0.1], [0.1, 0.0]]\n""")
        factory = ControllerFactory(str(config))
        ctrl = factory.create(backend=bk)
        from shinro.controllers.lqr import LQR
        assert isinstance(ctrl, LQR)
        assert ctrl.K is not None

    def test_create_pid(self, bk, tmp_path):
        """ControllerFactory creates a PID controller from a valid config."""
        from shinro.factories.controller_factory import ControllerFactory
        config = tmp_path / "pid.toml"
        config.write_text("""type = "PID"\ndt = 0.01\nkp = [1.0]\nki = [0.5]\nkd = [0.1]\n""")
        factory = ControllerFactory(str(config))
        ctrl = factory.create(backend=bk)
        from shinro.controllers.pid import PIDController
        assert isinstance(ctrl, PIDController)

    def test_create_mpc_lti(self, bk, tmp_path):
        """ControllerFactory creates an MPC_LTI controller from a valid config."""
        from shinro.factories.controller_factory import ControllerFactory
        config = tmp_path / "mpc.toml"
        config.write_text("""type = "MPC_LTI"\ndt = 0.1\nhorizon = 5\nstate_cost = [1.0, 1.0]\ncontrol_cost = [1.0, 1.0]\n""")
        factory = ControllerFactory(str(config))
        ctrl = factory.create(backend=bk)
        from shinro.controllers.mpc_lti import MPC_LTI
        assert isinstance(ctrl, MPC_LTI)
        assert ctrl.H is not None

    def test_create_unknown_type(self, tmp_path):
        """ControllerFactory raises KeyError for an unknown controller type."""
        from shinro.factories.controller_factory import ControllerFactory
        config = tmp_path / "unknown.toml"
        config.write_text("""type = "NonExistent"\ndt = 0.01\n""")
        factory = ControllerFactory(str(config))
        with pytest.raises(KeyError):
            factory.create()

    def test_create_missing_required_key(self, tmp_path):
        """ControllerFactory raises KeyError when a required config key is missing."""
        from shinro.factories.controller_factory import ControllerFactory
        config = tmp_path / "bad.toml"
        config.write_text("""type = "LQR"\n""")
        factory = ControllerFactory(str(config))
        with pytest.raises(KeyError):
            factory.create()


class TestEstimatorFactory:
    """Verify EstimatorFactory creates estimators from TOML configs."""

    def test_create_kalman(self, bk, tmp_path):
        """EstimatorFactory creates a KalmanFilter from a valid config."""
        from shinro.factories.estimator_factory import EstimatorFactory
        config = tmp_path / "kalman.toml"
        config.write_text("""type = "KalmanFilter"\ndt = 0.02\nprocess_noise = [0.1, 0.1, 0.1]\nmeasurement_noise = [0.01, 0.01, 0.01]\n""")
        factory = EstimatorFactory(str(config))
        est = factory.create(backend=bk)
        from shinro.estimators.kalman_filter import KalmanFilter
        assert isinstance(est, KalmanFilter)

    def test_create_luenberger(self, bk, tmp_path):
        """EstimatorFactory creates a LuenbergerObserver from a valid config."""
        from shinro.factories.estimator_factory import EstimatorFactory
        config = tmp_path / "luenberger.toml"
        config.write_text("""type = "LuenbergerObserver"\ndt = 0.02\nobserver_gain = [0.8, 0.8, 0.8]\n""")
        factory = EstimatorFactory(str(config))
        est = factory.create(backend=bk)
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        assert isinstance(est, LuenbergerObserver)

    def test_create_unknown_type(self, tmp_path):
        """EstimatorFactory raises KeyError for an unknown estimator type."""
        from shinro.factories.estimator_factory import EstimatorFactory
        config = tmp_path / "unknown.toml"
        config.write_text("""type = "NonExistent"\ndt = 0.01\n""")
        factory = EstimatorFactory(str(config))
        with pytest.raises(KeyError):
            factory.create()


class TestTrajectoryFactory:
    """Verify TrajectoryFactory creates trajectory generators from TOML configs."""

    def test_create_cubic_segments(self, bk, tmp_path):
        """TrajectoryFactory creates a cubic polynomial schedule from a valid config."""
        from shinro.factories.trajectory_factory import TrajectoryFactory
        config = tmp_path / "cubic.toml"
        config.write_text("""type = "cubic_segments"\ndt = 0.1\n[[segments]]\nstart = [0.0, 0.0, 0.0]\nend = [1.0, 0.0, 0.0]\nduration = 1.0\n""")
        factory = TrajectoryFactory(str(config))
        schedule = factory.create(backend=bk)
        assert _to_np(schedule, bk).shape[1] == 3

    def test_create_quintic_segments(self, bk, tmp_path):
        """TrajectoryFactory creates a quintic polynomial schedule from a valid config."""
        from shinro.factories.trajectory_factory import TrajectoryFactory
        config = tmp_path / "quintic.toml"
        config.write_text("""type = "quintic_segments"\ndt = 0.1\n[[segments]]\nstart = [0.0, 0.0, 0.0]\nend = [1.0, 0.0, 0.0]\nduration = 1.0\n""")
        factory = TrajectoryFactory(str(config))
        schedule = factory.create(backend=bk)
        assert _to_np(schedule, bk).shape[1] == 3

    def test_create_waypoints(self, bk, tmp_path):
        """TrajectoryFactory creates a waypoint schedule from a valid config."""
        from shinro.factories.trajectory_factory import TrajectoryFactory
        config = tmp_path / "waypoints.toml"
        config.write_text("""type = "waypoints"\ndt = 0.1\n[[waypoints]]\nposition = [0.0, 0.0, 0.0]\nduration = 1.0\n[[waypoints]]\nposition = [1.0, 0.0, 0.0]\nduration = 1.0\n""")
        factory = TrajectoryFactory(str(config))
        schedule = factory.create(backend=bk)
        assert _to_np(schedule, bk).shape[1] == 3

    def test_create_phase_list(self, bk, tmp_path):
        """TrajectoryFactory creates a phase schedule from a valid config."""
        from shinro.factories.trajectory_factory import TrajectoryFactory
        config = tmp_path / "phases.toml"
        config.write_text("""type = "phase_list"\ndt = 0.1\n[[phases]]\nduration = 1.0\narm = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\nbase = [0.0, 0.0, 0.0]\njaw = 0.0\n""")
        factory = TrajectoryFactory(str(config))
        result = factory.create(backend=bk)
        assert "arm" in result
        assert "base" in result
        assert "jaw" in result

    def test_create_unknown_type(self, tmp_path):
        """TrajectoryFactory raises KeyError for an unknown trajectory type."""
        from shinro.factories.trajectory_factory import TrajectoryFactory
        config = tmp_path / "unknown.toml"
        config.write_text("""type = "NonExistent"\ndt = 0.01\n""")
        factory = TrajectoryFactory(str(config))
        with pytest.raises(KeyError):
            factory.create()


class TestFactoryBackendPassthrough:
    """Verify backend is passed through correctly to created instances."""

    def test_controller_factory_backend_passthrough(self, bk, tmp_path):
        """ControllerFactory passes the backend to the created controller."""
        from shinro.factories.controller_factory import ControllerFactory
        config = tmp_path / "lqr.toml"
        config.write_text("""type = "LQR"\ndt = 0.02\nstate_cost = [1.0, 1.0]\ncontrol_cost = [0.1, 0.1]\n""")
        factory = ControllerFactory(str(config))
        ctrl = factory.create(backend=bk)
        assert ctrl.bk is bk

    def test_estimator_factory_backend_passthrough(self, bk, tmp_path):
        """EstimatorFactory passes the backend to the created estimator."""
        from shinro.factories.estimator_factory import EstimatorFactory
        config = tmp_path / "kalman.toml"
        config.write_text("""type = "KalmanFilter"\ndt = 0.02\nprocess_noise = [0.1, 0.1]\nmeasurement_noise = [0.01, 0.01]\n""")
        factory = EstimatorFactory(str(config))
        est = factory.create(backend=bk)
        assert est.bk is bk
