"""ScenarioFactory construction validation: composes every ABC role from one TOML."""

import pytest

from shinro.controllers.lqr import LQR
from shinro.controllers.pid import PIDController
from shinro.estimators.kalman_filter import KalmanFilter
from shinro.factories import ScenarioFactory

pytestmark = [pytest.mark.integration]

SCENARIOS = [
    "tests/integration/scenarios/base_tracking.toml",
    "tests/integration/scenarios/arm_cartesian.toml",
    "tests/integration/scenarios/pick_and_place.toml",
    "tests/integration/scenarios/adversarial_nan.toml",
]


class TestScenarioFactory:
    """Every shipped scenario TOML builds a consistent loop."""

    @pytest.mark.parametrize("path", SCENARIOS)
    def test_all_scenarios_build(self, mujoco_available, path):
        """Each scenario TOML composes sim + plant + trajectory (and loop roles)."""
        scenario = ScenarioFactory(path).build()
        assert scenario.sim is not None
        assert scenario.plant is not None
        assert scenario.trajectory is not None
        assert scenario.config["scenario"]["name"]

    @pytest.mark.parametrize("path", SCENARIOS)
    def test_plant_lookup_valid(self, mujoco_available, path):
        """The configured plant name resolves on the built RobotSim."""
        scenario = ScenarioFactory(path).build()
        name = scenario.config["plant"]["name"]
        assert scenario.sim.get_plant(name) is scenario.plant

    def test_base_tracking_composes_loop(self, mujoco_available):
        """Base tracking wires a concrete LQR + Kalman into the loop."""
        scenario = ScenarioFactory("tests/integration/scenarios/base_tracking.toml").build()
        assert isinstance(scenario.controller, LQR)
        assert isinstance(scenario.estimator, KalmanFilter)
        assert scenario.estimator.A.shape[0] == 3
        assert scenario.controller.B.shape[1] == 3

    def test_arm_cartesian_composes_loop(self, mujoco_available):
        """Arm Cartesian wires a concrete PID + Kalman (6D) into the loop."""
        scenario = ScenarioFactory("tests/integration/scenarios/arm_cartesian.toml").build()
        assert isinstance(scenario.controller, PIDController)
        assert isinstance(scenario.estimator, KalmanFilter)
        assert scenario.estimator.A.shape[0] == 6
        assert len(scenario.controller.kp) == 6

    def test_pick_and_place_is_feedforward(self, mujoco_available):
        """Pick-and-place has no feedback controller/estimator — schedule drives."""
        scenario = ScenarioFactory("tests/integration/scenarios/pick_and_place.toml").build()
        assert scenario.controller is None
        assert scenario.estimator is None
        assert isinstance(scenario.trajectory, dict)
        assert set(scenario.trajectory.keys()) == {"arm", "base", "jaw"}


class TestDimensionValidation:
    """Mismatched component dimensions are caught at build time."""

    def test_estimator_dimension_mismatch_raises(self, mujoco_available, tmp_path):
        """A 6D estimator on a 3D plant raises ValueError."""
        import textwrap

        bad = tmp_path / "bad_est.toml"
        bad.write_text(
            textwrap.dedent(
                """\
                [scenario]
                name = "bad_est"
                duration = 1.0
                dt = 0.02

                [physics]
                free_joint = true

                [plant]
                name = "base"

                [controller]
                type = "LQR"
                config = "configs/controllers/lqr_base.toml"

                [estimator]
                type = "KalmanFilter"
                config = "configs/estimators/kalman_arm.toml"

                [trajectory]
                type = "waypoints"
                config = "configs/trajectories/base_straight.toml"

                [sim]
                config = "robot_config.toml"
                """
            )
        )
        with pytest.raises(ValueError, match="state dimension"):
            ScenarioFactory(str(bad)).build()

    def test_missing_plant_name_raises(self, mujoco_available, tmp_path):
        """An unknown plant name raises KeyError at build time."""
        import textwrap

        bad = tmp_path / "bad_plant.toml"
        bad.write_text(
            textwrap.dedent(
                """\
                [scenario]
                name = "bad_plant"
                duration = 1.0
                dt = 0.02

                [physics]
                free_joint = true

                [plant]
                name = "nonexistent"

                [controller]
                type = "LQR"
                config = "configs/controllers/lqr_base.toml"

                [estimator]
                type = "KalmanFilter"
                config = "configs/estimators/kalman_base.toml"

                [trajectory]
                type = "waypoints"
                config = "configs/trajectories/base_straight.toml"

                [sim]
                config = "robot_config.toml"
                """
            )
        )
        with pytest.raises(KeyError):
            ScenarioFactory(str(bad)).build()
