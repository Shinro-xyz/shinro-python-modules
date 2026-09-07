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
    "tests/integration/scenarios/pendulum_balance.toml",
    "tests/integration/scenarios/cartpole_balance.toml",
]


class TestScenarioFactory:
    """Every shipped scenario TOML builds a consistent loop."""

    @pytest.mark.parametrize("path", SCENARIOS)
    def test_all_scenarios_build(self, mujoco_available, path):
        """Each scenario TOML composes sim + plant + trajectory (and loop roles)."""
        scenario = ScenarioFactory(path).build()
        assert scenario.plant is not None
        assert scenario.trajectory is not None
        assert scenario.config["scenario"]["name"]

    @pytest.mark.parametrize("path", SCENARIOS)
    def test_plant_lookup_valid(self, mujoco_available, path):
        """The configured plant resolves: by name on the RobotSim, or standalone."""
        scenario = ScenarioFactory(path).build()
        plant_cfg = scenario.config["plant"]
        if "name" in plant_cfg:
            assert scenario.sim is not None
            assert scenario.sim.get_plant(plant_cfg["name"]) is scenario.plant
        else:
            assert scenario.sim is None
            assert scenario.plant.__class__.__name__ == plant_cfg["type"]

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

    def test_pendulum_composes_plant_only_loop(self, mujoco_available):
        """Pendulum balance is a plant-only loop: no sim, derived 2D model, seeded state."""
        scenario = ScenarioFactory("tests/integration/scenarios/pendulum_balance.toml").build()
        assert scenario.sim is None
        assert scenario.estimator.A.shape[0] == 2
        assert scenario.controller.B.shape[1] == 1
        assert scenario.plant.get_state()[0] == pytest.approx(0.2)

    def test_cartpole_composes_plant_only_loop(self, mujoco_available):
        """Cartpole balance is a plant-only loop: no sim, derived 4D model, seeded state."""
        scenario = ScenarioFactory("tests/integration/scenarios/cartpole_balance.toml").build()
        assert scenario.sim is None
        assert scenario.estimator.A.shape[0] == 4
        assert scenario.controller.B.shape[1] == 1
        assert scenario.plant.get_state()[2] == pytest.approx(0.2)


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


class TestPlantOnlyValidation:
    """Plant-only scenarios (no [sim]) validate their [plant] section loudly."""

    def test_missing_plant_type_raises(self, tmp_path):
        """A plant-only scenario without [plant].type raises KeyError."""
        import textwrap

        bad = tmp_path / "no_type.toml"
        bad.write_text(
            textwrap.dedent(
                """\
                [scenario]
                name = "no_type"
                duration = 1.0
                dt = 0.01

                [plant]
                config = "configs/plants/inverted_pendulum.toml"

                [trajectory]
                type = "waypoints"
                config = "configs/trajectories/pendulum_upright.toml"
                """
            )
        )
        with pytest.raises(KeyError, match=r"\[plant\].type"):
            ScenarioFactory(str(bad)).build()

    def test_unknown_plant_type_raises(self, tmp_path):
        """An unregistered [plant].type raises KeyError."""
        import textwrap

        bad = tmp_path / "bad_type.toml"
        bad.write_text(
            textwrap.dedent(
                """\
                [scenario]
                name = "bad_type"
                duration = 1.0
                dt = 0.01

                [plant]
                type = "Nonexistent"
                config = "configs/plants/inverted_pendulum.toml"

                [trajectory]
                type = "waypoints"
                config = "configs/trajectories/pendulum_upright.toml"
                """
            )
        )
        with pytest.raises(KeyError, match="Nonexistent"):
            ScenarioFactory(str(bad)).build()

    def test_dt_mismatch_raises(self, tmp_path):
        """[scenario].dt must equal the plant's own dt in plant-only mode."""
        import textwrap

        bad = tmp_path / "dt_mismatch.toml"
        bad.write_text(
            textwrap.dedent(
                """\
                [scenario]
                name = "dt_mismatch"
                duration = 1.0
                dt = 0.02

                [plant]
                type = "InvertedPendulum"
                config = "configs/plants/inverted_pendulum.toml"

                [trajectory]
                type = "waypoints"
                config = "configs/trajectories/pendulum_upright.toml"
                """
            )
        )
        with pytest.raises(ValueError, match="must equal the plant's dt"):
            ScenarioFactory(str(bad)).build()

    def test_initial_state_wrong_length_raises(self, tmp_path):
        """[plant].initial_state length must match the plant state dimension."""
        import textwrap

        bad = tmp_path / "bad_init.toml"
        bad.write_text(
            textwrap.dedent(
                """\
                [scenario]
                name = "bad_init"
                duration = 1.0
                dt = 0.01

                [plant]
                type = "InvertedPendulum"
                config = "configs/plants/inverted_pendulum.toml"
                initial_state = [0.2, 0.0, 0.0]

                [trajectory]
                type = "waypoints"
                config = "configs/trajectories/pendulum_upright.toml"
                """
            )
        )
        with pytest.raises(ValueError, match="initial_state"):
            ScenarioFactory(str(bad)).build()
