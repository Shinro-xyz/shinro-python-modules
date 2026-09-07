"""Pendulum balance integration: plant-only closed-loop LQR + estimator on the inverted pendulum."""

import pytest

from shinro.factories import ScenarioFactory

from .helpers.assertions import (
    assert_estimator_recovery,
    assert_finite_state,
    assert_steady_state,
)
from .helpers.scenario_runner import run_scenario

pytestmark = [pytest.mark.integration]

SCENARIO = "tests/integration/scenarios/pendulum_balance.toml"


@pytest.fixture
def scenario(mujoco_available):
    """A freshly built pendulum-balance scenario (plant-only, no MuJoCo sim)."""
    return ScenarioFactory(SCENARIO).build()


class TestPendulumBalance:
    """LQR + Kalman regulate the pendulum from a 0.2 rad tilt to upright."""

    def test_regulates_from_perturbation(self, scenario):
        """The pendulum settles at upright within the declared tolerance."""
        records = run_scenario(scenario)
        tol = scenario.config["scenario"]["tolerance"]["steady_state"]
        assert_steady_state(records, tolerance=tol)
        assert_finite_state(records)

    def test_estimator_recovers_noisy_state(self, scenario):
        """The Kalman estimate stays close to the true (noisy-hidden) state."""
        records = run_scenario(scenario)
        tol = scenario.config["scenario"]["tolerance"]["estimator"]
        assert_estimator_recovery(records, tolerance=tol)

    def test_final_angle_near_upright(self, scenario):
        """The pendulum ends within 0.05 rad of upright."""
        records = run_scenario(scenario)
        final_theta = abs(records[-1].plant_state[0])
        assert final_theta < 0.05, f"final |theta| {final_theta:.4f} > 0.05"


@pytest.mark.parametrize(
    ("est_type", "estimator_cfg"),
    [
        ("KalmanFilter", "configs/estimators/kalman_pendulum.toml"),
        ("LuenbergerObserver", "configs/estimators/luenberger_pendulum.toml"),
    ],
)
class TestPendulumEstimatorVariants:
    """Different estimators both keep the pendulum balanced within a bounded error."""

    def test_tracking_bounded(self, mujoco_available, tmp_path, est_type, estimator_cfg):
        """Each estimator variant keeps the trailing tracking error under 0.06."""
        import textwrap

        variant = tmp_path / "pend_variant.toml"
        variant.write_text(
            textwrap.dedent(
                f"""\
                [scenario]
                name = "pend_variant"
                duration = 12.0
                dt = 0.01
                tolerance = {{ steady_state = 0.06, estimator = 0.03 }}
                input_limits = {{ min = [-2.0], max = [2.0] }}

                [plant]
                type = "InvertedPendulum"
                config = "configs/plants/inverted_pendulum.toml"
                initial_state = [0.2, 0.0]

                [controller]
                type = "LQR"
                config = "configs/controllers/lqr_pendulum.toml"

                [estimator]
                type = "{est_type}"
                config = "{estimator_cfg}"

                [trajectory]
                type = "waypoints"
                config = "configs/trajectories/pendulum_upright.toml"

                [noise.measurement]
                std = [0.005, 0.02]
                """
            )
        )

        scenario = ScenarioFactory(str(variant)).build()
        records = run_scenario(scenario)
        assert_steady_state(records, tolerance=0.06)
        assert_finite_state(records)
