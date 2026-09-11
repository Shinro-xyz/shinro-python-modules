"""Cartpole balance integration: plant-only closed-loop LQR + estimator on the cart-pole."""

import numpy as np
import pytest

from shinro.factories import ScenarioFactory

from .helpers.assertions import (
    assert_estimator_recovery,
    assert_finite_state,
    assert_steady_state,
)
from .helpers.scenario_runner import run_scenario

pytestmark = [pytest.mark.integration]

SCENARIO = "tests/integration/scenarios/cartpole_balance.toml"


@pytest.fixture
def scenario(mujoco_available):
    """A freshly built cartpole-balance scenario (plant-only, no MuJoCo sim)."""
    return ScenarioFactory(SCENARIO).build()


class TestCartPoleBalance:
    """LQR + Kalman regulate the cartpole from a 0.2 rad pole tilt to upright."""

    def test_regulates_from_perturbation(self, scenario):
        """The cartpole settles at upright within the declared tolerance."""
        records = run_scenario(scenario)
        tol = scenario.config["scenario"]["tolerance"]["steady_state"]
        assert_steady_state(records, tolerance=tol)
        assert_finite_state(records)

    def test_estimator_recovers_noisy_state(self, scenario):
        """The Kalman estimate stays close to the true (noisy-hidden) state."""
        records = run_scenario(scenario)
        tol = scenario.config["scenario"]["tolerance"]["estimator"]
        assert_estimator_recovery(records, tolerance=tol)

    def test_pole_angle_converges(self, scenario):
        """The pole ends within 0.05 rad of upright."""
        records = run_scenario(scenario)
        final_theta = abs(records[-1].plant_state[2])
        assert final_theta < 0.05, f"final |theta| {final_theta:.4f} > 0.05"

    def test_cart_stays_on_track(self, scenario):
        """The cart stays within the track limits throughout the run."""
        records = run_scenario(scenario)
        xs = np.array([r.plant_state[0] for r in records])
        assert xs.min() > -2.0 and xs.max() < 2.0, f"cart left track: [{xs.min():.3f}, {xs.max():.3f}]"


@pytest.mark.parametrize(
    ("est_type", "estimator_cfg"),
    [
        ("KalmanFilter", "tests/fixtures/configs/estimators/kalman_cartpole.toml"),
        ("LuenbergerObserver", "tests/fixtures/configs/estimators/luenberger_cartpole.toml"),
    ],
)
class TestCartPoleEstimatorVariants:
    """Different estimators both keep the cartpole balanced within a bounded error."""

    def test_tracking_bounded(self, mujoco_available, tmp_path, est_type, estimator_cfg):
        """Each estimator variant keeps the trailing tracking error under 0.06."""
        import textwrap

        variant = tmp_path / "cart_variant.toml"
        variant.write_text(
            textwrap.dedent(
                f"""\
                [scenario]
                name = "cart_variant"
                duration = 12.0
                dt = 0.01
                tolerance = {{ steady_state = 0.06, estimator = 0.03 }}
                input_limits = {{ min = [-10.0], max = [10.0] }}

                [plant]
                type = "CartPole"
                config = "configs/plants/cartpole.toml"
                initial_state = [0.0, 0.0, 0.2, 0.0]

                [controller]
                type = "LQR"
                config = "tests/fixtures/configs/controllers/lqr_cartpole.toml"

                [estimator]
                type = "{est_type}"
                config = "{estimator_cfg}"

                [trajectory]
                type = "waypoints"
                config = "tests/fixtures/configs/trajectories/cartpole_upright.toml"

                [noise.measurement]
                std = [0.005, 0.01, 0.005, 0.02]
                """
            )
        )

        scenario = ScenarioFactory(str(variant)).build()
        records = run_scenario(scenario)
        assert_steady_state(records, tolerance=0.06)
        assert_finite_state(records)
