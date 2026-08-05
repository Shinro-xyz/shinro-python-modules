"""Base tracking integration: full closed-loop LQR/MPC + estimator on the holonomic base."""

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

SCENARIO = "tests/integration/scenarios/base_tracking.toml"


@pytest.fixture
def scenario(mujoco_available):
    """A freshly built base-tracking scenario with the shared free-joint MJCF."""
    return ScenarioFactory(SCENARIO).build()


class TestBaseTrackingConvergence:
    """LQR + Kalman filter tracks the straight-line waypoint schedule."""

    def test_steady_state_tracking(self, scenario):
        """The base settles on each waypoint within the declared tolerance."""
        records = run_scenario(scenario)
        tol = scenario.config["scenario"]["tolerance"]["steady_state"]
        assert_steady_state(records, tolerance=tol)
        assert_finite_state(records)

    def test_estimator_recovers_noisy_state(self, scenario):
        """The Kalman estimate stays close to the true (noisy-hidden) state."""
        records = run_scenario(scenario)
        tol = scenario.config["scenario"]["tolerance"]["estimator"]
        assert_estimator_recovery(records, tolerance=tol)

    def test_final_position_matches_target(self, scenario):
        """The base ends at the final waypoint within 5cm."""
        records = run_scenario(scenario)
        final_ref = records[-1].reference
        final_state = records[-1].plant_state
        err = np.linalg.norm(final_ref - final_state)
        assert err < 0.05, f"final position error {err:.4f} > 0.05"


@pytest.mark.parametrize(
    ("controller_cfg", "estimator_cfg"),
    [
        ("configs/controllers/lqr_base.toml", "configs/estimators/kalman_base.toml"),
        ("configs/controllers/lqr_base.toml", "configs/estimators/luenberger_base.toml"),
        ("configs/controllers/mpc_lti_base.toml", "configs/estimators/kalman_base.toml"),
        ("configs/controllers/mpc_base.toml", "configs/estimators/kalman_base.toml"),
    ],
)
class TestBaseControllerVariants:
    """Different controller/estimator combos all track within a bounded error."""

    def test_tracking_bounded(self, mujoco_available, tmp_path, controller_cfg, estimator_cfg):
        """Each variant keeps the trailing tracking error under 0.06."""
        import textwrap

        variant = tmp_path / "variant.toml"
        variant.write_text(
            textwrap.dedent(
                f"""\
                [scenario]
                name = "variant"
                duration = 16.0
                dt = 0.02
                tolerance = {{ steady_state = 0.06, estimator = 0.03 }}
                input_limits = {{ min = [-0.5, -0.5, -1.0], max = [0.5, 0.5, 1.0] }}

                [physics]
                free_joint = true

                [plant]
                name = "base"

                [controller]
                type = "LQR"
                config = "{controller_cfg}"

                [estimator]
                type = "KalmanFilter"
                config = "{estimator_cfg}"

                [trajectory]
                type = "waypoints"
                config = "configs/trajectories/base_straight.toml"

                [sim]
                config = "robot_config.toml"

                [noise.measurement]
                std = [0.01, 0.01, 0.02]
                """
            )
        )

        scenario = ScenarioFactory(str(variant)).build()
        records = run_scenario(scenario)
        assert_steady_state(records, tolerance=0.06)
        assert_finite_state(records)
