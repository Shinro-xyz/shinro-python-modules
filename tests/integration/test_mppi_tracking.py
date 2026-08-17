"""MPPI base tracking integration: full closed-loop MPPI + estimator on the holonomic base.

MPPI is wired into the plant's model via ``BatchedDynamicsAdapter`` inside
``ScenarioFactory.build()`` — the controller's dynamics/cost are produced from
``HolonomicMobileRobot.get_model()`` (an LTI matmul path), so the rollout runs
as backend-native batched ops. This is the full Trajectory -> Controller ->
Estimator -> Plant -> Engine loop with MPPI as a first-class controller.
"""

import pytest

from shinro.factories import ScenarioFactory

from .helpers.assertions import (
    assert_finite_state,
    assert_steady_state,
)
from .helpers.scenario_runner import run_scenario

pytestmark = [pytest.mark.integration]

SCENARIO = "tests/integration/scenarios/mppi_base_tracking.toml"


@pytest.fixture
def scenario(mujoco_available):
    """A freshly built MPPI base-tracking scenario with the shared free-joint MJCF."""
    return ScenarioFactory(SCENARIO).build()


class TestMppiBaseTracking:
    """MPPI + Kalman filter tracks the straight-line waypoint schedule."""

    def test_controller_is_plant_wired(self, scenario):
        """ScenarioFactory attaches the plant to the MPPI controller via the adapter."""
        from shinro.controllers.mppi import MPPIController

        assert isinstance(scenario.controller, MPPIController)
        assert scenario.controller._adapter is not None
        assert scenario.controller.dynamics_fn is not None
        assert scenario.controller.cost_fn is not None

    def test_steady_state_tracking(self, scenario):
        """The base settles on each waypoint within the declared tolerance."""
        records = run_scenario(scenario)
        tol = scenario.config["scenario"]["tolerance"]["steady_state"]
        assert_steady_state(records, tolerance=tol)
        assert_finite_state(records)

    def test_final_position_matches_target(self, scenario):
        """The base ends at the final waypoint within 8cm."""
        records = run_scenario(scenario)
        final_ref = records[-1].reference
        final_state = records[-1].plant_state
        err = float(abs(final_ref[0] - final_state[0]))
        assert err < 0.08, f"final x error {err:.4f} > 0.08"
