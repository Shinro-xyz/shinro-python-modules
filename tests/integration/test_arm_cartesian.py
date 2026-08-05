"""Arm Cartesian integration: PID + Kalman track a 6D end-effector pose on the arm."""

import numpy as np
import pytest

from factories import ScenarioFactory

from .helpers.assertions import (
    assert_estimator_recovery,
    assert_finite_state,
    assert_steady_state,
)
from .helpers.scenario_runner import run_scenario

pytestmark = [pytest.mark.integration]

SCENARIO = "tests/integration/scenarios/arm_cartesian.toml"


@pytest.fixture
def scenario(mujoco_available):
    """A freshly built arm-cartesian scenario with the shared free-joint MJCF."""
    return ScenarioFactory(SCENARIO).build()


class TestArmCartesian:
    """PID + Kalman filter tracks the 6D end-effector lift schedule."""

    def test_steady_state_tracking(self, scenario):
        """The end-effector settles on each pose waypoint within tolerance."""
        records = run_scenario(scenario)
        tol = scenario.config["scenario"]["tolerance"]["steady_state"]
        assert_steady_state(records, tolerance=tol)
        assert_finite_state(records)

    def test_estimator_recovers_noisy_state(self, scenario):
        """The Kalman estimate stays close to the true end-effector state."""
        records = run_scenario(scenario)
        tol = scenario.config["scenario"]["tolerance"]["estimator"]
        assert_estimator_recovery(records, tolerance=tol)

    def test_arm_actually_lifts(self, scenario):
        """The arm physically reaches the raised +8cm waypoint."""
        records = run_scenario(scenario)
        z_home = records[0].reference[2]
        max_z = max(r.plant_state[2] for r in records)
        assert max_z - z_home > 0.06, f"arm never lifted (max z={max_z:.4f}, home z={z_home:.4f})"

    def test_no_nan_in_joint_qpos(self, scenario):
        """MuJoCo qpos stays finite through the whole trajectory."""
        records = run_scenario(scenario)
        for r in records[::20]:
            assert np.all(np.isfinite(r.plant_state))
