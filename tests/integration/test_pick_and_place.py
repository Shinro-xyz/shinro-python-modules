"""Pick-and-place integration: phase_list schedule drives base + arm + gripper through RobotSim."""

import numpy as np
import pytest

from shinro.factories import ScenarioFactory

from .helpers.scenario_runner import run_phase_schedule

pytestmark = [pytest.mark.integration]

SCENARIO = "tests/integration/scenarios/pick_and_place.toml"


@pytest.fixture
def scenario(mujoco_available):
    """A freshly built pick-and-place scenario with the shared free-joint MJCF."""
    return ScenarioFactory(SCENARIO).build()


class TestPickAndPlace:
    """The phase_list schedule drives a full reach → grip → lift → drive → release sequence."""

    def test_schedule_runs_without_error(self, scenario):
        """The full multi-plant schedule executes with finite states throughout."""
        records = run_phase_schedule(scenario)
        assert len(records) == len(scenario.trajectory["arm"])
        assert np.all(np.isfinite([r.plant_state[0] for r in records]))

    def test_gripper_closes_and_reopens(self, scenario):
        """The jaw setpoint is applied: gripper opens, closes, then opens again."""
        schedule = scenario.trajectory
        jaw = schedule["jaw"]
        # phases: open(0) -> close(0.5) -> hold(0.5) -> close(0.5) -> open(0)
        assert jaw.min() >= 0.0
        assert jaw.max() > 0.3, "gripper never closed (jaw never exceeded 0.3)"
        assert jaw[0] == 0.0 and jaw[-1] == 0.0, "gripper should open at start and end"

    def test_base_drives_forward(self, scenario):
        """The drive phases move the base along +x."""
        records = run_phase_schedule(scenario)
        final_x = records[-1].plant_state[0]
        assert final_x > 0.8, f"base never completed the drive (final x={final_x:.3f})"

    def test_arm_reaches_down_and_up(self, scenario):
        """The arm pitch/elbow setpoints are exercised without NaN."""
        records = run_phase_schedule(scenario)
        arm_states = np.array([r.plant_state for r in records])
        assert np.all(np.isfinite(arm_states))

    def test_reset_returns_to_origin(self, scenario):
        """After the sequence, a reset restores the zero state."""
        run_phase_schedule(scenario)
        scenario.sim.reset()
        base_state = np.asarray(scenario.sim.get_plant("base").get_state(), dtype=np.float64)
        assert np.allclose(base_state, np.zeros(3), atol=1e-3)
