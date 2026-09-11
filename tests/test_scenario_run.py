"""Scenario.run() / iter_run() / reset() — the public simulation API.

Plant-only scenarios (no MuJoCo): the cartpole balance loop self-integrates its
analytical dynamics, so these tests run on a minimal install.
"""

import numpy as np
import pytest

from shinro.factories import ScenarioFactory

BALANCE = "tests/integration/scenarios/cartpole_balance.toml"
DERIVED = "tests/fixtures/configs/scenarios/cartpole_lqr_kf_derived.toml"


@pytest.fixture()
def balance():
    return ScenarioFactory(BALANCE).build()


class TestRun:
    def test_run_returns_simresult(self, balance):
        result = balance.run(steps=200)
        assert len(result) == 200
        assert result.records[0].t == 0.0
        assert result.true_state.shape == (200, 4)
        assert result.control.shape == (200, 1)

    def test_run_defaults_to_duration(self, balance):
        result = balance.run()
        assert len(result) == 1200  # 12 s at 50 Hz

    def test_iter_run_yields_records(self, balance):
        records = list(balance.iter_run(steps=50))
        assert len(records) == 50
        assert records[0].plant_state.shape == (4,)

    def test_seed_determinism(self, balance):
        balance.reset()
        a = balance.run(steps=50, seed=7)
        balance.reset()
        b = balance.run(steps=50, seed=7)
        assert np.array_equal(a.true_state, b.true_state)

    def test_trajectory_cap_warns(self, balance):
        with pytest.warns(UserWarning, match="truncating"):
            result = balance.run(steps=5000)
        assert len(result) == 1200  # capped at the trajectory length


class TestReset:
    def test_reset_restores_initial_state(self, balance):
        balance.run(steps=50)
        balance.reset()
        init = balance.config["plant"]["initial_state"]
        assert np.allclose(np.asarray(balance.plant.get_state()), init)

    def test_reset_clears_estimator_state(self, balance):
        balance.run(steps=50)
        balance.reset()
        assert np.allclose(np.asarray(balance.estimator.x_hat), 0.0)


class TestCheck:
    def test_check_applies_toml_tolerances(self, balance):
        result = balance.run(steps=200)
        report = result.check()
        assert "steady_state" in report and "estimator" in report
        assert report["ok"] in (True, False)
        assert report["steady_state"]["tol"] == 0.05

    def test_check_never_raises(self, balance):
        result = balance.run(steps=5)  # far from settled
        report = result.check()
        assert report["ok"] in (True, False)


class TestFeedforwardBranch:
    def test_feedforward_requires_sim_backed(self):
        """A controller-less scenario routes to the phase runner, which needs a RobotSim."""
        from shinro.factories.scenario_factory import Scenario

        scenario = Scenario(
            sim=None,
            plant=None,
            controller=None,
            estimator=None,
            trajectory={"arm": [[0.0] * 6], "base": [[0.0] * 3], "jaw": [0.0]},
            config={"scenario": {"dt": 0.02}},
        )
        with pytest.raises(ValueError, match="sim-backed"):
            scenario.run()

    def test_phase_schedule_drives_jaw_explicitly(self):
        """The jaw setpoint is applied via the engine actuator, not twist channel 5."""
        from unittest.mock import MagicMock

        from shinro.factories.scenario_factory import Scenario
        from shinro.simulation.runner import iter_phase_schedule

        engine = MagicMock()
        engine.actuator_names = ["Jaw"]
        sim = MagicMock()
        sim.engine = engine
        arm_plant = MagicMock()
        base_plant = MagicMock()
        sim.plants = {"arm": arm_plant, "base": base_plant}
        sim.get_plant.side_effect = lambda name: {"arm": arm_plant, "base": base_plant}[name]

        scenario = Scenario(
            sim=sim,
            plant=arm_plant,  # primary plant: record/state carrier
            controller=None,
            estimator=None,
            trajectory={"arm": [[0.0] * 6, [0.0] * 6], "base": [[0.0] * 3, [0.0] * 3], "jaw": [0.0, 0.5]},
            config={"scenario": {"dt": 0.02}, "signals": {"jaw": {"actuator": "Jaw"}}},
        )
        list(iter_phase_schedule(scenario))

        jaw_calls = [c.args for c in engine.set_joint_ctrl.call_args_list]
        assert ("Jaw", 0.0) in jaw_calls
        assert ("Jaw", 0.5) in jaw_calls
        twists = [c.args[0] for c in arm_plant.step.call_args_list]
        assert all(len(t) == 6 for t in twists)


class TestPhaseScheduleGeneralization:
    """Schedule routing: plant names drive plants, [signals] declares actuator passthroughs,
    unknown keys and length mismatches are loud — nothing is robot-specific."""

    def _make_scenario(self, schedule, config):
        from unittest.mock import MagicMock

        from shinro.factories.scenario_factory import Scenario

        engine = MagicMock()
        engine.actuator_names = ["grip"]
        sim = MagicMock()
        sim.engine = engine
        left = MagicMock()
        right = MagicMock()
        sim.plants = {"left_arm": left, "right_arm": right}
        sim.get_plant.side_effect = lambda name: sim.plants[name]
        return Scenario(
            sim=sim,
            plant=left,
            controller=None,
            estimator=None,
            trajectory=schedule,
            config=config,
        ), engine, left, right

    def test_plant_renamed_still_routed(self):
        """A schedule key matching a plant name works regardless of what the plant is called."""
        scenario, engine, left, right = self._make_scenario(
            {
                "left_arm": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                "right_arm": [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
                "gripper": [[0.0], [0.5]],
            },
            {"scenario": {"dt": 0.02}, "signals": {"gripper": {"actuator": "grip"}}},
        )
        from shinro.simulation.runner import iter_phase_schedule

        records = list(iter_phase_schedule(scenario))
        assert len(records) == 2
        assert left.step.call_count == 2 and right.step.call_count == 2
        grip_calls = [c.args for c in engine.set_joint_ctrl.call_args_list]
        assert ("grip", 0.0) in grip_calls and ("grip", 0.5) in grip_calls
        # reference/control carry the full applied signal dict
        assert set(records[0].control) == {"left_arm", "right_arm", "gripper"}

    def test_unknown_signal_loud(self):
        """A schedule key matching neither a plant nor a [signals] entry fails loudly."""
        scenario, _, _, _ = self._make_scenario(
            {"left_arm": [[0.0] * 6], "armz": [[0.0] * 6]},
            {"scenario": {"dt": 0.02}, "signals": {}},
        )
        from shinro.simulation.runner import iter_phase_schedule

        with pytest.raises(ValueError, match="neither a declared plant"):
            list(iter_phase_schedule(scenario))

    def test_undeclared_actuator_loud(self):
        """A [signals] entry pointing at a missing engine actuator fails loudly."""
        scenario, engine, _, _ = self._make_scenario(
            {"gripper": [[0.0]]},
            {"scenario": {"dt": 0.02}, "signals": {"gripper": {"actuator": "nope"}}},
        )
        engine.actuator_names = ["grip"]
        from shinro.simulation.runner import iter_phase_schedule

        with pytest.raises(ValueError, match="no such actuator"):
            list(iter_phase_schedule(scenario))

    def test_length_mismatch_loud(self):
        """Signals with different schedule lengths fail before the loop starts."""
        scenario, _, _, _ = self._make_scenario(
            {"left_arm": [[0.0] * 6, [0.0] * 6], "right_arm": [[0.0] * 6]},
            {"scenario": {"dt": 0.02}},
        )
        from shinro.simulation.runner import iter_phase_schedule

        with pytest.raises(ValueError, match="disagree in length"):
            list(iter_phase_schedule(scenario))
