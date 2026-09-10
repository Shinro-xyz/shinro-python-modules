"""Scenario.run() / iter_run() / reset() — the public simulation API.

Plant-only scenarios (no MuJoCo): the cartpole balance loop self-integrates its
analytical dynamics, so these tests run on a minimal install.
"""

import numpy as np
import pytest

from shinro.factories import ScenarioFactory

BALANCE = "tests/integration/scenarios/cartpole_balance.toml"
DERIVED = "configs/scenarios/cartpole_lqr_kf_derived.toml"


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
