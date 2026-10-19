"""Tests for `simulate_scenario` — the design-time closed loop.

This is the tool that lets an agent *tune*: it re-runs the library's own
`run_scenario` and reports how well the loop tracked. The assertions below use the
SMC-on-a-pendulum fixture, whose controller takes its model from the plant, so a pass
also exercises the plant seam end to end.
"""

import json

import pytest

from shinro.mcp.server import simulate_scenario

SMC_RUN = "tests/fixtures/scenarios/smc_pendulum_run.toml"
COMPILE_ONLY = "tests/integration/scenarios/base_tracking.toml"


class TestSummary:
    """The summary an agent reads to decide whether a tuning change helped."""

    def test_runs_and_reports_the_expected_shape(self):
        """A plant scenario runs and reports steps, final state, tracking and action."""
        data = json.loads(simulate_scenario(SMC_RUN, steps=800, seed=0, samples=4))
        assert data["scenario"] == "smc_pendulum_run"
        assert data["steps"] == 800
        assert data["dt"] == pytest.approx(0.01)
        assert len(data["final"]["plant_state"]) == 2
        assert len(data["final"]["estimated"]) == 2
        assert data["all_finite"] is True
        assert data["max_abs_action"] is not None
        assert "check" in data

    def test_tracking_is_reported_and_improves(self):
        """The pendulum starts 0.2 rad off and the SMC drives it to ~zero."""
        data = json.loads(simulate_scenario(SMC_RUN, seed=0, samples=0))
        tracking = data["tracking"]
        # Starts at 0.2 rad; ends within a few milliradians.
        assert tracking["max"] == pytest.approx(0.2, abs=0.01)
        assert tracking["final"] < 0.05
        assert tracking["mean"] < tracking["max"]

    def test_steps_default_to_the_scenario_duration(self):
        """duration 8.0 s / dt 0.01 s = 800 steps when `steps` is omitted."""
        assert json.loads(simulate_scenario(SMC_RUN, samples=0))["steps"] == 800

    def test_deterministic_for_a_given_seed(self):
        """Measurement noise is seeded, so two identical runs agree exactly."""
        first = json.loads(simulate_scenario(SMC_RUN, steps=100, seed=7, samples=0))
        second = json.loads(simulate_scenario(SMC_RUN, steps=100, seed=7, samples=0))
        assert first["final"] == second["final"]
        assert first["tracking"] == second["tracking"]


class TestTrajectory:
    """The downsampled history, which is what makes a run inspectable."""

    def test_trajectory_is_evenly_downsampled(self):
        """`samples` points span the run, first to last."""
        data = json.loads(simulate_scenario(SMC_RUN, steps=100, samples=5))
        traj = data["trajectory"]
        assert len(traj["t"]) == 5
        assert traj["t"][0] == 0.0
        assert traj["t"][-1] == pytest.approx(0.99, abs=0.02)
        assert len(traj["plant_state"][0]) == 2

    def test_samples_zero_omits_the_trajectory(self):
        """Summary-only runs stay small."""
        data = json.loads(simulate_scenario(SMC_RUN, steps=20, samples=0))
        assert "trajectory" not in data


class TestErrors:
    """Bad input is an error payload, not an exception."""

    def test_missing_scenario_file(self):
        """A nonexistent path is reported."""
        data = json.loads(simulate_scenario("nope.toml"))
        assert "error" in data

    def test_compile_only_scenario_cannot_run(self):
        """A scenario with no [plant]/[sim] declares dims only — it cannot be simulated."""
        data = json.loads(simulate_scenario(COMPILE_ONLY))
        assert "error" in data
