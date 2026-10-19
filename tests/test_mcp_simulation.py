"""Tests for `simulate_scenario` — the design-time closed loop.

This is the tool that lets an agent *tune*: it re-runs the library's own
`run_scenario` and reports how well the loop tracked. The assertions below use the
SMC-on-a-pendulum fixture, whose controller takes its model from the plant, so a pass
also exercises the plant seam end to end.
"""

import json

import pytest

from shinro.mcp.notebook_server import compare_runs, record_run
from shinro.mcp.server import simulate_scenario, sweep_scenarios

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


SMC_CONFIG = "samples/controllers/smc_inverted_pendulum.toml"


def _smc_variant(tmp_path, label: str, **overrides) -> str:
    """Write an SMC config derived from the fixture, with keys overridden."""
    from pathlib import Path

    text = Path(SMC_CONFIG).read_text(encoding="utf-8")
    for key, value in overrides.items():
        # The fixture has one `key = value` line per setting.
        lines = [f"{key} = {value}" if line.startswith(f"{key} =") else line for line in text.splitlines()]
        text = "\n".join(lines) + "\n"
    target = tmp_path / f"{label}.toml"
    target.write_text(text, encoding="utf-8")
    return str(target)


class TestSweepScenarios:
    """Ranking trials — the compare step of a design loop."""

    def test_ranks_candidates_by_the_objective(self, tmp_path):
        """Two SMC gains are ranked ascending by final tracking error, ranks 1..n."""
        soft = _smc_variant(tmp_path, "soft", k1=0.5)
        stiff = _smc_variant(tmp_path, "stiff", k1=5.0)
        data = json.loads(
            sweep_scenarios(
                [{"label": "soft", "controller_config": soft}, {"label": "stiff", "controller_config": stiff}],
                scenario_path=SMC_RUN,
                steps=300,
                seed=0,
                objective="tracking.final",
                direction="min",
            )
        )
        assert data["trials"] == 2
        assert [r["rank"] for r in data["ranked"]] == [1, 2]
        values = [r["value"] for r in data["ranked"]]
        assert values == sorted(values), "min direction must sort ascending"
        assert data["best"]["label"] == data["ranked"][0]["label"]
        # The winner's file is reported so the caller can keep it.
        assert data["best"]["controller_config"] in (soft, stiff)

    def test_candidates_may_be_bare_scenario_paths(self):
        """The same scenario twice is two equal trials — candidates can be paths."""
        data = json.loads(sweep_scenarios([SMC_RUN, SMC_RUN], steps=50, seed=0, objective="tracking.final"))
        assert data["trials"] == 2
        assert len(data["ranked"]) == 2
        assert data["ranked"][0]["value"] == pytest.approx(data["ranked"][1]["value"])

    def test_a_bad_candidate_is_reported_not_fatal(self, tmp_path):
        """One unbuildable trial lands in `failed`; the others still rank."""
        good = _smc_variant(tmp_path, "good", k1=1.0)
        data = json.loads(
            sweep_scenarios(
                [
                    {"label": "good", "controller_config": good},
                    {"label": "broken", "controller_config": str(tmp_path / "missing.toml")},
                ],
                scenario_path=SMC_RUN,
                steps=50,
                seed=0,
            )
        )
        assert [r["label"] for r in data["ranked"]] == ["good"]
        assert [f["label"] for f in data["failed"]] == ["broken"]
        assert "error" in data["failed"][0]

    def test_max_trials_is_enforced(self):
        """A runaway sweep is refused rather than running for minutes."""
        data = json.loads(sweep_scenarios([SMC_RUN] * 3, steps=10, max_trials=2))
        assert "error" in data
        assert "max_trials" in data["error"]

    def test_unknown_objective_reports_which_candidates_lacked_it(self):
        """An objective that is not a number is per-trial failure, not a crash."""
        data = json.loads(sweep_scenarios([SMC_RUN], steps=20, objective="tracking.nope"))
        assert data["ranked"] == []
        assert data["best"] is None
        assert "tracking.nope" in data["failed"][0]["error"]

    def test_direction_max_sorts_descending(self):
        """`direction: "max"` ranks the largest value first."""
        data = json.loads(
            sweep_scenarios([SMC_RUN, SMC_RUN], steps=50, seed=0, objective="max_abs_action", direction="max")
        )
        values = [r["value"] for r in data["ranked"]]
        assert values == sorted(values, reverse=True)

    def test_bad_direction_is_an_error(self):
        """Only min/max are meaningful."""
        assert "error" in json.loads(sweep_scenarios([SMC_RUN], direction="sideways"))


class TestCompareRuns:
    """The notebook side of the same loop: rank history by a metric."""

    NOTEBOOK = None  # set per-test from tmp_path

    def _seed(self, tmp_path, rows):
        notebook = str(tmp_path / "exp.jsonl")
        for name, metrics, tags in rows:
            record_run(notebook, name, metrics=metrics, tags=tags)
        return notebook

    def test_ranks_ascending_by_default(self, tmp_path):
        """Lower is better by default; best mirrors rank 1."""
        notebook = self._seed(
            tmp_path,
            [("slow", {"final_error": 0.5}, []), ("fast", {"final_error": 0.1}, []), ("mid", {"final_error": 0.3}, [])],
        )
        data = json.loads(compare_runs(notebook, metric="metrics.final_error"))
        assert [r["value"] for r in data["ranked"]] == [0.1, 0.3, 0.5]
        assert data["best"]["name"] == "fast"
        assert data["considered"] == 3

    def test_direction_max(self, tmp_path):
        """`max` is for metrics like artifact size or stability margin."""
        notebook = self._seed(tmp_path, [("a", {"bytes": 10}, []), ("b", {"bytes": 30}, [])])
        data = json.loads(compare_runs(notebook, metric="metrics.bytes", direction="max"))
        assert [r["value"] for r in data["ranked"]] == [30.0, 10.0]

    def test_tags_and_name_filters(self, tmp_path):
        """Only matching runs are considered."""
        notebook = self._seed(
            tmp_path,
            [
                ("motor-a", {"e": 0.2}, ["motor"]),
                ("motor-b", {"e": 0.1}, ["motor", "baseline"]),
                ("pendulum", {"e": 0.05}, ["pendulum"]),
            ],
        )
        by_tag = json.loads(compare_runs(notebook, metric="metrics.e", tags=["motor"]))
        assert {r["name"] for r in by_tag["ranked"]} == {"motor-a", "motor-b"}
        by_name = json.loads(compare_runs(notebook, metric="metrics.e", name="pend"))
        assert [r["name"] for r in by_name["ranked"]] == ["pendulum"]

    def test_missing_metric_is_counted_not_ranked(self, tmp_path):
        """A run without the metric does not poison the comparison."""
        notebook = self._seed(tmp_path, [("has", {"e": 0.2}, []), ("has-not", {"other": 1.0}, [])])
        data = json.loads(compare_runs(notebook, metric="metrics.e"))
        assert data["considered"] == 2
        assert data["missing_metric"] == 1
        assert [r["name"] for r in data["ranked"]] == ["has"]

    def test_no_run_has_the_metric_lists_available_ones(self, tmp_path):
        """The error tells the caller what it could compare instead."""
        notebook = self._seed(tmp_path, [("a", {"final_error": 0.2}, [])])
        data = json.loads(compare_runs(notebook, metric="metrics.nope"))
        assert "error" in data
        assert "metrics.final_error" in data["available_metrics"]

    def test_nested_metric_path(self, tmp_path):
        """Dotted paths reach into nested metric dicts."""
        notebook = self._seed(tmp_path, [("a", {"check": {"steady_state": {"error": 0.04}}}, [])])
        data = json.loads(compare_runs(notebook, metric="metrics.check.steady_state.error"))
        assert data["best"]["value"] == 0.04
