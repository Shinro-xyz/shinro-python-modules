"""Tests for the graph-recipe registry (:mod:`shinro.codegen.recipes`).

The registry is the named seam the e2e pipeline dispatches through: a scenario's
``[compile].recipe`` selects the compose wiring. These tests pin the shipped
recipes, the dispatch, and the selection rules (explicit wins; only a policy
controller may omit ``[estimator]``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shinro.codegen.recipes import (
    available_graphs,
    build_base_graph,
    build_recipe,
    register_graph,
)
from shinro.codegen.scenario_gen import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_TRACKING = str(REPO_ROOT / "tests" / "integration" / "scenarios" / "base_tracking.toml")
TOY_POLICY = str(REPO_ROOT / "tests" / "fixtures" / "configs" / "scenarios" / "toy_mlp_policy.toml")


def _scenario(tmp_path, body: str):
    path = tmp_path / "scenario.toml"
    path.write_text(body)
    return str(path)


class TestRegistry:
    def test_ships_two_recipes(self):
        assert available_graphs() == ["closed_loop_tracking", "policy_only"]

    def test_unknown_recipe_raises(self):
        with pytest.raises(ValueError, match="unknown graph recipe"):
            build_recipe("nope", {})

    def test_duplicate_registration_raises(self):
        def _dup(spec):
            return build_base_graph()

        with pytest.raises(ValueError, match="already registered"):
            register_graph("closed_loop_tracking")(_dup)


class TestClosedLoopRecipe:
    def test_matches_shipped_base_graph(self):
        """The recipe reproduces the shipped KF+LQR graph node-for-node."""
        spec = load_scenario(BASE_TRACKING)
        assert spec["compile"]["recipe"] == "closed_loop_tracking"

        via_recipe = build_recipe("closed_loop_tracking", spec)
        shipped = build_base_graph()
        assert len(via_recipe.graph.nodes) == len(shipped.graph.nodes)
        assert via_recipe.inputs == shipped.inputs
        assert via_recipe.outputs == shipped.outputs
        assert via_recipe.state_outputs == shipped.state_outputs


class TestPolicyOnlyRecipe:
    def test_toy_policy(self):
        pytest.importorskip("onnx")
        spec = load_scenario(TOY_POLICY)
        assert spec["compile"]["recipe"] == "policy_only"

        cg = build_recipe("policy_only", spec)
        assert cg.inputs == ["state"]
        assert cg.outputs == ["u"]
        assert cg.state_outputs == []


class TestPlantDerivedDims:
    """A plant-only [plant] is authoritative: dims are derived, mismatches are loud."""

    @staticmethod
    def _cartpole(tmp_path, compile_body):
        return _scenario(
            tmp_path,
            '[plant]\ntype = "CartPole"\nconfig = "configs/plants/cartpole.toml"\n'
            '[controller]\ntype = "LQR"\nconfig = "configs/controllers/lqr_cartpole.toml"\n'
            '[estimator]\ntype = "KalmanFilter"\nconfig = "configs/estimators/kalman_cartpole.toml"\n'
            "[compile]\n" + compile_body,
        )

    def test_omitted_dims_are_derived(self, tmp_path):
        spec = load_scenario(self._cartpole(tmp_path, ""))
        assert spec["compile"]["n_x"] is None and spec["compile"]["n_u"] is None
        # CartPole -> (4, 1); the graph builds from the derived dims.
        build_recipe("closed_loop_tracking", spec)

    def test_matching_dims_accepted(self, tmp_path):
        spec = load_scenario(self._cartpole(tmp_path, "n_x = 4\nn_u = 1\n"))
        build_recipe("closed_loop_tracking", spec)

    def test_n_x_mismatch_is_loud(self, tmp_path):
        spec = load_scenario(self._cartpole(tmp_path, "n_x = 99\nn_u = 1\n"))
        with pytest.raises(ValueError, match="n_x=99 disagrees with plant"):
            build_recipe("closed_loop_tracking", spec)

    def test_n_u_mismatch_is_loud(self, tmp_path):
        spec = load_scenario(self._cartpole(tmp_path, "n_x = 4\nn_u = 99\n"))
        with pytest.raises(ValueError, match="n_u=99 disagrees with plant"):
            build_recipe("closed_loop_tracking", spec)


class TestRecipeSelection:
    def test_explicit_recipe_wins(self, tmp_path):
        """[compile].recipe is honoured even when [estimator] is present."""
        scenario = _scenario(
            tmp_path,
            '[controller]\nconfig = "configs/controllers/lqr_base.toml"\n'
            '[estimator]\nconfig = "configs/estimators/kalman_base.toml"\n'
            '[compile]\nn_x = 3\nn_u = 3\nrecipe = "closed_loop_tracking"\n',
        )
        assert load_scenario(scenario)["compile"]["recipe"] == "closed_loop_tracking"

    def test_inferred_default_is_closed_loop(self, tmp_path):
        scenario = _scenario(
            tmp_path,
            '[controller]\nconfig = "configs/controllers/lqr_base.toml"\n'
            '[estimator]\nconfig = "configs/estimators/kalman_base.toml"\n'
            '[compile]\nn_x = 3\nn_u = 3\n',
        )
        assert load_scenario(scenario)["compile"]["recipe"] == "closed_loop_tracking"

    def test_unknown_recipe_is_rejected(self, tmp_path):
        scenario = _scenario(
            tmp_path,
            '[controller]\nconfig = "configs/controllers/lqr_base.toml"\n'
            '[estimator]\nconfig = "configs/estimators/kalman_base.toml"\n'
            '[compile]\nn_x = 3\nn_u = 3\nrecipe = "bogus"\n',
        )
        with pytest.raises(ValueError, match=r"unknown \[compile\]\.recipe"):
            load_scenario(scenario)

    def test_non_policy_controller_still_needs_estimator(self, tmp_path):
        """Inferring policy_only from a missing [estimator] must not admit an LQR."""
        scenario = _scenario(
            tmp_path,
            '[controller]\nconfig = "configs/controllers/lqr_base.toml"\n'
            '[compile]\nn_x = 3\nn_u = 3\n',
        )
        with pytest.raises(ValueError, match=r"missing \[estimator\]"):
            load_scenario(scenario)
