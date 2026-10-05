"""Tests for gate A — ``interpret(composed graph)`` vs the live components.

Gate A is the pre-compile equivalence check the build now runs (see
:mod:`shinro.codegen.gate_a`): the composed graph must reproduce what the real
estimator/controller compute over a closed loop. These pin that the shipped
scenario cases pass with zero error and that a real divergence is caught.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shinro.codegen.gate_a import run_gate_a
from shinro.codegen.recipes import build_recipe, live_components
from shinro.codegen.scenario_gen import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = str(REPO_ROOT / "tests" / "integration" / "scenarios" / "base_tracking.toml")
MPPI = str(REPO_ROOT / "tests" / "integration" / "scenarios" / "mppi_compile.toml")
MPC = str(REPO_ROOT / "tests" / "integration" / "scenarios" / "mpc_compile.toml")

TOL = 1e-9


@pytest.mark.parametrize("scenario", [BASE, MPPI, MPC], ids=["kf_lqr", "mppi", "kf_mpc"])
def test_gate_a_matches_live_components(scenario):
    """The composed graph reproduces the live estimator+controller exactly."""
    spec = load_scenario(scenario)
    cg = build_recipe("closed_loop_tracking", spec)
    est, ctrl, n_x, n_u, limits = live_components(spec)
    err = run_gate_a(cg, est, ctrl, n_x, n_u, input_limits=limits, ticks=50)
    assert err <= TOL, f"gate A diverged: {err:.3e}"


def test_gate_a_flags_a_wrong_live_component():
    """A live controller whose gain differs from the graph's is caught."""
    spec = load_scenario(BASE)
    cg = build_recipe("closed_loop_tracking", spec)
    est, ctrl, n_x, n_u, limits = live_components(spec)
    ctrl.K = ctrl.K * 1.5  # perturb the live gain away from the traced one
    err = run_gate_a(cg, est, ctrl, n_x, n_u, input_limits=limits, ticks=50)
    assert err > TOL, f"gate A missed a real divergence: {err:.3e}"


SMC = str(REPO_ROOT / "tests" / "integration" / "scenarios" / "smc_pendulum_compile.toml")


def test_gate_a_drives_a_plant_derived_model():
    """SMC's f_x/g_x are composed in from the plant; gate A feeds the live side the
    same plant model (this is what makes a composed SMC meaningful at all)."""
    from shinro.codegen.recipes import live_components_and_plant

    spec = load_scenario(SMC)
    cg = build_recipe("closed_loop_tracking", spec)
    assert "f_x" not in cg.inputs and "g_x" not in cg.inputs, "model terms must be baked, not host ports"

    est, ctrl, n_x, n_u, limits, plant = live_components_and_plant(spec)
    err = run_gate_a(cg, est, ctrl, n_x, n_u, input_limits=limits, ticks=50, plant=plant)
    assert err <= TOL, f"gate A diverged: {err:.3e}"


def test_gate_a_without_a_plant_for_model_terms_is_loud():
    """A controller wanting f_x/g_x with no plant to derive them from raises."""
    from shinro.codegen.recipes import live_components

    spec = load_scenario(SMC)
    cg = build_recipe("closed_loop_tracking", spec)
    est, ctrl, n_x, n_u, limits = live_components(spec)
    with pytest.raises(ValueError, match="plant-model term"):
        run_gate_a(cg, est, ctrl, n_x, n_u, input_limits=limits, ticks=5, plant=None)
