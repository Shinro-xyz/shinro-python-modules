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
