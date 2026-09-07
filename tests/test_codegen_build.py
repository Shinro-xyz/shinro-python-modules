"""Generic composed-graph builder: equivalence with the shipped generators.

The generic :func:`shinro.codegen.build.build_composed_graph` must produce a
graph byte-identical to the LeKiwi-specific ``scripts/gen_base.py`` when given
the same configs and dims — this is the refactor-fidelity guard: the generic
path is a faithful generalization, not a reimplementation. It also proves the
two-pass state discovery works for components with recurrent state (PID's
integral / first-tick gate), which is what makes component swaps (KF →
Luenberger, LQR → PID) compose without per-component declaration.
"""

from __future__ import annotations

import numpy as np

from shinro.codegen import build_composed_graph
from shinro.codegen.compose import ComposedGraph

# Input limits from base_tracking.toml's [scenario.input_limits].
_BASE_LIMITS = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))


def _node_equal(a, b) -> bool:
    """Compare two graph nodes including ndarray-valued attrs."""
    if (a.op, a.inputs, a.shape) != (b.op, b.inputs, b.shape):
        return False
    if a.attrs.keys() != b.attrs.keys():
        return False
    for key in a.attrs:
        va, vb = a.attrs[key], b.attrs[key]
        if isinstance(va, np.ndarray):
            if not np.array_equal(va, vb):
                return False
        elif va != vb:
            return False
    return True


def _assert_graphs_identical(a: ComposedGraph, b: ComposedGraph) -> None:
    """Assert two composed graphs are node-for-node and port-for-port equal."""
    assert a.inputs == b.inputs
    assert a.outputs == b.outputs
    assert a.state_inputs == b.state_inputs
    assert a.state_outputs == b.state_outputs
    assert len(a.graph.nodes) == len(b.graph.nodes)
    for i, (na, nb) in enumerate(zip(a.graph.nodes, b.graph.nodes)):
        assert _node_equal(na, nb), f"node {i} diverged: {na} vs {nb}"


def test_generic_builder_matches_gen_base():
    """The generic path reproduces the shipped KF + LQR base graph exactly."""
    from scripts.gen_base import build_base_graph

    generic = build_composed_graph(
        "configs/estimators/kalman_base.toml",
        "configs/controllers/lqr_base.toml",
        n_x=3,
        n_u=3,
        input_limits=_BASE_LIMITS,
    )
    _assert_graphs_identical(generic, build_base_graph())


def test_generic_builder_discovers_pid_state(tmp_path):
    """Two-pass trace discovers PID's recurrent state without declaration."""
    # A 3-channel PID config (the shipped pid_arm.toml is 6-DOF; the KF is 3).
    pid_cfg = tmp_path / "pid3.toml"
    pid_cfg.write_text(
        'type = "PID"\n'
        'name = "pid3"\n'
        'dt = 0.02\n'
        'kp = [2.0, 2.0, 2.0]\n'
        'ki = [0.5, 0.5, 0.5]\n'
        'kd = [0.5, 0.5, 0.5]\n'
    )

    cg = build_composed_graph(
        "configs/estimators/kalman_base.toml",
        str(pid_cfg),
        n_x=3,
        n_u=3,
        input_limits=_BASE_LIMITS,
    )

    # PID's integral / prev-error / first-tick gate must be recurrent ports.
    for port in ("state_integral", "state_prev_error", "state_has_run"):
        assert port in cg.state_inputs, f"missing recurrent port {port}"
        assert port in cg.state_outputs, f"missing recurrent output {port}"

    # The estimator's covariance recursion must be live, not frozen.
    assert "state_P" in cg.state_inputs
    assert "state_P" in cg.state_outputs

    # Sanity: the composed graph is a real closed loop (u output, x_hat feed).
    assert cg.outputs == ["u"]
    assert "state_x_hat" in cg.state_inputs
