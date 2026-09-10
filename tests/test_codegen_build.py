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

import tomllib

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


def test_inject_model_derives_from_plant():
    """inject_model fills A/B from the plant and leaves explicit configs alone."""
    from shinro.factories.registry import _PLANT_REGISTRY
    from shinro.utils.array_backend import NumpyBackend
    from shinro.utils.config_resolver import resolve_config_path
    from shinro.utils.linearization import derive_model, inject_model

    with open(resolve_config_path("configs/plants/cartpole.toml"), "rb") as f:
        plant = _PLANT_REGISTRY["CartPole"].from_config(tomllib.load(f), backend=NumpyBackend())

    # Omitted A/B → derived from the plant (4x4 A, 4x1 B — not the dt*I default).
    cfg = inject_model({"type": "LQR", "dt": 0.01}, plant)
    A_d, B_d = derive_model(plant)
    assert np.allclose(cfg["A_dynamics"], A_d)
    assert np.allclose(cfg["B_dynamics"], B_d)
    assert np.asarray(cfg["B_dynamics"]).shape == (4, 1)

    # Explicit A/B → untouched (explicit model wins over derived).
    explicit = {"type": "LQR", "dt": 0.01, "A_dynamics": [[1.0, 0.0], [0.0, 1.0]]}
    cfg2 = inject_model(explicit, plant)
    assert cfg2["A_dynamics"] == explicit["A_dynamics"]
    assert "B_dynamics" not in cfg2


def test_derived_plant_scenario_matches_explicit(tmp_path):
    """A [plant] scenario (no A/B, no n_x/n_u) composes the SAME graph as the
    explicit-A/B path — the derivation is faithful, not a reimplementation."""
    from shinro.codegen.scenario_gen import gen_scenario
    from shinro.factories.registry import _PLANT_REGISTRY
    from shinro.utils.array_backend import NumpyBackend
    from shinro.utils.config_resolver import resolve_config_path
    from shinro.utils.linearization import derive_model

    # 1. Derived path: gen_scenario on the [plant] scenario (omits A/B + dims).
    cg_derived, _ = gen_scenario(
        "configs/scenarios/cartpole_lqr_kf_derived.toml", str(tmp_path / "derived")
    )

    # 2. Explicit path: write configs with A/B from the plant, build directly.
    with open(resolve_config_path("configs/plants/cartpole.toml"), "rb") as f:
        plant = _PLANT_REGISTRY["CartPole"].from_config(tomllib.load(f), backend=NumpyBackend())
    A_d, B_d = derive_model(plant)
    est = tmp_path / "est.toml"
    est.write_text(
        'type = "KalmanFilter"\nname = "kf"\ndt = 0.01\n'
        "process_noise = [0.001, 0.01, 0.001, 0.01]\n"
        "measurement_noise = [0.005, 0.05, 0.005, 0.05]\n"
        f"A_dynamics = {A_d.tolist()}\nB_dynamics = {B_d.tolist()}\n"
    )
    ctrl = tmp_path / "ctrl.toml"
    ctrl.write_text(
        'type = "LQR"\nname = "lqr"\ndt = 0.01\n'
        "state_cost = [20.0, 2.0, 100.0, 10.0]\ncontrol_cost = [0.1]\n"
        f"A_dynamics = {A_d.tolist()}\nB_dynamics = {B_d.tolist()}\n"
    )
    cg_explicit = build_composed_graph(
        str(est), str(ctrl), n_x=4, n_u=1,
        input_limits=(np.array([-10.0]), np.array([10.0])),
    )

    _assert_graphs_identical(cg_derived, cg_explicit)
