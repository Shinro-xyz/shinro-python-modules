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
from dataclasses import dataclass

import numpy as np
import pytest

from shinro.codegen import build_composed_graph, interpret
from shinro.codegen.compose import ComposedGraph, _lookup_input_shape
from shinro.components import StateEstimator
from shinro.factories.registry import register_estimator

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
        "samples/estimators/kalman_base.toml",
        "samples/controllers/lqr_base.toml",
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
        "samples/estimators/kalman_base.toml",
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

    with open(resolve_config_path("samples/plants/cartpole.toml"), "rb") as f:
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


def test_derive_model_falls_back_to_get_model():
    """Velocity-commanded plants (no dynamics/input_dim) derive via get_model."""
    from shinro.factories.registry import _PLANT_REGISTRY
    from shinro.utils.array_backend import NumpyBackend
    from shinro.utils.config_resolver import resolve_config_path
    from shinro.utils.linearization import derive_model

    with open(resolve_config_path("samples/plants/holonomic_base.toml"), "rb") as f:
        plant = _PLANT_REGISTRY["HolonomicMobileRobot"].from_config(
            tomllib.load(f), backend=NumpyBackend()
        )
    A_d, B_d = derive_model(plant)
    assert np.asarray(A_d).shape == (3, 3)
    assert np.asarray(B_d).shape == (3, 3)
    assert np.allclose(A_d, np.eye(3))  # velocity-commanded: A = I, B = dt*I


def test_derive_model_multi_input():
    """Multi-input plants derive the right B shape (n_u > 1)."""
    from shinro.factories.registry import _PLANT_REGISTRY
    from shinro.utils.array_backend import NumpyBackend
    from shinro.utils.config_resolver import resolve_config_path
    from shinro.utils.linearization import derive_model

    with open(resolve_config_path("samples/plants/double_pendulum.toml"), "rb") as f:
        plant = _PLANT_REGISTRY["DoublePendulum"].from_config(
            tomllib.load(f), backend=NumpyBackend()
        )
    A_d, B_d = derive_model(plant)
    assert np.asarray(A_d).shape == (4, 4)
    assert np.asarray(B_d).shape == (4, 2)  # n_u = 2, not the dt*I (4,4) default


class TestInjectPlantDerived:
    """inject_plant_derived: dt filled/checked, model derived — explicit wins, disagreement loud."""

    @pytest.fixture()
    def cartpole(self):
        from shinro.factories.registry import _PLANT_REGISTRY
        from shinro.utils.array_backend import NumpyBackend
        from shinro.utils.config_resolver import resolve_config_path

        with open(resolve_config_path("samples/plants/cartpole.toml"), "rb") as f:
            return _PLANT_REGISTRY["CartPole"].from_config(tomllib.load(f), backend=NumpyBackend())

    def _lqr_cfg(self):
        from shinro.controllers.lqr import LQRConfig
        return LQRConfig(state_cost=[20.0, 2.0, 100.0, 10.0], control_cost=[0.1])

    def test_dt_filled_from_plant_when_omitted(self, cartpole):
        from shinro.utils.linearization import inject_plant_derived

        cfg = inject_plant_derived(self._lqr_cfg(), cartpole, with_model=True)
        assert cfg.dt == cartpole.dt
        assert cfg.A_dynamics is not None and cfg.B_dynamics is not None

    def test_matching_dt_left_alone(self, cartpole):
        from dataclasses import replace

        from shinro.utils.linearization import inject_plant_derived

        cfg = replace(self._lqr_cfg(), dt=cartpole.dt)
        out = inject_plant_derived(cfg, cartpole, with_model=False)
        assert out.dt == cartpole.dt
        assert out.A_dynamics is None  # no derive_model: model untouched

    def test_dt_mismatch_is_loud(self, cartpole):
        from dataclasses import replace

        from shinro.utils.linearization import inject_plant_derived

        cfg = replace(self._lqr_cfg(), dt=cartpole.dt * 5)
        with pytest.raises(ValueError, match="disagrees with plant dt"):
            inject_plant_derived(cfg, cartpole, with_model=False)

    def test_model_derived_only_when_both_absent(self, cartpole):
        from dataclasses import replace

        from shinro.utils.linearization import inject_plant_derived

        # Explicit A blocks model injection even in derive_model mode.
        cfg = replace(self._lqr_cfg(), dt=cartpole.dt, A_dynamics=[[1.0, 0.0], [0.0, 1.0]])
        out = inject_plant_derived(cfg, cartpole, with_model=True)
        assert out.A_dynamics == [[1.0, 0.0], [0.0, 1.0]]
        assert out.B_dynamics is None

    def test_components_without_dt_fields_are_skipped(self, cartpole):
        """PIDConfig has no A/B fields; injection must not touch it beyond dt."""
        from dataclasses import replace

        from shinro.controllers.pid import PIDConfig
        from shinro.utils.linearization import inject_plant_derived

        cfg = replace(PIDConfig(kp=[1.0], ki=[0.1], kd=[0.01]), dt=None)
        out = inject_plant_derived(cfg, cartpole, with_model=True)
        assert out.dt == cartpole.dt


class TestLoadConfigPlantInjection:
    """ConfigDriven.load_config: parse → plant injection, end to end."""

    @pytest.fixture()
    def cartpole(self):
        from shinro.factories.registry import _PLANT_REGISTRY
        from shinro.utils.array_backend import NumpyBackend
        from shinro.utils.config_resolver import resolve_config_path

        with open(resolve_config_path("samples/plants/cartpole.toml"), "rb") as f:
            return _PLANT_REGISTRY["CartPole"].from_config(tomllib.load(f), backend=NumpyBackend())

    def test_lqr_constructs_from_dtless_config_with_plant(self, cartpole):
        """The shipped lqr_cartpole.toml (no dt) builds via the plant-derived path."""
        from shinro.controllers.lqr import LQR
        from shinro.utils.array_backend import NumpyBackend

        cfg = LQR.load_config("tests/fixtures/configs/controllers/lqr_cartpole.toml", plant=cartpole, derive_model=True)
        assert cfg.dt == cartpole.dt
        lqr = LQR.from_config(cfg, backend=NumpyBackend())
        assert lqr.K is not None

    def test_factory_create_with_plant_injects(self, cartpole):
        """ControllerFactory.create(plant=...) fills dt and derives the model."""
        from shinro.factories.controller_factory import ControllerFactory
        from shinro.utils.array_backend import NumpyBackend

        factory = ControllerFactory(config={"type": "LQR", "name": "x", "state_cost": [1.0, 1.0, 1.0, 1.0], "control_cost": [0.1]})
        ctrl = factory.create(backend=NumpyBackend(), plant=cartpole, derive_model=True)
        A_d, B_d = cartpole.get_model()
        assert np.allclose(np.asarray(ctrl.A), np.asarray(A_d))
        assert np.allclose(np.asarray(ctrl.B), np.asarray(B_d))
        assert np.asarray(ctrl.B).shape == (4, 1)


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
        "tests/fixtures/configs/scenarios/cartpole_lqr_kf_derived.toml", str(tmp_path / "derived")
    )

    # 2. Explicit path: write configs with A/B from the plant, build directly.
    with open(resolve_config_path("samples/plants/cartpole.toml"), "rb") as f:
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


def test_mppi_composes_with_a_host_filled_epsilon_port():
    """MPPI's sampled perturbations are a free host input, not an estimator feed.

    The compile path attaches the plant (MPPI's dynamics/cost come from its
    model) and then composes; ``epsilon`` is declared with MPPI's own shape —
    ``(N, K*D_u)``, sample-major — and appended last, so the
    estimator/controller ports ahead of it are unchanged.
    """
    from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
    from shinro.utils.array_backend import NumpyBackend

    plant = HolonomicMobileRobot(
        num_wheels=3, radius_robots=0.1, gamma=0.0, radius_wheels=0.03, dt=0.02, backend=NumpyBackend()
    )
    cg = build_composed_graph(
        "samples/estimators/kalman_base.toml",
        "samples/controllers/mppi_base.toml",
        3,
        3,
        input_limits=_BASE_LIMITS,
        plant=plant,
    )

    # The shipped port layout is unchanged; epsilon arrives last.
    assert cg.inputs[:5] == ["y", "x_ref", "u_prev", "state_x_hat", "state_P"]
    assert cg.inputs[-1] == "epsilon"
    assert _lookup_input_shape(cg.graph, "epsilon", default=()) == (200, 15 * 3)
    # MPPI's diagnostic survives composition, appended after the control output.
    assert cg.outputs == ["u", "costs"]

    feeds = {name: np.zeros(_lookup_input_shape(cg.graph, name, default=())) for name in cg.inputs}
    out = interpret(cg.graph, feeds)
    assert np.asarray(out["u"]).shape == (3,)
    assert np.asarray(out["costs"]).shape == (200,)


# ─── a hypothetical third estimator: does the pipeline generalize? ────────


@dataclass(frozen=True)
class _LeakyAverageConfig:
    """Strict config for the hypothetical estimator below."""

    n_x: int
    alpha: float = 0.5
    name: str = "leaky_average"


@register_estimator("LeakyAverage")
class _LeakyAverageEstimator(StateEstimator):
    """A plausible *new* estimator: an exponential moving average of the measurement.

    Deliberately not shipped — it exists to prove the compile pipeline
    generalizes past the two registered estimators. It has one recurrent attr
    (``x_hat``) that no tracer code declares: ``build_composed_graph``'s
    two-pass state discovery finds it by attr-diff and promotes it to a
    ``state_x_hat`` port. ``estimate`` is operator/``bk`` only, so it traces.
    """

    Config = _LeakyAverageConfig

    def __init__(self, n_x: int, alpha: float = 0.5, backend=None):
        from shinro.utils.array_backend import NumpyBackend

        self.bk = backend or NumpyBackend()
        self.n_x = n_x
        self.alpha = float(alpha)
        self.x_hat = self.bk.zeros(n_x)

    def estimate(self, measurement, control_input):
        y = self.bk.ravel(measurement)
        self.x_hat = (1.0 - self.alpha) * self.x_hat + self.alpha * y
        return self.bk.reshape(self.x_hat, (self.n_x, 1))

    def reset(self):
        self.x_hat = self.bk.zeros(self.n_x)

    @classmethod
    def from_config(cls, config, backend=None):
        cfg = cls.parse_config(config)
        return cls(n_x=cfg.n_x, alpha=cfg.alpha, backend=backend)


def test_composes_with_a_hypothetical_third_estimator():
    """A brand-new estimator composes with MPPI and matches the live loop.

    Nothing in the pipeline is estimator-aware: the class is registered ad hoc,
    its inputs follow the ``measurement`` / ``control_input`` naming contract,
    and its undiscovered state is promoted automatically. The composed graph
    must then run tick-for-tick like the live components, with MPPI's
    host-supplied ``epsilon`` alongside the new estimator's state port.
    """
    from shinro.codegen import interpret
    from shinro.factories.controller_factory import ControllerFactory
    from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
    from shinro.utils.array_backend import NumpyBackend

    bk = NumpyBackend()
    plant = HolonomicMobileRobot(
        num_wheels=3, radius_robots=0.1, gamma=0.0, radius_wheels=0.03, dt=0.02, backend=bk
    )
    est_cfg = {"type": "LeakyAverage", "n_x": 3, "alpha": 0.4}
    cg = build_composed_graph(
        est_cfg,
        "samples/controllers/mppi_base.toml",
        3,
        3,
        input_limits=_BASE_LIMITS,
        plant=plant,
    )

    # The new estimator's state became a recurrent port; epsilon is still free
    # and still last, and the controller never learned about either change.
    assert cg.inputs == ["y", "x_ref", "u_prev", "state_x_hat", "state_u", "epsilon"]
    assert "state_x_hat" in cg.state_outputs
    assert "state_u" in cg.state_outputs

    # Tick-for-tick against the live components (same order as the graph).
    est = _LeakyAverageEstimator.from_config(est_cfg, backend=NumpyBackend())
    ctrl = ControllerFactory("samples/controllers/mppi_base.toml").create(backend=NumpyBackend())
    ctrl.attach_plant(plant)

    rng = np.random.default_rng(5)
    n_x, n_u, n_samples, horizon = 3, 3, ctrl.N, ctrl.K
    x_hat = np.zeros(n_x)
    plan = np.zeros((horizon, n_u))
    u_prev = np.zeros(n_u)

    for trial in range(3):
        y = rng.normal(0.0, 0.2, n_x)
        x_ref = rng.normal(0.0, 0.5, n_x)
        eps = rng.normal(0.0, 0.3, (n_samples, horizon * n_u))

        traced = interpret(
            cg.graph,
            {
                "y": y,
                "x_ref": x_ref,
                "u_prev": u_prev,
                "state_x_hat": x_hat,
                "state_u": plan,
                "epsilon": eps,
            },
        )

        # Live reference: estimator, then controller, then the composed clip.
        x_hat_next = np.asarray(est.estimate(y.reshape(-1, 1), u_prev.reshape(-1, 1))).ravel()
        u_live = np.clip(
            np.asarray(ctrl.compute(x_hat_next, x_ref, eps)), _BASE_LIMITS[0], _BASE_LIMITS[1]
        )

        np.testing.assert_allclose(traced["u"], u_live, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            np.asarray(traced["state_x_hat"]).ravel(), x_hat_next, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(np.asarray(traced["state_u"]), ctrl.u, rtol=1e-12, atol=1e-12)

        # Carry the live state forward; the graph's state outputs match it.
        x_hat = x_hat_next
        plan = ctrl.u.copy()
        u_prev = u_live
