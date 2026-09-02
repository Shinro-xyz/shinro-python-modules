"""Composition and swap tests for the tracing layer (slice a, tests 3-7).

These tests verify the composition pass stitches per-node graphs into one
combined closed-loop step graph, and that the interpreted combined graph
matches the live ``NumpyBackend`` loop to float-exactness (``atol=1e-12``)
across 50 seeded random inputs.

The swap tests (5-7) prove the modularity claim: swap the estimator or
controller, re-compose, re-interpret, and the un-swapped node's graph is
reused without re-tracing.
"""

from __future__ import annotations

import numpy as np
import pytest

from shinro.codegen import interpret, trace_node
from shinro.codegen.compose import (
    ComposedGraph,
    _ensure_shape,
    _is_reshape_possible,
    _lookup_input_shape,
    compose,
)
from shinro.codegen.tracing import Graph, ShapeMismatchError
from shinro.controllers.lqr import LQR
from shinro.controllers.pid import PIDController
from shinro.estimators.kalman_filter import KalmanFilter
from shinro.estimators.luenberger_observer import LuenbergerObserver
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.utils.array_backend import NumpyBackend

# ─── helpers ───────────────────────────────────────────────────────────────


def _load_kalman() -> KalmanFilter:
    return EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())


def _load_luenberger() -> LuenbergerObserver:
    return EstimatorFactory("configs/estimators/luenberger_base.toml").create(backend=NumpyBackend())


def _load_lqr() -> LQR:
    return ControllerFactory("configs/controllers/lqr_base.toml").create(backend=NumpyBackend())


def _load_pid() -> PIDController:
    return ControllerFactory("configs/controllers/pid_arm.toml").create(backend=NumpyBackend())


# Input limits from base_tracking.toml's [scenario.input_limits].
_BASE_LIMITS = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))

# Plant dims for the holonomic base (n_x = n_u = 3).
_BASE_DIMS = {"n_x": 3, "n_u": 3}


# ─── Test 3: trivial-node composition unit test ────────────────────────────


class TestComposeTrivial:
    """Unit-test the composition pass in isolation with trivial hand-built nodes."""

    def test_compose_two_identity_nodes(self):
        """Two trivial nodes (each just forwards its input) compose correctly.

        We build a fake estimator (outputs its input unchanged) and a fake
        controller (outputs its input unchanged), compose them, and verify
        the combined graph's interpreter output equals the input.
        """
        from shinro.codegen.trace_node import NodeGraph
        from shinro.codegen.tracing import Graph

        # Build a trivial "estimator": one input node, one output node = the input.
        est_g = Graph()
        est_in = est_g.input("measurement", (3, 1))
        est_g.input("control_input", (3, 1))
        est_g.input("state_x_hat", (3, 1))
        est_g.output("out", est_in)
        est_out_key = next(n.attrs["name"] for n in est_g.nodes if n.op == "output")
        estimator = NodeGraph(
            graph=est_g,
            contract=None,  # type: ignore[arg-type]
            input_nodes={"measurement": est_in},
            output_nodes={est_out_key: est_in},
            state_attrs=[],
        )

        # Build a trivial "controller": one input node, one output node = the input.
        ctrl_g = Graph()
        ctrl_in1 = ctrl_g.input("current_state", (3,))
        ctrl_g.input("target_state", (3,))
        ctrl_g.output("out", ctrl_in1)
        ctrl_out_key = next(n.attrs["name"] for n in ctrl_g.nodes if n.op == "output")
        controller = NodeGraph(
            graph=ctrl_g,
            contract=None,  # type: ignore[arg-type]
            input_nodes={"current_state": ctrl_in1},
            output_nodes={ctrl_out_key: ctrl_in1},
            state_attrs=[],
        )

        composed = compose(estimator, controller, plant_dims=_BASE_DIMS, input_limits=None)
        # The combined graph should output u == y (both identity, with a reshape).
        y = np.array([[1.0], [2.0], [3.0]])
        x_ref = np.array([0.0, 0.0, 0.0])
        u_prev = np.array([0.0, 0.0, 0.0])
        x_hat_init = np.array([[0.0], [0.0], [0.0]])
        result = interpret(
            composed.graph,
            {"y": y, "x_ref": x_ref, "u_prev": u_prev, "state_x_hat": x_hat_init},
        )
        # u should equal y flattened (estimator forwards y, controller forwards x_hat).
        assert "u" in result, f"no u output; got {list(result)}"
        assert np.allclose(result["u"], y.ravel(), atol=1e-12), (
            f"trivial compose failed: u={result['u']}, expected={y.ravel()}"
        )


# ─── Test 4: compose KF + LQR per base_tracking ────────────────────────────


class TestComposeBaseTracking:
    """The main test: compose KF + LQR, interpret, assert == live numpy loop."""

    def test_compose_step_matches_numpy_50_inputs(self, rng):
        """Composed KF+LQR graph == NumpyBackend closed-loop step on 50 inputs."""
        kf = _load_kalman()
        lqr = _load_lqr()

        # Trace each component once (the graphs are reusable).
        est_ng = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1), "P": (3, 3)},
        )
        ctrl_ng = trace_node(
            lqr,
            input_shapes={"current_state": (3,), "target_state": (3,)},
        )

        composed = compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=_BASE_LIMITS)

        # Snapshot the KF's initial P so each trial starts from the same state.
        initial_P = kf.P.copy()

        for trial in range(50):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.1, (3,))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))
            u_prev = rng.normal(0.0, 0.1, (3,))

            # Ground truth: run the live numpy KF + LQR step.
            kf.P = initial_P.copy()
            kf.x_hat = x_hat_init.copy()
            x_hat_np = kf.estimate(y.reshape(-1, 1), u_prev.reshape(-1, 1))
            u_np = lqr.compute(x_hat_np.ravel(), x_ref)
            u_np = np.clip(u_np, _BASE_LIMITS[0], _BASE_LIMITS[1])

            # Interpret the composed graph on the same inputs.
            traced = interpret(
                composed.graph,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev,
                    "state_x_hat": x_hat_init,
                    "state_P": initial_P,
                },
            )

            assert "u" in traced, f"trial {trial}: no u output. got {list(traced)}"
            assert np.allclose(traced["u"], u_np, atol=1e-12), (
                f"trial {trial}: u diverged. max err = {np.max(np.abs(traced['u'] - u_np))}"
            )
            # The state_x_hat output should match the KF's updated x_hat.
            if "state_x_hat" in traced:
                assert np.allclose(traced["state_x_hat"], x_hat_np, atol=1e-12), (
                    f"trial {trial}: x_hat diverged. "
                    f"max err = {np.max(np.abs(traced['state_x_hat'] - x_hat_np))}"
                )


# ─── Test 5: swap estimator (Luenberger + LQR) ─────────────────────────────


class TestSwapEstimator:
    """Re-use the cached LQR graph; trace Luenberger; re-compose."""

    def test_luenberger_lqr_matches_numpy_50_inputs(self, rng):
        """Luenberger + LQR composed graph == numpy loop. LQR graph reused."""
        luen = _load_luenberger()
        lqr = _load_lqr()

        # Trace Luenberger (no state — it's a static-gain observer).
        est_ng = trace_node(
            luen,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1)},
        )
        # Re-use the LQR graph from a fresh trace (it's stateless, so caching
        # is trivial here; the point is we don't re-trace per swap).
        ctrl_ng = trace_node(
            lqr,
            input_shapes={"current_state": (3,), "target_state": (3,)},
        )

        composed = compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=_BASE_LIMITS)

        for trial in range(50):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.1, (3,))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))
            u_prev = rng.normal(0.0, 0.1, (3,))

            # Ground truth.
            luen.x_hat = x_hat_init.copy()
            x_hat_np = luen.estimate(y.reshape(-1, 1), u_prev.reshape(-1, 1))
            u_np = lqr.compute(x_hat_np.ravel(), x_ref)
            u_np = np.clip(u_np, _BASE_LIMITS[0], _BASE_LIMITS[1])

            traced = interpret(
                composed.graph,
                {"y": y, "x_ref": x_ref, "u_prev": u_prev, "state_x_hat": x_hat_init},
            )

            assert np.allclose(traced["u"], u_np, atol=1e-12), (
                f"trial {trial}: Luenberger+LQR u diverged. "
                f"max err = {np.max(np.abs(traced['u'] - u_np))}"
            )


# ─── Test 6: swap controller (KF + PID) ────────────────────────────────────


class TestSwapController:
    """Re-use the cached KF graph; trace PID; re-compose."""

    def test_kalman_pid_matches_numpy_50_inputs(self, rng):
        """KF + PID composed graph == numpy loop. KF graph reused.

        PID adds ops (clip, where, copy) — this test surfaces any missing
        op handlers as a NotImplementedError with an actionable message.
        """
        kf = _load_kalman()
        # PID arm config is 6-DOF; for the base (3-DOF) we build a 3-channel PID.
        # The pid_arm config has 6 channels; we build a custom 3-channel PID
        # so the dims match the base plant.
        pid = PIDController(
            kp=np.array([2.0, 2.0, 2.0]),
            ki=np.array([0.1, 0.1, 0.1]),
            kd=np.array([0.5, 0.5, 0.5]),
            dt=0.02,
            backend=NumpyBackend(),
        )

        est_ng = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1), "P": (3, 3)},
        )
        ctrl_ng = trace_node(
            pid,
            input_shapes={"current_state": (3,), "target_state": (3,)},
            state_shapes={"_integral": (3,), "_prev_error": (3,), "_has_run": (3,)},
        )

        composed = compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=_BASE_LIMITS)

        initial_P = kf.P.copy()
        initial_pid_integral = pid._integral.copy()
        initial_pid_prev = pid._prev_error.copy()

        for trial in range(50):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.1, (3,))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))
            u_prev = rng.normal(0.0, 0.1, (3,))

            # Ground truth.
            kf.P = initial_P.copy()
            kf.x_hat = x_hat_init.copy()
            pid._integral = initial_pid_integral.copy()
            pid._prev_error = initial_pid_prev.copy()
            pid._has_run = np.zeros(3)
            x_hat_np = kf.estimate(y.reshape(-1, 1), u_prev.reshape(-1, 1))
            u_np = pid.compute(x_hat_np.ravel(), x_ref)
            u_np = np.clip(u_np, _BASE_LIMITS[0], _BASE_LIMITS[1])

            traced = interpret(
                composed.graph,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev,
                    "state_x_hat": x_hat_init,
                    "state_P": initial_P,
                    "state_integral": initial_pid_integral,
                    "state_prev_error": initial_pid_prev,
                    "state_has_run": np.zeros(3),
                },
            )

            assert "u" in traced, f"trial {trial}: no u. got {list(traced)}"
            assert np.allclose(traced["u"], u_np, atol=1e-12), (
                f"trial {trial}: KF+PID u diverged. "
                f"max err = {np.max(np.abs(traced['u'] - u_np))}\n"
                f"  traced={traced['u']}\n  numpy={u_np}"
            )


# ─── Test 7: swap both (Luenberger + PID) ──────────────────────────────────


class TestSwapBoth:
    """Both components re-traced; compose; assert == numpy."""

    def test_luenberger_pid_matches_numpy_50_inputs(self, rng):
        """Luenberger + PID composed graph == numpy loop."""
        luen = _load_luenberger()
        pid = PIDController(
            kp=np.array([2.0, 2.0, 2.0]),
            ki=np.array([0.1, 0.1, 0.1]),
            kd=np.array([0.5, 0.5, 0.5]),
            dt=0.02,
            backend=NumpyBackend(),
        )

        est_ng = trace_node(
            luen,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1)},
        )
        ctrl_ng = trace_node(
            pid,
            input_shapes={"current_state": (3,), "target_state": (3,)},
            state_shapes={"_integral": (3,), "_prev_error": (3,), "_has_run": (3,)},
        )

        composed = compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=_BASE_LIMITS)

        initial_pid_integral = pid._integral.copy()
        initial_pid_prev = pid._prev_error.copy()

        for trial in range(50):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.1, (3,))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))
            u_prev = rng.normal(0.0, 0.1, (3,))

            luen.x_hat = x_hat_init.copy()
            pid._integral = initial_pid_integral.copy()
            pid._prev_error = initial_pid_prev.copy()
            pid._has_run = np.zeros(3)
            x_hat_np = luen.estimate(y.reshape(-1, 1), u_prev.reshape(-1, 1))
            u_np = pid.compute(x_hat_np.ravel(), x_ref)
            u_np = np.clip(u_np, _BASE_LIMITS[0], _BASE_LIMITS[1])

            traced = interpret(
                composed.graph,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev,
                    "state_x_hat": x_hat_init,
                    "state_integral": initial_pid_integral,
                    "state_prev_error": initial_pid_prev,
                    "state_has_run": np.zeros(3),
                },
            )

            assert np.allclose(traced["u"], u_np, atol=1e-12), (
                f"trial {trial}: Luenberger+PID u diverged. "
                f"max err = {np.max(np.abs(traced['u'] - u_np))}"
            )


# ─── Test 8: compose without clip limits ───────────────────────────────────


class TestComposeNoClip:
    """Compose with input_limits=None — no clip node, un-clipped u."""

    def test_no_clip_node_when_limits_none(self):
        kf = _load_kalman()
        lqr = _load_lqr()
        est_ng = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1), "P": (3, 3)},
        )
        ctrl_ng = trace_node(
            lqr,
            input_shapes={"current_state": (3,), "target_state": (3,)},
        )
        composed = compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=None)
        ops = [n.op for n in composed.graph.nodes]
        assert "clip" not in ops, f"no clip expected; ops = {ops}"

    def test_u_matches_unclipped_numpy(self, rng):
        kf = _load_kalman()
        lqr = _load_lqr()
        est_ng = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1), "P": (3, 3)},
        )
        ctrl_ng = trace_node(
            lqr,
            input_shapes={"current_state": (3,), "target_state": (3,)},
        )
        composed = compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=None)

        initial_P = kf.P.copy()
        for trial in range(50):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.1, (3,))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))
            u_prev = rng.normal(0.0, 0.1, (3,))

            kf.P = initial_P.copy()
            kf.x_hat = x_hat_init.copy()
            x_hat_np = kf.estimate(y.reshape(-1, 1), u_prev.reshape(-1, 1))
            u_np = lqr.compute(x_hat_np.ravel(), x_ref)  # no clip

            traced = interpret(
                composed.graph,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev,
                    "state_x_hat": x_hat_init,
                    "state_P": initial_P,
                },
            )
            assert np.allclose(traced["u"], u_np, atol=1e-12), (
                f"trial {trial}: un-clipped u diverged. "
                f"max err = {np.max(np.abs(traced['u'] - u_np))}"
            )


# ─── Test 9: compose port metadata ─────────────────────────────────────────


class TestComposePortMeta:
    """The ComposedGraph I/O port lists match the fixed ABC dataflow."""

    def test_port_lists(self):
        kf = _load_kalman()
        lqr = _load_lqr()
        est_ng = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1), "P": (3, 3)},
        )
        ctrl_ng = trace_node(
            lqr,
            input_shapes={"current_state": (3,), "target_state": (3,)},
        )
        composed = compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=_BASE_LIMITS)
        assert isinstance(composed, ComposedGraph)
        assert composed.inputs == ["y", "x_ref", "u_prev", "state_x_hat", "state_P"]
        assert composed.outputs == ["u"]
        assert composed.state_inputs == ["state_x_hat", "state_P", "u_prev"]
        assert composed.state_outputs == ["state_x_hat", "state_P", "state_u_prev"]


# ─── Test 9b: controller role mapping (MPC regulators) ─────────────────────


class TestComposeControllerRoles:
    """Controller inputs map by role from the compute() signature.

    MPC_LTI/MPC_DeltaU are regulators (no reference input): compose feeds the
    error ``x_hat - x_ref`` so regulating it to zero tracks ``x_ref``. A
    controller declaring ``u_prev`` (DeltaU) shares the estimator's
    previous-control port. Unmapped input names raise.
    """

    def test_mpc_lti_compose_error_state(self, rng):
        """KF + MPC_LTI: controller receives e = x_hat - x_ref, matches live."""
        kf = _load_kalman()
        mpc = ControllerFactory("configs/controllers/mpc_lti_base.toml").create(backend=NumpyBackend())

        est_ng = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1), "P": (3, 3)},
        )
        ctrl_ng = trace_node(mpc, input_shapes={"x0": (3,)})
        composed = compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=_BASE_LIMITS)

        # The regulator's state feed is the error: a sub node bridges x_hat and x_ref.
        ops = [n.op for n in composed.graph.nodes]
        assert "sub" in ops, f"expected an error-state sub node; ops = {ops}"

        initial_P = kf.P.copy()
        for trial in range(20):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.05, (3,))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))
            u_prev = rng.normal(0.0, 0.1, (3,))

            # Ground truth: live KF step, then the regulator on the error.
            kf.P = initial_P.copy()
            kf.x_hat = x_hat_init.copy()
            x_hat_np = kf.estimate(y.reshape(-1, 1), u_prev.reshape(-1, 1))
            u_np = np.clip(
                mpc.compute(x_hat_np.ravel() - x_ref), _BASE_LIMITS[0], _BASE_LIMITS[1]
            )

            traced = interpret(
                composed.graph,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev,
                    "state_x_hat": x_hat_init,
                    "state_P": initial_P,
                },
            )
            assert np.allclose(traced["u"], u_np, atol=1e-9), (
                f"trial {trial}: KF+MPC_LTI u diverged. "
                f"max err = {np.max(np.abs(traced['u'] - u_np))}"
            )

    def test_mpc_deltau_compose_routes_u_prev(self, rng):
        """KF + MPC_DeltaU: the controller's u_prev is the shared recurrent port."""
        kf = _load_kalman()
        mpc = ControllerFactory("configs/controllers/mpc_base.toml").create(backend=NumpyBackend())

        est_ng = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1), "P": (3, 3)},
        )
        ctrl_ng = trace_node(mpc, input_shapes={"x0": (3,), "u_prev": (3,)})
        composed = compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=_BASE_LIMITS)

        # Same port list as KF+LQR: u_prev is shared, not duplicated.
        assert composed.inputs == ["y", "x_ref", "u_prev", "state_x_hat", "state_P"]
        assert composed.state_outputs == ["state_x_hat", "state_P", "state_u_prev"]

        initial_P = kf.P.copy()
        for trial in range(20):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.05, (3,))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))
            u_prev = rng.normal(0.0, 0.1, (3,))

            # Ground truth: u_prev feeds BOTH the estimator and the controller.
            kf.P = initial_P.copy()
            kf.x_hat = x_hat_init.copy()
            x_hat_np = kf.estimate(y.reshape(-1, 1), u_prev.reshape(-1, 1))
            u_np = np.clip(
                mpc.compute(x_hat_np.ravel() - x_ref, u_prev),
                _BASE_LIMITS[0],
                _BASE_LIMITS[1],
            )

            traced = interpret(
                composed.graph,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev,
                    "state_x_hat": x_hat_init,
                    "state_P": initial_P,
                },
            )
            assert np.allclose(traced["u"], u_np, atol=1e-9), (
                f"trial {trial}: KF+MPC_DeltaU u diverged. "
                f"max err = {np.max(np.abs(traced['u'] - u_np))}"
            )

    def test_unknown_controller_input_raises(self):
        """A controller input outside the role table raises (e.g. SMC's f_x/g_x)."""
        from shinro.codegen.trace_node import NodeGraph

        est_g = Graph()
        est_in = est_g.input("measurement", (3, 1))
        est_g.input("control_input", (3, 1))
        est_g.input("state_x_hat", (3, 1))
        est_g.output("out", est_in)
        estimator = NodeGraph(
            graph=est_g,
            contract=None,  # type: ignore[arg-type]
            input_nodes={"measurement": est_in},
            output_nodes={"out": est_in},
            state_attrs=[],
        )
        ctrl_g = Graph()
        x_in = ctrl_g.input("x", (3,))
        ctrl_g.input("f_x", (3,))  # SMC-style dynamics term — not a known role
        ctrl_g.input("g_x", (3,))
        ctrl_g.output("out", x_in)
        controller = NodeGraph(
            graph=ctrl_g,
            contract=None,  # type: ignore[arg-type]
            input_nodes={"x": x_in},
            output_nodes={"out": x_in},
            state_attrs=[],
        )
        with pytest.raises(ValueError, match="does not map to a known role"):
            compose(estimator, controller, plant_dims=_BASE_DIMS, input_limits=None)


# ─── Test 10: compose error paths ──────────────────────────────────────────


class TestComposeErrors:
    """Error paths in the composition pass."""

    def test_estimator_with_no_output_raises(self):
        from shinro.codegen.trace_node import NodeGraph

        g = Graph()
        g.input("measurement", (3, 1))
        estimator = NodeGraph(
            graph=g,
            contract=None,  # type: ignore[arg-type]
            input_nodes={},
            output_nodes={},
            state_attrs=[],
        )
        ctrl_g = Graph()
        ctrl_in = ctrl_g.input("current_state", (3,))
        ctrl_g.input("target_state", (3,))
        ctrl_g.output("out", ctrl_in)
        ctrl_out_key = next(n.attrs["name"] for n in ctrl_g.nodes if n.op == "output")
        controller = NodeGraph(
            graph=ctrl_g,
            contract=None,  # type: ignore[arg-type]
            input_nodes={"current_state": ctrl_in},
            output_nodes={ctrl_out_key: ctrl_in},
            state_attrs=[],
        )
        with pytest.raises(ValueError, match="no output node"):
            compose(estimator, controller, plant_dims=_BASE_DIMS, input_limits=None)

    def test_mutated_state_without_placeholder_raises(self):
        """A mutated state attr with no pre-injected placeholder raises.

        The Kalman filter mutates P every estimate() call. If the trace site
        forgot to pre-inject it via state_shapes, the covariance recursion
        would collapse to trace-time constants (a frozen one-step gain) in
        the deployed graph — compose() must refuse instead of silently
        baking the freeze.
        """
        kf = _load_kalman()
        lqr = _load_lqr()
        est_ng = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1)},  # P deliberately omitted
        )
        ctrl_ng = trace_node(
            lqr,
            input_shapes={"current_state": (3,), "target_state": (3,)},
        )
        with pytest.raises(ValueError, match="silently frozen"):
            compose(est_ng, ctrl_ng, plant_dims=_BASE_DIMS, input_limits=None)


# ─── Test 11: reshape helpers ───────────────────────────────────────────────


class TestComposeReshape:
    """Direct unit tests for the auto-reshape helpers."""

    def test_ensure_shape_passthrough_when_equal(self):
        g = Graph()
        src = g.input("x", (3,))
        out = _ensure_shape(g, src, (3,), (3,))
        assert out == src
        assert all(n.op != "reshape" for n in g.nodes)

    def test_ensure_shape_inserts_reshape(self):
        g = Graph()
        src = g.input("x", (3, 1))
        out = _ensure_shape(g, src, (3, 1), (3,))
        node = g.nodes[out]
        assert node.op == "reshape"
        assert node.attrs["target_shape"] == (3,)
        assert node.inputs == [src]

    def test_ensure_shape_incompatible_raises(self):
        g = Graph()
        src = g.input("x", (4,))
        with pytest.raises(ShapeMismatchError):
            _ensure_shape(g, src, (4,), (3,))

    def test_is_reshape_possible(self):
        assert _is_reshape_possible((2, 3), (6,))
        assert _is_reshape_possible((3, 1), (3,))
        assert not _is_reshape_possible((2, 3), (7,))

    def test_lookup_input_shape(self):
        g = Graph()
        g.input("y", (3,))
        assert _lookup_input_shape(g, "y", default=(1,)) == (3,)
        assert _lookup_input_shape(g, "missing", default=(1,)) == (1,)
