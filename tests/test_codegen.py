"""Tests for the XLA-style tracing layer (slice a).

These tests verify that the tracer captures a component's math correctly:
the interpreted graph must match the live ``NumpyBackend`` computation to
float-exactness (``atol=1e-12``) across 50 seeded random inputs.

Tests 1-2 trace a single component (KalmanFilter, LQR) in isolation. The
composition and swap tests (3-7) live in :mod:`tests.test_codegen_compose`
and exercise the composition pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from shinro.codegen import TraceBackend, interpret, trace_node
from shinro.codegen.tracing import Graph, Tracer, _lift
from shinro.estimators.kalman_filter import KalmanFilter
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.factories.controller_factory import ControllerFactory
from shinro.controllers.lqr import LQR
from shinro.utils.array_backend import NumpyBackend


# ─── helpers ───────────────────────────────────────────────────────────────


def _load_kalman() -> KalmanFilter:
    return EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())


def _load_lqr() -> LQR:
    return ControllerFactory("configs/controllers/lqr_base.toml").create(backend=NumpyBackend())


# ─── Test 1: trace KalmanFilter alone ──────────────────────────────────────


class TestTraceKalman:
    """Tracing KalmanFilter.estimate captures the predict/update cycle."""

    def test_trace_matches_numpy_50_inputs(self, rng):
        """Interpreted KF graph == NumpyBackend KF on 50 random inputs."""
        kf_np = _load_kalman()
        kf_trace = _load_kalman()

        # KF.estimate signature: estimate(self, measurement, control_input)
        # Both are (n_x, 1) column vectors; n_x = 3 for the base config.
        input_shapes = {"measurement": (3, 1), "control_input": (3, 1)}

        for trial in range(50):
            y = rng.normal(0.0, 0.1, (3, 1))
            u = rng.normal(0.0, 0.1, (3, 1))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))

            # Ground truth: run the live numpy KF from this initial state.
            kf_np.x_hat = x_hat_init.copy()
            expected = kf_np.estimate(y, u)

            # Trace: inject the same initial x_hat as a state tracer.
            kf_trace.x_hat = x_hat_init.copy()
            node_graph = trace_node(
                kf_trace,
                input_shapes=input_shapes,
                state_shapes={"x_hat": (3, 1)},
            )

            # Interpret the captured graph on the same inputs.
            traced = interpret(
                node_graph.graph,
                {
                    "measurement": y,
                    "control_input": u,
                    "state_x_hat": x_hat_init,
                },
            )

            # The KF returns self.x_hat and mutates it — both should appear.
            # Primary output is "out" (the return value); state is "state_x_hat".
            got = traced.get("out", traced.get("state_x_hat"))
            assert got is not None, f"trial {trial}: no output. got {list(traced)}"
            assert np.allclose(got, expected, atol=1e-12), (
                f"trial {trial}: KF trace diverged. max err = {np.max(np.abs(got - expected))}"
            )

    def test_graph_has_matmul_and_inv_nodes(self):
        """The captured KF graph contains matmul and inv ops (predict + update)."""
        kf = _load_kalman()
        kf.x_hat = np.zeros((3, 1))
        node_graph = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1)},
        )
        ops = {n.op for n in node_graph.graph.nodes}
        assert "matmul" in ops, f"KF graph missing matmul; ops = {sorted(ops)}"
        assert "inv" in ops, f"KF graph missing inv; ops = {sorted(ops)}"

    def test_state_x_hat_detected(self):
        """The tracer detects x_hat as a mutated state attr."""
        kf = _load_kalman()
        kf.x_hat = np.zeros((3, 1))
        node_graph = trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1)},
        )
        assert "x_hat" in node_graph.state_attrs, (
            f"x_hat not detected as state; detected = {node_graph.state_attrs}"
        )


# ─── Test 2: trace LQR alone ───────────────────────────────────────────────


class TestTraceLQR:
    """Tracing LQR.compute captures the single gain matvec."""

    def test_trace_matches_numpy_50_inputs(self, rng):
        """Interpreted LQR graph == NumpyBackend LQR on 50 random inputs."""
        lqr_np = _load_lqr()
        lqr_trace = _load_lqr()

        # LQR.compute signature: compute(self, current_state, target_state)
        # Both are (n_x,) flat vectors; n_x = 3 for the base config.
        input_shapes = {"current_state": (3,), "target_state": (3,)}

        for trial in range(50):
            x = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.1, (3,))

            # Ground truth.
            expected = lqr_np.compute(x, x_ref)

            # Trace and interpret.
            node_graph = trace_node(lqr_trace, input_shapes=input_shapes)
            traced = interpret(
                node_graph.graph,
                {"current_state": x, "target_state": x_ref},
            )

            got = traced.get("out")
            assert got is not None, f"trial {trial}: no output. got {list(traced)}"
            assert np.allclose(got, expected, atol=1e-12), (
                f"trial {trial}: LQR trace diverged. max err = {np.max(np.abs(got - expected))}"
            )

    def test_graph_is_a_single_matmul(self):
        """LQR lowers to K @ (x_ref - x): one sub, one matmul, no inv."""
        lqr = _load_lqr()
        node_graph = trace_node(
            lqr,
            input_shapes={"current_state": (3,), "target_state": (3,)},
        )
        ops = [n.op for n in node_graph.graph.nodes]
        assert "matmul" in ops, f"LQR graph missing matmul; ops = {ops}"
        assert "inv" not in ops, f"LQR graph should have no inv; ops = {ops}"

    def test_lqr_has_no_state(self):
        """LQR is stateless — no attrs should be detected as mutated."""
        lqr = _load_lqr()
        node_graph = trace_node(
            lqr,
            input_shapes={"current_state": (3,), "target_state": (3,)},
        )
        assert node_graph.state_attrs == [], (
            f"LQR should be stateless; detected = {node_graph.state_attrs}"
        )


# ─── sanity: the tracer primitives themselves ──────────────────────────────


class TestTracerPrimitives:
    """Direct unit tests for the Tracer operator overloads."""

    def test_matmul_records_node(self):
        g = Graph()
        a = np.eye(3)
        b = np.ones((3, 1))
        ta = _lift(g, a)
        tb = _lift(g, b)
        tc = ta @ tb
        assert isinstance(tc, Tracer)
        assert tc.shape == (3, 1)
        assert g.nodes[tc.node].op == "matmul"

    def test_numpy_at_tracer_records_via_rmatmul(self):
        """The @ bypass: numpy array @ Tracer must record, not coerce."""
        g = Graph()
        a = np.eye(3)
        b = np.ones((3, 1))
        tb = _lift(g, b)
        tc = a @ tb  # should call Tracer.__rmatmul__
        assert isinstance(tc, Tracer), "numpy @ Tracer coerced instead of recording"
        assert g.nodes[tc.node].op == "matmul"

    def test_sub_and_neg(self):
        g = Graph()
        a = np.ones((3,))
        b = np.full((3,), 2.0)
        ta = _lift(g, a)
        tb = _lift(g, b)
        tc = ta - tb
        assert g.nodes[tc.node].op == "sub"
        td = -ta
        assert g.nodes[td.node].op == "neg"

    def test_transpose(self):
        g = Graph()
        a = np.ones((3, 2))
        ta = _lift(g, a)
        tb = ta.T
        assert tb.shape == (2, 3)
        assert g.nodes[tb.node].op == "transpose"