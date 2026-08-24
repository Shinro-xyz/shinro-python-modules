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

from shinro.codegen import (
    available_ops,
    has_op,
    interpret,
    interpret_step,
    register_op,
    trace_node,
    trace_node_with_state,
)
from shinro.codegen.infer_contract import (
    _is_array_like,
    detect_state,
    infer_contract,
    snapshot_instance_attrs,
)
from shinro.codegen.ops import OP_HANDLERS, missing_op_error
from shinro.codegen.trace_backend import TraceBackend
from shinro.codegen.trace_node import NodeGraph, _collect_output_nodes
from shinro.codegen.tracing import (
    Graph,
    ShapeMismatchError,
    Tracer,
    _broadcast_shape,
    _lift,
    _matmul_out_shape,
)
from shinro.components import Controller
from shinro.controllers.lqr import LQR
from shinro.estimators.kalman_filter import KalmanFilter
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.utils.array_backend import NumpyBackend  # ─── helpers ───────────────────────────────────────────────────────────────


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

        # Snapshot the initial state so each trial starts from the same P and
        # x_hat. The KF carries both across calls; if we don't reset them, the
        # numpy reference accumulates while the traced KF (which restores
        # attrs after each trace) doesn't, causing a spurious divergence.
        initial_P = kf_np.P.copy()

        for trial in range(50):
            y = rng.normal(0.0, 0.1, (3, 1))
            u = rng.normal(0.0, 0.1, (3, 1))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))

            # Ground truth: run the live numpy KF from a fresh state.
            kf_np.P = initial_P.copy()
            kf_np.x_hat = x_hat_init.copy()
            expected = kf_np.estimate(y, u)

            # Trace: inject the same initial x_hat and let P be the frozen
            # const value (the traced KF's P is restored to initial_P after
            # each trace, matching the numpy reference's reset above).
            kf_trace.P = initial_P.copy()
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
            assert np.allclose(got, expected, atol=1e-12), f"trial {trial}: KF trace diverged. max err = {np.max(np.abs(got - expected))}"

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
        assert "x_hat" in node_graph.state_attrs, f"x_hat not detected as state; detected = {node_graph.state_attrs}"


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
            assert np.allclose(got, expected, atol=1e-12), f"trial {trial}: LQR trace diverged. max err = {np.max(np.abs(got - expected))}"

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
        assert node_graph.state_attrs == [], f"LQR should be stateless; detected = {node_graph.state_attrs}"


# ─── Test 3: trace a deterministic MLP policy ──────────────────────────────


class _NumpyMLPPolicy(Controller):
    """Deterministic 2-layer MLP policy (tanh hidden, tanh output).

    Mirrors how a deterministic learned policy would be written against the
    ArrayBackend surface so it traces and lowers like a classical controller.
    """

    def __init__(self, W1, b1, W2, b2, backend=None):
        self.bk = backend or NumpyBackend()
        self.W1 = self.bk.array(W1)
        self.b1 = self.bk.array(b1)
        self.W2 = self.bk.array(W2)
        self.b2 = self.bk.array(b2)

    def compute(self, state, target=None):
        h = self.bk.tanh(state @ self.W1 + self.b1)
        u = self.bk.tanh(h @ self.W2 + self.b2)
        return u

    def reset(self):
        pass


class TestTraceMLPPolicy:
    """Tracing a deterministic MLP policy captures activations + matmuls."""

    def test_trace_matches_numpy_20_inputs(self, rng):
        """Interpreted MLP graph == NumpyBackend MLP on 20 random inputs."""
        W1 = rng.normal(size=(3, 16))
        b1 = rng.normal(size=(16,))
        W2 = rng.normal(size=(16, 2))
        b2 = rng.normal(size=(2,))
        policy_np = _NumpyMLPPolicy(W1, b1, W2, b2, backend=NumpyBackend())
        policy_tr = _NumpyMLPPolicy(W1, b1, W2, b2, backend=NumpyBackend())

        input_shapes = {"state": (3,), "target": (3,)}

        for trial in range(20):
            x = rng.normal(0.0, 0.1, (3,))
            expected = policy_np.compute(x)

            node_graph = trace_node(policy_tr, input_shapes=input_shapes)
            traced = interpret(node_graph.graph, {"state": x, "target": np.zeros((3,))})
            got = traced.get("out")
            assert got is not None, f"trial {trial}: no output. got {list(traced)}"
            assert np.allclose(got, expected, atol=1e-12), f"trial {trial}: MLP trace diverged. max err = {np.max(np.abs(got - expected))}"

    def test_graph_has_tanh_and_multiple_matmul(self):
        """The MLP graph contains tanh nodes and one matmul per layer."""
        rng = np.random.default_rng(0)
        policy = _NumpyMLPPolicy(
            rng.normal(size=(3, 16)),
            rng.normal(size=(16,)),
            rng.normal(size=(16, 2)),
            rng.normal(size=(2,)),
            backend=NumpyBackend(),
        )
        node_graph = trace_node(policy, input_shapes={"state": (3,), "target": (3,)})
        ops = [n.op for n in node_graph.graph.nodes]
        assert "tanh" in ops, f"MLP graph missing tanh; ops = {ops}"
        assert ops.count("matmul") == 2, f"expected 2 matmuls (one per layer); got {ops}"


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


# ─── _lift promotion paths ─────────────────────────────────────────────────


class TestLift:
    """Direct unit tests for _lift's promotion of concrete values to Tracers."""

    def test_tracer_passes_through(self):
        g = Graph()
        ta = _lift(g, np.ones((2, 2)))
        tb = _lift(g, ta)
        assert tb is ta
        assert tb.node == ta.node

    def test_ndarray_becomes_const(self):
        g = Graph()
        arr = np.arange(6.0).reshape(2, 3)
        t = _lift(g, arr)
        assert isinstance(t, Tracer)
        assert t.shape == (2, 3)
        node = g.nodes[t.node]
        assert node.op == "const"
        assert np.array_equal(node.attrs["value"], arr)

    def test_list_and_tuple_become_const(self):
        g = Graph()
        tl = _lift(g, [1.0, 2.0, 3.0])
        assert tl.shape == (3,)
        assert g.nodes[tl.node].op == "const"
        assert g.nodes[tl.node].attrs["value"].dtype == np.float64
        tt = _lift(g, (1.0, 2.0))
        assert tt.shape == (2,)
        assert g.nodes[tt.node].op == "const"

    def test_scalar_becomes_0d_const(self):
        g = Graph()
        t = _lift(g, 2.5)
        assert t.shape == ()
        node = g.nodes[t.node]
        assert node.op == "const"
        assert node.attrs["value"].shape == ()


# ─── _matmul_out_shape shape rules ─────────────────────────────────────────


class TestMatmulOutShape:
    """Direct unit tests for the matmul output-shape rules."""

    def test_2d_at_2d(self):
        assert _matmul_out_shape((3, 4), (4, 5)) == (3, 5)

    def test_2d_at_2d_mismatch_raises(self):
        with pytest.raises(ShapeMismatchError):
            _matmul_out_shape((3, 4), (3, 5))

    def test_2d_at_1d(self):
        assert _matmul_out_shape((3, 4), (4,)) == (3,)

    def test_2d_at_1d_mismatch_raises(self):
        with pytest.raises(ShapeMismatchError):
            _matmul_out_shape((3, 4), (3,))

    def test_1d_at_2d(self):
        assert _matmul_out_shape((4,), (4, 5)) == (5,)

    def test_1d_at_2d_mismatch_raises(self):
        with pytest.raises(ShapeMismatchError):
            _matmul_out_shape((4,), (3, 5))

    def test_0d_or_3d_raises(self):
        with pytest.raises(ShapeMismatchError):
            _matmul_out_shape((), (3, 3))
        with pytest.raises(ShapeMismatchError):
            _matmul_out_shape((2, 2, 2), (2, 2))


# ─── _broadcast_shape rules ────────────────────────────────────────────────


class TestBroadcastShape:
    """Direct unit tests for the broadcasting rules used by add/sub/mul."""

    def test_equal_shapes_pass_through(self):
        assert _broadcast_shape((3, 1), (3, 1)) == (3, 1)

    def test_0d_broadcasts_with_anything(self):
        assert _broadcast_shape((), (3, 4)) == (3, 4)
        assert _broadcast_shape((3, 4), ()) == (3, 4)

    def test_size_1_axis_broadcasts(self):
        assert _broadcast_shape((1, 4), (3, 4)) == (3, 4)
        assert _broadcast_shape((3, 1), (3, 4)) == (3, 4)

    def test_rank_mismatch_raises(self):
        with pytest.raises(ShapeMismatchError):
            _broadcast_shape((3,), (3, 1))

    def test_conflicting_dims_raise(self):
        with pytest.raises(ShapeMismatchError):
            _broadcast_shape((2, 3), (4, 3))


# ─── remaining Tracer operator overloads ────────────────────────────────────


class TestTracerArith:
    """Direct unit tests for the add/radd/mul/rmul/rsub overloads."""

    def test_add_and_radd(self):
        g = Graph()
        ta = _lift(g, np.ones((3,)))
        tb = _lift(g, np.ones((3,)))
        tc = ta + tb
        assert g.nodes[tc.node].op == "add"
        td = 1.0 + ta  # scalar on the left → __radd__
        assert g.nodes[td.node].op == "add"
        assert td.shape == (3,)

    def test_mul_and_rmul(self):
        g = Graph()
        ta = _lift(g, np.ones((3,)))
        tb = _lift(g, np.ones((3,)))
        tc = ta * tb
        assert g.nodes[tc.node].op == "mul"
        td = 2.0 * ta  # scalar on the left → __rmul__
        assert g.nodes[td.node].op == "mul"

    def test_rsub_operand_order(self):
        g = Graph()
        ta = _lift(g, np.ones((3,)))
        tc = 5.0 - ta  # __rsub__: (5.0) - ta
        node = g.nodes[tc.node]
        assert node.op == "sub"
        # First input is the lifted scalar const, second is ta.
        assert g.nodes[node.inputs[0]].op == "const"
        assert node.inputs[1] == ta.node

    def test_transpose_1d_raises(self):
        g = Graph()
        ta = _lift(g, np.ones((3,)))
        with pytest.raises(ShapeMismatchError):
            _ = ta.T


# ─── TraceBackend direct unit tests ────────────────────────────────────────


class TestTraceBackend:
    """Direct unit tests for the recording backend's named methods."""

    def _bk(self):
        return TraceBackend(Graph())

    def test_array_passthrough_and_const(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((2, 2)))
        assert bk.array(ta) is ta
        tb = bk.array([1.0, 2.0])
        assert tb.shape == (2,)
        assert bk.g.nodes[tb.node].op == "const"

    def test_zeros_variants(self):
        bk = self._bk()
        t1 = bk.zeros(3, 1)
        assert t1.shape == (3, 1)
        t2 = bk.zeros((3, 1))  # type: ignore[arg-type]
        assert t2.shape == (3, 1)
        assert bk.g.nodes[t1.node].op == "const"

    def test_zeros_like_matches_shape(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((4, 2)))
        tz = bk.zeros_like(ta)
        assert tz.shape == (4, 2)
        assert bk.g.nodes[tz.node].op == "const"

    def test_eye(self):
        bk = self._bk()
        te = bk.eye(3)
        assert te.shape == (3, 3)
        node = bk.g.nodes[te.node]
        assert node.op == "const"
        assert np.array_equal(node.attrs["value"], np.eye(3))

    def test_diag_1d(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((3,)))
        td = bk.diag(ta)
        assert td.shape == (3, 3)
        assert bk.g.nodes[td.node].op == "diag"

    def test_diag_2d_raises(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((3, 3)))
        with pytest.raises(NotImplementedError):
            bk.diag(ta)

    def test_copy_emits_copy_op(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((2, 2)))
        tc = bk.copy(ta)
        assert tc.shape == (2, 2)
        assert bk.g.nodes[tc.node].op == "copy"

    def test_inv_emits_inv_op(self):
        bk = self._bk()
        ta = _lift(bk.g, np.eye(3))
        ti = bk.inv(ta)
        assert ti.shape == (3, 3)
        assert bk.g.nodes[ti.node].op == "inv"

    def test_solve_2d_and_1d(self):
        bk = self._bk()
        A = _lift(bk.g, np.eye(3))
        b2 = _lift(bk.g, np.ones((3, 2)))
        x2 = bk.solve(A, b2)
        assert x2.shape == (3, 2)
        b1 = _lift(bk.g, np.ones((3,)))
        x1 = bk.solve(A, b1)
        assert x1.shape == (3,)

    def test_clip_records_lo_hi_attrs(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((3,)))
        tc = bk.clip(ta, -1.0, 1.0)
        node = bk.g.nodes[tc.node]
        assert node.op == "clip"
        assert np.array_equal(node.attrs["lo"], np.asarray(-1.0))
        assert np.array_equal(node.attrs["hi"], np.asarray(1.0))

    def test_where_input_order(self):
        bk = self._bk()
        cond = _lift(bk.g, np.array([True, False]))
        a = _lift(bk.g, np.ones((2,)))
        b = _lift(bk.g, np.zeros((2,)))
        tw = bk.where(cond, a, b)
        node = bk.g.nodes[tw.node]
        assert node.op == "where"
        assert node.inputs == [cond.node, a.node, b.node]

    def test_any_is_0d(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((3, 3)))
        tany = bk.any(ta)
        assert tany.shape == ()
        assert bk.g.nodes[tany.node].op == "any"

    def test_stack(self):
        bk = self._bk()
        a = _lift(bk.g, np.ones((2, 2)))
        b = _lift(bk.g, np.zeros((2, 2)))
        ts = bk.stack([a, b])
        assert ts.shape == (2, 2, 2)
        assert bk.g.nodes[ts.node].op == "stack"

    def test_stack_empty_raises(self):
        bk = self._bk()
        with pytest.raises(NotImplementedError):
            bk.stack([])

    def test_to_numpy_passthrough(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((2,)))
        assert bk.to_numpy(ta) is ta

    def test_from_numpy_lifts(self):
        bk = self._bk()
        t = bk.from_numpy(np.ones((2, 2)))
        assert t.shape == (2, 2)
        assert bk.g.nodes[t.node].op == "const"

    def test_allclose_rejects_traced_values(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((2,)))
        tb = _lift(bk.g, np.ones((2,)))
        with pytest.raises(NotImplementedError, match="allclose"):
            bk.allclose(ta, tb)

    def test_unknown_method_raises_actionable_error(self):
        bk = self._bk()
        with pytest.raises(NotImplementedError, match="register_op"):
            bk.some_unknown_op()

    def test_reshape_records_reshape_op(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((3, 1)))
        tr = bk.reshape(ta, (3,))  # type: ignore[arg-type]
        assert tr.shape == (3,)
        node = bk.g.nodes[tr.node]
        assert node.op == "reshape"
        assert node.attrs["target_shape"] == (3,)

    def test_policy_elementwise_ops(self):
        bk = self._bk()
        ta = _lift(bk.g, np.ones((4,)))
        assert bk.g.nodes[bk.tanh(ta).node].op == "tanh"
        assert bk.g.nodes[bk.relu(ta).node].op == "relu"
        assert bk.g.nodes[bk.exp(ta).node].op == "exp"
        tb = _lift(bk.g, np.full((4,), 2.0))
        td = bk.div(ta, tb)
        assert bk.g.nodes[td.node].op == "div"
        assert td.shape == (4,)

    def test_policy_discrete_ops(self):
        bk = self._bk()
        ta = _lift(bk.g, np.array([0.1, 0.5, 0.9, 0.3]))
        targ = bk.argmax(ta)
        assert targ.shape == ()
        assert bk.g.nodes[targ.node].op == "argmax"
        toh = bk.one_hot(targ, 4)
        assert toh.shape == (4,)
        node = bk.g.nodes[toh.node]
        assert node.op == "one_hot"
        assert node.attrs["depth"] == 4

    def test_slice_records_slice_op(self):
        bk = self._bk()
        ta = _lift(bk.g, np.arange(6.0).reshape(2, 3))
        ts = bk.slice_(ta, 0, 1)
        assert ts.shape == (1, 3)
        node = bk.g.nodes[ts.node]
        assert node.op == "slice"
        assert node.attrs["start"] == 0
        assert node.attrs["stop"] == 1


# ─── op registry and handlers ───────────────────────────────────────────────


class TestOps:
    """Direct unit tests for the op registry and its helpers."""

    def test_available_ops_sorted_and_complete(self):
        ops = available_ops()
        assert ops == sorted(ops)
        for expected in ("const", "input", "output", "matmul", "add", "sub", "mul", "neg", "transpose", "inv"):
            assert expected in ops, f"missing {expected} in {ops}"

    def test_has_op(self):
        assert has_op("matmul")
        assert not has_op("definitely_not_an_op")

    def test_missing_op_error_message(self):
        err = missing_op_error("foo")
        assert isinstance(err, NotImplementedError)
        assert "foo" in str(err)
        assert "register_op" in str(err)

    def test_register_op_round_trip(self):
        name = "test_only_double"

        @register_op(name)
        def _double(node, values, inputs):
            return 2.0 * values[node.inputs[0]]

        try:
            assert has_op(name)
            g = Graph()
            c = g.const(np.asarray(21.0))
            d = g.emit(name, [c], ())
            g.output("out", d)
            result = interpret(g, {})
            assert result["out"] == 42.0
        finally:
            OP_HANDLERS.pop(name, None)
        assert not has_op(name)


# ─── interpreter dispatch and errors ───────────────────────────────────────


class TestInterpreter:
    """Direct unit tests for interpret / interpret_step."""

    def test_const_to_output(self):
        g = Graph()
        c = g.const(np.asarray([1.0, 2.0]))
        g.output("out", c)
        result = interpret(g, {})
        assert np.array_equal(result["out"], [1.0, 2.0])

    def test_missing_input_raises_keyerror(self):
        g = Graph()
        i = g.input("y", (3,))
        g.output("out", i)
        with pytest.raises(KeyError, match="y"):
            interpret(g, {})

    def test_unknown_op_raises(self):
        g = Graph()
        c = g.const(np.asarray(1.0))
        g.emit("not_a_real_op", [c], ())
        g.output("out", c)
        with pytest.raises(NotImplementedError, match="not_a_real_op"):
            interpret(g, {})

    def test_interpret_step_splits_state(self):
        g = Graph()
        c = g.const(np.asarray([1.0]))
        g.output("out", c)
        g.output("state_x_hat", c)
        outputs, state = interpret_step(g, {})
        assert "out" in outputs
        assert "x_hat" in state
        assert "out" not in state
        assert "x_hat" not in outputs


# ─── contract inference ─────────────────────────────────────────────────────


class TestInferContract:
    """Direct unit tests for infer_contract and the state-detection helpers."""

    def test_lqr_contract(self):
        lqr = _load_lqr()
        contract = infer_contract(lqr, {"current_state": (3,), "target_state": (3,)})
        assert contract.method_name == "compute"
        assert contract.input_names == ["current_state", "target_state"]
        assert contract.input_shapes == {"current_state": (3,), "target_state": (3,)}

    def test_kalman_contract(self):
        kf = _load_kalman()
        contract = infer_contract(kf, {"measurement": (3, 1), "control_input": (3, 1)})
        assert contract.method_name == "estimate"
        assert contract.input_names == ["measurement", "control_input"]

    def test_non_abc_component_raises_typeerror(self):
        class NotAComponent:
            def compute(self, x):
                return x

        with pytest.raises(TypeError, match="ABC"):
            infer_contract(NotAComponent())

    def test_snapshot_and_detect_state(self):
        class Dummy:
            def __init__(self):
                self.a = np.zeros((2,))
                self.b = np.ones((2,))
                self.scalar = 3.0

        d = Dummy()
        before = snapshot_instance_attrs(d)
        assert set(before) == {"a", "b"}  # scalar excluded
        d.a = np.zeros((2,))  # reassignment → new id
        d.scalar = 4.0  # not array-like, ignored
        state = detect_state(d, before)
        assert state == ["a"]

    def test_is_array_like(self):
        assert _is_array_like(np.zeros((2,)))
        assert _is_array_like(Tracer(Graph(), (2,), 0))
        assert not _is_array_like(3.0)
        assert not _is_array_like([1.0, 2.0])


# ─── trace_node edges ──────────────────────────────────────────────────────


class TestTraceNode:
    """Direct unit tests for trace_node's error and restoration behavior."""

    def test_missing_input_shape_raises_keyerror(self):
        kf = _load_kalman()
        with pytest.raises(KeyError, match="measurement"):
            trace_node(kf, input_shapes={"control_input": (3, 1)})

    def test_backend_restored_after_trace(self):
        kf = _load_kalman()
        original_bk = kf.bk
        trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1)},
        )
        assert kf.bk is original_bk

    def test_attrs_restored_to_ndarrays_after_trace(self):
        kf = _load_kalman()
        trace_node(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_shapes={"x_hat": (3, 1)},
        )
        assert isinstance(kf.P, np.ndarray), "P leaked a Tracer"
        assert isinstance(kf.x_hat, np.ndarray), "x_hat leaked a Tracer"

    def test_trace_node_with_state_returns_restored_values(self):
        kf = _load_kalman()
        x_hat_init = np.zeros((3, 1))
        node_graph, state_outputs = trace_node_with_state(
            kf,
            input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
            state_inputs={"x_hat": x_hat_init},
        )
        assert isinstance(node_graph, NodeGraph)
        assert "x_hat" in state_outputs
        assert isinstance(state_outputs["x_hat"], np.ndarray), "state_outputs leaked a Tracer"
        assert np.array_equal(state_outputs["x_hat"], x_hat_init)

    def test_collect_output_nodes(self):
        g = Graph()
        c = g.const(np.asarray(1.0))
        g.output("out", c)
        g.output("state_x_hat", c)
        assert _collect_output_nodes(g) == {"out": c, "state_x_hat": c}
