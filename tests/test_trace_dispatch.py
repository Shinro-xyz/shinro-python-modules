"""Tests for the registry-driven `TraceBackend` op dispatcher (`__getattr__`).

The tracer's named-op methods are generated from the op registry — the handler
is the computation, the `VmOp` is the lowering, and a `TraceSpec` (only where
needed) declares the trace behaviour. These tests pin: ops are no longer
hand-written methods, dispatch emits the right node, renamed ops alias, and the
shape override is used where the default (handler on zeros) would fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from shinro.codegen.ops import OP_HANDLERS, TRACE_SPECS
from shinro.codegen.trace_backend import TraceBackend
from shinro.codegen.tracing import Graph, Tracer


def _bk() -> TraceBackend:
    return TraceBackend(Graph())


def _inp(bk: TraceBackend, name: str, shape: tuple[int, ...]) -> Tracer:
    return Tracer(bk.g, shape, bk.g.input(name, shape))


def _const(g: Graph, arr) -> Tracer:
    arr = np.asarray(arr, dtype=np.float64)
    return Tracer(g, arr.shape, g.const(arr))


class TestDispatcher:
    def test_op_methods_are_not_hand_written(self):
        # the whole point: no per-op method bodies survive in the class
        for op in ("tanh", "gemm", "clip", "where", "stack", "lstm", "inv", "one_hot"):
            assert op not in TraceBackend.__dict__, f"{op} should be dispatched, not a method"
        # only the irreducible helpers remain
        for kept in ("_emit", "array", "zeros", "reshape", "ravel", "hstack", "jacobian", "to_numpy", "from_numpy"):
            assert kept in TraceBackend.__dict__

    def test_generic_unary(self):
        bk = _bk()
        t = bk.tanh(_inp(bk, "x", (3,)))
        assert bk.g.nodes[t.node].op == "tanh"
        assert t.shape == (3,)
        assert bk.g.nodes[t.node].attrs == {}

    def test_attrs_from_vmop(self):
        bk = _bk()
        g = bk.g
        a = _inp(bk, "a", (4,))
        y = bk.gemm(a, _const(g, np.zeros((4, 3))), _const(g, np.zeros(3)), alpha=0.5, beta=2.0)
        node = g.nodes[y.node]
        assert node.op == "gemm" and y.shape == (3,)
        assert node.attrs == {"alpha": 0.5, "beta": 2.0, "transB": False}  # default applied

    def test_array_attrs_positional(self):
        bk = _bk()
        t = bk.clip(_inp(bk, "x", (3,)), [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        node = bk.g.nodes[t.node]
        assert node.op == "clip" and node.inputs == [0]
        np.testing.assert_array_equal(node.attrs["lo"], [-1.0, -1.0, -1.0])

    def test_renamed_op_aliases_to_slice(self):
        bk = _bk()
        t = bk.slice_(_inp(bk, "x", (5,)), 1, 3)
        node = bk.g.nodes[t.node]
        assert node.op == "slice"  # method is slice_, op is slice
        assert node.attrs == {"start": 1, "stop": 3}
        assert t.shape == (2,)

    def test_variadic_splices_into_inputs(self):
        bk = _bk()
        a = _inp(bk, "a", (2,))
        b = _inp(bk, "b", (2,))
        t = bk.stack([a, b])
        node = bk.g.nodes[t.node]
        assert node.op == "stack" and node.inputs == [0, 1] and t.shape == (2, 2)

    def test_shape_override_avoids_dummy_exec(self):
        # inv of a singular zero matrix would raise LinAlgError; the override
        # (shapes[0]) means the handler is never run at trace time.
        bk = _bk()
        t = bk.inv(_inp(bk, "M", (3, 3)))
        assert t.shape == (3, 3)

    def test_attr_driven_shape(self):
        bk = _bk()
        t = bk.one_hot(_inp(bk, "idx", ()), 4)
        node = bk.g.nodes[t.node]
        assert node.op == "one_hot" and t.shape == (4,) and node.attrs == {"depth": 4}

    def test_unknown_op_raises_actionable_error(self):
        bk = _bk()
        with pytest.raises(NotImplementedError, match="bogus_op"):
            bk.bogus_op(_inp(bk, "x", (2,)))

    def test_every_op_has_a_trace_spec(self):
        # register_op seeds TRACE_SPECS for every op, so the dispatcher never
        # hits a missing key for a registered op.
        assert set(OP_HANDLERS) <= set(TRACE_SPECS)
