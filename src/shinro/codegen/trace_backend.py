"""A recording :class:`ArrayBackend` that captures ops into a :class:`Graph`.

This is the JAX-style tracing backend — a third :class:`ArrayBackend`
alongside ``NumpyBackend`` and ``TorchBackend``. Components take
``backend=TraceBackend(graph)`` and trace themselves by being run once with
:class:`Tracer` values: every ``bk.*`` call emits a node instead of
computing a float, and concrete parameters (gain ``K``, matrices ``A``/``B``)
are frozen into the graph as ``const`` nodes the moment they touch a traced
operation.

The bare operators ``@``, ``+``, ``-``, ``*``, ``.T`` are handled by
:class:`Tracer`'s overloads (the :class:`ArrayBackend` docstring states that
``@`` is intentionally not wrapped). This backend implements the *named*
``ArrayBackend`` methods that components call (``eye``, ``zeros_like``,
``inv``, ``clip``, ``where``, ``copy``).

Any method not yet implemented raises ``NotImplementedError`` naming the op
to register — a loud, actionable signal when a new component uses a new
backend method. This is how the op set grows incrementally, driven by real
components, rather than being guessed up front.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shinro.codegen.ops import missing_op_error
from shinro.codegen.tracing import Graph, Tracer, _lift


class TraceBackend:
    """An :class:`ArrayBackend` that records operations into a :class:`Graph`.

    Not a subclass of :class:`shinro.utils.array_backend.ArrayBackend` because
    that ABC requires every method to be implemented. Instead we implement
    the methods components actually call and ``NotImplementedError`` the rest
    — the tracer grows its op set incrementally, driven by real components.
    Components receive this backend via ``self.bk`` and call ``self.bk.<op>``
    exactly as they would with ``NumpyBackend``.

    Args:
        graph: The graph to record ops into.
    """

    def __init__(self, graph: Graph) -> None:
        self.g = graph

    # --- helpers ---

    def _emit(self, op: str, inputs: list[Any], shape: tuple[int, ...], **attrs: Any) -> Tracer:
        # Lift any concrete ndarrays to const nodes. This happens when a
        # component's internal computation (e.g. the KF covariance update
        # P = A P A^T + Q) involves only real arrays — the @ between reals
        # returns a real, which then gets passed to a bk.* method here.
        # Lifting bakes it as a const, which is correct: the subgraph was
        # all-constant anyway and would be folded regardless.
        lifted = [_lift(self.g, x) for x in inputs]
        node = self.g.emit(op, [t.node for t in lifted], shape, **attrs)
        return Tracer(self.g, shape, node)

    # --- array creation (constants folded at trace time) ---

    def array(self, data: Any) -> Tracer:
        # A concrete list/ndarray becomes a const node. A Tracer passes through.
        if isinstance(data, Tracer):
            return data
        arr = np.asarray(data, dtype=np.float64)
        node = self.g.const(arr)
        return Tracer(self.g, arr.shape, node)

    def zeros(self, *shape: int) -> Tracer:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        arr = np.zeros(shape, dtype=np.float64)
        node = self.g.const(arr)
        return Tracer(self.g, arr.shape, node)

    def zeros_like(self, x: Tracer) -> Tracer:
        # x is a Tracer; its shape is known at trace time → fold the zeros.
        arr = np.zeros(x.shape, dtype=np.float64)
        node = self.g.const(arr)
        return Tracer(self.g, x.shape, node)

    def eye(self, n: int) -> Tracer:
        arr = np.eye(n, dtype=np.float64)
        node = self.g.const(arr)
        return Tracer(self.g, arr.shape, node)

    def diag(self, x: Tracer) -> Tracer:
        # diag of a tracer — emit a diag op. For the common case where x is a
        # const diagonal (Q/R weights from config), the caller usually passes
        # a concrete list which gets folded by array(); this handles the
        # tracer-input case.
        if x.shape == (1,) or len(x.shape) == 1:
            out_shape = (x.shape[0], x.shape[0])
        else:
            raise NotImplementedError("TraceBackend.diag only supports 1D -> 2D")
        return self._emit("diag", [x], out_shape)

    def copy(self, x: Tracer) -> Tracer:
        return self._emit("copy", [x], x.shape)

    # --- linear algebra ---

    def inv(self, x: Tracer) -> Tracer:
        return self._emit("inv", [x], x.shape)

    def solve(self, A: Tracer, b: Tracer) -> Tracer:
        # solve(A, b) returns x with A.shape[1] (= A.shape[0]) and b.shape[1]
        # if b is 2D, else b.shape[0] if 1D.
        if len(b.shape) == 2:
            out_shape = (A.shape[1], b.shape[1])
        else:
            out_shape = (A.shape[1],)
        return self._emit("solve", [A, b], out_shape)

    # --- elementwise / selection ---

    def clip(self, x: Tracer, lo: Any, hi: Any) -> Tracer:
        # lo/hi are concrete (from config) — fold them into attrs as consts.
        lo_arr = np.asarray(lo, dtype=np.float64) if not isinstance(lo, Tracer) else None
        hi_arr = np.asarray(hi, dtype=np.float64) if not isinstance(hi, Tracer) else None
        return self._emit("clip", [x], x.shape, lo=lo_arr, hi=hi_arr)

    def where(self, cond: Tracer, a: Tracer, b: Tracer) -> Tracer:
        a = _lift(self.g, a)
        b = _lift(self.g, b)
        cond = _lift(self.g, cond)
        return self._emit("where", [cond, a, b], a.shape)

    def any(self, x: Tracer) -> Tracer:
        out_shape: tuple[int, ...] = ()
        return self._emit("any", [x], out_shape)

    def stack(self, arrays: list[Tracer]) -> Tracer:
        # stack along a new leading axis — all inputs same shape.
        if not arrays:
            raise NotImplementedError("TraceBackend.stack of empty list")
        base = arrays[0].shape
        out_shape = (len(arrays),) + base
        return self._emit("stack", [a for a in arrays], out_shape)

    # --- conversions (no-ops under tracing) ---

    def to_numpy(self, x: Tracer) -> Tracer:
        # Under tracing we never materialize; the tracer passes through.
        return x

    def from_numpy(self, x: Any) -> Tracer:
        return _lift(self.g, x)

    def allclose(self, a: Tracer, b: Tracer) -> bool:
        # allclose on tracers is a runtime check — not meaningful at trace
        # time. Components don't call this in the compute path; if they do,
        # it's a control-flow-on-traced-value bug we want surfaced loudly.
        raise NotImplementedError(
            "TraceBackend.allclose is not traceable — it branches on traced "
            "values. Rewrite the component to avoid allclose in the compute path."
        )

    # --- everything else: loud NotImplementedError naming the op to add ---

    def __getattr__(self, name: str) -> Any:
        # Called only for attributes not found normally. Emit the standard
        # "register this op" error so new components surface missing ops
        # with an actionable message.
        raise missing_op_error(name)