"""Abstract values for tracing control-law computations into a primitive graph.

This is the XLA-style tracing layer: components run once with :class:`Tracer`
values instead of real arrays, and every operation they perform is recorded
as a node in a :class:`Graph`. The captured graph is then interpreted (see
:mod:`shinro.codegen.interpreter`) or, in a future slice, lowered to Zig.

The design mirrors JAX's tracing interpreter:

- A :class:`Tracer` stands in for an ndarray during tracing. It carries only
  its concrete shape and its graph node id — no data.
- Concrete numpy values seen during tracing (e.g. a precomputed gain ``K``)
  are lifted to ``const`` nodes via :func:`_lift`, freezing them into the
  graph. This is the "fixed as compiled" property: shapes and parameters are
  known at trace time.
- The ``@``, ``+``, ``-``, ``*`` operators and the ``.T`` property are
  overloaded on :class:`Tracer` because the :class:`ArrayBackend` docstring
  states that ``@`` is intentionally not wrapped by the backend. Setting
  ``__array_ufunc__ = None`` makes numpy defer to the tracer's
  ``__rmatmul__`` instead of coercing the tracer to an array.

The graph is a flat list of :class:`Node` records. Each node has an op name
(handled by the registry in :mod:`shinro.codegen.ops`), a list of input node
ids, a concrete shape, and an opaque ``attrs`` dict for op-specific data
(e.g. the baked ndarray for ``const`` nodes, the target shape for
``reshape`` nodes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Node:
    """A single operation in the captured graph.

    Args:
        op: The operation name, dispatched through the ``OP_HANDLERS``
            registry in :mod:`shinro.codegen.ops`.
        inputs: Node ids of the inputs (empty for ``const`` and ``input``).
        shape: Concrete output shape. This is the "fixed as compiled"
            property — every array's shape is known at trace time.
        attrs: Op-specific data. ``const`` nodes carry ``value`` (ndarray);
            ``input``/``output`` carry ``name``; ``reshape`` carries the
            target ``shape``; ``clip`` carries ``lo``/``hi``.
    """

    op: str
    inputs: list[int]
    shape: tuple[int, ...]
    attrs: dict[str, Any] = field(default_factory=dict)


class Graph:
    """A flat list of :class:`Node` records captured during tracing.

    The graph is built by emitting nodes in execution order. Each emit
    returns the new node's id so callers can wire it as an input to later
    nodes. Inputs and outputs are named (via :meth:`input` / :meth:`output`)
    so the interpreter and composition pass can refer to them symbolically.
    """

    def __init__(self) -> None:
        self.nodes: list[Node] = []

    def emit(self, op: str, inputs: list[int], shape: tuple[int, ...], **attrs: Any) -> int:
        """Append a node and return its id."""
        node = Node(op=op, inputs=list(inputs), shape=tuple(shape), attrs=dict(attrs))
        self.nodes.append(node)
        return len(self.nodes) - 1

    def const(self, value: np.ndarray) -> int:
        """Emit a ``const`` node carrying a baked ndarray."""
        return self.emit("const", [], value.shape, value=value)

    def input(self, name: str, shape: tuple[int, ...]) -> int:
        """Emit a named ``input`` node — a graph input port."""
        return self.emit("input", [], shape, name=name)

    def output(self, name: str, src: int) -> None:
        """Emit a named ``output`` node referencing an upstream node id."""
        self.emit("output", [src], self.nodes[src].shape, name=name)


class Tracer:
    """An abstract value standing in for an ndarray during tracing.

    A ``Tracer`` carries only its concrete shape and its graph node id — it
    holds no data. Operations on tracers (``@``, ``+``, ``-``, ``*``, ``.T``)
    emit nodes into the graph and return new tracers.

    ``__array_ufunc__ = None`` is critical: it tells numpy to defer to the
    tracer's reflected operators (``__rmatmul__`` etc.) when a numpy array
    interacts with a tracer, instead of coercing the tracer via
    ``np.asarray``. Without this, ``A @ tracer`` would silently compute with
    garbage instead of recording a ``matmul`` node.
    """

    __array_ufunc__ = None
    __array_priority__ = 1000

    def __init__(self, graph: Graph, shape: tuple[int, ...], node: int) -> None:
        self._g = graph
        self.shape = tuple(shape)
        self.node = node

    # --- operator overloads (the @ bypass; see ArrayBackend docstring) ---

    def __matmul__(self, other: Any) -> Tracer:
        other = _lift(self._g, other)
        out_shape = _matmul_out_shape(self.shape, other.shape)
        node = self._g.emit("matmul", [self.node, other.node], out_shape)
        return Tracer(self._g, out_shape, node)

    def __rmatmul__(self, other: Any) -> Tracer:
        other = _lift(self._g, other)
        out_shape = _matmul_out_shape(other.shape, self.shape)
        node = self._g.emit("matmul", [other.node, self.node], out_shape)
        return Tracer(self._g, out_shape, node)

    def __add__(self, other: Any) -> Tracer:
        other = _lift(self._g, other)
        out_shape = _broadcast_shape(self.shape, other.shape)
        node = self._g.emit("add", [self.node, other.node], out_shape)
        return Tracer(self._g, out_shape, node)

    def __radd__(self, other: Any) -> Tracer:
        other = _lift(self._g, other)
        out_shape = _broadcast_shape(other.shape, self.shape)
        node = self._g.emit("add", [other.node, self.node], out_shape)
        return Tracer(self._g, out_shape, node)

    def __sub__(self, other: Any) -> Tracer:
        other = _lift(self._g, other)
        out_shape = _broadcast_shape(self.shape, other.shape)
        node = self._g.emit("sub", [self.node, other.node], out_shape)
        return Tracer(self._g, out_shape, node)

    def __rsub__(self, other: Any) -> Tracer:
        other = _lift(self._g, other)
        out_shape = _broadcast_shape(other.shape, self.shape)
        node = self._g.emit("sub", [other.node, self.node], out_shape)
        return Tracer(self._g, out_shape, node)

    def __mul__(self, other: Any) -> Tracer:
        other = _lift(self._g, other)
        out_shape = _broadcast_shape(self.shape, other.shape)
        node = self._g.emit("mul", [self.node, other.node], out_shape)
        return Tracer(self._g, out_shape, node)

    def __rmul__(self, other: Any) -> Tracer:
        other = _lift(self._g, other)
        out_shape = _broadcast_shape(other.shape, self.shape)
        node = self._g.emit("mul", [other.node, self.node], out_shape)
        return Tracer(self._g, out_shape, node)

    def __neg__(self) -> Tracer:
        node = self._g.emit("neg", [self.node], self.shape)
        return Tracer(self._g, self.shape, node)

    @property
    def T(self) -> Tracer:
        """Transpose — reverses the shape."""
        if len(self.shape) != 2:
            raise ShapeMismatchError(f"transpose requires a 2D shape, got {self.shape}")
        out_shape = (self.shape[1], self.shape[0])
        node = self._g.emit("transpose", [self.node], out_shape)
        return Tracer(self._g, out_shape, node)

    def __repr__(self) -> str:
        return f"Tracer(shape={self.shape}, node={self.node})"


class ShapeMismatchError(ValueError):
    """Raised when a traced operation's input shapes are incompatible."""


def _lift(graph: Graph, value: Any) -> Tracer:
    """Promote a concrete value to a :class:`Tracer`.

    - A :class:`Tracer` is returned as-is.
    - A numpy array is baked into a ``const`` node (this is the "fixed as
      compiled" freezing — e.g. a precomputed gain ``K`` becomes a literal).
    - A list/tuple/scalar is converted to an ndarray first.

    Args:
        graph: The graph to emit the ``const`` node into (if needed).
        value: A Tracer, ndarray, list, tuple, or scalar.

    Returns:
        A :class:`Tracer` whose node is either the passed-in tracer's node
        or a freshly-emitted ``const`` node.
    """
    if isinstance(value, Tracer):
        return value
    if isinstance(value, np.ndarray):
        node = graph.const(value)
        return Tracer(graph, value.shape, node)
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float64)
        node = graph.const(arr)
        return Tracer(graph, arr.shape, node)
    # Scalar (int/float) — represent as a 0-d array.
    arr = np.asarray(value, dtype=np.float64)
    node = graph.const(arr)
    return Tracer(graph, arr.shape, node)


def _check_matmul_shapes(a: tuple[int, ...], b: tuple[int, ...]) -> None:
    """Validate matmul shapes: (m, k) @ (k, n) -> (m, n)."""
    if len(a) != 2 or len(b) != 2:
        raise ShapeMismatchError(f"matmul requires 2D inputs, got {a} @ {b}")
    if a[1] != b[0]:
        raise ShapeMismatchError(f"matmul shape mismatch: {a} @ {b}")


def _matmul_out_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the output shape of a matmul, following numpy's conventions.

    Numpy's ``@`` supports three shape combinations:

    - ``(m, k) @ (k, n)`` → ``(m, n)``  (standard 2D @ 2D)
    - ``(m, k) @ (k,)``   → ``(m,)``    (2D @ 1D; 1D treated as column vec)
    - ``(k,) @ (k, n)``   → ``(n,)``    (1D @ 2D; 1D treated as row vec)

    Any other shape combination raises :class:`ShapeMismatchError`.
    """
    if len(a) == 2 and len(b) == 2:
        if a[1] != b[0]:
            raise ShapeMismatchError(f"matmul shape mismatch: {a} @ {b}")
        return (a[0], b[1])
    if len(a) == 2 and len(b) == 1:
        if a[1] != b[0]:
            raise ShapeMismatchError(f"matmul shape mismatch: {a} @ {b}")
        return (a[0],)
    if len(a) == 1 and len(b) == 2:
        if a[0] != b[0]:
            raise ShapeMismatchError(f"matmul shape mismatch: {a} @ {b}")
        return (b[1],)
    raise ShapeMismatchError(f"matmul requires 1D or 2D inputs, got {a} @ {b}")


def _broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Compute the broadcasted shape of two traced shapes.

    Mirrors numpy broadcasting. Used by add/sub/mul so the recorded node
    has the shape the real numpy operation would produce.
    """
    if a == b:
        return a
    # Treat 0-d as broadcastable with anything.
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    if len(a) != len(b):
        raise ShapeMismatchError(f"cannot broadcast {a} and {b} (different ranks)")
    out = []
    for da, db in zip(a, b):
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            raise ShapeMismatchError(f"cannot broadcast {a} and {b}")
    return tuple(out)
