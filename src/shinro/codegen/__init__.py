"""XLA-style tracing and Zig lowering for the shinro control suite.

This subpackage implements the tracing model: components run once with
:class:`~shinro.codegen.tracing.Tracer` values instead of real arrays, and
every operation they perform is captured as a node in a
:class:`~shinro.codegen.tracing.Graph`. The captured graph is then
interpreted in Python (the correctness oracle) or lowered to Zig as a
bit-exact data table that a comptime graph VM compiles into a shared object.

Public API:

- :class:`~shinro.codegen.trace_backend.TraceBackend` — the recording
  :class:`ArrayBackend`; pass to ``from_config(..., backend=TraceBackend(g))``.
- :func:`~shinro.codegen.trace_node.trace_node` — trace one component.
- :func:`~shinro.codegen.compose.compose` — stitch per-node graphs into one.
- :func:`~shinro.codegen.interpreter.interpret` — run a captured graph on
  real numpy inputs (the correctness oracle).
- :func:`~shinro.codegen.ops.register_op` — register a handler for a new op.
- :func:`~shinro.codegen.lower_zig.lower_zig` — serialize a composed graph to
  a Zig data table (``runtime/graph_data.zig``); ``make zig-build`` compiles
  it to ``build/base.so`` with the handwritten comptime VM in
  ``runtime/lower.zig``.

See ``lab-notes/daily/`` for the design narrative.
"""

from shinro.codegen.interpreter import interpret, interpret_step
from shinro.codegen.lower_zig import lower_zig
from shinro.codegen.ops import available_ops, has_op, register_op
from shinro.codegen.trace_backend import TraceBackend
from shinro.codegen.trace_node import NodeGraph, trace_node, trace_node_with_state
from shinro.codegen.tracing import Graph, Node, ShapeMismatchError, Tracer

__all__ = [
    "Graph",
    "Node",
    "NodeGraph",
    "ShapeMismatchError",
    "TraceBackend",
    "Tracer",
    "available_ops",
    "has_op",
    "interpret",
    "interpret_step",
    "lower_zig",
    "register_op",
    "trace_node",
    "trace_node_with_state",
]
