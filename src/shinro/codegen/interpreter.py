"""Interpret a captured :class:`Graph` on real numpy inputs.

This is the correctness oracle for the tracer. A captured graph is a flat
list of :class:`Node` records in execution order; the interpreter walks
them, dispatches each through the ``OP_HANDLERS`` registry, and collects
the named outputs. If the interpreter's output matches the live
``NumpyBackend`` computation on the same inputs (to float-exactness), the
tracer is sound — it captured the math correctly.

The dispatch is a 5-line loop over the registry; adding a new op is a
``@register_op`` decorator in :mod:`shinro.codegen.ops`, not an edit here.
"""

from __future__ import annotations

import numpy as np

from shinro.codegen.ops import OP_HANDLERS, missing_op_error
from shinro.codegen.tracing import Graph, Node


def interpret(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run a captured graph on real numpy inputs and return named outputs.

    Args:
        graph: The graph to interpret (from :func:`shinro.codegen.trace_node.trace_node`
            or :func:`shinro.codegen.compose.compose`).
        inputs: Maps each graph ``input`` port name to a concrete ndarray.

    Returns:
        A dict mapping each ``output`` port name to its computed ndarray.

    Raises:
        KeyError: If an ``input`` port is missing from ``inputs``.
        NotImplementedError: If the graph contains an op with no registered
            handler (the error names the op to add).
    """
    values: dict[int, np.ndarray] = {}
    outputs: dict[str, np.ndarray] = {}

    for i, node in enumerate(graph.nodes):
        handler = OP_HANDLERS.get(node.op)
        if handler is None:
            raise missing_op_error(node.op)
        result = handler(node, values, inputs)
        values[i] = result
        if node.op == "output":
            outputs[node.attrs["name"]] = result

    return outputs


def interpret_step(
    graph: Graph,
    inputs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Interpret a graph, separating outputs from recurrent state outputs.

    A convenience for closed-loop interpretation: returns ``(outputs, state)``
    where ``state`` are the ``state_*`` outputs (recurrent edges that feed
    back as inputs next tick) and ``outputs`` are the non-state outputs.

    Args:
        graph: The graph to interpret.
        inputs: Maps each graph input port name to a concrete ndarray.

    Returns:
        A tuple ``(outputs, state)`` of named-array dicts.
    """
    all_outputs = interpret(graph, inputs)
    outputs: dict[str, np.ndarray] = {}
    state: dict[str, np.ndarray] = {}
    for name, value in all_outputs.items():
        if name.startswith("state_"):
            state[name[len("state_") :]] = value
        else:
            outputs[name] = value
    return outputs, state


# silence unused-import linters for Node (re-exported for type-checkers)
_ = Node