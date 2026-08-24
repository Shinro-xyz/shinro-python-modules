"""Trace a single component into a :class:`Graph`.

The tracer:

1. Infers the component's I/O contract from its ABC and method signature
   (see :mod:`shinro.codegen.infer_contract`).
2. Swaps the component's ``self.bk`` to a :class:`TraceBackend` so every
   ``bk.*`` call records into a fresh :class:`Graph`.
3. Creates :class:`Tracer` values for each declared input and injects them
   as graph ``input`` nodes.
4. Snapshots the component's instance attrs so state mutations can be
   detected after the call.
5. Calls the component's compute method. Every ``bk.*`` and ``@`` op
   appends a node to the graph; concrete parameters (``self.K``, ``self.A``)
   are lifted to ``const`` nodes the moment they touch a tracer.
6. Captures the method's return value (and any detected state mutations)
   as named ``output`` nodes.

The result is a small :class:`Graph` representing one step of the
component, plus an I/O descriptor naming its input and output ports. The
composition pass (:mod:`shinro.codegen.compose`) stitches these per-node
graphs together per the scenario wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shinro.codegen.infer_contract import (
    InferredContract,
    detect_state,
    infer_contract,
    snapshot_instance_attrs,
)
from shinro.codegen.trace_backend import TraceBackend
from shinro.codegen.tracing import Graph, Tracer, _lift


@dataclass
class NodeGraph:
    """A traced component: its graph plus an I/O descriptor.

    Args:
        graph: The captured primitive graph for one call of the component's
            compute method.
        contract: The inferred contract naming the method and its inputs.
        input_nodes: Maps input name → graph node id (the ``input`` nodes).
        output_nodes: Maps output name → graph node id (the upstream node
            feeding each ``output`` node).
        state_attrs: Instance attrs detected as mutated during the call —
            these are recurrent edges (their values feed back as inputs
            next tick).
    """

    graph: Graph
    contract: InferredContract
    input_nodes: dict[str, int] = field(default_factory=dict)
    output_nodes: dict[str, int] = field(default_factory=dict)
    state_attrs: list[str] = field(default_factory=list)


def trace_node(
    component: Any,
    input_shapes: dict[str, tuple[int, ...]],
    state_shapes: dict[str, tuple[int, ...]] | None = None,
) -> NodeGraph:
    """Trace one call of a component's compute method into a :class:`Graph`.

    Args:
        component: An instance of a registered Controller / StateEstimator.
            Its ``self.bk`` is temporarily swapped to a
            :class:`TraceBackend` for the duration of the call; the original
            backend is restored in a ``finally`` so a tracing error doesn't
            leave the component in a traced state.
        input_shapes: Maps each input name (from the inferred contract) to
            its concrete shape. The caller knows these from the scenario's
            plant dimensions.
        state_shapes: Optional shapes for known state attrs (e.g. on a
            re-trace, when state was detected on a prior trace). If provided,
            these attrs are pre-injected as tracers before the call; if not,
            state is detected after the call via attr-diff.

    Returns:
        A :class:`NodeGraph` holding the captured graph and I/O descriptor.

    Raises:
        KeyError: If an input name in the contract is not in ``input_shapes``.
        NotImplementedError: If the component uses an ``ArrayBackend`` op
            with no registered handler (the error names the op to add).
    """
    contract = infer_contract(component, input_shapes)
    graph = Graph()
    bk = TraceBackend(graph)

    # Validate that every input the method expects has a shape.
    for name in contract.input_names:
        if name not in input_shapes:
            raise KeyError(
                f"input '{name}' for {type(component).__name__}.{contract.method_name} "
                f"has no shape in input_shapes={list(input_shapes)}"
            )

    # Create Tracer inputs and emit input nodes.
    tracers: dict[str, Tracer] = {}
    for name in contract.input_names:
        shape = input_shapes[name]
        node = graph.input(name, shape)
        tracers[name] = Tracer(graph, shape, node)

    # Pre-inject state tracers if shapes were provided (re-trace case).
    state_shapes = state_shapes or {}
    for name, shape in state_shapes.items():
        node = graph.input(f"state_{name}", shape)
        setattr(component, name, Tracer(graph, shape, node))

    # Snapshot instance attrs to detect state mutations after the call.
    before = snapshot_instance_attrs(component)

    # Swap backend, call the method, restore backend.
    original_bk = getattr(component, "bk", None)
    component.bk = bk
    try:
        args = [tracers[n] for n in contract.input_names]
        result = getattr(component, contract.method_name)(*args)
    finally:
        component.bk = original_bk

    # Detect state mutations (attrs reassigned during the call).
    state_attrs = detect_state(component, before)
    node_graph = NodeGraph(
        graph=graph,
        contract=contract,
        input_nodes={n: t.node for n, t in tracers.items()},
        state_attrs=state_attrs,
    )

    # Capture outputs. The primary output is the method's return value;
    # any mutated state attrs are also emitted as outputs (recurrent edges).
    if result is not None:
        result_tracer = _lift(graph, result)
        # Use the first output slot name, or a generic "out" if none declared.
        out_name = "out"
        graph.output(out_name, result_tracer.node)
        node_graph.output_nodes[out_name] = result_tracer.node

    for name in state_attrs:
        value = getattr(component, name)
        value_tracer = _lift(graph, value)
        graph.output(f"state_{name}", value_tracer.node)
        node_graph.output_nodes[f"state_{name}"] = value_tracer.node

    return node_graph


def trace_node_with_state(
    component: Any,
    input_shapes: dict[str, tuple[int, ...]],
    state_inputs: dict[str, np.ndarray],
) -> tuple[NodeGraph, dict[str, np.ndarray]]:
    """Trace a component with explicit state inputs and return state outputs.

    This is the form used by the composition pass for components with
    recurrent state (e.g. ``KalmanFilter.x_hat``): the caller injects the
    current state as tracers, the trace captures the new state as outputs,
    and the returned state values can be fed back on the next trace.

    Args:
        component: The component to trace.
        input_shapes: Shapes for the method's non-state inputs.
        state_inputs: Maps state attr name → current ndarray value. These
            are injected as tracers before the call.

    Returns:
        A tuple ``(node_graph, state_outputs)`` where ``state_outputs`` maps
        state attr name → the new value after the call (for the caller to
        feed back next tick).
    """
    state_shapes = {n: v.shape for n, v in state_inputs.items()}
    node_graph = trace_node(component, input_shapes, state_shapes=state_shapes)
    # Read back the post-call state values.
    state_outputs = {n: getattr(component, n) for n in state_inputs}
    return node_graph, state_outputs