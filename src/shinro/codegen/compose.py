"""Stitch per-node graphs into one combined graph for a closed-loop step.

The composition pass takes the per-component graphs captured by
:func:`shinro.codegen.trace_node.trace_node` and wires them into a single
graph representing one tick of the closed loop:

    Trajectory ─x_ref─▶ Controller ─u─▶ [clip] ──▶ output
                        ▲
                        │ x_hat
                     Estimator ◀── y (measurement)

The wiring is the **fixed ABC dataflow** — it's the same for every scenario,
so no per-scenario edge dict is needed. What's scenario-specific (clip
limits, vector dims) comes from the scenario config.

Shape mismatches between estimator and controller (e.g. the KF produces
``(n,1)`` column vectors but the LQR expects ``(n,)`` flat) are resolved by
auto-inserting ``reshape`` nodes. This is how XLA handles it — shape
mismatches in a graph are resolved by explicit reshape ops, auto-inserted by
the compiler. The alternative (standardizing all components to flat
vectors) is deferred to a future refactor.

Recurrent state (the estimator's ``x_hat``, the previous control ``u_prev``)
is identified and emitted as state outputs that feed back as state inputs on
the next tick — these become the ``.so``'s internal ``var`` state in a
future slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shinro.codegen.trace_node import NodeGraph
from shinro.codegen.tracing import Graph, ShapeMismatchError


@dataclass
class ComposedGraph:
    """A composed closed-loop step graph plus its I/O ports.

    Args:
        graph: The combined graph (one ``shinro_step``).
        inputs: Names of the graph's input ports (fed each tick by the host).
        outputs: Names of the graph's output ports (returned to the host).
        state_inputs: Names of recurrent state input ports (fed from the
            previous tick's state outputs).
        state_outputs: Names of recurrent state output ports (fed to the
            next tick's state inputs).
    """

    graph: Graph
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    state_inputs: list[str] = field(default_factory=list)
    state_outputs: list[str] = field(default_factory=list)


def compose(
    estimator: NodeGraph,
    controller: NodeGraph,
    plant_dims: dict[str, int],
    input_limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> ComposedGraph:
    """Compose an estimator and controller into one closed-loop step graph.

    The wiring is the fixed ABC dataflow::

        y (measurement) ──▶ Estimator ──x_hat──▶ Controller ──u──▶ [clip] ──▶ output
        x_ref ───────────────────────────────▶ Controller
        state_x_hat (recurrent) ─▶ Estimator

    State (estimator's ``x_hat``) and the previous control (``u_prev``, fed
    to the estimator) are recurrent edges — emitted as state outputs that
    feed back as state inputs next tick.

    Args:
        estimator: The traced estimator node graph.
        controller: The traced controller node graph.
        plant_dims: Dict with ``n_x`` (state dim) and ``n_u`` (input dim)
            from the scenario's plant. Used to declare the measurement and
            reference input shapes.
        input_limits: Optional ``(lo, hi)`` clip bounds from
            ``[scenario.input_limits]``. If provided, a ``clip`` node is
            inserted on the controller output.

    Returns:
        A :class:`ComposedGraph` with the combined graph and port names.
    """
    n_x = plant_dims["n_x"]
    n_u = plant_dims["n_u"]

    combined = Graph()

    # --- declare combined-graph inputs first (the merge will reference them) ---
    y_id = combined.input("y", (n_x,))
    x_ref_id = combined.input("x_ref", (n_x,))
    u_prev_id = combined.input("u_prev", (n_u,))

    # The estimator's state_x_hat input — recurrent edge.
    state_x_hat_shape = _lookup_input_shape(estimator.graph, "state_x_hat", default=(n_x, 1))
    state_x_hat_id = combined.input("state_x_hat", state_x_hat_shape)

    # --- merge the estimator ---
    # Estimator inputs: measurement (y), control_input (u_prev), state_x_hat.
    # The KF expects (n,1) column vectors; y and u_prev are (n,) flat → reshape.
    est_input_map = {
        "measurement": y_id,
        "control_input": u_prev_id,
        "state_x_hat": state_x_hat_id,
    }
    est_remap, est_source_ids = _merge_and_rewire(combined, estimator.graph, est_input_map)

    # The estimator's output (x_hat) — flatten if it's (n,1) to match the
    # controller's (n,) expectation. The output_nodes value is a subgraph
    # node id; if it was an input placeholder, look it up via est_source_ids
    # (by name); otherwise use the remap.
    est_out_key = estimator.output_nodes.get("out")
    if est_out_key is None:
        est_out_key = estimator.output_nodes.get("state_x_hat")
    if est_out_key is None:
        raise ValueError(f"estimator has no output node; have {list(estimator.output_nodes)}")
    est_out_src_node = estimator.graph.nodes[est_out_key]
    if est_out_src_node.op == "input":
        est_out_id = est_source_ids[est_out_src_node.attrs["name"]]
    else:
        est_out_id = est_remap[est_out_key]
    x_hat_flat_id = _ensure_shape(combined, est_out_id, state_x_hat_shape, (n_x,))

    # --- merge the controller ---
    # Controller inputs: current_state (x_hat), target_state (x_ref).
    ctrl_input_map = {
        "current_state": x_hat_flat_id,
        "target_state": x_ref_id,
    }
    ctrl_remap, _ = _merge_and_rewire(combined, controller.graph, ctrl_input_map)

    # The controller's output (u) — clip if input_limits provided, then emit.
    ctrl_out_src_node = controller.graph.nodes[controller.output_nodes["out"]]
    if ctrl_out_src_node.op == "input":
        # Stateless controller whose output is a direct input — unusual but
        # handled for robustness.
        u_id = x_hat_flat_id
    else:
        u_id = ctrl_remap[controller.output_nodes["out"]]
    if input_limits is not None:
        lo, hi = input_limits
        u_id = combined.emit(
            "clip",
            [u_id],
            (n_u,),
            lo=np.asarray(lo, dtype=np.float64),
            hi=np.asarray(hi, dtype=np.float64),
        )

    # --- declare combined-graph outputs ---
    combined.output("u", u_id)
    # State outputs (recurrent): the new x_hat, and u (which becomes u_prev next tick).
    combined.output("state_x_hat", est_out_id)
    combined.output("state_u_prev", u_id)

    return ComposedGraph(
        graph=combined,
        inputs=["y", "x_ref", "u_prev", "state_x_hat"],
        outputs=["u"],
        state_inputs=["state_x_hat", "u_prev"],
        state_outputs=["state_x_hat", "state_u_prev"],
    )


# ─── helpers ───────────────────────────────────────────────────────────────


def _merge_and_rewire(
    combined: Graph,
    subgraph: Graph,
    input_map: dict[str, int],
) -> tuple[dict[int, int], dict[str, int]]:
    """Merge a subgraph into the combined graph, rewiring placeholder inputs.

    ``input`` nodes from the subgraph are placeholders — they're not copied.
    Instead, when a node consumes a placeholder input, that consumption is
    rewired to the real combined-graph source node from ``input_map``,
    keyed by the placeholder's name. If the source node's shape doesn't
    match the placeholder's shape, a ``reshape`` node is auto-inserted.

    Args:
        combined: The graph to merge into.
        subgraph: The source subgraph.
        input_map: Maps each placeholder input name → combined-graph source
            node id. Source shapes are looked up from ``combined``.

    Returns:
        A tuple ``(remap, source_ids)`` where ``remap`` maps old subgraph
        node id → new combined-graph node id, and ``source_ids`` maps
        placeholder input name → the (possibly reshaped) combined-graph
        source node id that consumers should reference.
    """
    # Build a name → source-id lookup, inserting reshapes where needed.
    # The placeholder's expected shape comes from the subgraph's input node.
    source_ids: dict[str, int] = {}
    for name, src_id in input_map.items():
        src_shape = combined.nodes[src_id].shape
        dst_shape = _lookup_input_shape(subgraph, name, src_shape)
        source_ids[name] = _ensure_shape(combined, src_id, src_shape, dst_shape)

    remap: dict[int, int] = {}
    for old_id, node in enumerate(subgraph.nodes):
        if node.op == "input":
            # Placeholder — not copied. Consumers will use source_ids[name].
            continue
        # Remap inputs: placeholders → source ids, others → already-copied ids.
        new_inputs: list[int] = []
        for inp in node.inputs:
            inp_node = subgraph.nodes[inp]
            if inp_node.op == "input":
                name = inp_node.attrs["name"]
                new_inputs.append(source_ids[name])
            else:
                new_inputs.append(remap[inp])
        new_id = combined.emit(node.op, new_inputs, node.shape, **node.attrs)
        remap[old_id] = new_id
    return remap, source_ids


def _lookup_input_shape(graph: Graph, name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Find an ``input`` node by name and return its shape, or ``default``."""
    for node in graph.nodes:
        if node.op == "input" and node.attrs.get("name") == name:
            return node.shape
    return default


def _ensure_shape(
    graph: Graph,
    src_id: int,
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
) -> int:
    """Insert a ``reshape`` node if src_shape != dst_shape, else pass through.

    This is the auto-reshape: when the estimator produces ``(n,1)`` but the
    controller expects ``(n,)``, a reshape node is inserted to bridge the
    mismatch. If the shapes already match, the source node is returned
    unchanged.
    """
    if src_shape == dst_shape:
        return src_id
    if _is_reshape_possible(src_shape, dst_shape):
        return graph.emit("reshape", [src_id], dst_shape, target_shape=dst_shape)
    raise ShapeMismatchError(
        f"cannot auto-reshape {src_shape} → {dst_shape} "
        f"(must have the same number of elements)"
    )


def _is_reshape_possible(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """Return True if a can be reshaped to b (same total number of elements)."""
    return int(np.prod(a)) == int(np.prod(b))
