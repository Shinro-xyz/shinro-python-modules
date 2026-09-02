"""Stitch per-node graphs into one combined graph for a closed-loop step.

The composition pass takes the per-component graphs captured by
:func:`shinro.codegen.trace_node.trace_node` and wires them into a single
graph representing one tick of the closed loop:

    Trajectory ─x_ref─▶ Controller ─u─▶ [clip] ──▶ output
                         ▲
                         │ x_hat
                      Estimator ◀── y (measurement)

The wiring is the **fixed ABC dataflow** — the same for every scenario, so no
per-scenario edge dict is needed. What's scenario-specific (clip limits, vector
dims) comes from the scenario config. Controller inputs are mapped by role
from the compute() signature; a regulator controller (MPC) gets the error
``x_hat - x_ref`` instead of a separate reference input.

Shape mismatches between estimator and controller (e.g. the KF produces
``(n,1)`` column vectors but the LQR expects ``(n,)`` flat) are resolved by
auto-inserting ``reshape`` nodes. This is how XLA handles it — shape
mismatches in a graph are resolved by explicit reshape ops, auto-inserted by
the compiler. The alternative (standardizing all components to flat
vectors) is deferred to a future refactor.

Recurrent state is identified and emitted as state outputs that feed back as
state inputs on the next tick (packed into the ``.so``'s ``state_out`` C-ABI
buffer; the host feeds it back as inputs). Every ``state_*`` input placeholder
the traced estimator declares becomes a recurrent port — ``x_hat`` specially
(feeds the controller), everything else (e.g. the Kalman filter's covariance
``P``) transparently. A state attribute the estimator mutated during the trace
*without* a matching pre-injected placeholder raises: that recursion would be
silently frozen in the deployed graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shinro.codegen.trace_node import NodeGraph, _state_port_name
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

    The controller's inputs are mapped by ROLE from its compute() signature
    (state / reference / u_prev — see ``_CONTROLLER_INPUT_ROLES``), not by
    hardcoded names. A controller that takes a reference (LQR, MPPI) receives
    ``x_hat`` and ``x_ref`` separately; a regulator without a reference input
    (MPC_LTI, MPC_DeltaU) receives the error ``x_hat - x_ref`` instead.
    ``u_prev`` routes the shared previous-control port into controllers that
    declare it (MPC_DeltaU).

    State (recurrent attributes on both sides — the estimator's ``x_hat`` and
    covariance ``P``, the controller's integral state, ...) and the previous
    control (``u_prev``, fed to the estimator) are recurrent edges — emitted
    as state outputs that feed back as state inputs next tick. The recurrent
    ports are derived from each traced graph: every ``state_*`` input
    placeholder it declares is wired, and every state attribute ``trace_node``
    detected (attr-diff) must have one — otherwise the recursion would be
    silently frozen at its trace-time value and this raises instead.

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

    # The estimator's recurrent state inputs — every "state_*" input
    # placeholder the traced graph declares (pre-injected via trace_node's
    # state_shapes). Each becomes a combined-graph input port fed from the
    # previous tick's state output. Emission order of the placeholders fixes
    # the port order (deterministic: contract inputs first, then
    # state_shapes iteration order).
    state_ports = [
        node.attrs["name"]
        for node in estimator.graph.nodes
        if node.op == "input" and str(node.attrs.get("name", "")).startswith("state_")
    ]

    # Guard rail: a state attribute the estimator mutated during the traced
    # call (detected by trace_node's attr-diff) MUST have a pre-injected
    # placeholder. Without one its recursion is computed from trace-time
    # constants and silently frozen in the deployed graph — e.g. the KF's
    # covariance P, which would bake a one-step gain instead of running the
    # live Riccati recursion.
    est_port_to_attr = {_state_port_name(a): a for a in estimator.state_attrs}
    for attr in estimator.state_attrs:
        port = _state_port_name(attr)
        if port not in state_ports:
            raise ValueError(
                f"estimator mutated attribute '{attr}' during the traced call "
                f"but '{port}' was not pre-injected via "
                f"trace_node(state_shapes=...); its recursion would be "
                f"silently frozen in the deployed graph. Add "
                f"'{attr}' to state_shapes at the trace site."
            )

    state_port_ids: dict[str, int] = {}
    for port in state_ports:
        shape = _lookup_input_shape(estimator.graph, port, default=(n_x, 1))
        state_port_ids[port] = combined.input(port, shape)

    # --- merge the estimator ---
    # Estimator inputs: measurement (y), control_input (u_prev), state ports.
    # The KF expects (n,1) column vectors; y and u_prev are (n,) flat → reshape.
    est_input_map = {
        "measurement": y_id,
        "control_input": u_prev_id,
        **state_port_ids,
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
    x_hat_flat_id = _ensure_shape(
        combined, est_out_id, _lookup_input_shape(estimator.graph, "state_x_hat", default=(n_x, 1)), (n_x,)
    )

    # --- merge the controller ---
    # Controller inputs are mapped by ROLE, driven by the input placeholders
    # the traced controller declares (its compute() signature):
    #   state     -> the state estimate x_hat, or the tracking error
    #                e = x_hat - x_ref when the controller declares no
    #                reference input. A regulator (MPC_LTI, MPC_DeltaU)
    #                drives e to zero, which tracks x_ref — exact for A = I
    #                plants; general A needs an (A - I) x_ref feedforward.
    #   reference -> the reference trajectory x_ref.
    #   u_prev    -> the previous control — the same recurrent port the
    #                estimator consumes (e.g. MPC_DeltaU's rate input).
    controller_takes_reference = False
    state_role_names: list[str] = []
    ctrl_input_map: dict[str, int] = {}
    for node in controller.graph.nodes:
        if node.op != "input":
            continue
        name = node.attrs["name"]
        if str(name).startswith("state_"):
            # Recurrent controller state — wired separately below.
            continue
        role = _CONTROLLER_INPUT_ROLES.get(name)
        if role == "reference":
            controller_takes_reference = True
            ctrl_input_map[name] = x_ref_id
        elif role == "u_prev":
            ctrl_input_map[name] = u_prev_id
        elif role == "state":
            state_role_names.append(name)
        else:
            raise ValueError(
                f"controller input '{name}' does not map to a known role "
                f"(state / reference / u_prev); extend _CONTROLLER_INPUT_ROLES "
                f"in shinro.codegen.compose"
            )
    if controller_takes_reference or not state_role_names:
        state_feed_id = x_hat_flat_id
    else:
        # Regulator: feed the error state e = x_hat - x_ref.
        state_feed_id = combined.emit("sub", [x_hat_flat_id, x_ref_id], (n_x,))
    for name in state_role_names:
        ctrl_input_map[name] = state_feed_id

    # Controller recurrent state — the same mechanism as the estimator's
    # state ports: every "state_*" input placeholder the traced controller
    # declares (e.g. PID's _integral/_prev_error/_has_run) becomes a
    # combined input port fed from the previous tick's state output, with
    # the same mutated-without-placeholder guard rail.
    ctrl_state_ports = [
        node.attrs["name"]
        for node in controller.graph.nodes
        if node.op == "input" and str(node.attrs.get("name", "")).startswith("state_")
    ]
    ctrl_port_to_attr = {_state_port_name(a): a for a in controller.state_attrs}
    for attr in controller.state_attrs:
        port = _state_port_name(attr)
        if port not in ctrl_state_ports:
            raise ValueError(
                f"controller mutated attribute '{attr}' during the traced call "
                f"but '{port}' was not pre-injected via "
                f"trace_node(state_shapes=...); its recursion would be "
                f"silently frozen in the deployed graph. Add "
                f"'{attr}' to state_shapes at the trace site."
            )
    for port in ctrl_state_ports:
        shape = _lookup_input_shape(controller.graph, port, default=(n_u,))
        ctrl_input_map[port] = combined.input(port, shape)

    ctrl_remap, ctrl_source_ids = _merge_and_rewire(combined, controller.graph, ctrl_input_map)

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
    # State outputs (recurrent): each wired state port's new value, plus u
    # (which becomes u_prev next tick). x_hat reuses the estimator's output
    # value; every other port (e.g. state_P) reads the estimator's own
    # state_<attr> output node.
    emitted_state_ports: list[str] = []
    for port in state_ports:
        if port == "state_x_hat":
            combined.output("state_x_hat", est_out_id)
            emitted_state_ports.append(port)
            continue
        attr = est_port_to_attr.get(port)
        if attr is None:
            # Placeholder fed but the trace detected no mutation — nothing
            # recurs; the port only feeds the estimator's computation.
            continue
        state_out_key = estimator.output_nodes.get(port)
        if state_out_key is None:
            raise ValueError(f"estimator has no {port} output node; have {list(estimator.output_nodes)}")
        state_out_src_node = estimator.graph.nodes[state_out_key]
        if state_out_src_node.op == "input":
            state_out_id = est_source_ids[state_out_src_node.attrs["name"]]
        else:
            state_out_id = est_remap[state_out_key]
        combined.output(port, state_out_id)
        emitted_state_ports.append(port)

    # Controller state outputs (recurrent): each mutated controller attr's
    # new value (e.g. PID's integral), read from the controller's own
    # state_<attr> output node.
    emitted_ctrl_state_ports: list[str] = []
    for port in ctrl_state_ports:
        if ctrl_port_to_attr.get(port) is None:
            # Placeholder fed but the trace detected no mutation.
            continue
        state_out_key = controller.output_nodes.get(port)
        if state_out_key is None:
            raise ValueError(f"controller has no {port} output node; have {list(controller.output_nodes)}")
        state_out_src_node = controller.graph.nodes[state_out_key]
        if state_out_src_node.op == "input":
            state_out_id = ctrl_source_ids[state_out_src_node.attrs["name"]]
        else:
            state_out_id = ctrl_remap[state_out_key]
        combined.output(port, state_out_id)
        emitted_ctrl_state_ports.append(port)
    combined.output("state_u_prev", u_id)

    return ComposedGraph(
        graph=combined,
        inputs=["y", "x_ref", "u_prev"] + state_ports + ctrl_state_ports,
        outputs=["u"],
        state_inputs=emitted_state_ports + emitted_ctrl_state_ports + ["u_prev"],
        state_outputs=emitted_state_ports + emitted_ctrl_state_ports + ["state_u_prev"],
    )


# ─── helpers ───────────────────────────────────────────────────────────────


# Controller input name → dataflow role. Exact names (no prefix matching) so
# `x_ref` never collides with `x0`/`x`. Controllers whose compute() uses other
# names for these roles should be added here; anything unmapped raises at
# compose time rather than silently mis-wiring (e.g. SMC's dynamics terms
# f_x/g_x, which need a different wiring model entirely).
_CONTROLLER_INPUT_ROLES: dict[str, str] = {
    "x0": "state",
    "current_state": "state",
    "state": "state",
    "x": "state",
    "x_ref": "reference",
    "target_state": "reference",
    "target": "reference",
    "u_prev": "u_prev",
}


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
        if node.op == "output":
            # Subgraph outputs are markers only — compose() declares the
            # combined graph's outputs itself. Skip them so output names
            # don't duplicate. output_nodes already holds the SOURCE node
            # id (not the output node's id), so no remap entry is needed.
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
