"""Compile-time constant folding for the composed control graph.

The composed graph is a pure dataflow DAG in execution order (every node's
inputs have smaller ids), so constant propagation is a single forward sweep: a
node is *constant* iff all of its inputs are constant. A constant node of an
intrinsic op is evaluated at lowering time with the **same numpy handler**
``interpret()`` uses (:data:`shinro.codegen.ops.OP_HANDLERS`), so the folded value
is bit-exact with the interpreter — the pass never changes what the graph
computes, only how much of it is baked.

The pass applies three compounding transforms in one sweep:

1. **Const-fold** — evaluate every all-constant intrinsic node and materialize
   its result as a ``const`` node.
2. **Const dedup** — merge constants with identical ``(shape, f64 bytes)``.
3. **Dead-node elimination** — drop nodes unreachable from the ``input`` and
   ``output`` roots (folded-away consts and their now-dead weight nodes).

Non-foldable ops (never evaluated here):

- ``input`` — a graph input port, not a computation.
- ``output`` — a marker; it is kept and its source is rewired, never folded.
- ``solve_qp`` — reads only ``q`` but depends on the baked solver; folding it
   would bake an interpreted solution and is deliberately out of scope.

The pass is applied inside :func:`shinro.codegen.lower_zig.lower_zig`, so
``interpret()`` (the correctness oracle, and gate A) still runs on the
**unfolded** graph while the emitted table and manifest describe the folded one.
The ComposedGraph's port lists are copied verbatim, so the C-ABI is unchanged.

:func:`canonicalize` runs alongside it (also lowering-only) to simplify identity
operations (``x*1``, ``x+0``, ``--x``, ``(x.T).T``, ...) — never arithmetic, so
it too is bit-exact.
"""

from __future__ import annotations

import numpy as np

from shinro.codegen.compose import ComposedGraph
from shinro.codegen.ops import OP_HANDLERS
from shinro.codegen.tracing import Graph, Node

#: Ops that are never folded: graph ports and the solver-coupled QP op.
_NON_FOLDABLE = frozenset({"input", "output", "solve_qp"})


def fold_constants(cg: ComposedGraph, *, dedupe: bool = True, dce: bool = True) -> ComposedGraph:
    """Return a copy of ``cg`` with all-constant subgraphs folded and deduped.

    Bit-exact with the input graph (the folded values are computed by the same
    numpy handlers :func:`shinro.codegen.interpreter.interpret` dispatches to),
    so it is safe to lower in place of ``cg`` while keeping ``interpret(cg)`` as
    the oracle.

    Args:
        cg: The composed graph to optimize. Not mutated.
        dedupe: Merge constants with identical ``(shape, f64 bytes)``.
        dce: Drop nodes unreachable from the ``input``/``output`` roots.

    Returns:
        A new :class:`ComposedGraph` with the same port lists and a folded graph.
    """
    nodes = cg.graph.nodes

    # --- forward sweep: constant propagation -------------------------------
    # The list is in topological order, so one pass resolves every node.
    values: dict[int, np.ndarray] = {}
    is_const = [False] * len(nodes)
    for i, node in enumerate(nodes):
        if node.op == "const":
            values[i] = np.asarray(node.attrs["value"], dtype=np.float64)
            is_const[i] = True
            continue
        if node.op in _NON_FOLDABLE:
            continue
        handler = OP_HANDLERS.get(node.op)
        if handler is None or not all(is_const[j] for j in node.inputs):
            # Unregistered op (interpret would fail on it too) or a live input.
            continue
        result = np.asarray(handler(node, values, {}), dtype=np.float64)
        expected = tuple(node.shape)
        if tuple(result.shape) != expected:
            # The tracer and the handler disagree on shape; folding would bake
            # a value the node's declared shape (rows/cols) does not describe.
            raise ValueError(
                f"constant fold of node {i} ({node.op}): handler produced shape "
                f"{tuple(result.shape)}, node declares {expected}"
            )
        values[i] = result
        is_const[i] = True

    # --- materialize: fold, dedupe, rewire -----------------------------------
    remap: dict[int, int] = {}
    out_nodes: list[Node] = []
    seen: dict[tuple[tuple[int, ...], bytes], int] = {}

    def _materialize(shape: tuple[int, ...], value: np.ndarray) -> int:
        key = (tuple(shape), np.ascontiguousarray(value, dtype=np.float64).tobytes())
        if dedupe:
            hit = seen.get(key)
            if hit is not None:
                return hit
        nid = len(out_nodes)
        out_nodes.append(Node(op="const", inputs=[], shape=tuple(shape), attrs={"value": value}))
        if dedupe:
            seen[key] = nid
        return nid

    for i, node in enumerate(nodes):
        if node.op == "const" or (is_const[i] and node.op not in _NON_FOLDABLE):
            remap[i] = _materialize(node.shape, values[i])
        else:
            remap[i] = len(out_nodes)
            out_nodes.append(Node(op=node.op, inputs=[remap[j] for j in node.inputs], shape=node.shape, attrs=node.attrs))

    if dce:
        out_nodes = _eliminate_dead(out_nodes)

    graph = Graph()
    graph.nodes = out_nodes
    return ComposedGraph(
        graph=graph,
        inputs=list(cg.inputs),
        outputs=list(cg.outputs),
        state_inputs=list(cg.state_inputs),
        state_outputs=list(cg.state_outputs),
    )


def _eliminate_dead(nodes: list[Node]) -> list[Node]:
    """Drop nodes unreachable from the ``input``/``output`` roots.

    Input nodes are roots even when unused: the lowerer resolves every name in
    ``ComposedGraph.inputs`` by scanning the node table, so they must stay.
    Output nodes are roots because they define the C-ABI ports.
    """
    live = [node.op in ("input", "output") for node in nodes]
    # Backward closure: inputs always have smaller ids, so a reverse sweep
    # propagates liveness from consumers to producers in one pass.
    for nid in range(len(nodes) - 1, -1, -1):
        if live[nid]:
            for j in nodes[nid].inputs:
                live[j] = True
    keep = [nid for nid in range(len(nodes)) if live[nid]]
    index = {old: new for new, old in enumerate(keep)}
    return [
        Node(
            op=nodes[old].op,
            inputs=[index[j] for j in nodes[old].inputs],
            shape=nodes[old].shape,
            attrs=nodes[old].attrs,
        )
        for old in keep
    ]


def canonicalize(cg: ComposedGraph, *, dce: bool = True) -> ComposedGraph:
    """Return a copy of ``cg`` with identity operations simplified away.

    Bit-exact: every rule replaces a node with a value that is numerically
    identical for *all* inputs, so ``interpret(canonicalize(cg))`` equals
    ``interpret(cg)`` exactly and the oracle stays intact. It never evaluates
    arithmetic — that is :func:`fold_constants` — so the two compose freely.

    Rules (the identity operand is a ``const`` whose entries all equal the
    value; ``-0.0`` counts as ``0.0``):

    - ``mul(x, 1)`` / ``mul(1, x)`` -> ``x``
    - ``div(x, 1)`` -> ``x``
    - ``add(x, 0)`` / ``add(0, x)`` -> ``x``
    - ``sub(x, 0)`` -> ``x``
    - ``pow(x, 1)`` -> ``x``
    - ``copy(x)`` -> ``x``
    - ``neg(neg(x))`` -> ``x``
    - ``transpose(transpose(x))`` -> ``x``
    - ``reshape(reshape(x, _), s)`` -> ``reshape(x, s)`` (inner reshape has
      this single consumer)

    Args:
        cg: The composed graph to simplify. Not mutated.
        dce: Drop nodes made dead by the rewrites (and their now-unused consts).

    Returns:
        A new :class:`ComposedGraph` with the same port lists.
    """
    nodes = cg.graph.nodes
    consumers: dict[int, list[int]] = {}
    for i, n in enumerate(nodes):
        for j in n.inputs:
            consumers.setdefault(j, []).append(i)
    const_val: dict[int, np.ndarray] = {}
    for i, n in enumerate(nodes):
        if n.op == "const":
            const_val[i] = np.asarray(n.attrs["value"], dtype=np.float64)

    canon = list(range(len(nodes)))

    def resolve(i: int) -> int:
        root = i
        while canon[root] != root:
            root = canon[root]
        while canon[i] != root:  # path compression
            canon[i], i = root, canon[i]
        return root

    def _is_const(idx: int, value: float) -> bool:
        v = const_val.get(idx)
        return v is not None and bool(np.all(v == value))

    new_inputs: list[list[int]] = [[] for _ in nodes]
    for i, n in enumerate(nodes):
        ins = [resolve(j) for j in n.inputs]
        new_inputs[i] = ins
        # A surviving operand is only equivalent if it already carries the
        # node's full shape — otherwise the op was a broadcast (e.g.
        # add((1,), zeros((n,))) -> (n,)) and dropping it changes the shape.
        fits = [nodes[j].shape == n.shape for j in ins]
        if n.op == "mul" and len(ins) == 2:
            if fits[1] and _is_const(ins[1], 1.0):
                canon[i] = ins[0]
            elif fits[0] and _is_const(ins[0], 1.0):
                canon[i] = ins[1]
        elif n.op == "div" and len(ins) == 2:
            if fits[0] and _is_const(ins[1], 1.0):
                canon[i] = ins[0]
        elif n.op == "add" and len(ins) == 2:
            if fits[0] and _is_const(ins[1], 0.0):
                canon[i] = ins[0]
            elif fits[1] and _is_const(ins[0], 0.0):
                canon[i] = ins[1]
        elif n.op == "sub" and len(ins) == 2:
            if fits[0] and _is_const(ins[1], 0.0):
                canon[i] = ins[0]
        elif n.op == "pow" and len(ins) == 2:
            if fits[0] and _is_const(ins[1], 1.0):
                canon[i] = ins[0]
        elif n.op == "copy" and len(ins) == 1:
            canon[i] = ins[0]
        elif n.op == "neg" and len(ins) == 1:
            inner = resolve(ins[0])
            if nodes[inner].op == "neg":
                canon[i] = resolve(nodes[inner].inputs[0])
        elif n.op == "transpose" and len(ins) == 1:
            inner = resolve(ins[0])
            if nodes[inner].op == "transpose":
                canon[i] = resolve(nodes[inner].inputs[0])
        elif n.op == "reshape" and len(ins) == 1:
            inner = resolve(ins[0])
            if nodes[inner].op == "reshape" and len(consumers.get(inner, [])) == 1:
                new_inputs[i] = [resolve(nodes[inner].inputs[0])]

    out_nodes: list[Node] = []
    remap: dict[int, int] = {}
    for i, n in enumerate(nodes):
        if canon[i] != i:
            continue  # simplified away; consumers resolve past it
        remap[i] = len(out_nodes)
        out_nodes.append(
            Node(op=n.op, inputs=[remap[resolve(j)] for j in new_inputs[i]], shape=n.shape, attrs=n.attrs)
        )

    if dce:
        out_nodes = _eliminate_dead(out_nodes)

    graph = Graph()
    graph.nodes = out_nodes
    return ComposedGraph(
        graph=graph,
        inputs=list(cg.inputs),
        outputs=list(cg.outputs),
        state_inputs=list(cg.state_inputs),
        state_outputs=list(cg.state_outputs),
    )
