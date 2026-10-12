"""Lower a :class:`~shinro.codegen.compose.ComposedGraph` to a Zig data table.

This is Option B of the lowering design: Python emits **data only** — the node
table, the packed constants, and the port layout — never Zig statements. The
actual program (the comptime graph VM) is handwritten once in
``src/shinro/runtime/lower.zig`` and specialized at compile time over this
table.

As of the schema refactor (see :mod:`shinro.codegen.vm_schema`), this serializer
carries **no per-op knowledge**: the op -> Zig mapping, the baked f64 scalar
tables, the packed ``aux`` word, and the manifest are all derived from the
declarative :class:`~shinro.codegen.vm_schema.VmOp` registrations attached to
``@register_op``. Adding a VM op therefore needs no edit here — it needs a
schema entry in ``ops.py`` and a switch arm in ``lower.zig``.

The generated file ``src/shinro/runtime/graph_data.zig`` contains:

- ``nodes``: the flat list of :class:`Node` records — op enum + input ids +
  rows/cols + an ``aux`` index whose meaning depends on the op.
- ``offsets``: per-node buffer offset (node ``i`` owns ``rows*cols`` f64s at
  ``buf[offsets[i]..offsets[i]+rows*cols]``).
- ``buf_len``: total f64 buffer size for one step.
- ``const_blob``: all non-f32 baked constants (NN biases, estimator gains,
  state matrices, ...). Too large to inline for a real policy, so they ship as
  a raw little-endian sidecar ``<stem>_consts.bin`` next to the generated file
  and are re-exposed here via ``@embedFile`` + ``std.mem.bytesAsSlice``.
- ``const_blob_f32``: the f32-exact contraction weights. Too large to inline
  as source (a 1M-parameter policy is ~20 MB of hex literals), so they are
  written to a raw little-endian sidecar ``<stem>_weights.bin`` next to the
  generated file and re-exposed here via ``@embedFile`` +
  ``std.mem.bytesAsSlice``. The bytes are baked into the compiled ``.so`` all
  the same — they just never pass through the Zig parser.
- per-op f64 tables (``gemm_alpha``, ``clip_lo``, ``layernorm_eps``, ...): the
  scalar/array attributes declared by each op's :class:`VmOp` schema.
- ``has_solve_qp``: whether any node performs QP, used by the Zig build to
  decide whether to link the OSQP codegen bake.
- ``n_outputs``, ``output_offsets``, ``state_offsets``: port packing into the
  C-ABI ``shinro_step(inputs, outputs, state_out)`` buffers.

Small scalars and per-node parameter arrays stay as Zig hex floats
(``float.hex()``), which bit-exactly round-trip the f64 the tracer baked in —
the "fixed at compile time" property holds across the language boundary. The
two large numeric blobs (``const_blob`` and ``const_blob_f32``) instead ship as
raw little-endian sidecars (see above).

Usage:

    from shinro.codegen.lower_zig import lower_zig
    lower_zig(composed, out_path="src/shinro/runtime/graph_data.zig")
"""

from __future__ import annotations

import json
import os

import numpy as np

from shinro.codegen import vm_schema
from shinro.codegen.compose import ComposedGraph
from shinro.codegen.ops import OP_HANDLERS
from shinro.codegen.optimize import fold_constants
from shinro.codegen.tracing import Graph, Node

#: The compiled VM and every C-ABI buffer are f64, so the manifest's byte
#: metrics are element counts times this.
_FLOAT_BYTES = 8
#: f32 bytes, for the manifest's f32 weight-blob metric.
_FLOAT32_BYTES = 4

#: The op -> Zig VM mapping is fully schema-driven; const/input/output declare
#: their ``cst``/``inp``/``out`` renames in ``ops.py``. Only their ``aux``
#: (blob offset / input offset / port index) is computed here rather than
#: declared, because it is plumbing over the graph's port lists.


def lower_zig(
    cg: ComposedGraph,
    out_path: str = "src/shinro/runtime/graph_data.zig",
    provenance: dict | None = None,
    *,
    fold: bool = True,
) -> None:
    """Serialize a composed graph into a Zig data table at ``out_path``.

    Also writes a deterministic JSON manifest next to it (``<stem>_manifest.json``)
    describing the compiled content — op histogram, port layout, buffer sizes,
    and the ``.solve_qp`` n_vars the graph expects. Same inputs ⇒ byte-identical
    manifest (no timestamps), so manifests are diffable audit records.

    ``provenance`` is an optional opaque dict (e.g. the config paths + hashes
    and tool versions the generator consumed) merged into the manifest under
    ``"provenance"``. It keeps the lowerer generic — the framework doesn't
    need to know what a generator records — while letting each generator pin
    exactly which inputs produced the graph.

    ``fold`` (default on) runs :func:`shinro.codegen.optimize.fold_constants`
    first, so the emitted table and manifest describe the folded graph while
    ``interpret(cg)`` — the oracle — still runs on the original. The fold is
    bit-exact.
    """
    if fold:
        cg = fold_constants(cg)
    g = cg.graph

    # Dimensionality contract: the VM is a flat 2-D row-major machine. A
    # rank-3+ shape would silently collapse to (1, 1) in _rows_cols — one f64
    # slot for a whole tensor, every downstream node misaligned, exit 0,
    # garbage everywhere. Refuse loudly instead.
    for i, node in enumerate(g.nodes):
        if len(node.shape) > 2:
            raise ValueError(
                f"node {i} ({node.op}) has rank-{len(node.shape)} shape "
                f"{tuple(node.shape)} — the lowered VM supports 1-D and 2-D "
                f"tensors only"
            )

    # Every registered op must declare its lowering schema (register_op requires
    # it); assert it here too so a hand-populated registry fails loudly rather
    # than emitting an op the VM could mis-lower.
    missing = vm_schema.missing_schemas(OP_HANDLERS)
    if missing:
        raise ValueError(
            "ops registered without a VmOp lowering schema: "
            + ", ".join(sorted(missing))
            + " — declare vm=VmOp(...) in @register_op"
        )

    # --- buffer layout: each node owns a slot sized by its shape ---
    offsets: list[int] = []
    total = 0
    for n in g.nodes:
        offsets.append(total)
        # Const nodes are read from the baked blob, never the workspace, so they
        # own no slot: not advancing keeps buf_len to the real working set (a
        # policy's weights were ~99% of it). The const offsets are left in the
        # table for indexing uniformity but are never sliced.
        if n.op != "const":
            total += _size(n.shape)
    buf_len = total

    # --- input port packing (cg.inputs order) ---
    input_offsets: dict[str, int] = {}
    acc = 0
    for name in cg.inputs:
        input_offsets[name] = acc
        acc += _input_size(g, name)

    # --- consumers, for the f32 weight-blob decision below ---
    # node index → [(consumer node index, input position)]. Bounds the f32 path
    # to constants that only ever feed a contraction's weight operand.
    consumers: dict[int, list[tuple[int, int]]] = {}
    for ci, cnode in enumerate(g.nodes):
        for pos, src in enumerate(cnode.inputs):
            consumers.setdefault(int(src), []).append((ci, pos))

    # A const node is baked into const_blob_f32 (instead of const_blob) when it
    # is f32-exact and only used as a contraction weight — a gemm's B (position
    # 1) or a recurrent cell's W/R (positions 1/2). const_offsets carries the
    # blob offset plus that dtype choice, so the VM selects the mixed-dtype
    # kernel at comptime. This halves the per-tick weight bytes while the kernel
    # widens each lane in a register, so the f64 result is unchanged.
    #
    # Coherence: a cell reads BOTH W and R from whichever blob they were baked
    # into, so the two must agree on the dtype — if only one is f32-eligible,
    # both stay f64 (otherwise one side would be read from the wrong blob).
    f32_nodes = {i for i, n in enumerate(g.nodes) if n.op == "const" and _is_f32_weight(i, n, consumers, g)}
    for n in g.nodes:
        if n.op in ("lstm", "gru", "rnn"):
            w_i, r_i = int(n.inputs[1]), int(n.inputs[2])
            if (w_i in f32_nodes) != (r_i in f32_nodes):
                f32_nodes.discard(w_i)
                f32_nodes.discard(r_i)

    # --- output port packing: separate zero-indexed offsets per buffer ---
    # `outputs` and `state_out` are separate C-ABI buffers, so each needs its
    # own cumulative offset table starting at 0.
    output_offsets: list[int] = []
    acc = 0
    for name in cg.outputs:
        output_offsets.append(acc)
        acc += _output_size(g, name)
    state_offsets: list[int] = []
    acc = 0
    for name in cg.state_outputs:
        state_offsets.append(acc)
        acc += _output_size(g, name)

    # --- schema-driven bake: collect per-op tables + encode every node ---
    # The tables are pre-seeded with every schema-declared name so a table is
    # always emitted (empty when the graph has no such node), matching the
    # hand-written emitter's behaviour. The node encodings are the single source
    # of truth consumed by both the emitted table and the manifest.
    const_blob: list[float] = []
    const_blob_f32: list[float] = []
    tables: dict[str, list[float]] = {name: [] for name in vm_schema.table_names()}
    encodings: list[tuple[str, int]] = [
        _encode_node(g, i, node, f32_nodes, input_offsets, cg, tables, const_blob, const_blob_f32)
        for i, node in enumerate(g.nodes)
    ]

    lines: list[str] = []
    lines.append("// Generated by shinro.codegen.lower_zig — DO NOT EDIT.")
    lines.append("// A ComposedGraph serialized as a comptime data table.")
    # The embedded numeric blobs below are decoded with std.mem.bytesAsSlice.
    if const_blob or const_blob_f32:
        lines.append('const std = @import("std");')
    lines.append("")
    lines.append("pub const Op = enum {")
    enum_names = _op_enum_names()
    for k in range(0, len(enum_names), 6):
        lines.append("    " + ", ".join(enum_names[k : k + 6]) + ",")
    lines.append("};")
    lines.append("")
    lines.append("pub const Node = struct {")
    lines.append("    op: Op,")
    lines.append("    inputs: []const usize,")
    lines.append("    rows: usize,")
    lines.append("    cols: usize,")
    lines.append("    aux: usize,")
    lines.append("    vec: bool,")
    lines.append("};")
    lines.append("")
    lines.append(f"pub const buf_len = {buf_len};")
    # The Zig build reads this flag and links the OSQP solver only when needed.
    lines.append("pub const has_solve_qp = " + ("true" if any(n.op == "solve_qp" for n in g.nodes) else "false") + ";")
    lines.append(f"pub const n_outputs = {len(cg.outputs)};")
    lines.append("")
    lines.append("pub const offsets = [_]usize{")
    lines.append("    " + ", ".join(str(o) for o in offsets) + ",")
    lines.append("};")
    lines.append("")
    lines.append("pub const nodes = [_]Node{")
    for i, node in enumerate(g.nodes):
        vm_op, aux = encodings[i]
        lines.append("    " + _node_line(node, vm_op, aux) + ",")
    lines.append("};")
    lines.append("")
    # --- baked numeric blobs → raw sidecars + @embedFile ---
    # Both blobs leave the generated source: the f64 constants (NN biases,
    # estimator gains, state matrices, ...) and the f32 contraction weights.
    # They ship as raw little-endian .bin sidecars next to this file and are
    # embedded at compile time, so the .so stays self-contained while the .zig
    # stays small. A blob that is empty (no f64 constants, or an all-f64 graph
    # with no f32 weights) keeps its empty literal and writes no sidecar; a
    # stale sidecar from a previous lowering of a different graph is removed so
    # nothing dangles.
    out_dir = os.path.dirname(out_path)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    consts_bin = f"{stem}_consts.bin"
    weights_bin = f"{stem}_weights.bin"
    if const_blob:
        raw = np.asarray(const_blob, dtype="<f8").tobytes(order="C")
        with open(os.path.join(out_dir, consts_bin), "wb") as f:
            f.write(raw)
        lines.append(
            f"const _consts: [{len(raw)}]u8 align(@alignOf(f64)) = "
            f'@embedFile("{consts_bin}").*;'
        )
        lines.append("pub const const_blob: []const f64 = std.mem.bytesAsSlice(f64, &_consts);")
    else:
        _remove_stale(out_dir, consts_bin)
        lines.append("pub const const_blob = [_]f64{};")
    if const_blob_f32:
        raw = np.asarray(const_blob_f32, dtype=np.float64).astype("<f4").tobytes(order="C")
        with open(os.path.join(out_dir, weights_bin), "wb") as f:
            f.write(raw)
        lines.append(
            f"const _weights: [{len(raw)}]u8 align(@alignOf(f32)) = "
            f'@embedFile("{weights_bin}").*;'
        )
        lines.append("pub const const_blob_f32: []const f32 = std.mem.bytesAsSlice(f32, &_weights);")
    else:
        _remove_stale(out_dir, weights_bin)
        lines.append("pub const const_blob_f32 = [_]f32{};")
    # Per-op f64 scalar/array tables, declared unconditionally (empty when the
    # graph has no node of that op). The VM only references a table from the
    # switch arm of its op, which is comptime-skipped when absent.
    for name, vals in tables.items():
        lines.append(f"pub const {name} = [_]f64{{" + _zig_floats(vals) + "};")
    lines.append("pub const output_offsets = [_]usize{" + ", ".join(str(o) for o in output_offsets) + "};")
    lines.append("pub const state_offsets = [_]usize{" + ", ".join(str(o) for o in state_offsets) + "};")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    manifest = _graph_manifest(cg, buf_len, len(const_blob), len(const_blob_f32), offsets, encodings, input_offsets)
    if provenance:
        manifest["provenance"] = provenance
    manifest_path = os.path.splitext(out_path)[0] + "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


# ─── node encoding (schema-driven) ───────────────────────────────────────────


def _encode_node(
    g: Graph,
    i: int,
    node: Node,
    f32_nodes: set[int],
    input_offsets: dict[str, int],
    cg: ComposedGraph,
    tables: dict[str, list[float]],
    const_blob: list[float],
    const_blob_f32: list[float],
) -> tuple[str, int]:
    """Encode one graph node as its Zig ``(vm_op, aux)`` pair.

    Port nodes (``const``/``input``/``output``) compute their own ``aux``
    (blob offset / input offset / port index). Every other op is looked up in
    the :mod:`shinro.codegen.vm_schema` registry and encoded generically:

    - scalar attrs → one ``<op>_<attr>`` table each, shared index in ``aux``;
    - flags → packed bits in ``aux`` (below the index);
    - array attrs → broadcast to the node shape, flat offset in ``aux``;
    - ``slot`` → structural selector (``attr:<name>`` / ``axis_code``).

    Args:
        g: The composed graph being serialized.
        i: Node index in ``g.nodes``.
        node: The node to encode.
        f32_nodes: Const node indices baked into ``const_blob_f32``.
        input_offsets: Input port name → offset in the packed input buffer.
        cg: The composed graph (for the ``outputs``/``state_outputs`` order).
        tables: Per-op f64 tables, mutated in place (pre-seeded with all names).
        const_blob: f64 constants blob, appended in place.
        const_blob_f32: f32 contraction-weights blob, appended in place.

    Returns:
        The ``(vm_op, aux)`` pair for the node.
    """
    if node.op == "const":
        vals = np.asarray(node.attrs["value"], dtype=np.float64).ravel()
        if i in f32_nodes:
            off = len(const_blob_f32)
            const_blob_f32.extend(vals.tolist())
            return "cst_f32", off
        off = len(const_blob)
        const_blob.extend(vals.tolist())
        return "cst", off
    if node.op == "input":
        return "inp", input_offsets[node.attrs["name"]]
    if node.op == "output":
        name = node.attrs["name"]
        if name in cg.outputs:
            return "out", cg.outputs.index(name)
        return "out", len(cg.outputs) + cg.state_outputs.index(name)

    spec = vm_schema.get_vm_op(node.op)
    vm_name = spec.vm_name or node.op

    if spec.array_attrs:
        # Array attrs (only clip's lo/hi) broadcast to the output shape; the
        # node's aux is the flat offset into the (identically-sized) tables.
        idx = len(tables.setdefault(f"{node.op}_{spec.array_attrs[0]}", []))
        for attr in spec.array_attrs:
            raw = np.asarray(node.attrs[attr], dtype=np.float64)
            try:
                vals = np.broadcast_to(raw, node.shape).ravel()
            except ValueError as exc:
                raise ValueError(
                    f"{node.op} node {i}: {attr!r} bound shape {raw.shape} cannot "
                    f"broadcast against shape {node.shape}"
                ) from exc
            tables.setdefault(f"{node.op}_{attr}", []).extend(vals.tolist())
        return vm_name, (idx << len(spec.flags)) | _flag_bits(node, spec)

    if spec.attrs:
        idx = len(tables.setdefault(f"{node.op}_{spec.attrs[0].name}", []))
        for a in spec.attrs:
            tables.setdefault(f"{node.op}_{a.name}", []).append(float(node.attrs.get(a.name, a.default)))
        # alpha/beta are baked into <op>_<attr> (index = aux >> nflags); the flag
        # bits ride in the low bits — the gemm transB pattern generalized.
        return vm_name, (idx << len(spec.flags)) | _flag_bits(node, spec)

    if spec.flags:
        return vm_name, _flag_bits(node, spec)

    if spec.slot == "none":
        return vm_name, 0
    if spec.slot == "axis_code":
        axis = node.attrs.get("axis")
        return vm_name, 0 if axis is None else int(axis) + 1
    if spec.slot.startswith("attr:"):
        return vm_name, int(node.attrs.get(spec.slot[len("attr:") :], 0))
    raise ValueError(f"unknown slot source {spec.slot!r} for op {node.op!r}")


def _flag_bits(node: Node, spec: vm_schema.VmOp) -> int:
    """Pack a schema's bool attrs into a bitfield (declaration order, bit 0 first)."""
    bits = 0
    for k, name in enumerate(spec.flags):
        if bool(node.attrs.get(name, False)):
            bits |= 1 << k
    return bits


def _op_enum_names() -> list[str]:
    """Derive the emitted Zig ``Op`` enum from the registry.

    Includes the structural port variants (``cst``/``cst_f32``/``inp``/``out``)
    and every registered op's VM name (renamed ops included). This is what makes
    adding an op turnkey: the enum can never drift from the registered set, and a
    missing ``lower.zig`` switch arm fails the Zig build with an exhaustive-switch
    error rather than silently mis-lowering.
    """
    names = {"cst_f32"}
    for op in OP_HANDLERS:
        names.add(vm_schema.vm_op_name(op))
    return ["cst", "cst_f32"] + sorted(n for n in names if n not in ("cst", "cst_f32"))


def _node_line(node: Node, vm_op: str, aux: int) -> str:
    """Render one node as a Zig ``Node`` struct literal."""
    rows, cols = _rows_cols(node.shape)
    inputs = "&.{" + ", ".join(str(x) for x in node.inputs) + "}"
    vec = str(len(node.shape) == 1).lower()
    return f".{{ .op = .{vm_op}, .inputs = {inputs}, .rows = {rows}, .cols = {cols}, .aux = {aux}, .vec = {vec} }}"


# ─── manifest ───────────────────────────────────────────────────────────────


def _graph_manifest(
    cg: ComposedGraph,
    buf_len: int,
    const_blob_len: int,
    const_blob_f32_len: int,
    offsets: list[int],
    encodings: list[tuple[str, int]],
    input_offsets: dict[str, int],
) -> dict:
    """Build the deterministic graph manifest for a composed graph.

    Describes what the compiled ``.so`` contains (the node table the VM
    executes): op histogram, the ordered node list (dual Python/Zig op names,
    wiring, shape, buffer offset, aux), the C-ABI port layout, buffer sizes,
    whether the graph contains a ``.solve_qp`` node, and, when present, the
    n_vars that node expects. No timestamps — the manifest is a pure function
    of the graph, so identical inputs produce identical bytes.

    Args:
        cg: The composed graph being serialized.
        buf_len: Total f64 buffer size for one step.
        const_blob_len: Number of f64s in the baked constants blob.
        const_blob_f32_len: Number of f32s in the baked contraction weights.
        offsets: Per-node buffer offset (node ``i`` owns ``rows*cols`` f64s at
            ``buf[offsets[i]..]``).
        encodings: Per-node ``(vm_op, aux)`` pairs from :func:`_encode_node`.
        input_offsets: Maps input port name → offset into the packed input
            buffer.

    Returns:
        A JSON-serializable dict (keys sorted for stable output).
    """
    g = cg.graph
    op_histogram: dict[str, int] = {}
    for node in g.nodes:
        op_histogram[node.op] = op_histogram.get(node.op, 0) + 1

    def _port(name: str) -> dict:
        for node in g.nodes:
            if node.op == "output" and node.attrs["name"] == name:
                return {"name": name, "shape": list(node.shape), "bytes": _size(node.shape) * _FLOAT_BYTES}
        raise KeyError(f"output port '{name}' not found in graph")

    solve_qp = None
    for node in g.nodes:
        if node.op == "solve_qp":
            solve_qp = {"expected_n_vars": _size(node.shape)}
            break

    nodes = []
    for i, node in enumerate(g.nodes):
        vm_op, aux = encodings[i]
        rows, cols = _rows_cols(node.shape)
        nodes.append(
            {
                "i": i,
                "op": node.op,
                "vm_op": vm_op,
                "inputs": list(node.inputs),
                "rows": rows,
                "cols": cols,
                "offset": offsets[i],
                "aux": aux,
            }
        )

    # Size metrics. Element counts above; byte figures here, because the three
    # sizes that matter to a deployment differ by orders of magnitude: the
    # C-ABI buffers the host packs per tick, the VM's stack buffer
    # (`buf: [buf_len]f64`), and the baked blobs. Recorded in the manifest so
    # every graph self-reports them (the audit trail already carried buf_len).
    input_bytes = sum(_input_size(g, n) for n in cg.inputs) * _FLOAT_BYTES
    output_bytes = sum(_output_size(g, n) for n in cg.outputs) * _FLOAT_BYTES
    state_bytes = sum(_output_size(g, n) for n in cg.state_outputs) * _FLOAT_BYTES
    # clip_blob_len counts one bound's elements (lo == hi in length); the VM
    # bakes both, so the byte figure doubles it.
    clip_blob_len = sum(_size(n.shape) for n in g.nodes if n.op == "clip")

    return {
        "float_type": "f64",
        "buf_len": buf_len,
        "buf_bytes": buf_len * _FLOAT_BYTES,
        "const_blob_len": const_blob_len,
        "const_blob_bytes": const_blob_len * _FLOAT_BYTES,
        "const_blob_f32_len": const_blob_f32_len,
        "const_blob_f32_bytes": const_blob_f32_len * _FLOAT32_BYTES,
        "clip_blob_len": clip_blob_len,
        "clip_blob_bytes": 2 * clip_blob_len * _FLOAT_BYTES,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "state_bytes": state_bytes,
        "has_solve_qp": solve_qp is not None,
        "nodes_total": len(g.nodes),
        "nodes": nodes,
        "ops": sorted(op_histogram),
        "op_histogram": op_histogram,
        "inputs": [
            {"name": n, "shape": list(_input_shape(g, n)), "bytes": _input_size(g, n) * _FLOAT_BYTES} for n in cg.inputs
        ],
        "outputs": [_port(n) for n in cg.outputs],
        "state_outputs": [_port(n) for n in cg.state_outputs],
        "solve_qp": solve_qp,
    }


def _input_shape(g: Graph, name: str) -> tuple[int, ...]:
    """Return the shape of the ``input`` port named ``name``."""
    for node in g.nodes:
        if node.op == "input" and node.attrs["name"] == name:
            return node.shape
    raise KeyError(f"input port '{name}' not found in graph")


# ─── helpers ───────────────────────────────────────────────────────────────


def _input_size(g: Graph, name: str) -> int:
    """Return the flat element count of the ``input`` port named ``name``.

    Args:
        g: The graph being serialized.
        name: The input port name.

    Returns:
        The number of f64s the port occupies in the packed input buffer.

    Raises:
        KeyError: If no ``input`` node with that name exists in the graph.
    """
    for node in g.nodes:
        if node.op == "input" and node.attrs["name"] == name:
            return _size(node.shape)
    raise KeyError(f"input port '{name}' not found in graph")


def _output_size(g: Graph, name: str) -> int:
    """Return the number of f64s an ``output`` port occupies in its C-ABI buffer.

    Args:
        g: The graph being serialized.
        name: The output port name.

    Returns:
        The element count the port takes in the packed ``outputs`` or
        ``state_out`` buffer.

    Raises:
        KeyError: If no output node with that name exists in the graph.
    """
    for node in g.nodes:
        if node.op == "output" and node.attrs["name"] == name:
            return _size(node.shape)
    raise KeyError(f"output port '{name}' not found in graph")


def _size(shape: tuple[int, ...]) -> int:
    """Return the total element count for a shape (empty shape counts as 1)."""
    if not shape:
        return 1
    out = 1
    for d in shape:
        out *= d
    return out


def _rows_cols(shape: tuple[int, ...]) -> tuple[int, int]:
    """Map a graph shape to Zig VM ``rows``/``cols`` (2D → as-is, 1D → (n, 1), else (1, 1))."""
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 1:
        return shape[0], 1
    return 1, 1


def _zig_float(x: float) -> str:
    """Format an f64 as a Zig hex float literal (bit-exact round-trip).

    ``float.hex()`` emits the exact binary representation numpy stored, so
    the "fixed at compile time" property holds across the language boundary.
    """
    return float(x).hex()


def _zig_floats(vals: list[float]) -> str:
    """Format a list of f64s as a comma-separated Zig hex-float literal."""
    return ", ".join(_zig_float(x) for x in vals)


def _remove_stale(directory: str, name: str) -> None:
    """Delete a sidecar left by a previous lowering of a different graph."""
    path = os.path.join(directory, name)
    if os.path.exists(path):
        os.remove(path)


def _is_f32_weight(node_index: int, node: Node, consumers: dict[int, list[tuple[int, int]]], g: Graph) -> bool:
    """Whether a constant can be baked as f32 (the contraction-weight fast path).

    True when every element round-trips through f32 exactly — the case for a
    model's float32 initializers, which the importer only widened — and every
    consumer is a contraction reading it as a weight operand: a ``gemm``'s B
    (input position 1) or a recurrent cell's W/R (positions 1/2). That bounds
    the mixed-dtype path to those VM arms, so no other arm sees a ``cst_f32``
    node. A cell's bias B (position 3) is left on the f64 blob — it is tiny.
    """
    uses = consumers.get(node_index)
    if not uses:
        return False
    for ci, pos in uses:
        cop = g.nodes[ci].op
        if cop == "gemm" and pos == 1:
            continue
        if cop in ("lstm", "gru", "rnn") and pos in (1, 2):
            continue
        return False
    vals = np.asarray(node.attrs["value"], dtype=np.float64).ravel()
    return bool(np.array_equal(vals.astype(np.float32).astype(np.float64), vals))
