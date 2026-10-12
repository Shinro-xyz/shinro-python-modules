"""The op -> Zig-VM lowering schema (single source of truth).

Adding an op to the compiled VM previously required three kinds of Python
edits in :mod:`shinro.codegen.lower_zig`: appending to a hand-written ``Op``
enum, threading a new f64-scalar table (``gemm_alpha``, ``layernorm_eps``, ...)
through three call sites plus a packing dict, and adding a branch to
``_node_vm_info``. This module replaces all of that with **one declarative
entry per op**, attached to the same ``@register_op`` decorator that carries
the numpy correctness oracle:

.. code-block:: python

    @register_op("gemm", vm=VmOp(attrs=(AttrSpec("alpha", 1.0),
                                       AttrSpec("beta", 1.0)),
                                flags=("transB",)))
    def _gemm(node, values, inputs): ...

The serializer (:func:`shinro.codegen.lower_zig.lower_zig`) is now fully
generic: it derives the Zig ``Op`` enum, the per-attr f64 tables, the packed
``aux`` word, and the manifest from this registry. Every op declares a schema —
the ~35 identity ops declare ``vm=VmOp()`` (the op's own name, no attrs, no
flags), so nothing lowers by silent default and the schema set can never drift
from the handler set.

Node encoding produced from a schema
------------------------------------
The Zig ``Node`` record keeps its existing shape
(``{op, inputs, rows, cols, aux, vec}``) so the VM, its fixtures, and the
C-ABI do not change. ``aux`` is packed per op as:

- **scalar attrs** — each declared ``AttrSpec`` becomes a table
  ``<op>_<attr>``; the node's ``aux`` high bits hold its index into those
  tables (all attrs of a node share one index). Missing attrs fall back to the
  ``AttrSpec.default``.
- **flags** — each declared bool becomes a bit in ``aux`` (declaration order,
  bit 0 first): ``aux = (index << len(flags)) | flagbits`` when the op also has
  attrs, else ``aux = flagbits``.
- **array attrs** — declared array operands (only ``clip``'s ``lo``/``hi``)
  become tables broadcast to the node's shape, with ``aux`` holding the flat
  offset (as the scalar-attr index would).
- **slot** — for the remaining structural/selector cases the schema names the
  source instead of a table: ``"attr:<name>"`` reads an integer attr (e.g.
  ``slice``'s ``start``, ``gather``/``concat``'s ``axis``), and
  ``"axis_code"`` maps ``min``'s ``axis`` (``None``/0/1) to the VM's ``0/1/2``
  selector.

``const``/``input``/``output`` are *port* nodes, not compute ops: they declare
only their ``cst``/``inp``/``out`` rename here, while their ``aux`` (blob offset /
input offset / port index) is plumbing the lowerer computes from the graph's
port lists.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class AttrSpec:
    """One f64 scalar attribute baked into a per-op table.

    Args:
        name: The ``node.attrs`` key.
        default: Value used when a node omits the attr (matches the handler's
            ``node.attrs.get(name, default)``).
    """

    name: str
    default: float = 0.0


@dataclass(frozen=True)
class VmOp:
    """Lowering schema for one op.

    Args:
        vm_name: Zig ``Op`` enum variant name. ``None`` means "same as the
            Python op name"; set it for the reserved/renamed ops
            (``where`` -> ``where_op``).
        attrs: f64 scalar attributes, in the order the VM arm reads them
            (one table ``<op>_<attr>`` each, shared index in ``aux``).
        flags: bool attributes packed into ``aux`` bits, declaration order.
        array_attrs: array attributes broadcast to the node shape (the
            ``clip`` bounds); ``aux`` holds the flat table offset.
        slot: ``"none"`` (no slot), ``"attr:<name>"`` (integer attr read
            directly as the slot), or ``"axis_code"`` (``None``/0/1 ->
            ``0``/``1``/``2``).
    """

    vm_name: str | None = None
    attrs: tuple[AttrSpec, ...] = ()
    flags: tuple[str, ...] = ()
    array_attrs: tuple[str, ...] = ()
    slot: str = "none"


#: python op name -> schema, populated by ``register_op(..., vm=...)``.
_REGISTRY: dict[str, VmOp] = {}


def register_vm_op(op: str, spec: VmOp) -> None:
    """Register the lowering schema for an op (called by ``register_op``)."""
    _REGISTRY[op] = spec


def get_vm_op(op: str) -> VmOp:
    """Return the schema for ``op``.

    Raises:
        KeyError: If ``op`` was registered without a schema — impossible through
            :func:`shinro.codegen.ops.register_op`, which requires ``vm=``; a
            direct call signals a programming error rather than silently
            defaulting an op to lower as itself.
    """
    try:
        return _REGISTRY[op]
    except KeyError:
        raise KeyError(
            f"op {op!r} has no VmOp lowering schema — declare vm=VmOp(...) in its @register_op"
        ) from None


def has_vm_op(op: str) -> bool:
    """Whether a schema was registered for ``op``."""
    return op in _REGISTRY


def missing_schemas(ops: Iterable[str]) -> list[str]:
    """Return the subset of ``ops`` with no declared :class:`VmOp` schema.

    Callers (the serializer and its tests) use this to fail loudly if a handler
    was registered without lowering metadata, rather than silently emitting an
    op the VM could mis-lower.
    """
    return [op for op in ops if op not in _REGISTRY]


def unregister_vm_op(op: str) -> None:
    """Remove a schema entry (test hygiene; mirrors ``OP_HANDLERS.pop``)."""
    _REGISTRY.pop(op, None)


def vm_op_name(op: str) -> str:
    """The Zig ``Op`` variant for ``op`` (schema name or the op itself)."""
    spec = _REGISTRY.get(op)
    if spec is not None and spec.vm_name is not None:
        return spec.vm_name
    return op


def table_names() -> list[str]:
    """All per-op f64 tables, in registration order.

    The serializer seeds these so a table is always declared (empty when the
    graph has no such node), matching the previous hand-written emitter's
    behaviour.
    """
    names: list[str] = []
    for op, spec in _REGISTRY.items():
        for attr in spec.array_attrs:
            names.append(f"{op}_{attr}")
        for attr in spec.attrs:
            names.append(f"{op}_{attr.name}")
    return names
