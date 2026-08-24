"""Op-handler registry for the graph interpreter.

This is the data-driven "switch" over op kinds. The interpreter
(:mod:`shinro.codegen.interpreter`) is a 5-line loop that dispatches each
node through ``OP_HANDLERS[node.op]``. Adding support for a new op is a
one-decorator affair — no interpreter edit, no if/elif ladder.

Each handler has the signature::

    handler(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray])
        -> np.ndarray

where ``values`` maps node ids to already-computed numpy arrays and
``inputs`` maps the graph's named input ports to their fed values. The
handler returns the numpy array for its node, which the interpreter stores
back into ``values``.

The initial op set covers what ``KalmanFilter.estimate`` and
``LQR.compute`` use (matmul, add, sub, mul, transpose, inv, const, input,
output) plus the glue ops the composition pass inserts (reshape, clip, neg).
PID's ``where`` and ``copy`` are included to support the swap test.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from shinro.codegen.tracing import Node

OpHandler = Callable[[Node, dict[int, np.ndarray], dict[str, np.ndarray]], np.ndarray]

OP_HANDLERS: dict[str, OpHandler] = {}


def register_op(name: str) -> Callable[[OpHandler], OpHandler]:
    """Register a handler for an op name. The data-driven switch."""

    def decorator(fn: OpHandler) -> OpHandler:
        OP_HANDLERS[name] = fn
        return fn

    return decorator


# ─── graph-structure ops ──────────────────────────────────────────────────


@register_op("const")
def _const(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return node.attrs["value"]


@register_op("input")
def _input(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    name = node.attrs["name"]
    if name not in inputs:
        raise KeyError(f"graph input '{name}' not provided")
    return inputs[name]


@register_op("output")
def _output(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    # output is a marker; the interpreter collects these by name separately.
    return values[node.inputs[0]]


# ─── linear-algebra ops ───────────────────────────────────────────────────


@register_op("matmul")
def _matmul(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]] @ values[node.inputs[1]]


@register_op("add")
def _add(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]] + values[node.inputs[1]]


@register_op("sub")
def _sub(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]] - values[node.inputs[1]]


@register_op("mul")
def _mul(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]] * values[node.inputs[1]]


@register_op("neg")
def _neg(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return -values[node.inputs[0]]


@register_op("transpose")
def _transpose(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]].T


@register_op("inv")
def _inv(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.linalg.inv(values[node.inputs[0]])


# ─── shape / selection glue ops (used by compose.py and PID) ──────────────


@register_op("reshape")
def _reshape(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]].reshape(node.attrs["shape"])


@register_op("clip")
def _clip(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.clip(values[node.inputs[0]], node.attrs["lo"], node.attrs["hi"])


@register_op("where")
def _where(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    cond = values[node.inputs[0]]
    a = values[node.inputs[1]]
    b = values[node.inputs[2]]
    return np.where(cond, a, b)


@register_op("copy")
def _copy(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]].copy()


@register_op("any")
def _any(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(np.any(values[node.inputs[0]]))


# ─── helpers exported for trace_backend.py ────────────────────────────────


def available_ops() -> list[str]:
    """Return the sorted list of registered op names. Useful for diagnostics."""
    return sorted(OP_HANDLERS.keys())


def has_op(name: str) -> bool:
    """Return True if an op handler is registered for ``name``."""
    return name in OP_HANDLERS


def missing_op_error(name: str) -> NotImplementedError:
    """Build the standard error for an unimplemented op.

    The message tells the caller exactly which op to register, so adding a
    new component that uses a new backend method produces an actionable
    signal rather than a silent failure.
    """
    return NotImplementedError(
        f"op '{name}' is not registered. Add a handler in shinro.codegen.ops via "
        f"@register_op('{name}'). Available ops: {available_ops()}"
    )


def __all__() -> list[str]:  # pragma: no cover - introspection helper
    return ["OP_HANDLERS", "register_op", "available_ops", "has_op", "missing_op_error"]


# silence unused-import linters for the Any re-export path
_ = Any