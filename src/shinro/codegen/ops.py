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
PID's ``where`` and ``copy`` are included to support the swap test. The
deterministic-policy ops (tanh, relu, div, exp, argmax, one_hot, slice)
cover NN controllers run in deterministic mode (no sampling), so a learned
policy's forward pass traces and lowers like any classical controller.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import osqp
from scipy import sparse

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


@register_op("gemm")
def _gemm(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Fused dense layer ``Y = alpha * (A @ B') + beta * C``.

    ``alpha``/``beta`` scale the product and the bias; ``transB`` reads the
    weight ``B`` as stored ``(n, k)`` (torch ``nn.Linear`` layout) and
    contracts its rows, i.e. the same values a lazy ``B.T`` would produce with
    no data movement. This is the numpy oracle for the Zig ``linalg.gemm``
    kernel — the broadcast on ``C`` (scalar / ``(n,)`` / full) comes free from
    numpy and is mirrored by the kernel's ``c_len`` dispatch.
    """
    a = values[node.inputs[0]]
    b = values[node.inputs[1]]
    c = values[node.inputs[2]]
    if node.attrs.get("transB", False):
        b = b.T
    return node.attrs.get("alpha", 1.0) * (a @ b) + node.attrs.get("beta", 1.0) * c


@register_op("add")
def _add(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]] + values[node.inputs[1]]


@register_op("sub")
def _sub(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]] - values[node.inputs[1]]


@register_op("mul")
def _mul(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]] * values[node.inputs[1]]


@register_op("ne")
def _ne(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Elementwise inequality as 1.0/0.0 floats — the graph's boolean repr."""
    return (values[node.inputs[0]] != values[node.inputs[1]]).astype(np.float64)


@register_op("lt")
def _lt(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Elementwise ``a < b`` as 1.0/0.0 floats — the ordered sibling of ``ne``.

    ``ne`` is the graph's only other boolean; it cannot express an ordering, so
    threshold/guard conditions (e.g. SMC's near-zero ``c^T g`` guard) need this
    op. Consumed by ``where``/``any`` exactly like ``ne``.
    """
    return (values[node.inputs[0]] < values[node.inputs[1]]).astype(np.float64)


@register_op("neg")
def _neg(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return -values[node.inputs[0]]


@register_op("transpose")
def _transpose(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]].T


@register_op("inv")
def _inv(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.linalg.inv(values[node.inputs[0]])


@register_op("solve_qp")
def _solve_qp(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Solve ``min ½ uᵀ H u + qᵀ u`` s.t. ``lb ≤ A u ≤ ub``.

    ``q`` is the node's sole input (``q = Fᵀ x₀`` for MPC); H/A/lb/ub are
    baked into the node's attrs at trace time (the Zig VM instead uses the
    codegen static ``solver`` global, so this handler must match its
    eps=1e-6). Returns the full solution; MPC slices ``u[:m]`` downstream.
    """
    q = np.asarray(values[node.inputs[0]]).ravel()
    lb = np.asarray(node.attrs["lb"]).ravel()
    ub = np.asarray(node.attrs["ub"]).ravel()
    prob = osqp.OSQP()
    prob.setup(
        sparse.csc_matrix(np.asarray(node.attrs["H"], dtype=np.float64)),
        q,
        node.attrs["A"],
        lb,
        ub,
        warm_starting=True,
        verbose=False,
    )
    prob.update_settings(eps_abs=1e-6, eps_rel=1e-6)
    return prob.solve().x


# ─── shape / selection glue ops (used by compose.py and PID) ──────────────


@register_op("reshape")
def _reshape(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    target = node.attrs.get("target_shape")
    if target is None:
        target = node.attrs.get("shape")
    return values[node.inputs[0]].reshape(target)


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


@register_op("stack")
def _stack(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    # stack([a, b, ...]) along a new leading axis (numpy's default axis=0).
    # All inputs share the same shape; output is (len(inputs),) + that shape.
    return np.stack([values[n] for n in node.inputs])


# ─── deterministic-policy ops (NN controllers in deterministic mode) ──────


@register_op("tanh")
def _tanh(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.tanh(values[node.inputs[0]])


@register_op("sigmoid")
def _sigmoid(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Logistic sigmoid, matching ONNX ``Sigmoid``: 1 / (1 + exp(-x))."""
    x = values[node.inputs[0]]
    return 1.0 / (1.0 + np.exp(-x))


@register_op("softmax")
def _softmax(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Softmax over the last axis (numpy ``axis=-1``), shape-preserving.

    Numerically stable: ``e = exp(x - x.max(axis=-1, keepdims=True));
    e / e.sum(axis=-1, keepdims=True)``. A 1-D input is a single row, which
    mirrors the VM's ``rows=n, cols=1, vec=true`` -> ``softmax_rows(1, n)``
    normalization. The Zig mirror is ``linalg.softmax_rows``.
    """
    x = values[node.inputs[0]]
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


@register_op("gelu")
def _gelu(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """GELU, tanh approximation: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))).

    The exact erf form is deliberately not implemented — this is the tanh
    approximation (GPT ``gelu_new``), which deviates from exact GELU by at most
    ~4.7e-4 absolute. The Zig mirror is ``linalg.gelu`` and must use the same
    expression so kernel/oracle agree bit-for-bit.
    """
    x = values[node.inputs[0]]
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


@register_op("elu")
def _elu(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """ELU: ``x if x > 0 else alpha * (exp(x) - 1)``; ``alpha`` defaults to 1.0."""
    x = values[node.inputs[0]]
    alpha = float(node.attrs.get("alpha", 1.0))
    return np.where(x > 0.0, x, alpha * (np.exp(x) - 1.0))


@register_op("leaky_relu")
def _leaky_relu(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """LeakyReLU: ``x if x >= 0 else alpha * x``; ``alpha`` defaults to 0.01.

    The linear sibling of ``elu`` — no exp, so the VM computes it inline with a
    select (there is no ``linalg`` kernel). At ``x == 0`` both branches give 0,
    so the ``>=`` / ``>`` boundary cannot diverge.
    """
    x = values[node.inputs[0]]
    alpha = float(node.attrs.get("alpha", 0.01))
    return np.where(x >= 0.0, x, alpha * x)


@register_op("layernorm")
def _layernorm(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Layer normalization over the last axis, the torch/ONNX canonical form.

    ``y = (x - mean) * (1 / sqrt(var + eps)) * scale + bias`` with the
    **biased** variance (``mean`` and ``var`` reduce the last axis, so a 1-D
    input is a single normalized row). ``scale``/``bias`` are the ONNX
    ``Scale``/``B`` operands, one entry per feature; ``eps`` is baked per node
    (default ONNX's 1e-5) and rides the same f64 table as elu's ``alpha``.

    The Zig mirror is ``linalg.layernorm_rows`` and evaluates the identical
    expression in the identical order (reciprocal-then-multiply), so the three
    engines agree to the oracle tolerance rather than merely mathematically.
    """
    x = values[node.inputs[0]]
    scale = values[node.inputs[1]]
    bias = values[node.inputs[2]]
    eps = float(node.attrs.get("eps", 1e-5))
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) * (1.0 / np.sqrt(var + eps)) * scale + bias


# ─── recurrent cells (one fused step; ONNX LSTM/GRU/RNN semantics) ────────
#
# Each handler is the numpy mirror of one runtime/linalg.zig kernel, so the
# interpreter, the ONNX importer's eager evaluation, and the compiled VM all
# agree by construction. Shapes are the ONNX contract: W is (ngates*H, I) and
# R is (ngates*H, H), both read as ``·ᵀ``; B is ``Wb ‖ Rb`` with the input
# biases first. Only the current timestep is computed — time is the host
# feeding h_next/c_next back next tick.


def _logistic(x: np.ndarray) -> np.ndarray:
    """Elementwise logistic sigmoid (a local helper — the ``sigmoid`` op's
    handler is a node-taking callable, not this map)."""
    return 1.0 / (1.0 + np.exp(-x))


@register_op("lstm")
def _lstm(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Fused LSTM cell: returns ``[h_next(H) ‖ c_next(H)]``.

    ONNX gate order is ``i, o, f, c`` with Sigmoid on i/o/f and Tanh on the
    cell candidate and the hidden output. The concatenated return is the
    ML-Agents ``recurrent_in``/``recurrent_out`` layout (h first). The Zig
    mirror is ``linalg.lstm_cell``.
    """
    x = values[node.inputs[0]].ravel()
    w = values[node.inputs[1]]
    r = values[node.inputs[2]]
    b = values[node.inputs[3]].ravel()
    h = values[node.inputs[4]].ravel()
    c = values[node.inputs[5]].ravel()
    H = w.shape[0] // 4
    z = x @ w.T + h @ r.T + b[: 4 * H] + b[4 * H :]
    i_g = _logistic(z[:H])
    o_g = _logistic(z[H : 2 * H])
    f_g = _logistic(z[2 * H : 3 * H])
    g_g = np.tanh(z[3 * H : 4 * H])
    c_next = f_g * c + i_g * g_g
    h_next = o_g * np.tanh(c_next)
    return np.concatenate([h_next, c_next])


@register_op("gru")
def _gru(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Fused GRU cell: returns ``h_next(H)``.

    ONNX gate order is ``z, r, h`` with Sigmoid on z/r and Tanh on the
    candidate. ``linear_before_reset`` moves the reset gate *across* the
    recurrent matmul:

    - ``lbr`` true  → ``g(X·Whᵀ + r ⊙ (H·Rhᵀ + Rbh) + Wbh)``
    - ``lbr`` false → ``g(X·Whᵀ + (r ⊙ H)·Rhᵀ + Rbh + Wbh)``

    Getting this backwards silently diverges ~1e-1, so it is an explicit node
    attribute (baked into the VM node's ``aux``). The Zig mirror is
    ``linalg.gru_cell``.
    """
    x = values[node.inputs[0]].ravel()
    w = values[node.inputs[1]]
    r = values[node.inputs[2]]
    b = values[node.inputs[3]].ravel()
    h = values[node.inputs[4]].ravel()
    H = w.shape[0] // 3
    gx = x @ w.T
    gh = h @ r.T
    z = _logistic(gx[:H] + gh[:H] + b[:H] + b[3 * H : 4 * H])
    r_g = _logistic(gx[H : 2 * H] + gh[H : 2 * H] + b[H : 2 * H] + b[4 * H : 5 * H])
    if node.attrs.get("linear_before_reset", False):
        n = np.tanh(gx[2 * H : 3 * H] + r_g * (gh[2 * H : 3 * H] + b[5 * H : 6 * H]) + b[2 * H : 3 * H])
    else:
        n = np.tanh(gx[2 * H : 3 * H] + (r_g * h) @ r[2 * H :].T + b[5 * H : 6 * H] + b[2 * H : 3 * H])
    return (1.0 - z) * n + z * h


@register_op("rnn")
def _rnn(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Vanilla Elman RNN cell: ``h_next = tanh(x·Wᵀ + h·Rᵀ + Wb + Rb)``.

    The Zig mirror is ``linalg.rnn_cell``.
    """
    x = values[node.inputs[0]].ravel()
    w = values[node.inputs[1]]
    r = values[node.inputs[2]]
    b = values[node.inputs[3]].ravel()
    h = values[node.inputs[4]].ravel()
    H = w.shape[0]
    return np.tanh(x @ w.T + h @ r.T + b[:H] + b[H:])


@register_op("concat")
def _concat(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate along an existing axis (ONNX ``Concat``).

    Negative axes are resolved against the assembled rank, matching numpy, so a
    ``-1`` axis on 1-D/2-D operands normalizes to 0/1 — the only two the VM can
    represent.
    """
    axis = int(node.attrs.get("axis", 0))
    return np.concatenate([values[i] for i in node.inputs], axis=axis)


@register_op("gather")
def _gather(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Index-select along one axis (ONNX ``Gather``).

    ``idx`` is a flat index vector; negatives count from the end (numpy/ONNX
    semantics), which is why the drone policies' ``[-1]`` last-step selection
    works. The index axis is replaced by the index length.
    """
    axis = int(node.attrs.get("axis", 0))
    idx = np.asarray(values[node.inputs[1]]).astype(np.int64)
    return np.take(values[node.inputs[0]], idx, axis=axis)


@register_op("sin")
def _sin(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.sin(values[node.inputs[0]])


@register_op("cos")
def _cos(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.cos(values[node.inputs[0]])


@register_op("relu")
def _relu(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.maximum(values[node.inputs[0]], 0.0)


@register_op("div")
def _div(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return values[node.inputs[0]] / values[node.inputs[1]]


@register_op("exp")
def _exp(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.exp(values[node.inputs[0]])


@register_op("sqrt")
def _sqrt(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Elementwise square root (ONNX ``Sqrt``); a negative operand is NaN."""
    return np.sqrt(values[node.inputs[0]])


@register_op("log")
def _log(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Elementwise natural logarithm (ONNX ``Log``, the C ``log``)."""
    return np.log(values[node.inputs[0]])


@register_op("mod")
def _mod(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Elementwise modulo (ONNX ``Mod``), numpy semantics.

    ``fmod=False`` (ONNX default, integer-modulo semantics) is ``np.mod`` — the
    result takes the sign of the divisor; ``fmod=True`` is ``np.fmod`` — the
    sign of the dividend (C ``fmod``). The VM's ``mod`` arm selects ``@mod`` /
    ``@rem`` on the same bit. Note ONNX constrains ``fmod=0`` to integer types;
    onnxruntime rejects a float ``fmod=0``, while this f64 machine extends
    Python's ``%`` to floats and the ONNX Python reference instead falls back to
    ``np.fmod`` (+ ``nan_to_num``) — matched here to onnxruntime, not the
    reference, since onnxruntime is the deployment target.
    """
    a = values[node.inputs[0]]
    b = values[node.inputs[1]]
    return np.fmod(a, b) if node.attrs.get("fmod", False) else np.mod(a, b)


@register_op("abs")
def _abs(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.abs(values[node.inputs[0]])


@register_op("sign")
def _sign(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Sign function matching ``np.sign``: -1 / 0 / +1 (0 maps to 0)."""
    return np.sign(values[node.inputs[0]])


@register_op("pow")
def _pow(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.power(values[node.inputs[0]], values[node.inputs[1]])


@register_op("argmax")
def _argmax(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(np.argmax(values[node.inputs[0]]))


@register_op("min")
def _min(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Minimum reduction, numpy ``min`` semantics.

    ``axis`` is a node attr: ``None`` collapses to a 0-d scalar, ``0``/``1``
    reduce that axis of a 2-D input. Introduced for MPPI's softmax shift
    (``beta = min(costs)``, the overflow guard); the axis support keeps the op
    a complete reduction rather than a one-off. The Zig mirror is
    ``linalg.min_all`` / ``min_axis0`` / ``min_axis1``, dispatched on the
    node's ``aux`` (0 = None, 1 = axis 0, 2 = axis 1).
    """
    return np.min(values[node.inputs[0]], axis=node.attrs.get("axis"))


@register_op("one_hot")
def _one_hot(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    # one_hot(x, depth): x is a scalar index → one-hot row vector.
    depth = node.attrs["depth"]
    idx = int(values[node.inputs[0]].item())
    out = np.zeros(depth, dtype=np.float64)
    out[idx] = 1.0
    return out


@register_op("slice")
def _slice(node: Node, values: dict[int, np.ndarray], inputs: dict[str, np.ndarray]) -> np.ndarray:
    # slice(x, start, stop): x[start:stop] along the first axis.
    start = node.attrs["start"]
    stop = node.attrs["stop"]
    return values[node.inputs[0]][start:stop]


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
        f"op '{name}' is not registered. Add a handler in shinro.codegen.ops via @register_op('{name}'). Available ops: {available_ops()}"
    )


def __all__() -> list[str]:  # pragma: no cover - introspection helper
    return ["OP_HANDLERS", "register_op", "available_ops", "has_op", "missing_op_error"]


# silence unused-import linters for the Any re-export path
_ = Any
