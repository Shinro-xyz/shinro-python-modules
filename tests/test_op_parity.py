"""Op parity: each registered handler must equal the eager `NumpyBackend` op.

The interpreter oracle (`interpret`) dispatches through `OP_HANDLERS`, and the
tracer's output shape is now derived by *running* those handlers on zeros — so
the whole compile path leans on the handlers being the reference implementation
of each op. This suite pins that directly: for every op `NumpyBackend` also
implements, the handler and the eager backend must agree on the same inputs.

Ops that the eager backends don't implement (``gemm``/``layernorm``/``lstm``/
``gru``/``rnn``/``mod``/``log``/``leaky_relu``, plus the operator-emitted ops)
are structurally out of scope here — they are trace→lower-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shinro.codegen.ops import OP_HANDLERS
from shinro.codegen.tracing import Node
from shinro.utils.array_backend import NumpyBackend

RNG = np.random.default_rng(0)


def _r(*shape: int) -> np.ndarray:
    return RNG.normal(size=shape)


def _pos(*shape: int) -> np.ndarray:
    return RNG.uniform(0.5, 2.0, size=shape)


def _spd(n: int) -> np.ndarray:
    m = RNG.normal(size=(n, n))
    return m @ m.T + n * np.eye(n)


def _handler(op: str, values: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    node = Node(op=op, inputs=list(range(len(values))), shape=np.asarray(values[0]).shape, attrs=dict(attrs))
    vals = {i: np.asarray(v, dtype=np.float64) for i, v in enumerate(values)}
    return np.asarray(OP_HANDLERS[op](node, vals, {}))


#: (op, values, attrs, eager call) — eager call takes (NumpyBackend, values).
CASES: list[tuple[str, list[np.ndarray], dict[str, Any], Any]] = [
    ("abs", [_r(4)], {}, lambda nb, v: nb.abs(v[0])),
    ("any", [np.array([0.0, 1.0, 0.0, 2.0])], {}, lambda nb, v: nb.any(v[0])),
    ("argmax", [_r(5)], {}, lambda nb, v: nb.argmax(v[0])),
    ("cholesky", [_spd(3)], {}, lambda nb, v: nb.cholesky(v[0])),
    ("clip", [_r(4)], {"lo": np.array(-0.5), "hi": np.array(0.5)}, lambda nb, v: nb.clip(v[0], -0.5, 0.5)),
    ("copy", [_r(3)], {}, lambda nb, v: nb.copy(v[0])),
    ("cos", [_r(3)], {}, lambda nb, v: nb.cos(v[0])),
    ("div", [_r(3), _pos(3)], {}, lambda nb, v: nb.div(v[0], v[1])),
    ("elu", [_r(4)], {"alpha": 1.0}, lambda nb, v: nb.elu(v[0], alpha=1.0)),
    ("exp", [_r(3)], {}, lambda nb, v: nb.exp(v[0])),
    ("gelu", [_r(4)], {}, lambda nb, v: nb.gelu(v[0])),
    ("inv", [_spd(3)], {}, lambda nb, v: nb.inv(v[0])),
    ("min", [_r(5)], {"axis": None}, lambda nb, v: nb.min(v[0], None)),
    ("min", [_r(3, 4)], {"axis": 0}, lambda nb, v: nb.min(v[0], 0)),
    ("one_hot", [np.array(2.0)], {"depth": 4}, lambda nb, v: nb.one_hot(v[0], 4)),
    ("relu", [_r(4)], {}, lambda nb, v: nb.relu(v[0])),
    ("sigmoid", [_r(4)], {}, lambda nb, v: nb.sigmoid(v[0])),
    ("sign", [_r(4)], {}, lambda nb, v: nb.sign(v[0])),
    ("sin", [_r(4)], {}, lambda nb, v: nb.sin(v[0])),
    ("softmax", [_r(5)], {}, lambda nb, v: nb.softmax(v[0])),
    ("sqrt", [_pos(4)], {}, lambda nb, v: nb.sqrt(v[0])),
    ("stack", [_r(2), _r(2)], {}, lambda nb, v: nb.stack(v)),
    ("concat", [_r(2, 3), _r(2, 2)], {"axis": 1}, lambda nb, v: nb.concat(v, axis=1)),
    ("tanh", [_r(4)], {}, lambda nb, v: nb.tanh(v[0])),
    (
        "where",
        [np.array([1.0, 0.0, 1.0, 0.0]), _r(4), _r(4)],
        {},
        lambda nb, v: nb.where(v[0], v[1], v[2]),
    ),
]


@pytest.mark.parametrize("op,values,attrs,eager", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_handler_matches_eager_backend(op, values, attrs, eager):
    nb = NumpyBackend()
    got = _handler(op, values, attrs)
    expected = np.asarray(eager(nb, [np.asarray(x, dtype=np.float64) for x in values]), dtype=np.float64)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_gemm_is_trace_only():
    # sanity: an op the eager backends don't implement has no NumpyBackend method,
    # so it is correctly out of scope for this parity suite.
    assert not hasattr(NumpyBackend(), "gemm")
    assert "gemm" in OP_HANDLERS
