"""Generate the committed toy ONNX policy fixture.

Deterministic — fixed hand-written weights, no RNG — so re-running reproduces
the same graph and the fixture is reviewable by reading this file. The model is
a 3 -> 4 -> 2 tanh MLP::

    h = tanh(W1 x + b1)
    action = W2 h + b2

with ``W1 = [[1,0,0],[0,1,0],[0,0,1],[1,1,0]]``, ``b1 = 0`` and
``W2 = [[1,0,0,0],[0,1,0,0]]``, ``b2 = [0.5, -0.5]``. Only the first two hidden
units feed the output, but all four are kept so the Gemm dimensions are
non-trivial (a real transpose, not a degenerate 1xN). The closed form is::

    action = [tanh(x0) + 0.5, tanh(x1) - 0.5]

It is used as the policy fixture for the ONNX importer/adapter unit tests and
for the committed policy-only compile scenario
(``tests/fixtures/configs/scenarios/toy_mlp_policy.toml``).

Usage::

    python3 scripts/gen_toy_onnx.py                         # the committed fixture
    python3 scripts/gen_toy_onnx.py --out path/to/policy.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUT = "tests/fixtures/models/toy_mlp.onnx"


def build_toy_mlp():
    """Build the toy 3 -> 4 -> 2 tanh MLP as an ``onnx.ModelProto``."""
    import numpy as np
    from onnx import TensorProto, helper

    w1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
    b1 = np.zeros(4, dtype=np.float32)
    w2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    b2 = np.array([0.5, -0.5], dtype=np.float32)

    def init(name, array):
        a = np.asarray(array, dtype=np.float32)
        return helper.make_tensor(name, TensorProto.FLOAT, a.shape, a.flatten().tolist())

    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, 3])
    action = helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, 2])
    nodes = [
        helper.make_node("Gemm", ["obs", "w1", "b1"], ["h"], transB=1),
        helper.make_node("Tanh", ["h"], ["a"]),
        helper.make_node("Gemm", ["a", "w2", "b2"], ["action"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "toy_mlp",
        [obs],
        [action],
        [init("w1", w1), init("b1", b1), init("w2", w2), init("b2", b2)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    import onnx

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_toy_mlp(), str(out))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
