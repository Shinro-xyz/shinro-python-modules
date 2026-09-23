"""Generate the committed toy ONNX *recurrent* policy fixture.

Deterministic — fixed arithmetic weights, no RNG — so re-running reproduces the
same graph and the fixture is reviewable by reading this file. The model is one
LSTM step with an explicit state interface::

    forward(obs(1, 3), h_in(1, 1, 4), c_in(1, 1, 4))
        -> action(1, 2), h_out(1, 1, 4), c_out(1, 1, 4)

The cell follows ONNX ``LSTM`` semantics (gate order ``i, o, f, c``, with
Sigmoid on i/o/f and Tanh on the candidate and the output), and the head is
``action = [h0, h1]`` — the first two hidden units. Weights are ``arange``
patterns so every value is visible here rather than hidden behind a seed::

    W  = (arange(4H*I) - 6) / 10      # (4H, I)   input weights
    R  = (arange(4H*H) - 8) / 20      # (4H, H)   recurrent weights
    B  = 0                            # (8H,)     Wb ‖ Rb
    head = eye(H, ACT)                # (H, ACT)  first ACT hidden units

It is the policy fixture for the recurrent ONNX importer/adapter tests and for
the committed recurrent policy-only compile scenario
(``tests/fixtures/configs/scenarios/toy_recurrent_policy.toml``). The graph
exposes the cell's state as ports, which is what the deployment path needs: the
host feeds ``state_h_0`` / ``state_c_0`` and reads the new state back.

Usage::

    python3 scripts/gen_toy_recurrent_onnx.py                    # the committed fixture
    python3 scripts/gen_toy_recurrent_onnx.py --out path/to/policy.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUT = "tests/fixtures/models/toy_lstm.onnx"

#: Hidden size, observation width, and action width of the toy cell.
HIDDEN = 4
OBS = 3
ACT = 2


def build_toy_recurrent(hidden: int = HIDDEN, obs: int = OBS, act: int = ACT):
    """Build the toy 3 -> (LSTM, H=4) -> 2 policy as an ``onnx.ModelProto``."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    w = ((np.arange(4 * hidden * obs, dtype=np.float32) - 6.0) / 10.0).reshape(4 * hidden, obs)
    r = ((np.arange(4 * hidden * hidden, dtype=np.float32) - 8.0) / 20.0).reshape(4 * hidden, hidden)
    b = np.zeros(8 * hidden, dtype=np.float32)
    head = np.eye(hidden, act, dtype=np.float32)

    def f_init(name: str, array):
        """A float initializer; ONNX stores W/R/B with a leading num_directions axis."""
        return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)

    def i_init(name: str, values):
        """An int64 initializer (Reshape/LSTM shape inputs are int64)."""
        return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)

    def vi(name: str, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    nodes = [
        helper.make_node("Reshape", ["obs", "shape_x"], ["X"]),
        helper.make_node(
            "LSTM",
            ["X", "W", "R", "B", "", "h_in", "c_in"],
            ["Y", "h_out", "c_out"],
            hidden_size=hidden,
        ),
        helper.make_node("Reshape", ["h_out", "shape_h"], ["hf"]),
        helper.make_node("MatMul", ["hf", "head"], ["action"]),
    ]
    graph = helper.make_graph(
        nodes,
        "toy_lstm",
        [vi("obs", (1, obs)), vi("h_in", (1, 1, hidden)), vi("c_in", (1, 1, hidden))],
        [vi("action", (1, act)), vi("h_out", (1, 1, hidden)), vi("c_out", (1, 1, hidden))],
        [
            f_init("W", w[None]),
            f_init("R", r[None]),
            f_init("B", b[None]),
            f_init("head", head),
            i_init("shape_x", [1, 1, obs]),
            i_init("shape_h", [1, hidden]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    import onnx

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_toy_recurrent(), str(out))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
