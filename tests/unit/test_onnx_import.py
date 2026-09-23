"""Tests for the ONNX policy → shinro graph importer.

The importer is the only path into the codegen machinery that does not use the
tracer: ``onnx.load(path).graph`` is already a dataflow graph, so these tests
build tiny models with ``onnx.helper`` and assert the *translated* graph runs
``interpret()`` to the hand-computed values. Zig parity for the same graphs is
covered separately in ``tests/test_zig_lowering.py``.
"""

from typing import Any

import numpy as np
import pytest

from shinro.codegen.interpreter import interpret
from shinro.codegen.onnx_import import EPSILON_PORT, OUTPUT_PORT, STATE_PORT, import_onnx_policy

onnx = pytest.importorskip("onnx")


def _vi(name, shape):
    """Build a float ValueInfoProto."""
    from onnx import TensorProto, helper

    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _init(name, array):
    """Build a float initializer from a numpy array."""
    from onnx import TensorProto, helper

    a = np.asarray(array, dtype=np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, a.shape, a.flatten().tolist())


def _save(nodes, inputs, outputs, initializers, tmp_path, name="policy.onnx", opset=13):
    """Serialize an ONNX graph to a temp file and return its path."""
    from onnx import helper

    graph = helper.make_graph(nodes, "g", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    path = tmp_path / name
    onnx.save(model, str(path))
    return str(path)


def _run(cg, x):
    """Interpret the imported graph on a state vector."""
    return interpret(cg.graph, {STATE_PORT: np.asarray(x, dtype=np.float64)})[OUTPUT_PORT]


def _sample(cg, x, epsilon):
    """Interpret the imported graph feeding both the state and the noise port."""
    feed = {
        STATE_PORT: np.asarray(x, dtype=np.float64),
        EPSILON_PORT: np.asarray(epsilon, dtype=np.float64),
    }
    return interpret(cg.graph, feed)[OUTPUT_PORT]


def _op_names(cg):
    return [n.op for n in cg.graph.nodes]


class TestPortLayout:
    def test_graph_is_memoryless_single_port(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Gemm", ["state", "w", "b"], ["y"], transB=1)],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2, 3)), _init("b", [0.0, 0.0])],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert cg.inputs == [STATE_PORT]
        assert cg.outputs == [OUTPUT_PORT]
        assert cg.state_inputs == []
        assert cg.state_outputs == []

    def test_all_nodes_rank_at_most_two(self, tmp_path):
        """The lowered VM is 2-D only — the importer must not emit rank-3 nodes."""
        from onnx import helper

        path = _save(
            [
                helper.make_node("Gemm", ["state", "w1", "b1"], ["h"], transB=1),
                helper.make_node("Tanh", ["h"], ["a"]),
                helper.make_node("Gemm", ["a", "w2", "b2"], ["y"], transB=1),
            ],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [
                _init("w1", np.ones((4, 3))),
                _init("b1", np.zeros(4)),
                _init("w2", np.ones((2, 4))),
                _init("b2", np.zeros(2)),
            ],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert all(len(n.shape) <= 2 for n in cg.graph.nodes)


class TestGemm:
    def test_torch_layout_transB(self, tmp_path):
        from onnx import helper

        w = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)  # (2, 3)
        b = np.array([0.5, -0.5], dtype=np.float32)
        path = _save(
            [helper.make_node("Gemm", ["state", "w", "b"], ["y"], transB=1)],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [_init("w", w), _init("b", b)],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(_run(cg, x), x @ w.T + b, rtol=1e-6)
        # The fused gemm op: one node, no materialized transpose / matmul / mul.
        assert _op_names(cg).count("gemm") == 1
        assert "transpose" not in _op_names(cg)
        assert "matmul" not in _op_names(cg)
        assert _op_names(cg).count("add") == 1  # the action bias; the Gemm bias is fused

    def test_default_layout_transB_off(self, tmp_path):
        from onnx import helper

        w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)  # (3, 2)
        path = _save(
            [helper.make_node("Gemm", ["state", "w"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [_init("w", w)],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        x = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(_run(cg, x), x @ w, rtol=1e-6)
        assert "gemm" in _op_names(cg)
        assert "transpose" not in _op_names(cg)

    def test_alpha_beta(self, tmp_path):
        from onnx import helper

        w = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)  # (2, 3) for transB
        c = np.array([2.0, 4.0], dtype=np.float32)
        path = _save(
            [helper.make_node("Gemm", ["state", "w", "c"], ["y"], transB=1, alpha=0.5, beta=2.0)],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [_init("w", w), _init("c", c)],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        x = np.array([2.0, 3.0, 0.0])
        np.testing.assert_allclose(_run(cg, x), 0.5 * (x @ w.T) + 2.0 * c, rtol=1e-6)
        # alpha/beta are baked into the fused gemm node (aux-indexed tables),
        # so the only remaining mul is the action-surface scale.
        assert _op_names(cg).count("mul") == 1

    def test_no_bias(self, tmp_path):
        from onnx import helper

        w = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        path = _save(
            [helper.make_node("Gemm", ["state", "w"], ["y"], transB=1)],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", w)],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        np.testing.assert_allclose(_run(cg, [3.0, 4.0]), [3.0, 4.0], rtol=1e-6)
        assert _op_names(cg).count("add") == 1  # the action bias only; the Gemm has none

    def test_transA_rejected(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Gemm", ["state", "w"], ["y"], transA=1)],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [_init("w", np.eye(3))],
            tmp_path,
        )
        with pytest.raises(NotImplementedError, match="transA"):
            import_onnx_policy(path)

    def test_unknown_attribute_rejected(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Gemm", ["state", "w"], ["y"], broadcast=1)],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [_init("w", np.eye(3))],
            tmp_path,
        )
        with pytest.raises(NotImplementedError, match="broadcast"):
            import_onnx_policy(path)


class TestActivations:
    def test_mlp_tanh(self, tmp_path):
        from onnx import helper

        w1 = np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0
        b1 = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        w2 = np.arange(8, dtype=np.float32).reshape(2, 4) / 5.0
        b2 = np.array([0.5, -0.5], dtype=np.float32)
        path = _save(
            [
                helper.make_node("Gemm", ["state", "w1", "b1"], ["h"], transB=1),
                helper.make_node("Tanh", ["h"], ["a"]),
                helper.make_node("Gemm", ["a", "w2", "b2"], ["y"], transB=1),
            ],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [_init("w1", w1), _init("b1", b1), _init("w2", w2), _init("b2", b2)],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        x = np.array([0.5, -1.5, 2.0])
        expected = np.tanh(x @ w1.T + b1) @ w2.T + b2
        np.testing.assert_allclose(_run(cg, x), expected, rtol=1e-6, atol=1e-6)

    def test_relu(self, tmp_path):
        from onnx import helper

        path = _save(
            [
                helper.make_node("Gemm", ["state", "w"], ["h"], transB=1),
                helper.make_node("Relu", ["h"], ["y"]),
            ],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2))],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        np.testing.assert_allclose(_run(cg, [3.0, -4.0]), [3.0, 0.0])
        assert "relu" in _op_names(cg)

    def test_sigmoid_lowered_to_fused_op(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Sigmoid", ["state"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        # Sigmoid is a single fused VM op — not the old exp/neg/add/div chain.
        # (`add`/`mul` remain from the action-space scale/bias baking.)
        assert "sigmoid" in _op_names(cg)
        assert not {"neg", "exp", "div"} & set(_op_names(cg)), _op_names(cg)
        x = np.array([-1.0, 0.0, 2.0])
        np.testing.assert_allclose(_run(cg, x), 1.0 / (1.0 + np.exp(-x)), rtol=1e-12)

    def test_softmax_last_axis(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Softmax", ["state"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert "softmax" in _op_names(cg)
        x = np.array([-1.0, 0.0, 2.0])
        e = np.exp(x - x.max())
        np.testing.assert_allclose(_run(cg, x), e / e.sum(), rtol=1e-12)

    def test_softmax_rank2_is_per_row(self, tmp_path):
        from onnx import helper

        w = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        path = _save(
            [
                helper.make_node("Softmax", ["w"], ["s"]),
                helper.make_node("MatMul", ["s", "state"], ["y"]),
            ],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [_init("w", w)],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert "softmax" in _op_names(cg)
        x = np.array([1.0, 1.0, 1.0])
        w64 = w.astype(np.float64)  # initializers are float32; the graph stores them as f64
        e = np.exp(w64 - w64.max(axis=-1, keepdims=True))
        expected = (e / e.sum(axis=-1, keepdims=True)) @ x
        np.testing.assert_allclose(_run(cg, x), expected, rtol=1e-12)

    def test_softmax_rejects_non_last_axis(self, tmp_path):
        from onnx import helper

        w = np.ones((2, 3), dtype=np.float32)
        path = _save(
            [
                helper.make_node("Softmax", ["w"], ["s"], axis=0),
                helper.make_node("MatMul", ["s", "state"], ["y"]),
            ],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [_init("w", w)],
            tmp_path,
        )
        with pytest.raises(NotImplementedError, match="last axis"):
            import_onnx_policy(path)

    def test_gelu_tanh(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Gelu", ["state"], ["y"], approximate="tanh")],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
            opset=20,
        )
        cg = import_onnx_policy(path)
        assert "gelu" in _op_names(cg)
        x = np.array([-2.0, 0.5, 2.0])
        expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))
        np.testing.assert_allclose(_run(cg, x), expected, rtol=1e-12)

    def test_gelu_exact_default_rejected(self, tmp_path):
        from onnx import helper

        # No approximate attr -> ONNX default "none" = exact erf, unsupported.
        path = _save(
            [helper.make_node("Gelu", ["state"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
            opset=20,
        )
        with pytest.raises(NotImplementedError, match="tanh"):
            import_onnx_policy(path)


class TestExpandedOps:
    """The elementwise / utility ONNX ops added for real small policies."""

    def test_elu(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Elu", ["state"], ["y"], alpha=1.0)],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert "elu" in _op_names(cg)
        x = np.array([-2.0, 0.0, 2.0])
        np.testing.assert_allclose(_run(cg, x), np.where(x > 0, x, np.exp(x) - 1.0), rtol=1e-12)

    def test_normalized_mlp_matches_manual(self, tmp_path):
        # Mirrors the real legged-locomotion policies: baked obs normalization
        # (Sub/Div) -> Gemm (torch layout) -> Elu.
        from onnx import helper

        w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # (out=2, in=3)
        b = np.array([0.1, -0.2], dtype=np.float32)
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        path = _save(
            [
                helper.make_node("Sub", ["state", "mean"], ["d"]),
                helper.make_node("Div", ["d", "std"], ["n"]),
                helper.make_node("Gemm", ["n", "w", "b"], ["y"], transB=1),
            ],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [_init("w", w), _init("b", b), _init("mean", mean), _init("std", std)],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert {"sub", "div", "gemm"} <= set(_op_names(cg))
        x = np.array([1.0, 2.0, 3.0])
        n = (x - mean.astype(np.float64)) / std.astype(np.float64)
        expected = n @ w.astype(np.float64).T + b.astype(np.float64)
        np.testing.assert_allclose(_run(cg, x), expected, rtol=1e-12)

    def test_binary_elementwise(self, tmp_path):
        from onnx import helper

        two = _init("two", np.full(3, 2.0))
        path = _save(
            [
                helper.make_node("Mul", ["state", "two"], ["m"]),
                helper.make_node("Sub", ["m", "state"], ["s"]),
                helper.make_node("Div", ["s", "two"], ["d"]),
                helper.make_node("Pow", ["d", "two"], ["y"]),
            ],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [two],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert {"mul", "sub", "div", "pow"} <= set(_op_names(cg))
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(_run(cg, x), ((x * 2 - x) / 2) ** 2, rtol=1e-12)

    def test_unary_elementwise(self, tmp_path):
        from onnx import helper

        path = _save(
            [
                helper.make_node("Neg", ["state"], ["n"]),
                helper.make_node("Abs", ["n"], ["a"]),
                helper.make_node("Exp", ["a"], ["y"]),
            ],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert {"neg", "abs", "exp"} <= set(_op_names(cg))
        x = np.array([0.0, 1.0, -2.0])
        np.testing.assert_allclose(_run(cg, x), np.exp(np.abs(-x)), rtol=1e-12)

    def test_clip_attrs(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Clip", ["state"], ["y"], min=-1.0, max=1.0)],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert "clip" in _op_names(cg)
        np.testing.assert_allclose(_run(cg, np.array([-5.0, 0.5, 5.0])), [-1.0, 0.5, 1.0], rtol=1e-12)

    def test_constant(self, tmp_path):
        from onnx import helper

        c = helper.make_node(
            "Constant", [], ["c"], value=helper.make_tensor("v", onnx.TensorProto.FLOAT, [3], [1.0, 2.0, 3.0])
        )
        path = _save(
            [c, helper.make_node("Add", ["state", "c"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert "const" in _op_names(cg)
        np.testing.assert_allclose(_run(cg, np.zeros(3)), [1.0, 2.0, 3.0], rtol=1e-12)

    def test_identity(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Identity", ["state"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert "copy" in _op_names(cg)
        np.testing.assert_allclose(_run(cg, np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0], rtol=1e-12)

    def test_reshape_and_flatten(self, tmp_path):
        from onnx import helper

        shape = _init("shape", np.array([1, 3], dtype=np.int64))
        path = _save(
            [
                helper.make_node("Reshape", ["state", "shape"], ["r"]),
                helper.make_node("Flatten", ["r"], ["y"], axis=1),
            ],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [shape],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert "reshape" in _op_names(cg)
        np.testing.assert_allclose(_run(cg, np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0], rtol=1e-12)

    def test_transpose_2d(self, tmp_path):
        from onnx import helper

        shape = _init("shape", np.array([3, 1], dtype=np.int64))
        path = _save(
            [
                helper.make_node("Reshape", ["state", "shape"], ["r"]),
                helper.make_node("Transpose", ["r"], ["t"], perm=[1, 0]),
            ],
            [_vi("state", [None, 3])],
            [_vi("t", [None, 3])],
            [shape],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert "transpose" in _op_names(cg)
        np.testing.assert_allclose(_run(cg, np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0], rtol=1e-12)

    def test_transpose_rank1_rejected(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Transpose", ["state"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
        )
        with pytest.raises(NotImplementedError, match="Transpose"):
            import_onnx_policy(path)


class TestPointwiseGraphs:
    def test_matmul_add(self, tmp_path):
        from onnx import helper

        w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)  # (3, 2)
        b = np.array([1.0, -1.0], dtype=np.float32)
        path = _save(
            [
                helper.make_node("MatMul", ["state", "w"], ["m"]),
                helper.make_node("Add", ["m", "b"], ["y"]),
            ],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 2])],
            [_init("w", w), _init("b", b)],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(_run(cg, x), x @ w + b, rtol=1e-6)


class TestObservationEncoder:
    def test_selection_normalize_clip(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Gemm", ["state", "w"], ["y"], transB=1)],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2))],
            tmp_path,
        )
        cg = import_onnx_policy(
            path,
            n_x=3,
            obs_cfg={
                "state_keys": [2, 0],
                "normalize": True,
                "obs_mean": [1.0, 2.0],
                "obs_std": [2.0, 4.0],
                "clip": [-1.0, 1.0],
            },
        )
        # state [10, 0, 5] -> obs [5, 10] -> ([4, 8])/[2,4] = [2,2] -> clip [1,1]
        np.testing.assert_allclose(_run(cg, [10.0, 0.0, 5.0]), [1.0, 1.0], rtol=1e-6)

    def test_identity_selection_skips_matmul(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("MatMul", ["state", "w"], ["y"])],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2))],
            tmp_path,
        )
        cg = import_onnx_policy(path, n_x=2, obs_cfg={"state_keys": [0, 1]})
        assert _op_names(cg).count("matmul") == 1  # only the policy's own matmul

    def test_state_keys_infer_n_x(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("MatMul", ["state", "w"], ["y"])],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2))],
            tmp_path,
        )
        cg = import_onnx_policy(path, obs_cfg={"state_keys": [2, 0]})
        # n_x = max(state_keys)+1 = 3; state [7, 8, 9] -> obs [9, 7]
        np.testing.assert_allclose(_run(cg, [7.0, 8.0, 9.0]), [9.0, 7.0], rtol=1e-6)

    def test_normalize_without_stats_rejected(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("MatMul", ["state", "w"], ["y"])],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2))],
            tmp_path,
        )
        with pytest.raises(ValueError, match="obs_mean"):
            import_onnx_policy(path, obs_cfg={"normalize": True})

    def test_obs_dim_mismatch_rejected(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("MatMul", ["state", "w"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [_init("w", np.eye(3))],
            tmp_path,
        )
        with pytest.raises(ValueError, match="state_keys"):
            import_onnx_policy(path, obs_cfg={"state_keys": [0, 1]})

    def test_state_keys_out_of_range_rejected(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("MatMul", ["state", "w"], ["y"])],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2))],
            tmp_path,
        )
        with pytest.raises(ValueError, match="out of range"):
            import_onnx_policy(path, n_x=2, obs_cfg={"state_keys": [0, 5]})

    def test_unknown_observation_key_rejected(self, tmp_path):
        """A typo like `obs_means` must not silently drop normalization."""
        from onnx import helper

        path = _save(
            [helper.make_node("MatMul", ["state", "w"], ["y"])],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2))],
            tmp_path,
        )
        with pytest.raises(ValueError, match="unknown key"):
            import_onnx_policy(path, obs_cfg={"obs_means": [0.0, 0.0]})


class TestRejections:
    def test_unsupported_op_names_itself(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Sqrt", ["state"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [],
            tmp_path,
        )
        with pytest.raises(NotImplementedError, match="Sqrt"):
            import_onnx_policy(path)

    def test_unreachable_unsupported_node_ignored(self, tmp_path):
        from onnx import helper

        path = _save(
            [
                helper.make_node("Sqrt", ["state"], ["junk"]),
                helper.make_node("MatMul", ["state", "w"], ["y"]),
            ],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2))],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        assert "sqrt" not in _op_names(cg)
        np.testing.assert_allclose(_run(cg, [1.0, 2.0]), [1.0, 2.0], rtol=1e-6)

    def test_multi_input_policy_rejected(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Add", ["a", "b"], ["y"])],
            [_vi("a", [None, 2]), _vi("b", [None, 2])],
            [_vi("y", [None, 2])],
            [],
            tmp_path,
        )
        with pytest.raises(ValueError, match="exactly one"):
            import_onnx_policy(path)

    def test_rank2_batch1_output_flattened(self, tmp_path):
        from onnx import helper

        # (3,) + (1, 3) broadcasts to (1, 3), a rank-2 batch-1 result the
        # importer must reshape down to the (n_u,) action port.
        path = _save(
            [helper.make_node("Add", ["state", "c"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [_init("c", np.array([[1.0, 2.0, 3.0]], dtype=np.float32))],
            tmp_path,
        )
        cg = import_onnx_policy(path)
        u = _run(cg, [1.0, 1.0, 1.0])
        assert u.shape == (3,)
        np.testing.assert_allclose(u, [2.0, 3.0, 4.0], rtol=1e-6)
        assert "reshape" in _op_names(cg)

    def test_batched_output_rejected(self, tmp_path):
        from onnx import helper

        # (3,) + (2, 1) broadcasts to (2, 3) — a genuine batch, not a single
        # action vector, so the importer must refuse it rather than drop a row.
        path = _save(
            [helper.make_node("Add", ["state", "c"], ["y"])],
            [_vi("state", [None, 3])],
            [_vi("y", [None, 3])],
            [_init("c", np.array([[1.0], [2.0]], dtype=np.float32))],
            tmp_path,
        )
        with pytest.raises(ValueError, match="action vector"):
            import_onnx_policy(path)

    def test_unknown_tensor_rejected(self, tmp_path):
        from onnx import helper

        path = _save(
            [helper.make_node("Add", ["state", "missing"], ["y"])],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [],
            tmp_path,
        )
        with pytest.raises(ValueError, match="unknown tensor"):
            import_onnx_policy(path)


def _gemm_policy(tmp_path, w, b=None, *, name="policy.onnx"):
    """Build a single-Gemm policy in torch layout (transB=1)."""
    from onnx import helper

    w = np.asarray(w, dtype=np.float32)
    inputs = ["state", "w"] + (["b"] if b is not None else [])
    inits = [_init("w", w)] + ([_init("b", np.asarray(b, dtype=np.float32))] if b is not None else [])
    return _save(
        [helper.make_node("Gemm", inputs, ["y"], transB=1)],
        [_vi("state", [None, w.shape[1]])],
        [_vi("y", [None, w.shape[0]])],
        inits,
        tmp_path,
        name,
    )


class TestActionSurface:
    """The baked post-processing must mirror the old runtime `_postprocess`."""

    def _tiny(self, tmp_path):
        """2-output policy: raw(x) = [x0 + 0.5, 2*x1 - 0.5]."""
        return _gemm_policy(tmp_path, [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], [0.5, -0.5])

    def _stochastic(self, tmp_path):
        """4-output policy: raw(x) = [x0+1, x1+2, x2+3, 4] = [mean; log_std]."""
        w = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        return _gemm_policy(tmp_path, w, [1.0, 2.0, 3.0, 4.0], name="stochastic.onnx")

    def test_continuous_default_passthrough(self, tmp_path):
        cg = import_onnx_policy(self._tiny(tmp_path))
        assert cg.inputs == [STATE_PORT]
        assert "clip" not in _op_names(cg)  # no clip configured
        np.testing.assert_allclose(_run(cg, [1.0, 2.0, 3.0]), [1.5, 3.5], rtol=1e-6)

    def test_continuous_default_still_emits_scale_bias(self, tmp_path):
        """Uniform lowering: the default 1.0 / 0.0 still produce the mul/add nodes."""
        from onnx import helper

        path = _save(
            [helper.make_node("MatMul", ["state", "w"], ["y"])],
            [_vi("state", [None, 2])],
            [_vi("y", [None, 2])],
            [_init("w", np.eye(2, dtype=np.float32))],
            tmp_path,
        )
        cg = import_onnx_policy(path)  # default continuous action config
        ops = _op_names(cg)
        assert ops.count("mul") == 1  # action scale, emitted even though it is 1.0
        assert ops.count("add") == 1  # action bias, emitted even though it is 0.0

    def test_continuous_scale_bias_clip(self, tmp_path):
        cg = import_onnx_policy(
            self._tiny(tmp_path),
            action_cfg={"action_scale": 2.0, "action_bias": 1.0, "action_clip_low": -3.0, "action_clip_high": 3.0},
        )
        # raw [1.5, 1.5] -> *2+1 = [4, 4] -> clipped to 3
        np.testing.assert_allclose(_run(cg, [1.0, 1.0, 0.0]), [3.0, 3.0], rtol=1e-6)

    def test_continuous_vector_scale_bias(self, tmp_path):
        path = _gemm_policy(tmp_path, [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], name="vec.onnx")
        cg = import_onnx_policy(path, action_cfg={"action_scale": [2.0, 3.0], "action_bias": [1.0, -1.0]})
        # raw [1, 2] -> [1*2+1, 2*3-1] = [3, 5]
        np.testing.assert_allclose(_run(cg, [1.0, 1.0, 0.0]), [3.0, 5.0], rtol=1e-6)

    def test_discrete_deterministic_one_hot(self, tmp_path):
        cg = import_onnx_policy(self._tiny(tmp_path), action_cfg={"action_space": "discrete"})
        assert cg.inputs == [STATE_PORT]  # deterministic: no noise port
        assert {"argmax", "one_hot"} <= set(_op_names(cg))
        # equal logits [1.5, 1.5] -> first-max wins, matching numpy argmax
        np.testing.assert_allclose(_run(cg, [1.0, 1.0, 1.0]), [1.0, 0.0], rtol=1e-6)

    def test_discrete_ignores_scale_bias(self, tmp_path):
        cg = import_onnx_policy(
            self._tiny(tmp_path),
            action_cfg={"action_space": "discrete", "action_scale": 5.0, "action_bias": 1.0},
        )
        np.testing.assert_allclose(_run(cg, [1.0, 1.0, 1.0]), [1.0, 0.0], rtol=1e-6)

    def test_discrete_gumbel_max_uses_epsilon(self, tmp_path):
        cg = import_onnx_policy(self._tiny(tmp_path), action_cfg={"action_space": "discrete", "deterministic": False})
        assert cg.inputs == [STATE_PORT, EPSILON_PORT]
        state = np.array([1.0, 1.0, 1.0])
        # A huge positive Gumbel draw flips the argmax; the kernel only adds.
        np.testing.assert_allclose(_sample(cg, state, [0.0, 100.0]), [0.0, 1.0], rtol=1e-6)
        np.testing.assert_allclose(_sample(cg, state, [100.0, 0.0]), [1.0, 0.0], rtol=1e-6)

    def test_discrete_gumbel_max_matches_softmax(self, tmp_path):
        # logits = [0, ln 3] -> softmax p(1) = 0.75; Gumbel-max must reproduce it.
        path = _gemm_policy(tmp_path, np.zeros((2, 3), dtype=np.float32), [0.0, np.log(3.0)], name="dist.onnx")
        cg = import_onnx_policy(path, action_cfg={"action_space": "discrete", "deterministic": False})
        rng = np.random.default_rng(0)
        n = 2000
        draws = -np.log(-np.log(rng.uniform(size=(n, 2))))
        state = np.zeros(3)
        hits = sum(int(np.argmax(_sample(cg, state, eps))) for eps in draws)
        assert abs(hits / n - 0.75) < 0.06

    def test_stochastic_deterministic_returns_mean(self, tmp_path):
        cg = import_onnx_policy(self._stochastic(tmp_path), action_cfg={"action_space": "stochastic"})
        assert cg.inputs == [STATE_PORT]
        # raw = [11, 22, 33, 4] -> mean = [11, 22]
        np.testing.assert_allclose(_run(cg, [10.0, 20.0, 30.0]), [11.0, 22.0], rtol=1e-6)

    def test_stochastic_epsilon_formula(self, tmp_path):
        cg = import_onnx_policy(self._stochastic(tmp_path), action_cfg={"action_space": "stochastic", "deterministic": False})
        assert cg.inputs == [STATE_PORT, EPSILON_PORT]
        # raw = [11, 22, 33, 4]; log_std clipped to 2 -> std = e^2
        u = _sample(cg, [10.0, 20.0, 30.0], [0.5, -1.0])
        std = np.exp(2.0)
        np.testing.assert_allclose(u, [11.0 + std * 0.5, 22.0 - std], rtol=1e-6)

    def test_stochastic_odd_output_rejected(self, tmp_path):
        path = _gemm_policy(tmp_path, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], name="odd.onnx")
        with pytest.raises(ValueError, match="even"):
            import_onnx_policy(path, action_cfg={"action_space": "stochastic"})

    def test_invalid_action_space_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="action_space"):
            import_onnx_policy(self._tiny(tmp_path), action_cfg={"action_space": "bogus"})

    def test_single_sided_clip_rejected(self, tmp_path):
        # The missing bound would be ±inf, which Zig cannot represent as a literal.
        with pytest.raises(ValueError, match="together"):
            import_onnx_policy(self._tiny(tmp_path), action_cfg={"action_clip_low": -1.0})

    def test_infinite_clip_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="finite"):
            import_onnx_policy(
                self._tiny(tmp_path),
                action_cfg={"action_clip_low": -np.inf, "action_clip_high": np.inf},
            )


# ─── recurrent policies (LSTM / GRU / RNN) ──────────────────────────────────


def _int_init(name, values):
    """Build an int64 initializer (Slice/Reshape shape inputs are int64)."""
    from onnx import TensorProto, helper

    a = np.asarray(values, dtype=np.int64)
    return helper.make_tensor(name, TensorProto.INT64, a.shape, a.flatten().tolist())


# The three fused cells, and the ONNX gate-stack width each one expects.
_RECURRENT_OPS = ("LSTM", "GRU", "RNN")
_RECURRENT_GATES = {"LSTM": 4, "GRU": 3, "RNN": 1}


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def _ref_cell(op, x, h, c, w, r, b, H, lbr=0):
    """Independent numpy implementation of the ONNX cell (test oracle).

    Deliberately hand-written rather than routed through ``ops.py`` so a bug in
    both the handler and the kernel cannot hide behind a shared implementation.
    Returns ``(h_next, c_next)`` (``c_next`` is None for GRU/RNN).
    """
    gx = x @ w.T
    gh = h @ r.T
    if op == "LSTM":
        z = gx + gh + b[: 4 * H] + b[4 * H :]
        i_g = _sigmoid(z[:H])
        o_g = _sigmoid(z[H : 2 * H])
        f_g = _sigmoid(z[2 * H : 3 * H])
        cand = np.tanh(z[3 * H : 4 * H])
        c_next = f_g * c + i_g * cand
        return o_g * np.tanh(c_next), c_next
    if op == "GRU":
        z = _sigmoid(gx[:H] + gh[:H] + b[:H] + b[3 * H : 4 * H])
        r_g = _sigmoid(gx[H : 2 * H] + gh[H : 2 * H] + b[H : 2 * H] + b[4 * H : 5 * H])
        if lbr:
            cand = np.tanh(gx[2 * H : 3 * H] + r_g * (gh[2 * H : 3 * H] + b[5 * H : 6 * H]) + b[2 * H : 3 * H])
        else:
            cand = np.tanh(gx[2 * H : 3 * H] + (r_g * h) @ r[2 * H :].T + b[5 * H : 6 * H] + b[2 * H : 3 * H])
        return (1.0 - z) * cand + z * h, None
    return np.tanh(gx + gh + b[:H] + b[H:]), None


def _recurrent_policy(
    op,
    tmp_path,
    *,
    H=3,
    I=4,
    OBS=5,
    ACT=2,
    layers=1,
    lbr=0,
    direction="forward",
    activations=None,
    clip=None,
    sequence_lens=False,
    peepholes=False,
    consume_sequence=False,
    batch=1,
    seq=1,
    extra_direction=False,
):
    """Build a torch-style recurrent export: ``forward(x, h[, c]) -> action, h'[, c']``.

    Layer ``L``'s hidden state enters as ``h{L}_in`` / ``c{L}_in`` and leaves as
    ``h_out`` (last layer, also a graph output) or ``h_{L}`` (intermediate, fed
    to the next cell). Returns ``(path, ref)`` where ``ref`` is everything the
    independent numpy oracle needs.
    """
    from onnx import helper

    rng = np.random.default_rng(7)
    ng = _RECURRENT_GATES[op]
    nodes = []
    inits = [_init("Win", rng.normal(0.0, 0.3, (OBS, I))), _init("Whead", rng.normal(0.0, 0.3, (H, ACT)))]
    inits += [_int_init("sx", [1, seq, I]), _int_init("sh", [1, H]), _int_init("sl", [1])]
    weights = []
    ndir = 2 if extra_direction else 1
    for layer in range(layers):
        in_w = I if layer == 0 else H
        w = rng.normal(0.0, 0.3, (ndir, ng * H, in_w)).astype(np.float32)
        r = rng.normal(0.0, 0.3, (ndir, ng * H, H)).astype(np.float32)
        b = rng.normal(0.0, 0.3, (ndir, 2 * ng * H)).astype(np.float32)
        weights.append((w[0].astype(np.float64), r[0].astype(np.float64), b[0].astype(np.float64)))
        inits += [_init(f"W{layer}", w), _init(f"R{layer}", r), _init(f"B{layer}", b)]

    nodes.append(helper.make_node("MatMul", ["obs", "Win"], ["x2"]))
    nodes.append(helper.make_node("Reshape", ["x2", "sx"], ["X0"]))

    cell_attrs: dict[str, Any] = {"hidden_size": H}
    if op == "GRU":
        cell_attrs["linear_before_reset"] = int(lbr)
    if direction != "forward":
        cell_attrs["direction"] = direction
    if activations is not None:
        cell_attrs["activations"] = activations
    if clip is not None:
        cell_attrs["clip"] = clip

    prev = "X0"
    for layer in range(layers):
        last = layer == layers - 1
        h_name = "h_out" if last else f"h_{layer}"
        cell_in = [prev, f"W{layer}", f"R{layer}", f"B{layer}", "sl" if sequence_lens else "", f"h{layer}_in"]
        cell_out = [f"Y_{layer}", h_name]
        if op == "LSTM":
            cell_in.append(f"c{layer}_in")
            cell_out.append(f"c{layer}_out")
            if peepholes:
                inits.append(_init(f"P{layer}", np.zeros(3 * H, dtype=np.float32)))
                cell_in.append(f"P{layer}")
        nodes.append(helper.make_node(op, cell_in, cell_out, **cell_attrs))
        prev = h_name

    # Head: consume h_out (or the sequence output Y_0 when consume_sequence).
    head_src = "h_out"
    if consume_sequence:
        nodes.append(helper.make_node("Reshape", ["Y_0", "sh"], ["yflat"]))
        head_src = "yflat"
    else:
        nodes.append(helper.make_node("Reshape", ["h_out", "sh"], ["hflat"]))
        head_src = "hflat"
    nodes.append(helper.make_node("MatMul", [head_src, "Whead"], ["action"]))

    ins = [_vi("obs", [batch, OBS])]
    outs = [_vi("action", [batch, ACT]), _vi("h_out", [1, seq, H])]
    for layer in range(layers):
        ins.append(_vi(f"h{layer}_in", [1, seq, H]))
        if op == "LSTM":
            ins.append(_vi(f"c{layer}_in", [1, seq, H]))
    if op == "LSTM":
        outs.append(_vi("c_out", [1, seq, H]))

    path = _save(nodes, ins, outs, inits, tmp_path, name=f"{op.lower()}.onnx", opset=14)
    ref = {"weights": weights, "Win": None, "Whead": None, "H": H, "lbr": lbr}
    # Recover W1/W2 as f64 for the oracle.
    ref["Win"] = np.asarray([t for t in inits if t.name == "Win"][0].float_data).reshape(OBS, I).astype(np.float64)
    ref["Whead"] = np.asarray([t for t in inits if t.name == "Whead"][0].float_data).reshape(H, ACT).astype(np.float64)
    return path, ref


def _ref_forward(op, obs, states, ref):
    """Run the numpy oracle through every layer; returns (action, new_states)."""
    H = ref["H"]
    x = np.asarray(obs, dtype=np.float64) @ ref["Win"]
    new_states = []
    for layer, (w, r, b) in enumerate(ref["weights"]):
        h, c = states[layer]
        h_next, c_next = _ref_cell(op, x, h, c, w, r, b, H, ref["lbr"])
        new_states.append((h_next, c_next))
        x = h_next
    return x @ ref["Whead"], new_states


def _state_names(op, layers=1):
    ports = []
    for layer in range(layers):
        ports.append(f"state_h_{layer}")
        if op == "LSTM":
            ports.append(f"state_c_{layer}")
    return ports


def _feed_states(op, states, layers=1):
    feed = {}
    for layer, (h, c) in enumerate(states):
        feed[f"state_h_{layer}"] = h
        if op == "LSTM":
            feed[f"state_c_{layer}"] = c
    return feed


@pytest.mark.parametrize("op", _RECURRENT_OPS)
class TestRecurrent:
    def test_ports_and_state_ports(self, op, tmp_path):
        path, _ = _recurrent_policy(op, tmp_path)
        cg = import_onnx_policy(path)
        assert cg.inputs == [STATE_PORT, *_state_names(op)]
        assert cg.outputs == [OUTPUT_PORT]
        assert cg.state_inputs == _state_names(op)
        assert cg.state_outputs == ["state_hc_0"] if op == "LSTM" else cg.state_outputs == ["state_h_0"]

    def test_cell_is_one_fused_node(self, op, tmp_path):
        """The whole cell is a single VM node, not a gate-by-gate decomposition."""
        path, _ = _recurrent_policy(op, tmp_path)
        cg = import_onnx_policy(path)
        assert _op_names(cg).count(op.lower()) == 1
        # and nothing from the decomposed form leaked in
        for leaked in ("sigmoid", "tanh"):
            assert leaked not in _op_names(cg)

    def test_matches_numpy_reference(self, op, tmp_path):
        path, ref = _recurrent_policy(op, tmp_path)
        cg = import_onnx_policy(path)
        H = ref["H"]
        rng = np.random.default_rng(11)
        obs = rng.normal(0.0, 0.5, 5)
        h0 = rng.normal(0.0, 0.5, H)
        c0 = rng.normal(0.0, 0.5, H)
        feed = {STATE_PORT: obs, **_feed_states(op, [(h0, c0)])}
        got = interpret(cg.graph, feed)
        exp_u, exp_states = _ref_forward(op, obs, [(h0, c0)], ref)
        np.testing.assert_allclose(got[OUTPUT_PORT], exp_u, atol=1e-12)
        h_next, c_next = exp_states[0]
        if op == "LSTM":
            assert c_next is not None
            np.testing.assert_allclose(got["state_hc_0"], np.concatenate([h_next, c_next]), atol=1e-12)
        else:
            np.testing.assert_allclose(got["state_h_0"], h_next, atol=1e-12)

    def test_all_nodes_rank_at_most_two(self, op, tmp_path):
        path, _ = _recurrent_policy(op, tmp_path)
        cg = import_onnx_policy(path)
        assert all(len(n.shape) <= 2 for n in cg.graph.nodes)

    def test_state_feedback_changes_the_next_tick(self, op, tmp_path):
        """The recurrence is live: feeding the state back must change the action."""
        path, ref = _recurrent_policy(op, tmp_path)
        cg = import_onnx_policy(path)
        H = ref["H"]
        obs = np.linspace(-0.4, 0.4, 5)
        first = interpret(cg.graph, {STATE_PORT: obs, **_feed_states(op, [(np.zeros(H), np.zeros(H))])})
        state = first["state_hc_0"] if op == "LSTM" else first["state_h_0"]
        h = state[:H]
        c = state[H:] if op == "LSTM" else np.zeros(H)
        second = interpret(cg.graph, {STATE_PORT: obs, **_feed_states(op, [(h, c)])})
        assert not np.allclose(first[OUTPUT_PORT], second[OUTPUT_PORT])

    def test_stacked_cells_get_distinct_state_ports(self, op, tmp_path):
        """Two stacked cells (a real G1-humanoid shape) get one port pair each."""
        path, ref = _recurrent_policy(op, tmp_path, layers=2)
        cg = import_onnx_policy(path)
        assert cg.state_inputs == _state_names(op, layers=2)
        assert cg.state_outputs == (["state_hc_0", "state_hc_1"] if op == "LSTM" else ["state_h_0", "state_h_1"])
        H = ref["H"]
        rng = np.random.default_rng(3)
        states = [(rng.normal(0, 0.5, H), rng.normal(0, 0.5, H)) for _ in range(2)]
        obs = rng.normal(0, 0.5, 5)
        got = interpret(cg.graph, {STATE_PORT: obs, **_feed_states(op, states, layers=2)})
        exp_u, _ = _ref_forward(op, obs, states, ref)
        np.testing.assert_allclose(got[OUTPUT_PORT], exp_u, atol=1e-12)
        assert _op_names(cg).count(op.lower()) == 2


class TestRecurrentGruFlag:
    def test_linear_before_reset_is_baked_and_changes_the_result(self, tmp_path):
        """The two reset placements must produce different graphs and numbers."""
        h = np.array([0.3, -0.2, 0.5])
        x = np.linspace(-0.5, 0.5, 5)  # the fixture obs port is 5-wide
        outs = {}
        for lbr in (0, 1):
            path, ref = _recurrent_policy("GRU", tmp_path, lbr=lbr)
            cg = import_onnx_policy(path)
            node = next(n for n in cg.graph.nodes if n.op == "gru")
            assert bool(node.attrs.get("linear_before_reset", False)) == bool(lbr)
            outs[lbr] = interpret(cg.graph, {STATE_PORT: x, "state_h_0": h})[OUTPUT_PORT]
            exp, _ = _ref_forward("GRU", x, [(h, None)], ref)
            np.testing.assert_allclose(outs[lbr], exp, atol=1e-12)
        assert not np.allclose(outs[0], outs[1])


class TestRecurrentMlAgentsLayout:
    def test_recurrent_in_and_out_are_intercepted(self, tmp_path):
        """An ML-Agents LSTM slices ``recurrent_in`` and concats ``recurrent_out``.

        Both are glue around the cell's state; the importer must bind its own
        state ports and drop the glue (the cell's ``[h ‖ c]`` output already is
        ``recurrent_out``).
        """
        from onnx import helper

        H, I, OBS, ACT = 3, 4, 5, 2
        rng = np.random.default_rng(2)
        ng = 4
        inits = [
            _init("Win", rng.normal(0, 0.3, (OBS, I))),
            _init("Whead", rng.normal(0, 0.3, (H, ACT))),
            _init("W", rng.normal(0, 0.3, (1, ng * H, I))),
            _init("R", rng.normal(0, 0.3, (1, ng * H, H))),
            _init("B", rng.normal(0, 0.3, (1, 2 * ng * H))),
            _int_init("sx", [1, 1, I]),
            _int_init("sh", [1, H]),
            _int_init("s0", [0]),
            _int_init("e0", [H]),
            _int_init("sH", [H]),
            _int_init("eMAX", [2**31 - 1]),
            _int_init("ax", [2]),
        ]
        nodes = [
            helper.make_node("MatMul", ["obs", "Win"], ["x2"]),
            helper.make_node("Reshape", ["x2", "sx"], ["X"]),
            helper.make_node("Slice", ["recurrent_in", "s0", "e0", "ax"], ["h0"]),
            helper.make_node("Slice", ["recurrent_in", "sH", "eMAX", "ax"], ["c0"]),
            helper.make_node("LSTM", ["X", "W", "R", "B", "", "h0", "c0"], ["Y", "Y_h", "Y_c"], hidden_size=H),
            helper.make_node("Reshape", ["Y_h", "sh"], ["hf"]),
            helper.make_node("MatMul", ["hf", "Whead"], ["action"]),
            helper.make_node("Concat", ["Y_h", "Y_c"], ["recurrent_out"], axis=2),
        ]
        ins = [_vi("obs", [1, OBS]), _vi("recurrent_in", [1, 1, 2 * H])]
        outs = [_vi("action", [1, ACT]), _vi("recurrent_out", [1, 1, 2 * H])]
        path = _save(nodes, ins, outs, inits, tmp_path, name="mlagents.onnx", opset=14)

        cg = import_onnx_policy(path)
        assert cg.inputs == [STATE_PORT, "state_h_0", "state_c_0"]
        assert cg.state_outputs == ["state_hc_0"]
        # the ONNX glue is unreachable from the action, so it is not translated
        assert "concat" not in _op_names(cg)

        H_ = H
        rng2 = np.random.default_rng(5)
        obs = rng2.normal(0, 0.5, OBS)
        h0 = rng2.normal(0, 0.5, H_)
        c0 = rng2.normal(0, 0.5, H_)
        got = interpret(cg.graph, {STATE_PORT: obs, "state_h_0": h0, "state_c_0": c0})
        w = np.asarray([t for t in inits if t.name == "W"][0].float_data).reshape(4 * H_, I).astype(np.float64)
        r = np.asarray([t for t in inits if t.name == "R"][0].float_data).reshape(4 * H_, H_).astype(np.float64)
        b = np.asarray([t for t in inits if t.name == "B"][0].float_data).reshape(8 * H_).astype(np.float64)
        w1 = np.asarray([t for t in inits if t.name == "Win"][0].float_data).reshape(OBS, I).astype(np.float64)
        w2 = np.asarray([t for t in inits if t.name == "Whead"][0].float_data).reshape(H_, ACT).astype(np.float64)
        h_next, c_next = _ref_cell("LSTM", obs @ w1, h0, c0, w, r, b, H_)
        assert c_next is not None
        np.testing.assert_allclose(got[OUTPUT_PORT], h_next @ w2, atol=1e-12)
        np.testing.assert_allclose(got["state_hc_0"], np.concatenate([h_next, c_next]), atol=1e-12)


class TestRecurrentRejections:
    def _expect(self, op, tmp_path, match, **kwargs):
        path, _ = _recurrent_policy(op, tmp_path, **kwargs)
        with pytest.raises(NotImplementedError, match=match):
            import_onnx_policy(path)

    def test_bidirectional_rejected(self, tmp_path):
        self._expect("LSTM", tmp_path, "direction", direction="reverse")

    def test_gate_clip_rejected(self, tmp_path):
        self._expect("LSTM", tmp_path, "clip", clip=2.0)

    def test_custom_activations_rejected(self, tmp_path):
        self._expect("LSTM", tmp_path, "activations", activations=["Sigmoid", "Tanh", "Relu"])

    def test_sequence_lens_rejected(self, tmp_path):
        self._expect("GRU", tmp_path, "sequence_lens", sequence_lens=True)

    def test_peepholes_rejected(self, tmp_path):
        self._expect("LSTM", tmp_path, "peepholes", peepholes=True)

    def test_sequence_output_rejected_when_consumed(self, tmp_path):
        self._expect("LSTM", tmp_path, "sequence output", consume_sequence=True)

    def test_extra_direction_axis_rejected(self, tmp_path):
        self._expect("GRU", tmp_path, "num_directions", extra_direction=True)
