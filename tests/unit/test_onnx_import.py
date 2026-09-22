"""Tests for the ONNX policy → shinro graph importer.

The importer is the only path into the codegen machinery that does not use the
tracer: ``onnx.load(path).graph`` is already a dataflow graph, so these tests
build tiny models with ``onnx.helper`` and assert the *translated* graph runs
``interpret()`` to the hand-computed values. Zig parity for the same graphs is
covered separately in ``tests/test_zig_lowering.py``.
"""

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
