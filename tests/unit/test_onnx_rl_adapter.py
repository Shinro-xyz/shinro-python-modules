"""Tests for the ONNX RL policy adapter (eager mode).

The adapter no longer calls ``onnxruntime``: ``from_config`` imports the ONNX
graph into a shinro graph and runs it with the interpreter. The graph-level
behavior (encoder folding, action spaces, epsilon ports) is pinned in
``tests/unit/test_onnx_import.py``; these tests cover the *adapter* contract —
strict config parsing, backend conversion, RNG seeding, and error surfaces.
Compiled-artifact mode is exercised end-to-end by the Zig oracle suite.
"""

import dataclasses
import json

import numpy as np
import pytest

from shinro.controllers.onnx_rl_adapter import KERNEL_FILENAME, OnnxRLAdapter, OnnxRLConfig, _CompiledPolicy

onnx = pytest.importorskip("onnx")


def _save_model(w, b, tmp_path, *, input_name="obs", output_name="output", name="policy.onnx"):
    """Write a single-Gemm (torch layout, transB=1) policy and return its path."""
    from onnx import TensorProto, helper

    w = np.asarray(w, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    assert w.shape[0] == b.shape[0], (w.shape, b.shape)
    x = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, w.shape[1]])
    y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, w.shape[0]])
    node = helper.make_node("Gemm", [input_name, "w", "b"], [output_name], transB=1)
    w_init = helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten().tolist())
    b_init = helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.flatten().tolist())
    graph = helper.make_graph([node], "g", [x], [y], [w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / name
    onnx.save(model, str(path))
    return str(path)


# raw(x) = [x0 + 0.5, 2*x1 - 0.5]
_W_2OUT = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
_B_2OUT = [0.5, -0.5]
# raw(x) = [x0 + 1, x1 + 2, x2 + 3, 4] = [mean; log_std] for a 2-action policy
_W_4OUT = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
_B_4OUT = [1.0, 2.0, 3.0, 4.0]


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    return _save_model(_W_2OUT, _B_2OUT, tmp_path_factory.mktemp("onnx"))


@pytest.fixture(scope="module")
def stochastic_path(tmp_path_factory):
    return _save_model(_W_4OUT, _B_4OUT, tmp_path_factory.mktemp("onnx"), name="stochastic.onnx")


def _ctrl(model_path, **overrides):
    cfg = {"model_path": str(model_path), "action_space": "continuous"}
    cfg.update(overrides)
    return OnnxRLAdapter.from_config(cfg)


class TestConfigSurface:
    def test_declares_a_frozen_config_dataclass(self):
        """The registry checks this (and it becomes a hard error eventually)."""
        assert dataclasses.is_dataclass(OnnxRLConfig)
        assert OnnxRLAdapter.Config is OnnxRLConfig
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(OnnxRLConfig(), "name", "mutated")

    def test_unknown_key_rejected(self, model_path):
        with pytest.raises(ValueError, match="unknown key"):
            OnnxRLAdapter.from_config({"model_path": str(model_path), "bogus": 1})

    def test_missing_mode_rejected(self):
        with pytest.raises(ValueError, match="model_path"):
            OnnxRLAdapter.from_config({"action_space": "continuous"})

    def test_action_space_invalid(self, model_path):
        with pytest.raises(ValueError, match="action_space"):
            OnnxRLAdapter.from_config({"model_path": str(model_path), "action_space": "bogus"})

    def test_default_action_space_is_continuous(self, model_path):
        assert _ctrl(model_path).compute(np.array([1.0, 2.0, 3.0])).shape == (2,)

    def test_single_sided_action_clip_rejected(self, model_path):
        # A missing bound would be ±inf, which the lowerer cannot emit.
        with pytest.raises(ValueError, match="together"):
            OnnxRLAdapter.from_config({"model_path": str(model_path), "action_clip_low": 2.0})

    def test_normalize_without_stats_rejected(self, model_path):
        with pytest.raises(ValueError, match="obs_mean"):
            OnnxRLAdapter.from_config({"model_path": str(model_path), "observation": {"normalize": True}})

    def test_config_file_round_trip(self, model_path, tmp_path):
        """The TOML shape the shipped config uses parses strictly."""
        config = tmp_path / "rl.toml"
        config.write_text(
            f'type = "onnx_rl"\nname = "ppo_policy"\nmodel_path = "{model_path}"\n'
            'action_space = "continuous"\ndeterministic = true\naction_scale = 1.0\n'
            "action_bias = 0.0\nseed = 0\n\n[observation]\nstate_keys = [0, 1, 2]\nnormalize = false\n"
        )
        from shinro.factories.controller_factory import ControllerFactory

        ctrl = ControllerFactory(str(config)).create()
        np.testing.assert_allclose(ctrl.compute(np.array([1.0, 2.0, 3.0])), [1.5, 3.5], rtol=1e-9)


class TestContinuous:
    def test_values(self, model_path):
        np.testing.assert_allclose(_ctrl(model_path).compute(np.array([1.0, 2.0, 3.0])), [1.5, 3.5], rtol=1e-9)

    def test_scale_bias_clip(self, model_path):
        ctrl = _ctrl(model_path, action_scale=2.0, action_bias=1.0, action_clip_low=-3.0, action_clip_high=3.0)
        # raw [1.5, 1.5] -> *2+1 = [4, 4] -> clipped to 3
        np.testing.assert_allclose(ctrl.compute(np.array([1.0, 1.0, 0.0])), [3.0, 3.0], rtol=1e-9)

    def test_vector_scale_bias(self, model_path):
        ctrl = _ctrl(model_path, action_scale=[2.0, 3.0], action_bias=[1.0, -1.0])
        # raw [1.5, 3.5] -> [1.5*2 + 1, 3.5*3 - 1] = [4, 9.5]
        np.testing.assert_allclose(ctrl.compute(np.array([1.0, 2.0, 0.0])), [4.0, 9.5], rtol=1e-9)

    def test_obs_normalization_from_config(self, model_path):
        ctrl = _ctrl(model_path, observation={"normalize": True, "obs_mean": [1.0, 1.0, 1.0], "obs_std": [2.0, 2.0, 2.0]})
        # obs = [(3-1)/2, (5-1)/2, 0] = [1, 2, 0] -> [1.5, 3.5]
        np.testing.assert_allclose(ctrl.compute(np.array([3.0, 5.0, 1.0])), [1.5, 3.5], rtol=1e-9)

    def test_obs_clip_from_config(self, model_path):
        ctrl = _ctrl(model_path, observation={"clip": [-1.0, 1.0]})
        # obs[0] = min(5, 1) = 1 -> [1.5, -0.5]
        np.testing.assert_allclose(ctrl.compute(np.array([5.0, 0.0, 0.0])), [1.5, -0.5], rtol=1e-9)

    def test_state_keys_override(self, model_path):
        ctrl = _ctrl(model_path, observation={"state_keys": [1, 2, 0]})
        # obs = [x1, x2, x0] = [1, 0, 0] -> [1.5, -0.5]
        np.testing.assert_allclose(ctrl.compute(np.array([0.0, 1.0, 0.0])), [1.5, -0.5], rtol=1e-9)

    def test_n_x_beyond_observation_reach(self, model_path):
        """A state larger than the observed entries needs only n_x + state_keys."""
        ctrl = _ctrl(model_path, n_x=4, observation={"state_keys": [0, 1, 2]})
        # state is 4 long, obs reads the first three
        np.testing.assert_allclose(ctrl.compute(np.array([1.0, 2.0, 3.0, 9.0])), [1.5, 3.5], rtol=1e-9)

    def test_state_size_mismatch_raises(self, model_path):
        with pytest.raises(ValueError, match="expected a state of 3"):
            _ctrl(model_path).compute(np.array([1.0, 2.0]))

    def test_output_name_override(self, tmp_path):
        path = _save_model(_W_2OUT, _B_2OUT, tmp_path, output_name="action")
        ctrl = OnnxRLAdapter.from_config({"model_path": path, "output_name": "action"})
        np.testing.assert_allclose(ctrl.compute(np.array([1.0, 0.0, 0.0])), [1.5, -0.5], rtol=1e-9)

    def test_custom_input_name(self, tmp_path):
        path = _save_model(_W_2OUT, _B_2OUT, tmp_path, input_name="policy_in")
        ctrl = OnnxRLAdapter.from_config({"model_path": path, "observation": {"input_name": "policy_in"}})
        np.testing.assert_allclose(ctrl.compute(np.array([1.0, 0.0, 0.0])), [1.5, -0.5], rtol=1e-9)

    def test_wrong_input_name_override_rejected(self, model_path):
        with pytest.raises(ValueError, match="input_name"):
            OnnxRLAdapter.from_config({"model_path": str(model_path), "observation": {"input_name": "nope"}})

    def test_compute_accepts_list(self, model_path):
        np.testing.assert_allclose(_ctrl(model_path).compute([1.0, 2.0, 3.0]), [1.5, 3.5], rtol=1e-9)

    def test_compute_ignores_target(self, model_path):
        ctrl = _ctrl(model_path)
        a1 = ctrl.compute(np.array([1.0, 0.0, 0.0]))
        a2 = ctrl.compute(np.array([1.0, 0.0, 0.0]), target=np.array([9.0, 9.0]))
        np.testing.assert_allclose(a1, a2, rtol=0, atol=0)

    def test_action_dtype_is_float64(self, model_path):
        """The graph is f64 throughout, so the action is too (no f32 downcast)."""
        action = _ctrl(model_path).compute(np.array([1.0, 2.0, 3.0]))
        assert action.dtype == np.float64
        assert action.shape == (2,)

    def test_deterministic_action_has_no_noise_port(self, model_path):
        ctrl = _ctrl(model_path)
        assert ctrl.policy.noise_port is None
        np.testing.assert_allclose(ctrl.compute(np.array([1.0, 1.0, 1.0])), ctrl.compute(np.array([1.0, 1.0, 1.0])))


class TestDiscrete:
    def test_deterministic_one_hot(self, model_path):
        ctrl = _ctrl(model_path, action_space="discrete", deterministic=True)
        action = ctrl.compute(np.array([1.0, 1.0, 1.0]))  # logits [1.5, 1.5] -> first max
        np.testing.assert_allclose(action, [1.0, 0.0], rtol=0, atol=0)

    def test_deterministic_has_no_noise_port(self, model_path):
        assert _ctrl(model_path, action_space="discrete", deterministic=True).policy.noise_port is None

    def test_sampling_uses_epsilon_and_is_one_hot(self, model_path):
        ctrl = _ctrl(model_path, action_space="discrete", deterministic=False, seed=1)
        assert ctrl.policy.noise_port == "epsilon"
        assert ctrl.policy.gumbel
        action = ctrl.compute(np.array([1.0, 1.0, 1.0]))
        assert set(np.unique(action)) <= {0.0, 1.0}
        assert action.sum() == 1.0

    def test_sampling_reproducible_with_seed(self, model_path):
        cfg = {"model_path": str(model_path), "action_space": "discrete", "deterministic": False, "seed": 5}
        a1 = OnnxRLAdapter.from_config(cfg).compute(np.array([1.0, 1.0, 1.0]))
        a2 = OnnxRLAdapter.from_config(cfg).compute(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(a1, a2, rtol=0, atol=0)


class TestStochastic:
    def test_deterministic_returns_mean(self, stochastic_path):
        ctrl = _ctrl(stochastic_path, action_space="stochastic")
        # raw = [11, 22, 33, 4] -> mean = [11, 22]
        np.testing.assert_allclose(ctrl.compute(np.array([10.0, 20.0, 30.0])), [11.0, 22.0], rtol=1e-9)
        assert ctrl.policy.noise_port is None

    def test_deterministic_mean_gets_scale_bias(self, stochastic_path):
        ctrl = _ctrl(stochastic_path, action_space="stochastic", action_scale=2.0, action_bias=1.0)
        np.testing.assert_allclose(ctrl.compute(np.array([10.0, 20.0, 30.0])), [23.0, 45.0], rtol=1e-9)

    def test_sampling_reproducible_with_seed(self, stochastic_path):
        cfg = {"model_path": str(stochastic_path), "action_space": "stochastic", "deterministic": False, "seed": 3}
        a1 = OnnxRLAdapter.from_config(cfg).compute(np.array([0.0, 0.0, 0.0]))
        a2 = OnnxRLAdapter.from_config(cfg).compute(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(a1, a2, rtol=0, atol=0)
        assert a1.shape == (2,)

    def test_reset_reseeds_and_repeats(self, stochastic_path):
        """reset() must restore the stream, so the first draw repeats."""
        cfg = {"model_path": str(stochastic_path), "action_space": "stochastic", "deterministic": False, "seed": 7}
        ctrl = OnnxRLAdapter.from_config(cfg)
        first = ctrl.compute(np.array([0.0, 0.0, 0.0]))
        ctrl.compute(np.array([0.0, 0.0, 0.0]))  # advance the stream
        ctrl.reset()
        again = ctrl.compute(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(first, again, rtol=0, atol=0)

    def test_sampling_is_not_standard_normal_scale(self, stochastic_path):
        """The sampled action is mean + exp(clip(log_std, -10, 2)) * noise."""
        ctrl = _ctrl(stochastic_path, action_space="stochastic", deterministic=False, seed=0)
        # raw = [1, 2, 3, 4]; log_std clipped to 2 -> sigma = e^2; mean = [1, 2]
        u = ctrl.compute(np.array([0.0, 0.0, 0.0]))
        sigma = np.exp(2.0)
        assert np.all(np.abs(u - [1.0, 2.0]) <= 4.0 * sigma)


class TestBackendAgnostic:
    def test_torch_backend_returns_tensor(self, model_path):
        torch = pytest.importorskip("torch")
        from shinro.utils.array_backend import TorchBackend

        ctrl = OnnxRLAdapter.from_config({"model_path": str(model_path)}, backend=TorchBackend(device="cpu"))
        action = ctrl.compute(torch.tensor([1.0, 2.0, 3.0]))
        assert isinstance(action, torch.Tensor)
        torch.testing.assert_close(action, torch.tensor([1.5, 3.5], dtype=torch.float64))

    def test_torch_backend_discrete(self, model_path):
        torch = pytest.importorskip("torch")
        from shinro.utils.array_backend import TorchBackend

        cfg = {"model_path": str(model_path), "action_space": "discrete", "deterministic": True}
        ctrl = OnnxRLAdapter.from_config(cfg, backend=TorchBackend(device="cpu"))
        torch.testing.assert_close(ctrl.compute(torch.tensor([1.0, 1.0, 1.0])), torch.tensor([1.0, 0.0], dtype=torch.float64))

    def test_factory_passes_backend_through(self, model_path, tmp_path):
        torch = pytest.importorskip("torch")
        from shinro.factories.controller_factory import ControllerFactory
        from shinro.utils.array_backend import TorchBackend

        config = tmp_path / "rl.toml"
        config.write_text(f'type = "onnx_rl"\nmodel_path = "{model_path}"\naction_space = "continuous"\n')
        ctrl = ControllerFactory(str(config)).create(backend=TorchBackend(device="cpu"))
        assert isinstance(ctrl.compute(torch.tensor([1.0, 0.0, 0.0])), torch.Tensor)


class TestCompiledModeSurface:
    """Artifact-mode failures that need no compiled binary (the Zig oracle
    suite covers a real ``.so`` end-to-end)."""

    def _manifest(self, artifact_dir, *, inputs=None, outputs=None):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "inputs": inputs if inputs is not None else [{"name": "state", "shape": [3], "bytes": 24}],
            "outputs": outputs if outputs is not None else [{"name": "u", "shape": [2], "bytes": 16}],
            "state_outputs": [],
            "op_histogram": {"matmul": 1},
            "buf_len": 16,
        }
        (artifact_dir / "graph_data_manifest.json").write_text(json.dumps(manifest))
        return artifact_dir

    def test_expected_kernel_filename(self):
        """The adapter looks for the renamed NN kernel, not the generic libbase.so."""
        assert KERNEL_FILENAME == "lib_neural_network.so"

    def test_missing_manifest(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no graph manifest"):
            _CompiledPolicy(tmp_path / "nope")

    def test_missing_state_port(self, tmp_path):
        d = self._manifest(tmp_path / "art", inputs=[{"name": "y", "shape": [3], "bytes": 24}])
        with pytest.raises(ValueError, match="no 'state' input port"):
            _CompiledPolicy(d)

    def test_missing_u_port(self, tmp_path):
        d = self._manifest(tmp_path / "art", outputs=[{"name": "logits", "shape": [2], "bytes": 16}])
        with pytest.raises(ValueError, match="no 'u' output port"):
            _CompiledPolicy(d)

    def test_manifest_ok_but_no_binary(self, tmp_path):
        d = self._manifest(tmp_path / "art")
        with pytest.raises(FileNotFoundError, match="no compiled kernel"):
            _CompiledPolicy(d)

    def test_finds_kernel_under_expected_filename(self, tmp_path):
        """A file at lib/<KERNEL_FILENAME> must satisfy the existence check."""
        d = self._manifest(tmp_path / "art")
        (d / "lib").mkdir()
        (d / "lib" / KERNEL_FILENAME).write_bytes(b"")  # not a loadable .so
        with pytest.raises(OSError):  # got past the existence check to dlopen
            _CompiledPolicy(d)

    def test_corrupt_manifest(self, tmp_path):
        d = tmp_path / "art"
        d.mkdir()
        (d / "graph_data_manifest.json").write_text("{ not json")
        with pytest.raises(ValueError, match="corrupt"):
            _CompiledPolicy(d)
