"""Tests for the ONNX RL policy adapter."""

import numpy as np
import pytest

from shinro.controllers.onnx_rl_adapter import OnnxRLAdapter, _ObsEncoder

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")


def _build_model(input_name: str = "obs", output_name: str = "output"):
    """Build a tiny ONNX linear model: obs -> Gemm -> output."""
    import tempfile

    from onnx import TensorProto, helper

    w = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    b = np.array([0.5, -0.5], dtype=np.float32)
    X = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, 3])
    Y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, 2])
    w_init = helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten().tolist())
    b_init = helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.flatten().tolist())
    node = helper.make_node("Gemm", [input_name, "w", "b"], [output_name])
    graph = helper.make_graph([node], "g", [X], [Y], [w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
    onnx.save(model, path)
    return path


@pytest.fixture(scope="module")
def model_path():
    return _build_model()


def _make_encoder(
    *,
    input_name: str = "obs",
    state_keys: list[int] | None = None,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
    clip: tuple[float, float] | None = None,
    add_batch_dim: bool = True,
) -> _ObsEncoder:
    if state_keys is None:
        state_keys = [0, 1, 2]
    return _ObsEncoder(input_name, state_keys, obs_mean, obs_std, clip, add_batch_dim)


def test_obs_encoder_index_selection():
    enc = _make_encoder(state_keys=[2, 0])
    feed = enc.encode(np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(feed["obs"], [[30.0, 10.0]])


def test_obs_encoder_normalize_and_clip():
    enc = _make_encoder(obs_mean=np.array([1.0, 2.0, 3.0]), obs_std=np.array([2.0, 2.0, 2.0]), clip=(-1.0, 1.0))
    feed = enc.encode(np.array([10.0, 10.0, 10.0]))
    # (10-1)/2=4.5 clipped to 1.0
    np.testing.assert_allclose(feed["obs"], [[1.0, 1.0, 1.0]])


def test_obs_encoder_no_batch_dim():
    enc = _make_encoder(add_batch_dim=False)
    feed = enc.encode(np.array([1.0, 2.0, 3.0]))
    assert feed["obs"].shape == (3,)


class TestOnnxRLAdapter:
    def test_from_config_continuous(self, model_path, tmp_path):
        config = tmp_path / "rl.toml"
        config.write_text(f'type = "onnx_rl"\nmodel_path = "{model_path}"\naction_space = "continuous"\n')
        cfg = {"type": "onnx_rl", "model_path": str(model_path), "action_space": "continuous"}
        ctrl = OnnxRLAdapter.from_config(cfg)
        action = ctrl.compute(np.array([1.0, 2.0, 3.0]))
        assert action.shape == (2,)
        expected = np.array([1.0 * 1.0 + 0.5, 2.0 * 2.0 - 0.5])
        np.testing.assert_allclose(action, expected)

    def test_continuous_action_scale_bias_clip(self, model_path):
        cfg = {
            "model_path": str(model_path),
            "action_space": "continuous",
            "action_scale": 2.0,
            "action_bias": 1.0,
            "action_clip_low": -3.0,
            "action_clip_high": 3.0,
        }
        ctrl = OnnxRLAdapter.from_config(cfg)
        action = ctrl.compute(np.array([1.0, 1.0, 0.0]))
        # w[:,0] = [1,0,0], bias 0.5 -> (1*2+1)=3, clipped to 3.0
        np.testing.assert_allclose(action, [3.0, 3.0])

    def test_action_space_invalid(self, model_path):
        with pytest.raises(ValueError, match="action_space"):
            OnnxRLAdapter.from_config({"model_path": str(model_path), "action_space": "bogus"})

    def test_reset_reseeds(self, model_path):
        cfg = {"model_path": str(model_path), "action_space": "stochastic", "deterministic": False, "seed": 7}
        ctrl = OnnxRLAdapter.from_config(cfg)
        ctrl.compute(np.array([0.0, 0.0, 0.0]))
        ctrl.reset()
        ctrl2 = OnnxRLAdapter.from_config(cfg)
        a1 = ctrl.compute(np.array([0.0, 0.0, 0.0]))
        a2 = ctrl2.compute(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(a1, a2)


class TestDiscreteActionSpace:
    def test_deterministic_argmax(self, model_path):
        cfg = {"model_path": str(model_path), "action_space": "discrete", "deterministic": True}
        ctrl = OnnxRLAdapter.from_config(cfg)
        # output for obs [1,1,1]: [1.5, 1.5] -> argmax=0 (first max)
        action = ctrl.compute(np.array([1.0, 1.0, 1.0]))
        assert action.shape == (2,)
        assert action.dtype == np.float32
        assert action[0] == 1.0 and action[1] == 0.0

    def test_stochastic_samples_one_hot(self, model_path):
        cfg = {"model_path": str(model_path), "action_space": "discrete", "deterministic": False, "seed": 1}
        ctrl = OnnxRLAdapter.from_config(cfg)
        action = ctrl.compute(np.array([1.0, 1.0, 1.0]))
        assert set(np.unique(action)) <= {0.0, 1.0}
        assert action.sum() == 1.0


class TestStochasticActionSpace:
    def test_deterministic_returns_mean(self, model_path):
        cfg = {"model_path": str(model_path), "action_space": "stochastic", "deterministic": True}
        ctrl = OnnxRLAdapter.from_config(cfg)
        action = ctrl.compute(np.array([1.0, 2.0, 3.0]))
        # mean part = same as continuous output
        expected = np.array([1.0 * 1.0 + 0.5, 2.0 * 2.0 - 0.5])
        np.testing.assert_allclose(action, expected)

    def test_sample_reproducible(self, model_path):
        cfg = {"model_path": str(model_path), "action_space": "stochastic", "deterministic": False, "seed": 5}
        ctrl = OnnxRLAdapter.from_config(cfg)
        ctrl2 = OnnxRLAdapter.from_config(cfg)
        a1 = ctrl.compute(np.array([0.0, 0.0, 0.0]))
        a2 = ctrl2.compute(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(a1, a2)


class TestFromConfigSurface:
    """Cover the config-parsing branches in from_config()."""

    def test_default_action_space_continuous(self, model_path):
        """No action_space field -> defaults to continuous."""
        ctrl = OnnxRLAdapter.from_config({"model_path": str(model_path)})
        assert ctrl.action_space == "continuous"

    def test_default_state_keys_from_input_shape(self, model_path):
        """No state_keys -> defaults to all input dims, so compute works."""
        ctrl = OnnxRLAdapter.from_config({"model_path": str(model_path)})
        action = ctrl.compute(np.array([1.0, 2.0, 3.0]))
        expected = np.array([1.5, 3.5])
        np.testing.assert_allclose(action, expected)

    def test_custom_input_output_names(self, tmp_path):
        """Non-default ONNX I/O tensor names are honored."""
        path = _build_model(input_name="policy_in", output_name="policy_out")
        ctrl = OnnxRLAdapter.from_config({"model_path": path, "action_space": "continuous"})
        action = ctrl.compute(np.array([1.0, 0.0, 0.0]))
        # obs[0]=1 -> w[0]=1, bias 0.5 -> [1.5, -0.5]
        np.testing.assert_allclose(action, [1.5, -0.5])

    def test_custom_input_name_override(self, model_path):
        """observation.input_name overrides the session's input name."""
        ctrl = OnnxRLAdapter.from_config(
            {"model_path": str(model_path), "observation": {"input_name": "obs", "state_keys": [1, 2, 0]}}
        )
        action = ctrl.compute(np.array([0.0, 1.0, 0.0]))
        # obs = [1,0,0] -> [1.5, -0.5]
        np.testing.assert_allclose(action, [1.5, -0.5])

    def test_output_name_override(self, model_path):
        """output_name field selects a different ONNX output."""
        ctrl = OnnxRLAdapter.from_config({"model_path": str(model_path), "output_name": "output"})
        action = ctrl.compute(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(action, [1.5, -0.5])

    def test_obs_normalization_from_config(self, model_path):
        """observation.normalize applies mean/std from config."""
        ctrl = OnnxRLAdapter.from_config(
            {
                "model_path": str(model_path),
                "observation": {"normalize": True, "obs_mean": [1.0, 1.0, 1.0], "obs_std": [2.0, 2.0, 2.0]},
            }
        )
        action = ctrl.compute(np.array([3.0, 5.0, 1.0]))
        # obs = [(3-1)/2, (5-1)/2, 0] = [1,2,0] -> [1.5, 3.5]
        np.testing.assert_allclose(action, [1.5, 3.5])

    def test_obs_clip_from_config(self, model_path):
        """observation.clip clamps observations."""
        ctrl = OnnxRLAdapter.from_config({"model_path": str(model_path), "observation": {"clip": [-1.0, 1.0]}})
        action = ctrl.compute(np.array([5.0, 0.0, 0.0]))
        # obs[0] clipped to 1.0 -> [1.5, -0.5]
        np.testing.assert_allclose(action, [1.5, -0.5])

    def test_action_clip_one_sided(self, model_path):
        """action_clip_low alone -> clip at inf high bound."""
        ctrl = OnnxRLAdapter.from_config({"model_path": str(model_path), "action_clip_low": 2.0})
        action = ctrl.compute(np.array([1.0, 0.0, 0.0]))
        # raw = [1.5, -0.5]; low-clip to 2.0
        np.testing.assert_allclose(action, [2.0, 2.0])

    def test_action_clip_high_only(self, model_path):
        """action_clip_high alone -> clip at -inf low bound."""
        ctrl = OnnxRLAdapter.from_config({"model_path": str(model_path), "action_clip_high": -1.0})
        action = ctrl.compute(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(action, [-1.0, -1.0])

    def test_missing_stats_with_normalize_raises(self, model_path):
        """normalize=true without obs_mean/obs_std raises KeyError."""
        with pytest.raises(KeyError):
            OnnxRLAdapter.from_config({"model_path": str(model_path), "observation": {"normalize": True}})


class TestPostprocessDirect:
    """Exercise _postprocess branches that the stub model cannot reach."""

    def _ctrl(self, **overrides):
        cfg = {"model_path": str(_build_model()), "action_space": "continuous"}
        cfg.update(overrides)
        return OnnxRLAdapter.from_config(cfg)

    def test_discrete_reshape_flat(self):
        """Multi-dimensional logits are flattened before argmax."""
        ctrl = self._ctrl(action_space="discrete", deterministic=True)
        action = ctrl._postprocess(np.array([[5.0], [2.0]]))
        assert action.shape == (2,)
        assert action[0] == 1.0 and action[1] == 0.0

    def test_discrete_extreme_logits(self):
        """Stochastic discrete with extreme logits doesn't overflow (max subtraction)."""
        ctrl = self._ctrl(action_space="discrete", deterministic=False, seed=0)
        action = ctrl._postprocess(np.array([1e6, 0.0]))
        assert action[0] == 1.0 and action[1] == 0.0

    def test_discrete_deterministic_tie_picks_first(self):
        ctrl = self._ctrl(action_space="discrete", deterministic=True)
        action = ctrl._postprocess(np.array([1.0, 1.0]))
        assert action[0] == 1.0 and action[1] == 0.0

    def test_stochastic_logstd_clamped(self):
        """Out-of-range log_std is clamped to [-10, 2]."""
        ctrl = self._ctrl(action_space="stochastic", deterministic=False, seed=3)
        # mean=0, log_std=100 -> clamped to 2 -> sigma ~7.39 -> sample stays ~O(20),
        # not exp(100) ~ 2.7e43
        raw = np.array([0.0, 0.0, 100.0, 100.0])
        u = ctrl._postprocess(raw)
        assert np.all(np.abs(u) <= 40.0)
        # log_std=-100 -> clamped to -10 -> sigma ~4.5e-5 -> sample ~mean
        u2 = ctrl._postprocess(np.array([5.0, 5.0, -100.0, -100.0]))
        np.testing.assert_allclose(u2, [5.0, 5.0], atol=1e-3)

    def test_stochastic_mean_passthrough_astype(self):
        """Deterministic stochastic postprocess returns mean as float32."""
        ctrl = self._ctrl(action_space="stochastic", deterministic=True, action_scale=2.0, action_bias=1.0)
        u = ctrl._postprocess(np.array([1.0, 2.0, -3.0, -3.0]))
        assert u.dtype == np.float32
        np.testing.assert_allclose(u, [3.0, 5.0])

    def test_continuous_astype_float32(self):
        ctrl = self._ctrl()
        action = ctrl._postprocess(np.array([1.0, 2.0], dtype=np.float64))
        assert action.dtype == np.float32

    def test_batched_compute_shapes(self):
        """Raw ONNX output with batch dim is flattened back to (n_u,)."""
        ctrl = self._ctrl(action_space="continuous")
        action = ctrl.compute(np.array([1.0, 2.0, 3.0]))
        assert action.shape == (2,)
        np.testing.assert_allclose(action, [1.5, 3.5])

    def test_compute_list_input(self):
        """compute accepts a plain Python list as state."""
        ctrl = self._ctrl(action_space="continuous")
        action = ctrl.compute([1.0, 2.0, 3.0])
        np.testing.assert_allclose(action, [1.5, 3.5])

    def test_compute_ignores_target(self):
        """target is ignored for learned policies."""
        ctrl = self._ctrl(action_space="continuous")
        a1 = ctrl.compute(np.array([1.0, 0.0, 0.0]))
        a2 = ctrl.compute(np.array([1.0, 0.0, 0.0]), target=np.array([9.0, 9.0]))
        np.testing.assert_allclose(a1, a2)


class TestBackendAgnostic:
    """Verify the adapter converts at the ONNX boundary, not in the framework."""

    def test_torch_backend_returns_tensor(self, model_path):
        torch = pytest.importorskip("torch")
        from shinro.utils.array_backend import TorchBackend

        bk = TorchBackend(device="cpu")
        ctrl = OnnxRLAdapter.from_config({"model_path": str(model_path), "action_space": "continuous"}, backend=bk)
        action = ctrl.compute(torch.tensor([1.0, 2.0, 3.0]))
        assert isinstance(action, torch.Tensor)
        expected = torch.tensor([1.5, 3.5])
        torch.testing.assert_close(action, expected)

    def test_torch_backend_observations_normalized(self, model_path):
        torch = pytest.importorskip("torch")
        from shinro.utils.array_backend import TorchBackend

        bk = TorchBackend(device="cpu")
        ctrl = OnnxRLAdapter.from_config(
            {"model_path": str(model_path), "observation": {"normalize": True, "obs_mean": [1.0, 1.0, 1.0], "obs_std": [2.0, 2.0, 2.0]}},
            backend=bk,
        )
        action = ctrl.compute(torch.tensor([3.0, 5.0, 1.0]))
        # obs = [(3-1)/2, (5-1)/2, 0] = [1,2,0] -> [1.5, 3.5]
        torch.testing.assert_close(action, torch.tensor([1.5, 3.5]))

    def test_torch_backend_discrete(self, model_path):
        torch = pytest.importorskip("torch")
        from shinro.utils.array_backend import TorchBackend

        bk = TorchBackend(device="cpu")
        ctrl = OnnxRLAdapter.from_config(
            {"model_path": str(model_path), "action_space": "discrete", "deterministic": True}, backend=bk
        )
        action = ctrl.compute(torch.tensor([1.0, 1.0, 1.0]))
        assert isinstance(action, torch.Tensor)
        torch.testing.assert_close(action, torch.tensor([1.0, 0.0]))

    def test_factory_passthrough_backend(self, model_path, tmp_path):
        """ControllerFactory passes the backend through from_config."""
        torch = pytest.importorskip("torch")
        from shinro.factories.controller_factory import ControllerFactory
        from shinro.utils.array_backend import TorchBackend

        config = tmp_path / "rl.toml"
        config.write_text(f'type = "onnx_rl"\nmodel_path = "{model_path}"\naction_space = "continuous"\n')
        factory = ControllerFactory(str(config))
        bk = TorchBackend(device="cpu")
        ctrl = factory.create(backend=bk)
        action = ctrl.compute(torch.tensor([1.0, 0.0, 0.0]))
        assert isinstance(action, torch.Tensor)
