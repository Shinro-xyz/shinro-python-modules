"""ONNX RL policy adapter — wraps any ONNX-exported reinforcement-learning policy as a Controller.

Allows swapping between classical control (LQR, MPC) and policies trained in
*any* external RL stack (Stable-Baselines3, RLlib, CleanRL, custom PyTorch,
JAX/Flax, ...) provided the actor network is exported to ONNX. Inference runs
through ``onnxruntime`` — no torch / gym / framework dependency at deploy time.

The adapter supports three action-space conventions:

- ``continuous``: output is a real-valued action vector, optionally scaled and
  biased for tanh-squashed policies (``u = scale * tanh(a) + bias``).
- ``discrete``: output is a logits vector; greedy ``argmax`` by default, or
  sampled from ``softmax`` when ``deterministic = false``.
- ``stochastic``: output is ``[mean; log_std]``; the mean is used when
  ``deterministic = true``, otherwise a Gaussian sample is drawn with the
  configured seed.

Observations are built from the flat plant state via integer index selection
plus optional per-dimension normalization and clipping. The adapter is
backend-agnostic: state may arrive as a numpy array or torch tensor, and the
action is returned in the same backend's native type (conversion happens at
the ONNX boundary, which requires numpy feeds).

Usage:
    # In configs/controllers/onnx_rl.toml:
    #   type = "onnx_rl"
    #   model_path = "path/to/policy.onnx"
    #   action_space = "continuous"
    #
    # python -m demos.demo_base_tracking --controller onnx_rl
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shinro.components import Controller
from shinro.factories.registry import register_controller
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


class _ObsEncoder:
    """Config-driven observation encoder: plant state -> ONNX feed dict.

    Accepts any backend-native state (numpy array or torch tensor), converts
    it to numpy at the ONNX boundary via ``bk.to_numpy``, subselects integer
    indices, optionally applies per-dimension normalization
    ``(x - mean) / std`` and clipping, then packs the result into a
    batch-ready float32 array for a single ONNX input.
    """

    def __init__(
        self,
        input_name: str,
        state_keys: list[int],
        obs_mean: np.ndarray | None = None,
        obs_std: np.ndarray | None = None,
        clip: tuple[float, float] | None = None,
        add_batch_dim: bool = True,
        backend: ArrayBackend | None = None,
    ) -> None:
        self.input_name = input_name
        self.state_keys = np.asarray(state_keys, dtype=int)
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.clip = clip
        self.add_batch_dim = add_batch_dim
        self.bk = backend or NumpyBackend()

    def encode(self, state: Any) -> dict[str, np.ndarray]:
        """Convert a backend-native plant state into an ``{input_name: tensor}`` feed dict."""
        s = self.bk.to_numpy(state)
        obs = np.asarray(s, dtype=np.float32)[self.state_keys].astype(np.float32, copy=True)
        if self.obs_mean is not None:
            obs = obs - self.obs_mean
        if self.obs_std is not None:
            obs = obs / self.obs_std
        if self.clip is not None:
            obs = np.clip(obs, *self.clip)
        if self.add_batch_dim:
            obs = obs[None, :]
        return {self.input_name: obs}


@register_controller("onnx_rl")
class OnnxRLAdapter(Controller):
    """Wrap an ONNX-exported RL policy as a Controller.

    The policy is loaded from a local ``.onnx`` file via ``onnxruntime``.
    ``compute()`` encodes the plant state (normalization / clipping / index
    selection), runs the model, post-processes the raw output into a control
    action, and returns it as a numpy array.

    Args:
        session: Loaded ``onnxruntime.InferenceSession``.
        obs_encoder: Encoder mapping plant state to the ONNX feed dict.
        output_name: ONNX output tensor name.
        action_space: ``"continuous"``, ``"discrete"``, or ``"stochastic"``.
        deterministic: For discrete/stochastic policies, return the greedy
            argmax / mean instead of sampling (default: ``True``).
        action_scale: Post-policy per-action scaling (tanh-squash support).
        action_bias: Post-policy per-action bias.
        action_clip: Optional ``(low, high)`` tuple to clip the final action.
        seed: RNG seed for sampling action spaces.
        backend: Array backend for state input and action output. ONNX
            inference itself always runs on numpy arrays, but the adapter
            converts the backend-native state to numpy at the boundary and
            the resulting action back to the backend's native type.
    """

    def __init__(
        self,
        weights,
        obs_encoder: _ObsEncoder,
        output_name: str,
        action_space: str = "continuous",
        deterministic: bool = True,
        action_scale: float | np.ndarray = 1.0,
        action_bias: float | np.ndarray = 0.0,
        action_clip: tuple[float, float] | None = None,
        seed: int = 0,
        backend: ArrayBackend | None = None,
    ) -> None:
        if action_space not in ("continuous", "discrete", "stochastic"):
            raise ValueError(f"action_space must be continuous/discrete/stochastic, got {action_space!r}")
        self.session = weights
        self.obs_encoder = obs_encoder
        self.output_name = output_name
        self.action_space = action_space
        self.deterministic = deterministic
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.action_bias = np.asarray(action_bias, dtype=np.float32)
        self.action_clip = action_clip
        self.seed = seed
        self.bk = backend or NumpyBackend()
        self._rng = np.random.default_rng(seed)

    def _postprocess(self, raw: np.ndarray) -> np.ndarray:
        """Turn raw network output into a control action."""
        raw = raw.astype(np.float32, copy=False)

        if self.action_space == "continuous":
            action = raw * self.action_scale + self.action_bias
        elif self.action_space == "discrete":
            logits = raw.reshape(-1)
            if self.deterministic:
                action = np.zeros_like(logits)
                action[np.argmax(logits)] = 1.0
            else:
                probs = np.exp(logits - np.max(logits))
                probs = probs / probs.sum()
                idx = self._rng.choice(len(logits), p=probs)
                action = np.zeros_like(logits)
                action[idx] = 1.0
        else:  # stochastic: raw = [mean; log_std]
            half = raw.size // 2
            mean = raw[:half]
            if self.deterministic:
                u = mean
            else:
                log_std = np.clip(raw[half:], -10.0, 2.0)
                u = mean + np.exp(log_std) * self._rng.standard_normal(mean.size)
            u = u * self.action_scale + self.action_bias
            if self.action_clip is not None:
                u = np.clip(u, *self.action_clip)
            return u.astype(np.float32, copy=False)

        if self.action_clip is not None:
            action = np.clip(action, *self.action_clip)
        return action

    def compute(self, state, target=None):
        """Run the ONNX policy on the current state.

        Args:
            state: Plant state vector in the configured backend's native
                type (numpy array or torch tensor).
            target: Ignored for learned policies — they generate actions
                from observation alone.

        Returns:
            Action vector (n_u,) in the backend-native type.
        """
        feed = self.obs_encoder.encode(state)
        raw = self.session.run([self.output_name], feed)[0]
        action = self._postprocess(raw).reshape(-1)
        return self.bk.from_numpy(action)

    def reset(self):
        """Reset the policy RNG to the configured seed."""
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an OnnxRLAdapter from a TOML config dict.

        Config fields:
            model_path: Path to the ``.onnx`` model file (required).
            action_space: ``"continuous"``, ``"discrete"``, or ``"stochastic"``
                (default: ``"continuous"``).
            deterministic: Whether to return argmax/mean instead of sampling
                (default: ``true``).
            action_scale: Post-policy action scale (default: 1.0).
            action_bias: Post-policy action bias (default: 0.0).
            action_clip_low / action_clip_high: Clip the final action
                (default: no clipping).
            seed: RNG seed for stochastic sampling (default: 0).

        ``[observation]`` subtable fields:
            input_name: ONNX input tensor name (defaults to the model's first
                input).
            state_keys: Integer indices into the plant state to use as
                observations (default: ``[0, 1, ..., n-1]``).
            normalize: Apply mean/std normalization (default: false).
            obs_mean / obs_std: Arrays for normalization.
            clip: ``[low, high]`` observation clipping (default: none).
            add_batch_dim: Prepend a batch axis (default: true).

        Args:
            config: TOML config dict.
            backend: Array backend for state input and action output.
                Defaults to NumpyBackend.

        Returns:
            OnnxRLAdapter instance.
        """
        import onnxruntime  # type: ignore

        bk = backend or NumpyBackend()

        session = onnxruntime.InferenceSession(config["model_path"], providers=["CPUExecutionProvider"])
        output_name = config.get("output_name")
        if output_name is None:
            output_name = session.get_outputs()[0].name

        obs_cfg = config.get("observation", {})
        input_name = obs_cfg.get("input_name")
        if input_name is None:
            input_name = session.get_inputs()[0].name

        state_keys = obs_cfg.get("state_keys")
        if state_keys is None:
            state_keys = list(range(session.get_inputs()[0].shape[1] or 0))

        obs_mean = None
        obs_std = None
        if obs_cfg.get("normalize", False):
            obs_mean = np.asarray(obs_cfg["obs_mean"], dtype=np.float32)
            obs_std = np.asarray(obs_cfg["obs_std"], dtype=np.float32)

        obs_clip = None
        if "clip" in obs_cfg:
            obs_clip = (float(obs_cfg["clip"][0]), float(obs_cfg["clip"][1]))

        encoder = _ObsEncoder(
            input_name=input_name,
            state_keys=state_keys,
            obs_mean=obs_mean,
            obs_std=obs_std,
            clip=obs_clip,
            add_batch_dim=obs_cfg.get("add_batch_dim", True),
        )

        action_clip = None
        if "action_clip_low" in config or "action_clip_high" in config:
            action_clip = (float(config.get("action_clip_low", -np.inf)), float(config.get("action_clip_high", np.inf)))

        return cls(
            weights=session,
            obs_encoder=encoder,
            output_name=output_name,
            action_space=config.get("action_space", "continuous"),
            deterministic=config.get("deterministic", True),
            action_scale=config.get("action_scale", 1.0),
            action_bias=config.get("action_bias", 0.0),
            action_clip=action_clip,
            seed=config.get("seed", 0),
            backend=bk,
        )
