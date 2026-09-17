"""ONNX RL policy adapter — run an ONNX-exported policy as a Controller.

``onnxruntime`` is gone. The policy's ONNX graph is translated into a shinro
graph by :mod:`shinro.codegen.onnx_import` — observation encoding, the network,
and the action post-processing all become arithmetic on baked constants — and
that graph is executed through one of two interchangeable backends:

- **eager** (``model_path``): the imported graph is run in-process by
  :func:`shinro.codegen.interpreter.interpret` (pure numpy, f64). No build step
  and no compiled artifact are needed; this is the testing/reference path.
- **compiled** (``artifact_dir``): the scenario's compiled kernel
  (``lib/lib_neural_network.so``) is dlopen'd and driven through the
  ``shinro_step`` C ABI, with the graph manifest next to it describing the port
  layout. This is the deployment path: no Python array framework, no ONNX
  runtime, no dependencies at all.

Both backends see the same graph, so they agree bit-for-bit (the compile gate
checks exactly that). The only input port is the raw plant state; a sampling
action space adds an ``epsilon`` port that the host fills with noise each tick
(Gumbel for ``discrete``, standard normal for ``stochastic``) — the kernel does
the arithmetic, RNG stays on the host, matching MPPI's contract.

Action spaces (baked at import time, mirroring the historical runtime):

- ``continuous``: ``u = scale * a + bias`` (optionally clipped).
- ``discrete``: argmax one-hot (deterministic) or Gumbel-max sampling.
- ``stochastic``: ``[mean; log_std]`` — the mean, or
  ``mean + exp(clip(log_std, -10, 2)) * epsilon``, then ``scale * u + bias``.

The host-side noise is drawn from a seeded generator; :meth:`reset` reseeds it,
so a run is reproducible.

Usage (configs/controllers/onnx_rl.toml)::

    #   type = "onnx_rl"
    #   model_path = "path/to/policy.onnx"     # eager mode
    #   # artifact_dir = "build/my_policy"     # compiled mode (make compile --out)
    #   action_space = "continuous"
"""

from __future__ import annotations

import ctypes
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shinro.components import Controller
from shinro.factories.registry import register_controller
from shinro.utils.array_backend import ArrayBackend, NumpyBackend

#: Clamp for the ``U(0, 1)`` draws feeding the Gumbel transform, so a draw of
#: exactly 0 cannot produce ``-inf`` noise.
_GUMBEL_EPS = 1e-12

#: Filename the compiled policy kernel must be installed as under
#: ``<artifact_dir>/lib/`` (the ``make compile`` output for this scenario).
KERNEL_FILENAME = "lib_neural_network.so"


@dataclass(frozen=True)
class OnnxRLConfig:
    """Strict TOML schema for :class:`OnnxRLAdapter`.

    Exactly one of ``model_path`` / ``artifact_dir`` must be given: the former
    imports and interprets the ONNX graph in-process, the latter loads an
    already-compiled kernel. ``artifact_dir`` wins if both are set, so a config
    can keep the model path for provenance while deploying the ``.so``.

    The observation sub-table is left as a plain dict because its keys map
    straight onto the importer's ``obs_cfg`` (which validates them); every other
    field is the same action-space surface the old adapter exposed.
    """

    model_path: str | None = None
    artifact_dir: str | None = None
    n_x: int | None = None
    output_name: str | None = None
    action_space: str = "continuous"
    deterministic: bool = True
    action_scale: Any = 1.0
    action_bias: Any = 0.0
    action_clip_low: float | None = None
    action_clip_high: float | None = None
    seed: int = 0
    observation: dict[str, Any] = field(default_factory=dict)
    name: str = "onnx_rl"


class _GraphPolicy:
    """Eager artifact: an imported shinro graph executed by the interpreter."""

    def __init__(self, cg) -> None:
        """Wrap a composed graph as a runnable policy.

        Args:
            cg: The :class:`~shinro.codegen.compose.ComposedGraph` returned by
                :func:`shinro.codegen.onnx_import.import_onnx_policy`.
        """
        from shinro.codegen.onnx_import import EPSILON_PORT, STATE_PORT

        self._cg = cg
        self.inputs = list(cg.inputs)
        self.ops = frozenset(node.op for node in cg.graph.nodes)
        self._u_port = cg.outputs[0]
        self.state_port = STATE_PORT
        self.state_size = _graph_port_size(cg.graph, STATE_PORT)
        self.noise_port = EPSILON_PORT if EPSILON_PORT in self.inputs else None
        self.noise_size = _graph_port_size(cg.graph, EPSILON_PORT) if self.noise_port else 0

    @property
    def gumbel(self) -> bool:
        """True when the graph expects Gumbel noise (i.e. it samples discretely)."""
        return "one_hot" in self.ops

    def step(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        """Run one tick through the interpreter."""
        from shinro.codegen.interpreter import interpret

        return interpret(self._cg.graph, feed)[self._u_port]


class _CompiledPolicy:
    """Deployment artifact: a compiled policy kernel plus its graph manifest.

    The manifest (written by :func:`shinro.codegen.lower_zig.lower_zig` next to
    the graph) is the artifact's self-description, so the loader reads the port
    order, shapes, and op histogram from it rather than from the original ONNX
    model — the ``.onnx`` file is not needed at deploy time.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        """Load ``<artifact_dir>/lib/lib_neural_network.so`` and its manifest.

        Args:
            artifact_dir: A ``make compile --out`` directory (contains
                ``graph_data_manifest.json`` and ``lib/lib_neural_network.so``).

        Raises:
            FileNotFoundError: If the manifest or the shared object is missing.
            ValueError: If the artifact does not expose the expected ``state``
                input and ``u`` output ports.
        """
        from shinro.codegen.onnx_import import EPSILON_PORT, OUTPUT_PORT, STATE_PORT

        root = Path(artifact_dir)
        manifest_path = root / "graph_data_manifest.json"
        so_path = root / "lib" / KERNEL_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no graph manifest at {manifest_path} — run `make compile --out {root}` first")

        try:
            self.manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"graph manifest at {manifest_path} is unreadable or corrupt: {exc}") from exc
        self.inputs = [port["name"] for port in self.manifest["inputs"]]
        self.ops = frozenset(self.manifest["op_histogram"])
        if STATE_PORT not in self.inputs:
            raise ValueError(f"artifact {root} has no '{STATE_PORT}' input port (inputs: {self.inputs})")

        out_names = [port["name"] for port in self.manifest["outputs"]]
        if OUTPUT_PORT not in out_names:
            raise ValueError(f"artifact {root} has no '{OUTPUT_PORT}' output port (outputs: {out_names})")

        self._in_sizes = [_flat_size(port["shape"]) for port in self.manifest["inputs"]]
        out_sizes = [_flat_size(port["shape"]) for port in self.manifest["outputs"]]
        self._n_out = sum(out_sizes)
        self._n_state = sum(_flat_size(port["shape"]) for port in self.manifest["state_outputs"])
        u_index = out_names.index(OUTPUT_PORT)
        u_start = sum(out_sizes[:u_index])
        self._u_slice = (u_start, u_start + out_sizes[u_index])

        self.state_port = STATE_PORT
        self.state_size = _flat_size(self.manifest["inputs"][self.inputs.index(STATE_PORT)]["shape"])
        self.noise_port = EPSILON_PORT if EPSILON_PORT in self.inputs else None
        self.noise_size = _flat_size(self.manifest["inputs"][self.inputs.index(EPSILON_PORT)]["shape"]) if self.noise_port else 0

        if not so_path.exists():
            raise FileNotFoundError(f"no compiled kernel at {so_path} — run `make compile --out {root}` first")
        lib = ctypes.CDLL(str(so_path))
        lib.shinro_step.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3
        lib.shinro_step.restype = None
        self._lib = lib

    @property
    def gumbel(self) -> bool:
        """True when the compiled graph expects Gumbel noise (it samples discretely)."""
        return "one_hot" in self.ops

    def step(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        """Pack the ports, call ``shinro_step``, and return the ``u`` slice."""
        packed = np.concatenate([np.asarray(feed[name], dtype=np.float64).ravel() for name in self.inputs])
        out = np.zeros(self._n_out, dtype=np.float64)
        # A memoryless policy declares no state outputs; the C ABI still wants a
        # non-null pointer, so give it a one-element scratch buffer.
        state = np.zeros(max(self._n_state, 1), dtype=np.float64)
        ptr = ctypes.POINTER(ctypes.c_double)
        self._lib.shinro_step(
            packed.ctypes.data_as(ptr),
            out.ctypes.data_as(ptr),
            state.ctypes.data_as(ptr),
        )
        start, stop = self._u_slice
        return out[start:stop].copy()


@register_controller("onnx_rl")
class OnnxRLAdapter(Controller):
    """Run an ONNX-exported RL policy as a Controller.

    The policy is prepared once (imported + interpreted, or a compiled ``.so``
    is loaded) and each :meth:`compute` call runs one tick on the plant state.

    Args:
        policy: A loaded policy artifact — :class:`_GraphPolicy` (eager) or
            :class:`_CompiledPolicy` (compiled). Built by :meth:`from_config`.
        seed: RNG seed for action sampling. Sampling action spaces (discrete
            non-deterministic, stochastic non-deterministic) draw their noise
            from a generator seeded here.
        backend: Array backend for the state input and action output. The
            kernel itself always works on numpy f64; the adapter converts at the
            boundary, so a torch state yields a torch action.
    """

    Config = OnnxRLConfig

    def __init__(self, policy: _GraphPolicy | _CompiledPolicy, *, seed: int = 0, backend: ArrayBackend | None = None) -> None:
        self.policy = policy
        self.seed = seed
        self.bk = backend or NumpyBackend()
        self._rng = np.random.default_rng(self.seed)

    def compute(self, state, target=None):
        """Run the policy on the current plant state.

        Args:
            state: Plant state vector in the configured backend's native type
                (numpy array, torch tensor, or a sequence). Must have the
                compiled graph's ``state`` port length.
            target: Ignored — learned policies act on the observation alone.

        Returns:
            The action vector (n_u,) in the backend's native type.
        """
        x = np.asarray(self.bk.to_numpy(state), dtype=np.float64).ravel()
        if x.size != self.policy.state_size:
            raise ValueError(f"onnx_rl: expected a state of {self.policy.state_size} elements, got {x.size}")
        feed = {self.policy.state_port: x}
        if self.policy.noise_port is not None:
            feed[self.policy.noise_port] = self._draw_noise(self.policy.noise_size)
        return self.bk.from_numpy(self.policy.step(feed))

    def _draw_noise(self, size: int) -> np.ndarray:
        """Draw the host noise the epsilon port expects.

        For a discretely-sampling graph this is Gumbel noise, which makes
        ``argmax(logits + g)`` an exact categorical draw from
        ``softmax(logits)``; otherwise it is a standard normal.
        """
        if self.policy.gumbel:
            uniform = np.clip(self._rng.uniform(0.0, 1.0, size=size), _GUMBEL_EPS, 1.0 - _GUMBEL_EPS)
            return -np.log(-np.log(uniform))
        return self._rng.standard_normal(size)

    def reset(self):
        """Reseed the action-sampling RNG (a fresh run is reproducible)."""
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an OnnxRLAdapter from a TOML config dict or :class:`OnnxRLConfig`.

        Config fields:
            model_path: Path to the ``.onnx`` model (eager mode).
            artifact_dir: A ``make compile --out`` directory (compiled mode).
            action_space: ``"continuous"``, ``"discrete"``, or ``"stochastic"``
                (default: continuous).
            deterministic: Return argmax/mean instead of sampling (default true).
            action_scale / action_bias: Post-policy affine transform.
            action_clip_low / action_clip_high: Clip the final action; both are
                required together (a missing bound would be ``±inf``, which the
                lowerer cannot represent).
            seed: RNG seed for sampling action spaces.
            n_x: Plant state dimension, when the observation sub-table does not
                reach the last state entry.
            output_name: ONNX tensor to import as the action.
            observation: The importer's observation table (``state_keys``,
                ``normalize``, ``obs_mean``, ``obs_std``, ``clip``, ...).

        Args:
            config: TOML config dict or :class:`OnnxRLConfig`.
            backend: Array backend for state input and action output.

        Returns:
            OnnxRLAdapter instance.

        Raises:
            ValueError: On a missing/invalid mode or an invalid action config.
            FileNotFoundError: If a compiled artifact is missing.
        """
        cfg = cls.parse_config(config)
        bk = backend or NumpyBackend()

        if cfg.artifact_dir is not None:
            policy: _GraphPolicy | _CompiledPolicy = _CompiledPolicy(cfg.artifact_dir)
        elif cfg.model_path is not None:
            from shinro.codegen.onnx_import import import_onnx_policy

            action_cfg: dict[str, Any] = {
                "action_space": cfg.action_space,
                "deterministic": cfg.deterministic,
                "action_scale": cfg.action_scale,
                "action_bias": cfg.action_bias,
            }
            if cfg.action_clip_low is not None:
                action_cfg["action_clip_low"] = cfg.action_clip_low
            if cfg.action_clip_high is not None:
                action_cfg["action_clip_high"] = cfg.action_clip_high
            cg = import_onnx_policy(
                cfg.model_path,
                n_x=cfg.n_x,
                obs_cfg=cfg.observation,
                action_cfg=action_cfg,
                output_name=cfg.output_name,
            )
            policy = _GraphPolicy(cg)
        else:
            raise ValueError("onnx_rl: config needs model_path (eager) or artifact_dir (compiled)")

        return cls(policy, seed=cfg.seed, backend=bk)


def _flat_size(shape: Any) -> int:
    """Flat element count of a manifest shape (an empty shape is a scalar)."""
    dims = list(shape) if shape is not None else []
    return math.prod(dims) if dims else 1


def _graph_port_size(graph, name: str) -> int:
    """Flat element count of a named input port of a shinro graph.

    Raises:
        KeyError: If the graph declares no such input port.
    """
    for node in graph.nodes:
        if node.op == "input" and node.attrs["name"] == name:
            return _flat_size(node.shape)
    raise KeyError(f"input port '{name}' not found in graph")
