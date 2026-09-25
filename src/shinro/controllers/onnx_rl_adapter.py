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

Usage (samples/controllers/onnx_rl.toml)::

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
        # In-network noise ports (``noise_<k>`` from RandomNormalLike): the host
        # supplies standard-normal draws every tick, because the kernel does no
        # RNG. Distinct from ``epsilon``, whose kind depends on the action space.
        self.noise_ports = [(n, _graph_port_size(cg.graph, n)) for n in self.inputs if n.startswith("noise_")]
        # Recurrent feedback: what each published state output refills next tick.
        self._state_plan = _state_feedback_plan(cg.state_outputs, lambda n: _graph_port_size(cg.graph, n))
        self.state = _zero_state(self._state_plan, lambda n: _graph_port_size(cg.graph, n))

    @property
    def gumbel(self) -> bool:
        """True when the graph expects Gumbel noise (i.e. it samples discretely)."""
        return "one_hot" in self.ops

    def step(self, feed: dict[str, np.ndarray]) -> np.ndarray:
        """Run one tick through the interpreter, carrying the recurrent state."""
        from shinro.codegen.interpreter import interpret

        out = interpret(self._cg.graph, {**feed, **self.state})
        for out_port, chunks in self._state_plan:
            values = np.asarray(out[out_port]).ravel()
            for in_port, start, stop in chunks:
                self.state[in_port] = values[start:stop].copy()
        return out[self._u_port]

    def reset(self) -> None:
        """Zero the recurrent state (a fresh run starts from the zero state)."""
        for name in self.state:
            self.state[name] = np.zeros_like(self.state[name])


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

        # Recurrent feedback: each state output's slot in the state buffer, and
        # which input ports it refills next tick.
        self._state_sizes = {
            port["name"]: _flat_size(port["shape"]) for port in self.manifest["state_outputs"]
        }
        self._state_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for port in self.manifest["state_outputs"]:
            size = self._state_sizes[port["name"]]
            self._state_slices[port["name"]] = (offset, offset + size)
            offset += size
        self._state_plan = _state_feedback_plan(self._state_slices, lambda n: self._state_sizes[n])
        self.state = _zero_state(self._state_plan, lambda n: self._state_sizes[n])

        self.state_port = STATE_PORT
        self.state_size = _flat_size(self.manifest["inputs"][self.inputs.index(STATE_PORT)]["shape"])
        self.noise_port = EPSILON_PORT if EPSILON_PORT in self.inputs else None
        self.noise_size = _flat_size(self.manifest["inputs"][self.inputs.index(EPSILON_PORT)]["shape"]) if self.noise_port else 0
        self.noise_ports = [
            (port["name"], _flat_size(port["shape"])) for port in self.manifest["inputs"] if port["name"].startswith("noise_")
        ]

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
        """Pack the ports (plus the carried state), call ``shinro_step``, and
        return the ``u`` slice."""
        merged = {**feed, **self.state}
        packed = np.concatenate([np.asarray(merged[name], dtype=np.float64).ravel() for name in self.inputs])
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
        for out_port, chunks in self._state_plan:
            start, stop = self._state_slices[out_port]
            values = state[start:stop]
            for in_port, a, b in chunks:
                self.state[in_port] = values[a:b].copy()
        start, stop = self._u_slice
        return out[start:stop].copy()

    def reset(self) -> None:
        """Zero the recurrent state (a fresh run starts from the zero state)."""
        for name in self.state:
            self.state[name] = np.zeros_like(self.state[name])


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
        feed: dict[str, np.ndarray] = {self.policy.state_port: x}
        if self.policy.noise_port is not None:
            feed[self.policy.noise_port] = self._draw_noise(self.policy.noise_size)
        for name, size in self.policy.noise_ports:
            feed[name] = self._rng.standard_normal(size)
        return self.bk.from_numpy(self.policy.step(feed))

    def _draw_noise(self, size: int) -> np.ndarray:
        """Draw the host noise the action-space ``epsilon`` port expects.

        For a discretely-sampling graph this is Gumbel noise, which makes
        ``argmax(logits + g)`` an exact categorical draw from
        ``softmax(logits)``; otherwise it is a standard normal. In-network
        ``noise_<k>`` ports are always standard normal and are drawn in
        :meth:`compute`.
        """
        if self.policy.gumbel:
            uniform = np.clip(self._rng.uniform(0.0, 1.0, size=size), _GUMBEL_EPS, 1.0 - _GUMBEL_EPS)
            return -np.log(-np.log(uniform))
        return self._rng.standard_normal(size)

    def reset(self):
        """Reseed the action-sampling RNG and clear any recurrent state."""
        self._rng = np.random.default_rng(self.seed)
        self.policy.reset()

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


def _state_feedback_plan(state_outputs, size_of) -> list[tuple[str, list[tuple[str, int, int]]]]:
    """Map each published state output to the input ports it refills next tick.

    Follows the importer's port convention: a GRU/RNN publishes ``state_h_<k>``
    (H) which refills the same-named input; an LSTM publishes ``state_hc_<k>``
    (2H, ``[h ‖ c]`` — the kernel's output order and ML-Agents' ``recurrent_in``
    layout) which refills ``state_h_<k>`` / ``state_c_<k>`` in that order.

    Args:
        state_outputs: State-output port names (a mapping's keys work too).
        size_of: Callable returning an output port's flat element count.

    Returns:
        ``[(output_port, [(input_port, start, stop), ...]), ...]``.
    """
    plan: list[tuple[str, list[tuple[str, int, int]]]] = []
    for out in state_outputs:
        n = size_of(out)
        if out.startswith("state_hc_"):
            suffix = out[len("state_hc_") :]
            plan.append((out, [(f"state_h_{suffix}", 0, n // 2), (f"state_c_{suffix}", n // 2, n)]))
        else:
            plan.append((out, [(out, 0, n)]))
    return plan


def _zero_state(plan, size_of) -> dict[str, np.ndarray]:
    """Initial recurrent state: zeros for every input port the plan refills."""
    return {name: np.zeros(stop - start, dtype=np.float64) for _, chunks in plan for name, start, stop in chunks}


def _flat_size(shape: Any) -> int:
    """Flat element count of a manifest shape (an empty shape is a scalar)."""
    dims = list(shape) if shape is not None else []
    return math.prod(dims) if dims else 1


def _graph_port_size(graph, name: str) -> int:
    """Flat element count of a named input *or* output port of a shinro graph.

    Both directions are needed: recurrent state ports are looked up on the
    input side (``state_h_0``) and the output side (``state_hc_0``).

    Raises:
        KeyError: If the graph declares no such port.
    """
    for node in graph.nodes:
        if node.op in ("input", "output") and node.attrs["name"] == name:
            return _flat_size(node.shape)
    raise KeyError(f"port '{name}' not found in graph")
