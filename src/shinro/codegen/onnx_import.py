"""Import an ONNX-exported policy as a shinro graph.

Unlike every other component in the framework, this does **not** go through the
tracer. ONNX is already a dataflow graph — ``onnx.load(path).graph`` is a
topologically-sorted list of ``(op_type, inputs, outputs, attributes)`` records
with the weights inlined as initializers — so the importer translates it
directly into :class:`~shinro.codegen.tracing.Graph` nodes. The result is a
memoryless :class:`~shinro.codegen.compose.ComposedGraph` that runs through the
same two execution paths as every other compiled graph:

- :func:`shinro.codegen.interpreter.interpret` — the pure-numpy f64 reference,
  and also the eager controller path when no ``.so`` has been built, and
- :func:`shinro.codegen.lower_zig.lower_zig` + ``zig build`` — the deployable
  C-ABI ``.so``.

The policy's observation encoder (integer index selection, mean/std
normalization, clipping) is folded into the graph as arithmetic on baked
constants, so the compiled artifact's only input port is the raw plant state
(``state``). ``onnxruntime`` is not involved anywhere: ``onnx`` is a
compile-time-only dependency (the graph reader), and the deployed artifact has
no runtime dependency at all.

Supported ONNX surface — everything else raises ``NotImplementedError``:

- ``Gemm`` (``alpha`` / ``beta`` / ``transB``; ``transA=1`` is rejected) —
  lowered to the fused ``gemm`` op: one node does the contraction, the
  alpha/beta scales, the transposed-weight read, and the bias add
- ``MatMul``, ``Add``
- ``Relu``, ``Tanh``, ``Sigmoid`` — lowered to the fused ``relu`` / ``tanh`` /
  ``sigmoid`` VM ops (no composition needed)
- ``Softmax`` — lowered to the fused ``softmax`` op (last-axis only; any other
  ``axis`` is rejected loudly)
- ``Gelu`` — lowered to the fused tanh-approx ``gelu`` op; ``approximate``
  must be ``"tanh"`` (the exact erf default is rejected loudly)

Nodes that do not contribute to the declared output are ignored, so an
exporter's stray logging/cast node does not fail the import.

The batch axis is interpreted as a single sample: the policy's declared input
shape ``(None, n_obs)`` becomes the 1-D ``state`` port, and the emitted graph
stays 1-D (or 2-D where an initializer forces it) exactly like the classical
controllers. ``observation.add_batch_dim`` is consequently a no-op in the
compiled path — it only mattered for feeding ``onnxruntime``.

The action space is baked too (``action_cfg``): ``continuous`` applies
scale/bias, ``discrete`` emits an argmax one-hot, and ``stochastic`` splits
``[mean; log_std]``. A non-deterministic ``discrete`` / ``stochastic`` policy
gains an ``epsilon`` input port — the host supplies the noise (Gumbel for
discrete, standard normal for stochastic) and the kernel does only the
arithmetic, exactly like MPPI's port. Deterministic policies have no
``epsilon`` port at all.

Usage::

    from shinro.codegen.interpreter import interpret
    from shinro.codegen.onnx_import import import_onnx_policy

    cg = import_onnx_policy("policy.onnx", n_x=6, obs_cfg={"state_keys": [0, 1, 2]})
    u = interpret(cg.graph, {"state": state})["u"]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shinro.codegen.compose import ComposedGraph
from shinro.codegen.ops import OP_HANDLERS, has_op, missing_op_error
from shinro.codegen.tracing import Graph, Node

#: C-ABI input port carrying the raw plant state (mirrors ``Controller.compute(state, ...)``).
STATE_PORT = "state"
#: C-ABI output port carrying the policy's action.
OUTPUT_PORT = "u"
#: C-ABI input port carrying host noise for a non-deterministic action space.
EPSILON_PORT = "epsilon"

#: Action-space names the importer understands.
_ACTION_SPACES = frozenset({"continuous", "discrete", "stochastic"})
#: Observation-config keys the importer reads. Unknown keys are rejected so a
#: typo (``obs_means``) cannot silently drop normalization or clipping.
_OBS_KEYS = frozenset({"input_name", "state_keys", "normalize", "obs_mean", "obs_std", "clip", "add_batch_dim"})
#: ONNX ops the importer can translate. Everything else is rejected loudly.
_SUPPORTED_OPS = frozenset({"Gemm", "MatMul", "Add", "Relu", "Tanh", "Sigmoid", "Softmax", "Gelu"})
#: ONNX ops translated straight to a same-named shinro op.
_UNARY_OPS = {"Relu": "relu", "Tanh": "tanh", "Sigmoid": "sigmoid"}
#: Attributes Gemm may carry; any other attribute is rejected.
_GEMM_ATTRS = frozenset({"alpha", "beta", "transA", "transB"})


class _OnnxImporter:
    """Translate one ONNX policy into a memoryless shinro graph.

    One object owns the whole translation state — the graph under construction,
    the eagerly-computed value of every node (the shape oracle), the ONNX
    tensor-name → node-id map, the baked initializers, and the resolved
    observation config — so the op translators are methods referring to a
    single graph rather than free functions threading a builder through every
    call.

    Each emitted node is immediately evaluated with its registered numpy
    handler from :mod:`shinro.codegen.ops`, and the resulting array's shape
    becomes the node's declared shape. The ops registry is therefore the single
    source of truth for interpreter *and* lowering semantics — there is no
    parallel hand-written shape table to drift — and an unregistered op fails
    here, at import time, with the registry's actionable message.

    Initializers are emitted lazily (first :meth:`tensor` reference), so weights
    the graph never reads do not land in the compiled constant blob.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_x: int | None = None,
        obs_cfg: dict | None = None,
        action_cfg: dict | None = None,
        output_name: str | None = None,
    ) -> None:
        """Store the import request; parsing and emission happen in :meth:`build`.

        Args:
            model_path: Path to the ``.onnx`` policy file.
            n_x: Plant state dimension — the length of the ``state`` port.
                Defaults to ``max(state_keys) + 1`` (or the model's observation
                dimension when no ``state_keys`` are given).
            obs_cfg: Observation-encoder config, mirroring the old adapter's
                ``[observation]`` TOML table.
            action_cfg: Action-space config, mirroring the old adapter's
                top-level TOML fields.
            output_name: ONNX tensor to import as the action. Defaults to the
                model's first declared output.
        """
        self.model_path = model_path
        self.requested_n_x = n_x
        self.obs_cfg = dict(obs_cfg or {})
        self.action_cfg = dict(action_cfg or {})
        self.output_name = output_name

        self.g = Graph()
        self.values: dict[int, np.ndarray] = {}
        self.tensors: dict[str, int] = {}
        self.inputs: dict[str, np.ndarray] = {}
        self.initializers: dict[str, np.ndarray] = {}

        # Overwritten by _resolve_action_cfg(); the defaults keep the object
        # introspectable before build() runs.
        self.action_space = "continuous"
        self.deterministic = True
        self.action_scale = np.asarray(1.0, dtype=np.float64)
        self.action_bias = np.asarray(0.0, dtype=np.float64)
        self.action_clip: tuple[float, float] | None = None

    # ── orchestration ─────────────────────────────────────────────────────

    def build(self) -> ComposedGraph:
        """Parse the model, emit the graph, and return the composed result.

        Returns:
            A :class:`ComposedGraph` with a ``state`` input port (plus
            ``epsilon`` when the action space samples), a ``u`` output port,
            and no recurrent state.

        Raises:
            ValueError: On a malformed model/config, a multi-input policy, an
                unresolvable tensor, a batched (leading dim ≠ 1) output, or an
                inconsistent action config.
            NotImplementedError: On an unsupported ONNX op or attribute.
        """
        self._resolve_action_cfg()
        onnx, numpy_helper = _onnx_modules()
        graph_proto = onnx.load(self.model_path).graph
        self.initializers = {t.name: np.asarray(numpy_helper.to_array(t), dtype=np.float64) for t in graph_proto.initializer}

        input_name, state_keys, n_x = self._resolve_ports(graph_proto)
        resolved_output = self.output_name or graph_proto.output[0].name
        nodes = list(graph_proto.node)
        needed = _needed_node_indices(nodes, resolved_output, self.initializers, input_name)

        self.inputs = {STATE_PORT: np.ones(n_x, dtype=np.float64)}
        state_id = self.emit("input", [], name=STATE_PORT)
        # The encoder output takes the place of the ONNX input, so the network
        # reads the encoded observation while the graph's only port stays the
        # raw state.
        self.bind(input_name, self.fold_encoder(state_id, n_x=n_x, state_keys=state_keys))

        for idx in sorted(needed):
            node = nodes[idx]
            self.emit_node(node, _onnx_attrs(node))

        if resolved_output not in self.tensors:
            raise ValueError(f"ONNX output {resolved_output!r} was not produced by any reachable node")

        action_id = self.flatten_output(self.tensors[resolved_output])
        ports = [STATE_PORT]
        epsilon_id = self._emit_epsilon(action_id, ports) if self.samples_actions else None
        u = self.apply_action(action_id, epsilon_id)
        self.emit("output", [u], name=OUTPUT_PORT)
        return ComposedGraph(graph=self.g, inputs=ports, outputs=[OUTPUT_PORT])

    def _resolve_ports(self, graph_proto: Any) -> tuple[str, list[int], int]:
        """Resolve the input tensor name, observation indices, and ``n_x``.

        Validates the model's declared input against the observation config, so
        a ``state_keys`` list that disagrees with the ONNX input (or reaches
        past ``n_x``) fails here rather than producing a silently mis-wired
        encoder.

        Args:
            graph_proto: The model's ``GraphProto``.

        Returns:
            ``(input_name, state_keys, n_x)``.

        Raises:
            ValueError: On multiple graph inputs, an ``input_name`` override
                that does not match, an unknown observation key, an
                un-inferable observation dimension, or out-of-range
                ``state_keys``.
        """
        unknown = set(self.obs_cfg) - _OBS_KEYS
        if unknown:
            raise ValueError(f"observation has unknown key(s): {sorted(unknown)} — valid keys: {sorted(_OBS_KEYS)}")
        real_inputs = [i for i in graph_proto.input if i.name not in self.initializers]
        if len(real_inputs) != 1:
            raise ValueError(
                f"ONNX policy must declare exactly one non-initializer input "
                f"(got {[i.name for i in real_inputs]}); multi-input policies are not supported"
            )
        input_name = self.obs_cfg.get("input_name") or real_inputs[0].name
        if input_name != real_inputs[0].name:
            raise ValueError(f"observation.input_name {input_name!r} is not the model's input {real_inputs[0].name!r}")

        state_keys = self.obs_cfg.get("state_keys")
        declared_obs = _declared_last_dim(real_inputs[0])
        if state_keys is None:
            if declared_obs is None:
                raise ValueError("cannot infer the observation dimension from the ONNX input shape — set observation.state_keys")
            state_keys = list(range(declared_obs))
        state_keys = [int(k) for k in state_keys]
        if declared_obs is not None and declared_obs != len(state_keys):
            raise ValueError(f"observation.state_keys selects {len(state_keys)} entries but the ONNX input declares {declared_obs}")

        n_x = self.requested_n_x
        if n_x is None:
            n_x = max(state_keys) + 1 if state_keys else 0
        if n_x <= 0 or any(k < 0 or k >= n_x for k in state_keys):
            raise ValueError(f"observation.state_keys {state_keys} out of range for n_x={n_x}")
        return input_name, state_keys, n_x

    # ── graph emission ────────────────────────────────────────────────────

    def emit(self, op: str, inputs: list[int], **attrs: Any) -> int:
        """Evaluate ``op`` eagerly, then append it with the resulting shape.

        Args:
            op: Registered shinro op name.
            inputs: Node ids of the operands.
            **attrs: Op-specific attributes (baked ``value``, ``target_shape``,
                ``name``, ...).

        Returns:
            The new node's id.

        Raises:
            NotImplementedError: If ``op`` is not in the registry.
        """
        if not has_op(op):
            raise missing_op_error(op)
        probe = Node(op=op, inputs=list(inputs), shape=(), attrs=dict(attrs))
        value = np.asarray(OP_HANDLERS[op](probe, self.values, self.inputs), dtype=np.float64)
        node_id = self.g.emit(op, inputs, value.shape, **attrs)
        self.values[node_id] = value
        return node_id

    def const(self, value: Any) -> int:
        """Emit a ``const`` node carrying ``value`` as an f64 array."""
        return self.emit("const", [], value=np.asarray(value, dtype=np.float64))

    def tensor(self, name: str) -> int:
        """Resolve an ONNX tensor name to its node id, baking initializers lazily.

        Raises:
            ValueError: If ``name`` is neither already produced nor an initializer.
        """
        node_id = self.tensors.get(name)
        if node_id is None:
            value = self.initializers.get(name)
            if value is None:
                raise ValueError(f"ONNX tensor {name!r} is neither a produced value nor an initializer")
            node_id = self.const(value)
            self.tensors[name] = node_id
        return node_id

    def bind(self, name: str, node_id: int) -> None:
        """Record that ONNX tensor ``name`` is now produced by ``node_id``."""
        self.tensors[name] = node_id

    def value_of(self, node_id: int) -> np.ndarray:
        """Return the eagerly-computed value of ``node_id`` (the shape oracle)."""
        return self.values[node_id]

    # ── observation encoder ───────────────────────────────────────────────

    def fold_encoder(self, state_id: int, *, n_x: int, state_keys: list[int]) -> int:
        """Fold the observation encoder into the graph as arithmetic nodes.

        Mirrors the old runtime encoder exactly, in order: integer index
        selection (a baked 0/1 selection matrix, i.e. a matmul), mean/std
        normalization (``sub`` / ``div`` with baked constants), then clipping
        (``clip``). When the selection is the identity the matmul is skipped and
        the state feeds straight through.

        Args:
            state_id: Node id of the raw ``state`` input port.
            n_x: Plant state dimension.
            state_keys: Integer indices of the state used as observations.

        Returns:
            Node id of the encoded observation vector.

        Raises:
            ValueError: On a normalization request without mean/std, or a
                constant whose length does not match the observation dimension.
        """
        n_obs = len(state_keys)
        obs = state_id
        if state_keys != list(range(n_obs)) or n_x != n_obs:
            selection = np.zeros((n_x, n_obs), dtype=np.float64)
            selection[state_keys, np.arange(n_obs)] = 1.0
            obs = self.emit("matmul", [obs, self.const(selection)])

        if self.obs_cfg.get("normalize", False):
            mean = self.obs_cfg.get("obs_mean")
            std = self.obs_cfg.get("obs_std")
            if mean is None or std is None:
                raise ValueError("observation.normalize requires both obs_mean and obs_std")
            obs = self.emit("sub", [obs, self.const(_obs_vector(mean, n_obs, "obs_mean"))])
            obs = self.emit("div", [obs, self.const(_obs_vector(std, n_obs, "obs_std"))])

        if "clip" in self.obs_cfg:
            lo, hi = self.obs_cfg["clip"]
            obs = self.emit("clip", [obs], lo=float(lo), hi=float(hi))
        return obs

    # ── ONNX op translation ───────────────────────────────────────────────

    def emit_node(self, node: Any, attrs: dict[str, Any]) -> None:
        """Translate one ONNX node into shinro node(s) and bind its output tensor.

        Args:
            node: The ``onnx.NodeProto`` to translate.
            attrs: Pre-extracted node attributes.

        Raises:
            NotImplementedError: On an unsupported op, attribute, or arity.
        """
        op_type = node.op_type
        inputs = [n for n in node.input if n]
        outputs = [n for n in node.output if n]
        if op_type not in _SUPPORTED_OPS:
            raise NotImplementedError(
                f"ONNX op {op_type!r} is not supported by the policy importer. Supported ops: "
                f"{sorted(_SUPPORTED_OPS)}. Decompose the policy to Gemm/MatMul/Add + Relu/Tanh/Sigmoid/Softmax/Gelu, "
                f"or extend shinro.codegen.onnx_import."
            )
        if len(outputs) != 1:
            raise NotImplementedError(f"ONNX op {op_type!r} must have exactly one output (got {outputs})")

        if op_type == "Gemm":
            result = self.emit_gemm(inputs, attrs)
        elif op_type == "MatMul":
            _require_no_attrs(op_type, attrs)
            result = self.emit("matmul", [self.tensor(inputs[0]), self.tensor(inputs[1])])
        elif op_type == "Add":
            _require_no_attrs(op_type, attrs)
            result = self.emit("add", [self.tensor(inputs[0]), self.tensor(inputs[1])])
        elif op_type in _UNARY_OPS:
            _require_no_attrs(op_type, attrs)
            result = self.emit(_UNARY_OPS[op_type], [self.tensor(inputs[0])])
        elif op_type == "Softmax":
            result = self.emit_softmax(self.tensor(inputs[0]), attrs)
        elif op_type == "Gelu":
            result = self.emit_gelu(self.tensor(inputs[0]), attrs)
        else:  # pragma: no cover — _SUPPORTED_OPS is disjoint from the above
            raise NotImplementedError(f"ONNX op {op_type!r} has no importer branch")

        self.bind(outputs[0], result)

    def emit_gemm(self, inputs: list[str], attrs: dict[str, Any]) -> int:
        """Emit one fused ``gemm`` node for ``Y = alpha * A' * B' + beta * C``.

        The fused kernel (``linalg.gemm``) does the contraction, the
        ``alpha``/``beta`` scales, the ``transB`` read, and the bias add in a
        single pass, so a torch-style Gemm costs one node and one output-sized
        buffer slot instead of the old transpose + matmul + mul (+ mul + add)
        chain — ``transB`` is realized by striding the baked weight, never by
        moving data. The attributes ride on the node; the lowerer bakes
        ``alpha``/``beta`` into ``gemm_alpha``/``gemm_beta`` and packs the
        table index plus the ``transB`` flag into the node's ``aux``.

        A missing bias (2 inputs) is emitted as an explicit ``const(0.0)``
        scalar so the kernel's three-operand contract stays uniform; the zero
        broadcasts and annihilates the ``beta * C`` term.

        Raises:
            NotImplementedError: On unknown attributes, wrong arity, or ``transA=1``.
        """
        unknown = set(attrs) - _GEMM_ATTRS
        if unknown:
            raise NotImplementedError(f"ONNX Gemm carries unsupported attribute(s): {sorted(unknown)}")
        if len(inputs) not in (2, 3):
            raise NotImplementedError(f"ONNX Gemm must have 2 or 3 inputs (got {len(inputs)})")
        if int(attrs.get("transA", 0)):
            raise NotImplementedError("ONNX Gemm transA=1 is not supported (transpose the activation upstream)")

        a = self.tensor(inputs[0])
        b = self.tensor(inputs[1])
        c = self.tensor(inputs[2]) if len(inputs) == 3 else self.const(0.0)
        return self.emit(
            "gemm",
            [a, b, c],
            alpha=float(attrs.get("alpha", 1.0)),
            beta=float(attrs.get("beta", 1.0)),
            transB=bool(int(attrs.get("transB", 0))),
        )

    def emit_softmax(self, x_id: int, attrs: dict[str, Any]) -> int:
        """Emit a last-axis softmax for ONNX ``Softmax``.

        The VM op is softmax over the last axis (numpy ``axis=-1``), so the only
        accepted axis is the rank-1 one: ``axis=-1`` or ``axis=rank-1``. ONNX
        opset < 13 defaults ``axis`` to 1 (which for the rank-1/2 tensors the
        importer supports is also the last axis); opset >= 13 defaults to -1.
        Any other axis (e.g. per-column softmax on a 2-D input) is rejected
        loudly rather than silently reinterpreted.

        Raises:
            NotImplementedError: On unknown attributes, a rank-0 input, or an
                axis that is not the last axis.
        """
        unknown = set(attrs) - {"axis"}
        if unknown:
            raise NotImplementedError(f"ONNX Softmax carries unsupported attribute(s): {sorted(unknown)}")
        rank = self.value_of(x_id).ndim
        axis = int(attrs.get("axis", -1))
        # The accepted axis is the last one. For a rank-1 vector the softmax axis
        # is the only axis, so opset<13's coerced default (axis=1), opset>=13's
        # -1, and 0 all name it. For rank-2 the last axis is 1/-1; a per-column
        # softmax (axis=0) is rejected rather than silently reinterpreted.
        ok = axis in (-1, 0, 1) if rank <= 1 else axis in (-1, rank - 1)
        if not ok:
            raise NotImplementedError(
                f"ONNX Softmax axis={axis} on a rank-{rank} input is not the last axis; "
                "only last-axis softmax is supported"
            )
        return self.emit("softmax", [x_id])

    def emit_gelu(self, x_id: int, attrs: dict[str, Any]) -> int:
        """Emit the fused tanh-approx ``gelu`` op for ONNX ``Gelu``.

        Only the tanh approximation is implemented (the GPT ``gelu_new``
        formula). ONNX's default ``approximate="none"`` is the exact erf form,
        which this importer deliberately does not support — it is rejected
        loudly rather than silently substituted (the two differ by up to
        ~4.7e-4 absolute).

        Raises:
            NotImplementedError: On unknown attributes, or any approximate mode
                other than ``"tanh"``.
        """
        unknown = set(attrs) - {"approximate"}
        if unknown:
            raise NotImplementedError(f"ONNX Gelu carries unsupported attribute(s): {sorted(unknown)}")
        approx = attrs.get("approximate", "none")
        if isinstance(approx, bytes):
            approx = approx.decode()
        if approx != "tanh":
            raise NotImplementedError(
                f"ONNX Gelu approximate={approx!r} is not supported; only the tanh "
                f"approximation (approximate='tanh') is implemented"
            )
        return self.emit("gelu", [x_id])

    def flatten_output(self, node_id: int) -> int:
        """Reduce a batch-1 output to a 1-D action vector.

        The graph's action port is ``(n_u,)`` like every classical controller. A
        network whose last op produced ``(1, n_u)`` (a rank-2 initializer can
        force that) is reshaped; a genuinely batched output is rejected.

        Args:
            node_id: Node id producing the raw policy output.

        Returns:
            Node id of the flattened ``(n_u,)`` action.

        Raises:
            ValueError: If the output is not 1-D or batch-1 2-D.
        """
        value = self.value_of(node_id)
        if value.ndim == 1:
            return node_id
        if value.ndim == 2 and value.shape[0] == 1:
            return self.emit("reshape", [node_id], target_shape=(value.shape[1],))
        raise ValueError(
            f"policy output shape {value.shape} is not a single action vector — "
            f"only batch-1 policies (a leading dimension of 1) are supported"
        )

    # ── action space ──────────────────────────────────────────────────────

    @property
    def samples_actions(self) -> bool:
        """Whether the graph consumes host noise instead of a deterministic action."""
        return self.action_space != "continuous" and not self.deterministic

    def _resolve_action_cfg(self) -> None:
        """Validate the action config and resolve the constants it bakes.

        Raises:
            ValueError: On an unknown action space, a single-sided clip, or a
                non-finite clip bound. The lowerer writes floats as Zig hex
                literals and ``inf`` is not a Zig identifier, so an ``±inf``
                bound would fail the build — rejecting it here keeps the error
                at import time.
        """
        space = self.action_cfg.get("action_space", "continuous")
        if space not in _ACTION_SPACES:
            raise ValueError(f"action_space must be one of {sorted(_ACTION_SPACES)}, got {space!r}")
        self.action_space = space
        self.deterministic = bool(self.action_cfg.get("deterministic", True))
        self.action_scale = np.asarray(self.action_cfg.get("action_scale", 1.0), dtype=np.float64)
        self.action_bias = np.asarray(self.action_cfg.get("action_bias", 0.0), dtype=np.float64)

        has_low = "action_clip_low" in self.action_cfg
        has_high = "action_clip_high" in self.action_cfg
        if has_low != has_high:
            raise ValueError(
                "action_clip_low and action_clip_high must be given together: a missing bound "
                "would default to ±inf, which the lowerer cannot emit (inf is not a Zig literal)"
            )
        clip = None
        if has_low:
            clip = (float(self.action_cfg["action_clip_low"]), float(self.action_cfg["action_clip_high"]))
            if not all(np.isfinite(clip)):
                raise ValueError(f"action_clip_low/action_clip_high must be finite (got {clip})")
        self.action_clip = clip

    def apply_action(self, raw_id: int, epsilon_id: int | None = None) -> int:
        """Translate the raw policy output into the graph's action port.

        Mirrors the old runtime post-processing exactly: ``continuous`` applies
        scale/bias; ``discrete`` emits an argmax one-hot (scale/bias do not
        apply to a one-hot action, matching the adapter); ``stochastic`` splits
        ``[mean; log_std]`` and either returns the mean or adds
        ``exp(clip(log_std, -10, 2)) * epsilon``. The optional clip is applied
        last in every space.

        Args:
            raw_id: Node id of the flattened raw policy output.
            epsilon_id: Node id of the host-noise port when the space samples,
                else ``None``.

        Returns:
            Node id of the final action.
        """
        if self.action_space == "continuous":
            u = self._scale_bias(raw_id)
        elif self.action_space == "discrete":
            logits = raw_id if epsilon_id is None else self.emit("add", [raw_id, epsilon_id])
            u = self.emit("one_hot", [self.emit("argmax", [logits])], depth=self.value_of(raw_id).size)
        else:  # stochastic
            u = self._stochastic(raw_id, epsilon_id)
        if self.action_clip is not None:
            lo, hi = self.action_clip
            u = self.emit("clip", [u], lo=lo, hi=hi)
        return u

    def _emit_epsilon(self, raw_id: int, ports: list[str]) -> int:
        """Emit the host-noise input port and return its node id.

        The noise kind is part of the deployment contract: ``stochastic``
        expects standard-normal draws of length ``n_u``; ``discrete`` expects
        Gumbel noise of length ``n_actions`` (``g = -log(-log(u))`` from
        ``u ~ U(0, 1)``), which turns ``argmax(logits + g)`` into exact
        categorical sampling from ``softmax(logits)``.
        """
        n_noise = self.value_of(raw_id).size if self.action_space == "discrete" else self._stochastic_half(raw_id)
        self.inputs[EPSILON_PORT] = np.zeros(n_noise, dtype=np.float64)
        ports.append(EPSILON_PORT)
        return self.emit("input", [], name=EPSILON_PORT)

    def _scale_bias(self, x_id: int) -> int:
        """Apply ``scale * x + bias``, emitting both nodes unconditionally.

        Like Gemm's ``alpha`` / ``beta`` multipliers, the default ``1.0`` /
        ``0.0`` still produce their ``mul`` / ``add``: a no-op node is cheap and
        it keeps the action lowering uniform instead of branching on configured
        values.
        """
        x = x_id
        x = self.emit("mul", [x, self.const(self.action_scale)])
        x = self.emit("add", [x, self.const(self.action_bias)])
        return x

    def _stochastic_half(self, raw_id: int) -> int:
        """Return ``n_u`` for a ``[mean; log_std]`` output, validating its size.

        Raises:
            ValueError: If the output size is zero or odd.
        """
        size = self.value_of(raw_id).size
        if size == 0 or size % 2:
            raise ValueError(f"stochastic policy output must be [mean; log_std] with an even, non-zero size (got {size})")
        return size // 2

    def _stochastic(self, raw_id: int, epsilon_id: int | None) -> int:
        """Split ``[mean; log_std]`` and, when sampling, add the scaled noise."""
        half = self._stochastic_half(raw_id)
        mean = self.emit("slice", [raw_id], start=0, stop=half)
        if epsilon_id is None:
            return self._scale_bias(mean)
        log_std = self.emit("slice", [raw_id], start=half, stop=2 * half)
        std = self.emit("exp", [self.emit("clip", [log_std], lo=-10.0, hi=2.0)])
        return self._scale_bias(self.emit("add", [mean, self.emit("mul", [std, epsilon_id])]))


def import_onnx_policy(
    model_path: str,
    *,
    n_x: int | None = None,
    obs_cfg: dict | None = None,
    action_cfg: dict | None = None,
    output_name: str | None = None,
) -> ComposedGraph:
    """Translate an ONNX policy into a memoryless composed graph.

    The graph's declared output is resolved from the model (or ``output_name``),
    only the nodes it depends on are imported, and the observation encoder is
    folded in front of the network. The returned graph has one input port
    (``state``), one output port (``u``), and no recurrent state.

    Args:
        model_path: Path to the ``.onnx`` policy file.
        n_x: Plant state dimension — the length of the ``state`` port. Defaults
            to ``max(state_keys) + 1`` (or the model's observation dimension
            when no ``state_keys`` are given).
        obs_cfg: Observation-encoder config, mirroring the old adapter's
            ``[observation]`` TOML table. Supported keys: ``input_name``,
            ``state_keys``, ``normalize``, ``obs_mean``, ``obs_std``, ``clip``,
            and (accepted but ignored in the compiled path) ``add_batch_dim``.
        action_cfg: Action-space config, mirroring the old adapter's top-level
            TOML fields: ``action_space``, ``deterministic``, ``action_scale``,
            ``action_bias``, and ``action_clip_low`` / ``action_clip_high``
            (which must be given together — the lowerer cannot emit ``±inf``).
        output_name: ONNX tensor to import as the action. Defaults to the
            model's first declared output.

    Returns:
        A :class:`ComposedGraph` ready for ``interpret`` or ``lower_zig``. Its
        input ports are ``state`` and, for a sampling action space,
        ``epsilon``.

    Raises:
        ImportError: If the ``onnx`` package is not installed.
        ValueError: On a malformed model/config or an unsupported layout.
        NotImplementedError: On an unsupported ONNX op or attribute.
    """
    return _OnnxImporter(model_path, n_x=n_x, obs_cfg=obs_cfg, action_cfg=action_cfg, output_name=output_name).build()


def _onnx_modules() -> tuple[Any, Any]:
    """Import and return the lazily-required ``(onnx, numpy_helper)`` modules.

    Raises:
        ImportError: If the ``onnx`` package (the ``onnx-rl`` extra) is missing.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("the ONNX policy importer needs the 'onnx' package — install with `pip install \"shinro[onnx-rl]\"`") from exc
    return onnx, numpy_helper


def _needed_node_indices(nodes: list[Any], output_name: str, initializers: dict[str, np.ndarray], input_name: str) -> set[int]:
    """Collect the node indices the declared output actually depends on.

    A backwards walk from ``output_name`` through tensor producers. Exporters
    routinely leave nodes that do not feed the output (logging, unused
    branches), and those must not fail the import — only the reachable subgraph
    is translated, so an unsupported op is rejected exactly when it matters.

    Args:
        nodes: The ONNX graph's nodes, in topological order.
        output_name: ONNX tensor name of the requested output.
        initializers: Initializer names (leaves of the walk).
        input_name: The model's real input tensor name (leaf of the walk).

    Returns:
        Indices into ``nodes`` of the reachable nodes.

    Raises:
        ValueError: If the graph references a tensor nothing produces.
    """
    producer: dict[str, int] = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                producer[out] = idx

    needed: set[int] = set()
    visited: set[str] = set()
    stack = [output_name]
    while stack:
        name = stack.pop()
        if name in visited:
            continue
        visited.add(name)
        if name in initializers:
            continue
        idx = producer.get(name)
        if idx is None:
            if name == input_name:
                continue
            raise ValueError(f"ONNX graph references unknown tensor {name!r}")
        needed.add(idx)
        stack.extend(n for n in nodes[idx].input if n)
    return needed


def _require_no_attrs(op_type: str, attrs: dict[str, Any]) -> None:
    """Raise if a pointwise ONNX op carries attributes the importer ignores."""
    if attrs:
        raise NotImplementedError(f"ONNX op {op_type!r} carries unsupported attribute(s): {sorted(attrs)}")


def _declared_last_dim(value_info: Any) -> int | None:
    """Return the model input's declared last dimension, or None if symbolic."""
    dims = value_info.type.tensor_type.shape.dim
    if not dims:
        return None
    last = dims[-1]
    return int(last.dim_value) if last.dim_value and last.dim_value > 0 else None


def _obs_vector(values: Any, n_obs: int, field: str) -> np.ndarray:
    """Validate and flatten an encoder constant to length ``n_obs``.

    Raises:
        ValueError: If the constant does not have exactly ``n_obs`` entries.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size != n_obs:
        raise ValueError(f"observation.{field} must have {n_obs} entries (got {arr.size})")
    return arr


def _onnx_attrs(node: Any) -> dict[str, Any]:
    """Extract an ONNX node's attributes as plain Python values."""
    from onnx import helper

    return {a.name: helper.get_attribute_value(a) for a in node.attribute}
