"""A recording :class:`ArrayBackend` that captures ops into a :class:`Graph`.

This is the JAX-style tracing backend — a third array backend alongside
``NumpyBackend`` and ``TorchBackend``. Components take
``backend=TraceBackend(graph)`` and trace themselves by being run once with
:class:`Tracer` values: every ``bk.*`` call emits a node instead of computing a
float, and concrete parameters (gain ``K``, matrices ``A``/``B``) are frozen
into the graph as ``const`` nodes the moment they touch a traced operation.

The named-op surface is **data-driven**, not hand-written. For every op
registered in :mod:`shinro.codegen.ops`, the method is generated on demand by
:meth:`TraceBackend.__getattr__`:

- the operands are the signature params that are **not** attributes — the
  attribute set is already declared by the op's :class:`~shinro.codegen.vm_schema.VmOp`
  (``attrs``/``flags``/``array_attrs``/``slot``) plus any
  :class:`~shinro.codegen.ops.TraceSpec` extras;
- the output shape defaults to running the op's numpy handler on
  ``zeros(input_shape)`` — one source of truth with the interpreter oracle —
  with a :class:`TraceSpec` shape override where that can't work (e.g. ``inv``
  of a zero matrix is singular);
- the call signature comes from the eager ``ArrayBackend`` when it declares the
  op, else from ``TraceSpec.params``.

So adding an op is one ``@register_op`` in :mod:`shinro.codegen.ops` (handler +
``VmOp`` + optional ``TraceSpec``), never a new method here. The only methods
that stay hand-written are the ones that are not single-node computations:
array construction (``array``/``zeros``/``eye``/``diag``), shape glue
(``reshape``/``ravel``/``hstack``), the FD meta-op ``jacobian`` (which emits a
whole subgraph), and the boundary helpers (``to_numpy``/``from_numpy``/
``emit_named_output``/``allclose``). Any method with no registered op still
raises ``NotImplementedError`` naming the op to register — a loud, actionable
signal when a new component uses a new backend method.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from shinro.codegen.ops import OP_HANDLERS, TRACE_SPECS, TraceSpec, missing_op_error
from shinro.codegen.tracing import Graph, Node, ShapeMismatchError, Tracer, _lift
from shinro.codegen.vm_schema import VmOp, get_vm_op
from shinro.utils.array_backend import ArrayBackend

#: backend-method name -> op name, where they differ (the op is emitted under
#: a different name than the method a component calls).
_OP_ALIASES = {"slice_": "slice"}


def _attr_param_names(vm: VmOp) -> set[str]:
    """The signature params that are emit attrs (per the op's ``VmOp`` schema)."""
    names = {a.name for a in vm.attrs} | set(vm.flags) | set(vm.array_attrs)
    if vm.slot.startswith("attr:"):
        names.add(vm.slot.split(":", 1)[1])
    elif vm.slot == "axis_code":
        names.add("axis")
    return names


def _handler_shape(op: str, operands: list[Tracer], attrs: dict[str, Any]) -> tuple[int, ...]:
    """Default output shape = the op's numpy handler run on zeros(input_shape).

    The handler is the single source of truth for the oracle value; running it
    on zero arrays of the operand shapes makes it the shape source too, so the
    trace-time shape and the interpreted value can't drift. Warnings (log of 0,
    division by 0, ...) are irrelevant — only the returned shape matters.
    """
    node = Node(op=op, inputs=list(range(len(operands))), shape=(), attrs=dict(attrs))
    values = {i: np.zeros(t.shape, dtype=np.float64) for i, t in enumerate(operands)}
    with np.errstate(all="ignore"):
        result = OP_HANDLERS[op](node, values, {})
    return tuple(np.asarray(result).shape)


class TraceBackend:
    """An :class:`ArrayBackend` that records operations into a :class:`Graph`.

    Not a subclass of :class:`shinro.utils.array_backend.ArrayBackend` because
    that ABC requires every method to be implemented and, crucially, its
    declared methods would *shadow* :meth:`__getattr__` (so the generated
    dispatch would never run). Components receive this backend via ``self.bk``
    and call ``self.bk.<op>`` exactly as they would with ``NumpyBackend``.

    Args:
        graph: The graph to record ops into.
    """

    def __init__(self, graph: Graph) -> None:
        self.g = graph

    # --- helpers ---

    def _emit(self, op: str, inputs: list[Any], shape: tuple[int, ...], **attrs: Any) -> Tracer:
        # Lift any concrete ndarrays to const nodes. This happens when a
        # component's internal computation (e.g. the KF covariance update
        # P = A P A^T + Q) involves only real arrays — the @ between reals
        # returns a real, which then gets passed to a bk.* method here.
        # Lifting bakes it as a const, which is correct: the subgraph was
        # all-constant anyway and would be folded regardless.
        lifted = [_lift(self.g, x) for x in inputs]
        node = self.g.emit(op, [t.node for t in lifted], shape, **attrs)
        return Tracer(self.g, shape, node)

    # --- array creation (constants folded at trace time) ---

    def array(self, data: Any) -> Tracer:
        # A concrete list/ndarray becomes a const node. A Tracer passes through.
        if isinstance(data, Tracer):
            return data
        arr = np.asarray(data, dtype=np.float64)
        node = self.g.const(arr)
        return Tracer(self.g, arr.shape, node)

    def zeros(self, *shape: int) -> Tracer:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        arr = np.zeros(shape, dtype=np.float64)
        node = self.g.const(arr)
        return Tracer(self.g, arr.shape, node)

    def zeros_like(self, x: Tracer) -> Tracer:
        # x is a Tracer; its shape is known at trace time → fold the zeros.
        arr = np.zeros(x.shape, dtype=np.float64)
        node = self.g.const(arr)
        return Tracer(self.g, x.shape, node)

    def eye(self, n: int) -> Tracer:
        arr = np.eye(n, dtype=np.float64)
        node = self.g.const(arr)
        return Tracer(self.g, arr.shape, node)

    def diag(self, x: Tracer) -> Tracer:
        # diag of a tracer — emit a diag op. For the common case where x is a
        # const diagonal (Q/R weights from config), the caller usually passes
        # a concrete list which gets folded by array(); this handles the
        # tracer-input case.
        if x.shape == (1,) or len(x.shape) == 1:
            out_shape = (x.shape[0], x.shape[0])
        else:
            raise NotImplementedError("TraceBackend.diag only supports 1D -> 2D")
        return self._emit("diag", [x], out_shape)

    # --- shape glue (emitted as reshape, not a dedicated op) ---

    def reshape(self, x: Tracer, *shape: int) -> Tracer:
        # Converts the (shape,) or *shape call forms, then emits a reshape op.
        # The reshape node stores its target shape in attrs, same as the
        # composition pass's auto-inserted reshapes.
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out_shape = tuple(shape)
        return self._emit("reshape", [x], out_shape, target_shape=out_shape)

    def ravel(self, x: Tracer) -> Tracer:
        # Flatten to 1-D. Pure data movement — emitted as a reshape to the
        # trace-time-known element count, so the VM reuses its reshape arm
        # rather than growing a dedicated op (same trick as hstack).
        n = 1
        for d in x.shape:
            n *= d
        return self._emit("reshape", [x], (n,), target_shape=(n,))

    def hstack(self, arrays: list[Tracer]) -> Tracer:
        # 1-D hstack == stack along a new leading axis + flatten: pure data
        # movement, bit-exact with np.hstack. Lets MPC_DeltaU's
        # x0_aug = hstack([x0, u_prev]) trace without a new VM op.
        if not arrays:
            raise NotImplementedError("TraceBackend.hstack of empty list")
        if any(len(a.shape) != 1 for a in arrays):
            raise NotImplementedError(f"TraceBackend.hstack only supports 1-D arrays (got shapes {[a.shape for a in arrays]})")
        stacked = self.stack(arrays)
        n = sum(a.shape[0] for a in arrays)
        return self._emit("reshape", [stacked], (n,), target_shape=(n,))

    # --- linear algebra that is not a single-node op ---

    def solve(self, A: Tracer, b: Tracer) -> Tracer:
        # solve(A, b) returns x with A.shape[1] (= A.shape[0]) and b.shape[1]
        # if b is 2D, else b.shape[0] if 1D.
        if len(b.shape) == 2:
            out_shape = (A.shape[1], b.shape[1])
        else:
            out_shape = (A.shape[1],)
        return self._emit("solve", [A, b], out_shape)

    # --- the FD meta-op: emits a whole subgraph, not a node ---

    def jacobian(self, f: Any, x: Tracer, eps: float = 1e-6, *, batched: bool = True) -> Tracer:
        # Central finite differences, mirroring NumpyBackend.jacobian — but as
        # graph nodes, so a traced nonlinear model (the EKF's f/h) lowers with
        # everything else instead of being the one eager-only op.
        #
        # The perturbation is batched: f is called ONCE on the (2n, n) block of
        # x ± eps*e_i probes and the columns are cut out of that single
        # evaluation. That relies on f being batch-capable — the Plant.dynamics
        # contract (``(N, n_x) -> (N, n_x)``) already guarantees it and MPPI's
        # batched rollout already depends on it, so the probes ride the same
        # batch axis the model is written in. One f-subgraph instead of 2n
        # copies: the derivative costs the same nodes as one model evaluation.
        # The arithmetic per probe is unchanged, so the result matches the
        # per-column form to finite-difference roundoff — bit-identical when the
        # model is elementwise-only (as the plant dynamics are), and within
        # ~1/2eps ulps when a shape-sensitive kernel (e.g. BLAS matmul) is in the
        # model, since the probe batch changes its blocking.
        x = _lift(self.g, x)
        if len(x.shape) != 1:
            raise ShapeMismatchError(f"jacobian requires a 1-D point, got {x.shape}")
        n = x.shape[0]

        if not batched:
            # Escape hatch for a callable that cannot carry a leading batch axis
            # (e.g. a user-supplied measurement model): 2n separate emissions.
            cols = []
            for i in range(n):
                e = np.zeros(n, dtype=np.float64)
                e[i] = eps
                cols.append((f(x + e) - f(x - e)) * (1.0 / (2.0 * eps)))
            return self.stack(cols).T

        n_probes = 2 * n
        e = np.vstack([np.eye(n, dtype=np.float64), -np.eye(n, dtype=np.float64)]) * eps
        out = f(self.reshape(x, 1, n) + e)
        if len(out.shape) != 2 or out.shape[0] != n_probes:
            raise ShapeMismatchError(
                f"jacobian: f returned shape {tuple(out.shape)} for a ({n_probes}, {n}) batched probe — "
                f"a batch-capable callable must accept a leading probe axis and return ({n_probes}, m), "
                f"like Plant.dynamics (wrap the callable so it passes the batch through, or call with "
                f"batched=False)"
            )
        plus = self.slice_(out, 0, n)
        minus = self.slice_(out, n, n_probes)
        return ((plus - minus) * (1.0 / (2.0 * eps))).T

    # --- conversions (no-ops under tracing) ---

    def to_numpy(self, x: Tracer) -> Tracer:
        # Under tracing we never materialize; the tracer passes through.
        return x

    def from_numpy(self, x: Any) -> Tracer:
        return _lift(self.g, x)

    def emit_named_output(self, name: str, value: Any) -> None:
        # Record an auxiliary graph output port. This is how a component
        # publishes a diagnostic (e.g. SMC's `healthy` controllability flag)
        # alongside its primary return value without changing its return
        # contract — the eager backends ignore it (see ArrayBackend).
        self.g.output(name, _lift(self.g, value).node)

    def allclose(self, a: Tracer, b: Tracer) -> bool:
        # allclose on tracers is a runtime check — not meaningful at trace
        # time. Components don't call this in the compute path; if they do,
        # it's a control-flow-on-traced-value bug we want surfaced loudly.
        raise NotImplementedError(
            "TraceBackend.allclose is not traceable — it branches on traced "
            "values. Rewrite the component to avoid allclose in the compute path."
        )

    # --- everything else: the registry-driven op dispatcher ---

    def __getattr__(self, name: str) -> Any:
        """Generate the method for a registered op (see the module docstring).

        Called only for attributes not found normally — i.e. op methods, which
        are no longer hand-written. An unknown name raises the standard
        "register this op" error so new components surface missing ops loudly.
        """
        op = _OP_ALIASES.get(name, name)
        if op not in OP_HANDLERS:
            raise missing_op_error(name)
        spec = TRACE_SPECS.get(op, TraceSpec())
        vm = get_vm_op(op)
        attr_params = _attr_param_names(vm) | set(spec.attrs)
        array_attrs = set(vm.array_attrs)
        sig = _signature(op, spec)

        def method(*args: Any, **kwargs: Any) -> Tracer:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            operands: list[Any] = []
            attrs: dict[str, Any] = {}
            for pname, val in bound.arguments.items():
                if spec.variadic is not None and pname == spec.variadic:
                    operands.extend(val)
                elif pname in attr_params:
                    attrs[pname] = np.asarray(val, dtype=np.float64) if pname in array_attrs else val
                else:
                    operands.append(val)
            lifted = [_lift(self.g, x) for x in operands]
            if spec.validate is not None:
                spec.validate(tuple(t.shape for t in lifted), attrs)
            if spec.shape is not None:
                shape = spec.shape(tuple(t.shape for t in lifted), attrs)
            else:
                shape = _handler_shape(op, lifted, attrs)
            return self._emit(op, lifted, shape, **attrs)

        method.__name__ = name
        return method


def _signature(op: str, spec: TraceSpec) -> inspect.Signature:
    """The op's public call signature.

    ``TraceSpec.params`` (a throwaway function whose signature *is* the spec)
    wins; otherwise the eager ``ArrayBackend`` method named ``op`` is the
    reference (self stripped). Ops that are neither must declare ``params``.
    """
    if spec.params is not None:
        return inspect.signature(spec.params)
    eager = getattr(ArrayBackend, op, None)
    if eager is None:
        raise NotImplementedError(f"op '{op}' has no TraceSpec.params and is not on ArrayBackend")
    params = list(inspect.signature(eager).parameters.values())[1:]  # drop self
    return inspect.Signature(parameters=params)
