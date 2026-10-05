"""Generic estimator + controller → :class:`ComposedGraph` builder.

This is the framework-side entry point for compiling an arbitrary scenario
into a deployable graph. It is deliberately robot-agnostic: the caller
supplies the estimator/controller config paths and the plant dimensions, and
the builder returns a composed closed-loop step graph ready for
:func:`shinro.codegen.lower_zig.lower_zig`.

The two-pass trace is the mechanism that makes this generic:

1. **Pass 1** — :func:`shinro.codegen.trace_node.trace_node` runs the
   component once with :class:`Tracer` values and detects recurrent state via
   attr-diff (any array-valued instance attr whose ``id()`` changed during
   the call is state).
2. **Pass 2** — the component is re-traced with those attrs pre-injected as
   tracers, so they become ``state_*`` recurrent ports instead of being frozen
   at their trace-time values. This is what lets a *new* component (PID's
   ``_integral``, a custom estimator's observer state, ...) compose correctly
   with zero per-component declaration — the state is discovered, not listed.

Controller inputs are mapped to shapes by role, mirroring the role logic in
:mod:`shinro.codegen.compose`: ``u_prev`` is ``(n_u,)``, every other input
(state / reference) is ``(n_x,)``. Estimator inputs follow the repo's column-
vector convention ``(n, 1)``; ``compose`` auto-inserts reshapes from the flat
``y`` / ``u_prev`` ports.
"""

from __future__ import annotations

import inspect

from shinro.codegen.compose import ComposedGraph, _measurement_dim, compose
from shinro.codegen.trace_node import NodeGraph, trace_node
from shinro.factories import ControllerFactory, EstimatorFactory
from shinro.utils.array_backend import NumpyBackend


def _trace_with_state(component, input_shapes: dict[str, tuple[int, ...]]) -> NodeGraph:
    """Trace a component, discovering and pre-injecting its recurrent state.

    Pass 1 traces without ``state_shapes`` so attr-diff can detect which
    instance attrs the component reassigns during the call. Pass 2 re-traces
    with those attrs pre-injected as tracers — they become ``state_*``
    recurrent ports, so ``compose`` threads them as feedback edges instead of
    freezing the recursion at trace-time constants.

    Args:
        component: A registered Controller / StateEstimator instance.
        input_shapes: Maps each method input name to its concrete shape.

    Returns:
        A :class:`NodeGraph` with the component's state pre-injected.
    """
    ng = trace_node(component, input_shapes)
    state_shapes = {attr: getattr(component, attr).shape for attr in ng.state_attrs}
    return trace_node(component, input_shapes, state_shapes=state_shapes)


def _factory(factory_cls, config):
    """Instantiate a component from a config TOML path or an in-memory dict."""
    if isinstance(config, dict):
        return factory_cls(config=config).create(backend=NumpyBackend())
    return factory_cls(config).create(backend=NumpyBackend())


def build_composed_graph(
    estimator_config: str | dict,
    controller_config: str | dict,
    n_x: int,
    n_u: int,
    input_limits: tuple | None = None,
    plant: object | None = None,
) -> ComposedGraph:
    """Trace an estimator + controller and compose the closed-loop step graph.

    Instantiates both components with a :class:`NumpyBackend`, traces them
    with the two-pass state discovery, and composes them into the fixed ABC
    dataflow (``y → estimator → x̂ → controller → u → [clip]``). The
    estimator's recurrent state (e.g. the Kalman filter's ``x_hat`` / ``P``)
    and the controller's (e.g. PID's integral) become ``state_*`` ports the
    host feeds back each tick.

    Args:
        estimator_config: Estimator config TOML path (resolved via
            :func:`shinro.utils.config_resolver.resolve_config_path`) or an
            in-memory config dict.
        controller_config: Controller config TOML path or in-memory dict.
        n_x: Plant state dimension.
        n_u: Plant input dimension.
        input_limits: Optional ``(lo, hi)`` clip bounds for the controller
            output, from ``[scenario].input_limits``.
        plant: Optional plant instance for controllers whose model comes from
            it rather than from their config (MPPI's dynamics/cost). Passed
            through ``attach_plant`` when the controller supports it — the
            same wiring the simulation path uses, so sim and compile agree.
            Ignored by controllers that do not need it (LQR, PID, MPC).

    Returns:
        A :class:`ComposedGraph` for one closed-loop step.
    """
    est, ctrl = instantiate(estimator_config, controller_config, plant)
    return build_composed_graph_from_instances(est, ctrl, n_x, n_u, input_limits, plant=plant)


def instantiate(estimator_config, controller_config, plant=None):
    """Instantiate the estimator + controller (attaching the plant to both).

    The single construction site: :func:`build_composed_graph` composes the
    traced graphs from these instances, and gate A
    (:mod:`shinro.codegen.gate_a`) drives the *same* instances live, so the
    graph and the live loop cannot be built from different components.

    The plant is attached to any component exposing ``attach_plant`` — the
    controller (its model, e.g. MPPI's dynamics/cost) and the estimator (the EKF's
    process model, so a compiled filter traces the plant's own ``dynamics``).
    """
    est = _factory(EstimatorFactory, estimator_config)
    ctrl = _factory(ControllerFactory, controller_config)
    if plant is not None:
        if hasattr(est, "attach_plant"):
            est.attach_plant(plant)
        if hasattr(ctrl, "attach_plant"):
            ctrl.attach_plant(plant)
    return est, ctrl


def _trace_plant_model(plant: object, n_x: int, n_u: int):
    """Trace the plant-derived model a controller can consume as ``f_x`` / ``g_x``.

    Emits the control-affine pair the SMC form assumes — the drift
    ``f(x) = dynamics(x, 0)`` and the control matrix ``g(x) = control_matrix(x, 0)``
    — as ONE subgraph whose input is the state estimate. ``compose`` wires it into
    a controller that declares those inputs, so the model lowers into the binary
    instead of being evaluated by the host (the SMC gap).

    Both terms are evaluated at zero control, the drift convention of
    ``ẋ = f(x) + g(x)u``. A plant whose actuation is nonlinear in ``u`` (rotor
    thrusts ~ ``w²``) has ``g(x, 0) = 0`` and needs an operating control instead —
    pass one through :meth:`Plant.control_matrix` when that case is plumbed.

    Args:
        plant: Plant exposing ``dynamics`` and ``control_matrix``.
        n_x: State dimension.
        n_u: Control dimension.

    Returns:
        A :class:`~shinro.codegen.trace_node.NodeGraph` with input ``x`` and
        outputs ``f_x`` / ``g_x``.

    Raises:
        ValueError: If the plant has no dynamics to differentiate.
    """
    from shinro.codegen.infer_contract import InferredContract
    from shinro.codegen.trace_backend import TraceBackend
    from shinro.codegen.trace_node import NodeGraph
    from shinro.codegen.tracing import Graph, Tracer

    if getattr(plant, "dynamics", None) is None:
        raise ValueError(f"{type(plant).__name__} exposes no dynamics() to derive f_x/g_x from.")

    g = Graph()
    tb = TraceBackend(g)
    x_node = g.input("x", (n_x,))
    x = Tracer(g, (n_x,), x_node)
    f_x = plant.dynamics(x, 0.0, bk=tb)
    g_x = plant.control_matrix(x, 0.0, bk=tb)
    g.output("f_x", f_x.node)
    g.output("g_x", g_x.node)
    return NodeGraph(
        graph=g,
        contract=InferredContract(method_name="plant_model", input_names=["x"]),
        input_nodes={"x": x_node},
        output_nodes={"f_x": f_x.node, "g_x": g_x.node},
        state_attrs=[],
    )


def _controller_input_shape(name: str, n_x: int, n_u: int) -> tuple[int, ...]:
    """Default traced-input shape for a controller ``compute()`` parameter, by role.

    ``u_prev`` is the control; ``g_x`` is the control matrix, one column per input;
    everything else (the state estimate, and ``f_x`` the drift) is state-shaped.
    """
    if name == "u_prev":
        return (n_u,)
    if name == "g_x":
        return (n_x, n_u)
    return (n_x,)


def build_composed_graph_from_instances(est, ctrl, n_x, n_u, input_limits=None, plant=None):
    """Trace + compose already-instantiated components into a closed-loop step graph."""
    n_y = _measurement_dim(est, n_x)
    est_input_shapes = {"measurement": (n_y, 1), "control_input": (n_u, 1)}
    ctrl_input_shapes = {
        name: _controller_input_shape(name, n_x, n_u)
        for name in inspect.signature(ctrl.compute).parameters
        if name != "self"
    }
    # Free host inputs (e.g. MPPI's epsilon) declare their own shapes — the
    # role-based defaults above cannot know them.
    host_shapes = ctrl.host_input_shapes() if hasattr(ctrl, "host_input_shapes") else {}
    ctrl_input_shapes.update(host_shapes)

    # A controller that takes its model as inputs (SMC's f_x/g_x) gets a plant-derived
    # subgraph composed in — the model is baked, not host-fed. Without a plant compose
    # raises loudly if the controller still declares those inputs.
    model = _trace_plant_model(plant, n_x, n_u) if plant is not None and {"f_x", "g_x"} & set(ctrl_input_shapes) else None

    return compose(
        _trace_with_state(est, est_input_shapes),
        _trace_with_state(ctrl, ctrl_input_shapes),
        plant_dims={"n_x": n_x, "n_u": n_u},
        input_limits=input_limits,
        host_inputs=tuple(host_shapes),
        model=model,
    )
