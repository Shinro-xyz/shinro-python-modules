"""Generic estimator + controller → :class:`ComposedGraph` builder.

This is the framework-side entry point for compiling an arbitrary scenario
into a deployable graph. It is deliberately LeKiwi-agnostic: the caller
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

from shinro.codegen.compose import ComposedGraph, compose
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


def build_composed_graph(
    estimator_config: str,
    controller_config: str,
    n_x: int,
    n_u: int,
    input_limits: tuple | None = None,
) -> ComposedGraph:
    """Trace an estimator + controller and compose the closed-loop step graph.

    Instantiates both components from their config TOMLs with a
    :class:`NumpyBackend`, traces them with the two-pass state discovery, and
    composes them into the fixed ABC dataflow (``y → estimator → x̂ →
    controller → u → [clip]``). The estimator's recurrent state (e.g. the
    Kalman filter's ``x_hat`` / ``P``) and the controller's (e.g. PID's
    integral) become ``state_*`` ports the host feeds back each tick.

    Args:
        estimator_config: Estimator config TOML path (resolved via
            :func:`shinro.utils.config_resolver.resolve_config_path`).
        controller_config: Controller config TOML path.
        n_x: Plant state dimension.
        n_u: Plant input dimension.
        input_limits: Optional ``(lo, hi)`` clip bounds for the controller
            output, from ``[scenario].input_limits``.

    Returns:
        A :class:`ComposedGraph` for one closed-loop step.
    """
    est = EstimatorFactory(estimator_config).create(backend=NumpyBackend())
    ctrl = ControllerFactory(controller_config).create(backend=NumpyBackend())

    est_input_shapes = {"measurement": (n_x, 1), "control_input": (n_u, 1)}
    ctrl_input_shapes = {
        name: (n_u,) if name == "u_prev" else (n_x,)
        for name in inspect.signature(ctrl.compute).parameters
        if name != "self"
    }

    return compose(
        _trace_with_state(est, est_input_shapes),
        _trace_with_state(ctrl, ctrl_input_shapes),
        plant_dims={"n_x": n_x, "n_u": n_u},
        input_limits=input_limits,
    )
