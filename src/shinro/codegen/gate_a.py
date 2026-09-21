"""Gate A: ``interpret(composed graph)`` vs the live estimator + controller loop.

The pre-compile equivalence check: does the composed graph reproduce what the
*real* components compute, driven through the same closed-loop wiring? Gate B
(``.so`` vs ``interpret``) proves the Zig VM executes the graph faithfully;
gate A proves the graph faithfully represents the components. Only both
together make a stamped artifact's "verified" claim true end to end.

The loop mirrors :func:`shinro.codegen.compose.compose`'s fixed ABC dataflow::

    y ─▶ estimator ─x_hat─▶ controller ─u─▶ [clip] ─▶ u

with the same role mapping for the controller inputs (state / reference /
u_prev / a host input), the same recurrent ``state_*`` threading, and the same
clip. It runs **more than one tick** on purpose: at tick 0 both sides start
from the same seeded state, so a frozen recurrence (the class the closed-loop
test oracles guard — e.g. the Kalman covariance not updating) only diverges
after a tick.

A mismatch means the tracer or composer lost or garbled a computation, so the
build refuses to stamp a "verified" artifact for the wrong math.

The live side is built by :func:`shinro.codegen.recipes.live_components` — the
same construction the recipe traces — so gate A and the graph can never be
built from different components.
"""

from __future__ import annotations

import numpy as np

from shinro.codegen.compose import _CONTROLLER_INPUT_ROLES
from shinro.codegen.infer_contract import infer_contract
from shinro.codegen.interpreter import interpret_step


def _state_port_owner(est, ctrl, port: str):
    """Resolve a ``state_*`` port to ``(component, attr)`` via the attr name.

    ``trace_node._state_port_name`` strips a leading underscore, so try the
    port's suffix and its underscore-prefixed form on each component.
    """
    base = port[len("state_") :]
    for attr in (base, "_" + base):
        if hasattr(est, attr):
            return est, attr
        if hasattr(ctrl, attr):
            return ctrl, attr
    return None, None


def _initial_state(cg, est, ctrl, n_u) -> dict[str, np.ndarray]:
    """Seed the graph's recurrent state inputs from the live components' state.

    Both sides must start identical, otherwise the comparison is meaningless.
    ``u_prev`` starts at zero (no previous control).
    """
    state: dict[str, np.ndarray] = {}
    for port in cg.state_inputs:
        if port == "u_prev":
            state[port] = np.zeros(n_u)
            continue
        comp, attr = _state_port_owner(est, ctrl, port)
        if comp is None or attr is None:
            raise ValueError(f"gate A: cannot map state port '{port}' to a component attribute")
        state[port] = np.asarray(getattr(comp, attr), dtype=np.float64).copy()
    return state


def _controller_kwargs(input_names, x_hat, x_ref, u_prev, host_values) -> dict:
    """Map the controller's compute() inputs to live values by role (as compose does)."""
    takes_reference = any(_CONTROLLER_INPUT_ROLES.get(n) == "reference" for n in input_names)
    kwargs: dict[str, np.ndarray] = {}
    for name in input_names:
        role = _CONTROLLER_INPUT_ROLES.get(name)
        if role == "reference":
            kwargs[name] = x_ref
        elif role == "u_prev":
            kwargs[name] = u_prev
        elif role == "state":
            # Regulator (no reference input) gets the error x_hat - x_ref; a
            # tracker gets x_hat and x_ref separately (exactly like compose).
            kwargs[name] = x_hat if takes_reference else (x_hat - x_ref)
        else:
            # A free host input the host fills each tick (e.g. MPPI's epsilon).
            kwargs[name] = host_values.get(name, np.array(0.0))
    return kwargs


def run_gate_a(cg, est, ctrl, n_x, n_u, input_limits=None, ticks: int = 50, seed: int = 0) -> float:
    """Drive the live loop and ``interpret_step(cg)`` over ``ticks``; return max abs err.

    Args:
        cg: The composed graph (from :func:`shinro.codegen.recipes.build_recipe`).
        est: The live estimator instance.
        ctrl: The live controller instance.
        n_x: Plant state dimension.
        n_u: Plant input dimension.
        input_limits: Optional ``(lo, hi)`` clip bounds (as composed).
        ticks: Number of ticks to run (default 50; > 1 to exercise recurrence).
        seed: RNG seed for the shared y / x_ref / host inputs.

    Returns:
        The maximum absolute error between the graph's ``u`` and the live
        ``u`` across the run — 0.0 when both sides agree bit-for-bit.
    """
    rng = np.random.default_rng(seed)
    state = _initial_state(cg, est, ctrl, n_u)
    input_names = infer_contract(ctrl).input_names
    host_shapes = ctrl.host_input_shapes() if hasattr(ctrl, "host_input_shapes") else {}
    # state input port -> the state output key that feeds it next tick
    # (interpret_step strips the leading "state_" from its state dict keys)
    state_in_to_out = {port: port.removeprefix("state_") for port in cg.state_inputs}

    u_prev = np.zeros(n_u)
    max_err = 0.0
    for _ in range(ticks):
        y = rng.normal(0.0, 0.1, (n_x,))
        x_ref = rng.normal(0.0, 0.05, (n_x,))
        host_values = {name: (rng.normal(0.0, 0.1, shape) if shape else np.array(0.0)) for name, shape in host_shapes.items()}

        graph_inputs = {"y": y, "x_ref": x_ref, **state, **host_values}
        outs, state_outs = interpret_step(cg.graph, graph_inputs)
        u_graph = np.asarray(outs["u"], dtype=np.float64).ravel()

        x_hat = np.asarray(est.estimate(y.reshape(-1, 1), u_prev.reshape(-1, 1)), dtype=np.float64).ravel()
        kwargs = _controller_kwargs(input_names, x_hat, x_ref, u_prev, host_values)
        u_live = np.asarray(ctrl.compute(**kwargs), dtype=np.float64).ravel()
        if input_limits is not None:
            u_live = np.clip(u_live, np.asarray(input_limits[0], dtype=np.float64), np.asarray(input_limits[1], dtype=np.float64))

        max_err = max(max_err, float(np.max(np.abs(u_graph - u_live))))
        state = {inp: np.asarray(state_outs[out], dtype=np.float64) for inp, out in state_in_to_out.items()}
        u_prev = u_live
    return max_err
