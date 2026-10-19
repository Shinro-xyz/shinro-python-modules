"""MCP tools for controllers: create, compute, reset, constraints, listing."""

import json

from shinro.codegen.compose import controller_input_kwargs
from shinro.codegen.infer_contract import infer_contract
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.registry import _CONTROLLER_REGISTRY
from shinro.mcp._common import (
    _resolve_backend,
    _set_mpc_default_constraints,
    _store,
    _to_list,
    attached_plant,
    server,
)
from shinro.utils.array_backend import NumpyBackend

# ── Controller tools ──────────────────────────────────────────────────────────


@server.tool()
def create_controller(
    name: str,
    config_path: str | None = None,
    type: str | None = None,
    params: dict | None = None,
    backend: str | None = None,
) -> str:
    """Create a controller from a TOML config file or inline parameters.

    Args:
        name: Unique name to store this controller instance.
        config_path: Path to a TOML config file (e.g. samples/controllers/lqr_base.toml).
        type: Controller type when using inline params (PID, LQR, MPC_DeltaU, MPC_LTI).
        params: Inline parameters (same keys as TOML config, e.g. dt, kp, ki, kd).
        backend: Array backend ('numpy' or 'torch'). Defaults to 'numpy'.
    """
    bk = _resolve_backend(backend)
    if config_path:
        factory = ControllerFactory(config_path)
        ctrl = factory.create(backend=bk)
    elif type:
        if type not in _CONTROLLER_REGISTRY:
            available = list(_CONTROLLER_REGISTRY.keys())
            return f"Unknown controller type '{type}'. Available: {available}"
        if not params:
            return f"Provide 'params' for controller type '{type}'"
        cls = _CONTROLLER_REGISTRY[type]
        ctrl = cls.from_config(params, backend=bk)
    else:
        return "Provide either 'config_path' or both 'type' and 'params'"

    registry_name = getattr(ctrl, "_registry_name", "")
    if registry_name in ("MPC_LTI", "MPC_DeltaU"):
        _set_mpc_default_constraints(ctrl)

    _store[name] = ctrl
    return f"Created {type or 'config-based'} controller '{name}'"


@server.tool()
def controller_compute(
    name: str,
    state: list[float],
    reference: list[float] | None = None,
    u_prev: list[float] | None = None,
) -> str:
    """Compute a control action from a named controller.

    Inputs are mapped to the controller's ``compute()`` parameters **by role** — the
    same dispatcher the simulation runner and gate A use — so the general form works:
    SMC receives ``f_x``/``g_x`` derived from its attached plant, MPPI receives
    ``epsilon=None`` (so it samples its own), and MPC receives ``u_prev``.

    Args:
        name: Name of the stored controller instance.
        state: Current state vector (the state estimate in a closed loop).
        reference: Target/reference state vector (defaults to zeros). A controller
            declaring no reference input sees the tracking error ``state - reference``.
        u_prev: Previous control input (required for MPC_DeltaU; defaults to zeros).
    """
    if name not in _store:
        return f"No controller named '{name}'. Create one first."

    ctrl = _store[name]
    bk = ctrl.bk
    state_arr = bk.array(state)
    ref_arr = bk.array(reference) if reference is not None else bk.zeros_like(state_arr)
    plant = attached_plant(name)

    # u_prev is needed by MPC_DeltaU and to size the plant's model terms for SMC
    # (g_x is n_x x n_u), so take its width from the controller or the plant.
    n_u = getattr(ctrl, "m", None)
    if n_u is None and plant is not None:
        n_u = getattr(plant, "input_dim", None)
    if u_prev is not None:
        u_prev_arr = bk.array(u_prev)
    elif n_u:
        u_prev_arr = bk.zeros(int(n_u))
    else:
        u_prev_arr = bk.zeros_like(state_arr)

    try:
        input_names = infer_contract(ctrl).input_names
        kwargs = controller_input_kwargs(input_names, state_arr, ref_arr, u_prev_arr, plant=plant)
        action = ctrl.compute(**kwargs)
    except Exception as e:
        return f"Error in compute: {e}"

    return json.dumps({"action": _to_list(action)})


@server.tool()
def controller_reset(name: str) -> str:
    """Reset a named controller's internal state.

    Args:
        name: Name of the stored controller instance.
    """
    if name not in _store:
        return f"No controller named '{name}'"
    _store[name].reset()
    return f"Controller '{name}' reset"


@server.tool()
def set_mpc_constraints(
    name: str,
    upper: list[float],
    lower: list[float],
    matrix: list[list[float]] | None = None,
) -> str:
    """Set or update linear constraints on a stored MPC controller.

    Defines F u_k <= upper and F u_k >= lower for each step k,
    tiled across the horizon. If matrix is omitted, defaults to [I; -I].

    Args:
        name: Name of the stored MPC controller instance.
        upper: Upper bound vector (n_c,).
        lower: Lower bound vector (n_c,).
        matrix: Per-step constraint matrix F (n_c, n_u). Defaults to [I; -I].
    """
    if name not in _store:
        return f"No controller named '{name}'. Create one first."
    ctrl = _store[name]
    registry_name = getattr(ctrl, "_registry_name", "")
    if registry_name not in ("MPC_LTI", "MPC_DeltaU"):
        return f"Controller '{name}' is not an MPC type."

    bk = ctrl.bk
    m = ctrl.m
    if matrix is not None:
        F = bk.array(matrix)
    else:
        F = bk.vstack([bk.eye(m), -bk.eye(m)])
    ctrl.constraints(F, bk.array(upper), bk.array(lower))
    return f"Constraints set on MPC controller '{name}'"


@server.tool()
def get_mpc_constraints(name: str) -> str:
    """Get the current constraints of a stored MPC controller.

    Returns the per-step constraint matrix F, upper bounds, and lower bounds.

    Args:
        name: Name of the stored MPC controller instance.
    """
    if name not in _store:
        return json.dumps({"error": f"No controller named '{name}'"})
    ctrl = _store[name]
    registry_name = getattr(ctrl, "_registry_name", "")
    if registry_name not in ("MPC_LTI", "MPC_DeltaU"):
        return json.dumps({"error": f"Controller '{name}' is not an MPC type."})
    if not hasattr(ctrl, "A_constraints"):
        return json.dumps({"error": "No constraints set on this MPC controller."})

    n_c = ctrl.lcons.shape[0] // ctrl.N
    lo = ctrl.lcons[:n_c]
    hi = ctrl.ucons[:n_c]
    return json.dumps(
        {
            "upper": _to_list(hi),
            "lower": _to_list(lo),
        }
    )


@server.tool()
def set_pid_output_limits(
    name: str,
    min: list[float],
    max: list[float],
) -> str:
    """Set output clamping limits on a stored PID controller.

    Enables anti-windup: when output is clamped, the integral term
    back-calculates on saturated channels.

    Args:
        name: Name of the stored PID controller instance.
        min: Minimum output limits per channel (n,).
        max: Maximum output limits per channel (n,).
    """
    if name not in _store:
        return f"No controller named '{name}'. Create one first."
    ctrl = _store[name]
    registry_name = getattr(ctrl, "_registry_name", "")
    if registry_name != "PID":
        return f"Controller '{name}' is not a PID type."

    bk = NumpyBackend()
    ctrl.min_limits = bk.array(min)
    ctrl.max_limits = bk.array(max)
    return f"Output limits set on PID controller '{name}'"


@server.tool()
def list_controller_types() -> str:
    """List all registered controller types available for creation."""
    return json.dumps({"controller_types": list(_CONTROLLER_REGISTRY.keys())})


@server.tool()
def list_controllers() -> str:
    """List all created controller instances."""
    info = {}
    for name, ctrl in _store.items():
        registry_name = getattr(ctrl, "_registry_name", type(ctrl).__name__)
        info[name] = registry_name
    return json.dumps({"controllers": info})
