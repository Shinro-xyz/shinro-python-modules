"""MCP tools for state estimators: create, estimate, reset, listing."""

import json

from shinro.factories.estimator_factory import EstimatorFactory
from shinro.factories.registry import _ESTIMATOR_REGISTRY
from shinro.mcp._common import _resolve_backend, _store, _to_list, server

# ── Estimator tools ──────────────────────────────────────────────────────────


@server.tool()
def create_estimator(
    name: str,
    config_path: str | None = None,
    type: str | None = None,
    params: dict | None = None,
    backend: str | None = None,
) -> str:
    """Create a state estimator from a TOML config file or inline parameters.

    Args:
        name: Unique name to store this estimator instance.
        config_path: Path to a TOML config file (e.g. samples/estimators/luenberger_base.toml).
        type: Estimator type when using inline params (KalmanFilter, LuenbergerObserver).
        params: Inline parameters (same keys as TOML config, e.g. dt, process_noise, measurement_noise).
        backend: Array backend ('numpy' or 'torch'). Defaults to 'numpy'.
    """
    bk = _resolve_backend(backend)
    if config_path:
        factory = EstimatorFactory(config_path)
        est = factory.create(backend=bk)
    elif type:
        if type not in _ESTIMATOR_REGISTRY:
            available = list(_ESTIMATOR_REGISTRY.keys())
            return f"Unknown estimator type '{type}'. Available: {available}"
        if not params:
            return f"Provide 'params' for estimator type '{type}'"
        cls = _ESTIMATOR_REGISTRY[type]
        est = cls.from_config(params, backend=bk)
    else:
        return "Provide either 'config_path' or both 'type' and 'params'"

    _store[name] = est
    return f"Created {type or 'config-based'} estimator '{name}'"


@server.tool()
def estimator_estimate(
    name: str,
    measurement: list[float],
    control_input: list[float],
) -> str:
    """Run one predict-update cycle of a named estimator.

    Estimators take flat vectors; the server passes the lists through directly.

    Args:
        name: Name of the stored estimator instance.
        measurement: Observation vector from sensors.
        control_input: Control vector applied at this step.
    """
    if name not in _store:
        return f"No estimator named '{name}'. Create one first."

    est = _store[name]
    bk = est.bk
    meas = bk.array(measurement)
    ctrl = bk.array(control_input)

    try:
        x_hat = est.estimate(meas, ctrl)
    except Exception as e:
        return f"Error in estimate: {e}"

    return json.dumps({"state_estimate": _to_list(x_hat)})


@server.tool()
def estimator_reset(name: str) -> str:
    """Reset a named estimator's internal state.

    Args:
        name: Name of the stored estimator instance.
    """
    if name not in _store:
        return f"No estimator named '{name}'"
    _store[name].reset()
    return f"Estimator '{name}' reset"


@server.tool()
def list_estimator_types() -> str:
    """List all registered estimator types available for creation."""
    return json.dumps({"estimator_types": list(_ESTIMATOR_REGISTRY.keys())})


@server.tool()
def list_estimators() -> str:
    """List all created estimator instances."""
    info = {}
    for name, ctrl in _store.items():
        registry_name = getattr(ctrl, "_registry_name", type(ctrl).__name__)
        info[name] = registry_name
    return json.dumps({"estimators": info})
