"""MCP tools for trajectory generators: create, generate, evaluate, listing."""

import json

import numpy as np

from shinro.factories.registry import _TRAJECTORY_REGISTRY
from shinro.factories.trajectory_factory import TrajectoryFactory
from shinro.mcp._common import _from_list, _resolve_backend, _store, _to_list, server

# ── Trajectory tools ──────────────────────────────────────────────────────────


@server.tool()
def create_trajectory(
    name: str,
    config_path: str | None = None,
    type: str | None = None,
    params: dict | None = None,
    backend: str | None = None,
) -> str:
    """Create a trajectory generator from a TOML config file or inline parameters.

    Config-based creation (cubic_segments, quintic_segments, waypoints) returns
    a pre-computed waypoint array. Inline creation of cubic or quintic returns
    a live generator object with generate() and position_at() methods.

    Args:
        name: Unique name to store this trajectory instance.
        config_path: Path to a TOML config file (e.g. samples/trajectories/arm_extension.toml).
        type: Trajectory type when using inline params (cubic_segments, quintic_segments, waypoints, phase_list).
        params: Inline parameters (same keys as TOML config).
        backend: Array backend ('numpy' or 'torch'). Defaults to 'numpy'.
    """
    bk = _resolve_backend(backend)
    if config_path:
        factory = TrajectoryFactory(config_path)
        traj = factory.create(backend=bk)
    elif type:
        if type not in _TRAJECTORY_REGISTRY:
            available = list(_TRAJECTORY_REGISTRY.keys())
            return f"Unknown trajectory type '{type}'. Available: {available}"
        if not params:
            return f"Provide 'params' for trajectory type '{type}'"
        if type == "cubic_segments":
            from shinro.trajectories.cubic_polynomial import CubicPolynomial

            traj = CubicPolynomial(backend=bk)
        elif type == "quintic_segments":
            from shinro.trajectories.quintic_polynomial import QuinticPolynomial

            traj = QuinticPolynomial(backend=bk)
        else:
            cls = _TRAJECTORY_REGISTRY[type]
            traj = cls.from_config(params, backend=bk)
    else:
        return "Provide either 'config_path' or both 'type' and 'params'"

    _store[name] = traj
    return f"Created {type or 'config-based'} trajectory '{name}'"


@server.tool()
def trajectory_generate(
    name: str,
    start_position: list[float],
    end_position: list[float],
    duration: float,
    start_vel: list[float] | None = None,
    end_vel: list[float] | None = None,
    start_acc: list[float] | None = None,
    end_acc: list[float] | None = None,
) -> str:
    """Generate a trajectory by computing polynomial coefficients.

    Works with cubic_segments and quintic_segments generators created inline.
    After calling generate(), use trajectory_position_at() to evaluate.

    Args:
        name: Name of the stored trajectory generator instance.
        start_position: Initial position vector.
        end_position: Final position vector.
        duration: Total trajectory time in seconds.
        start_vel: Initial velocity vector (defaults to zeros).
        end_vel: Final velocity vector (defaults to zeros).
        start_acc: Initial acceleration vector (quintic only, defaults to zeros).
        end_acc: Final acceleration vector (quintic only, defaults to zeros).
    """
    if name not in _store:
        return f"No trajectory named '{name}'. Create one first."

    traj = _store[name]
    p0 = _from_list(start_position)
    pf = _from_list(end_position)
    v0 = _from_list(start_vel) if start_vel is not None else None
    vf = _from_list(end_vel) if end_vel is not None else None
    a0 = _from_list(start_acc) if start_acc is not None else None
    af = _from_list(end_acc) if end_acc is not None else None

    try:
        if hasattr(traj, "generate"):
            registry_name = getattr(traj, "_registry_name", "")
            if registry_name == "quintic_segments":
                traj.generate(p0, pf, duration, v0, vf, a0, af)
            else:
                v0 = v0 if v0 is not None else np.zeros_like(p0)
                vf = vf if vf is not None else np.zeros_like(pf)
                traj.generate(p0, pf, duration, v0, vf)
            return json.dumps({"status": "generated", "duration": duration})
        return "This trajectory type does not support inline generate(). Use from_config instead."
    except Exception as e:
        return f"Error in generate: {e}"


@server.tool()
def trajectory_position_at(name: str, t: float) -> str:
    """Evaluate a generated trajectory at time t.

    Returns position, velocity, and acceleration vectors.

    Args:
        name: Name of the stored trajectory generator instance.
        t: Time in seconds (clipped to [0, duration]).
    """
    if name not in _store:
        return f"No trajectory named '{name}'. Create one first."

    traj = _store[name]

    try:
        if hasattr(traj, "position_at"):
            pos, vel, acc = traj.position_at(t)
            return json.dumps(
                {
                    "position": _to_list(pos),
                    "velocity": _to_list(vel),
                    "acceleration": _to_list(acc),
                }
            )
        return "This trajectory type does not support position_at()."
    except Exception as e:
        return f"Error in position_at: {e}"


@server.tool()
def list_trajectory_types() -> str:
    """List all registered trajectory types available for creation."""
    return json.dumps({"trajectory_types": list(_TRAJECTORY_REGISTRY.keys())})


@server.tool()
def list_trajectories() -> str:
    """List all created trajectory instances."""
    info = {}
    for name, traj in _store.items():
        registry_name = getattr(traj, "_registry_name", type(traj).__name__)
        info[name] = registry_name
    return json.dumps({"trajectories": info})
