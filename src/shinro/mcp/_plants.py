"""MCP tools for plants — the model seam for nonlinear controllers and estimators.

Linear components are self-contained: LQR carries its A/B, PID its gains. Everything
nonlinear needs the *plant*:

- **SMC** takes the control-affine pair ``f_x`` / ``g_x`` as ``compute`` arguments,
  derived from the plant at its operating point.
- **MPPI** rolls out ``plant.dynamics`` and scores with a cost.
- **EKF / UKF** differentiate / propagate the plant's process model.
- **UKF** also takes its ``dt`` from the plant in scenario builds.

Without a plant those components construct fine and then fail at the first call
("dynamics_fn and cost_fn must be set", "missing 1 required positional argument:
'g_x'"). This module supplies the missing seam: create a plant, attach it, and the
controller/estimator tools drive the general form.

These register on the **component** server (``shinro-mcp``), alongside the components
they serve — not on the codegen server, which needs no live plant instance.
"""

import json

from shinro.codegen.infer_contract import infer_contract
from shinro.factories.registry import _PLANT_REGISTRY
from shinro.mcp._common import _attached, _plant_store, _store, server
from shinro.utils.array_backend import NumpyBackend


def _plant_summary(plant) -> dict:
    """Type, dims, and the attachment for a stored plant."""
    state = plant.get_state()
    n_x = int(state.shape[0])
    n_u = getattr(plant, "input_dim", None)
    attached_to = sorted(name for name, pname in _attached.items() if _plant_store.get(pname) is plant)
    return {
        "type": getattr(plant, "_registry_name", type(plant).__name__),
        "n_x": n_x,
        "n_u": n_u,
        "attached_to": attached_to,
    }


@server.tool()
def create_plant(
    name: str,
    type: str | None = None,
    config_path: str | None = None,
    params: dict | None = None,
) -> str:
    """Create a plant from a TOML config file or inline parameters.

    A plant supplies the model that nonlinear controllers and estimators need. Attach
    it with ``attach_plant``; linear components (LQR, PID, KF, Luenberger) work
    without one.

    Args:
        name: Unique name to store this plant instance.
        type: Plant type when using inline params (e.g. CartPole, InvertedPendulum,
            Quadrotor) — see ``list_plant_types``.
        config_path: Path to a plant config TOML (relative to the server's CWD).
        params: Inline parameters (same keys as the plant's TOML, e.g. dt, gravity).
    """
    bk = NumpyBackend()
    if config_path:
        import tomllib

        from shinro.utils.config_resolver import resolve_config_path

        try:
            with open(resolve_config_path(config_path), "rb") as f:
                cfg = tomllib.load(f)
        except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
            return json.dumps({"error": str(e)})
        ptype = cfg.get("type")
        if ptype is None:
            return json.dumps({"error": f"{config_path}: no 'type' field"})
        if ptype not in _PLANT_REGISTRY:
            return json.dumps({"error": f"Unknown plant type '{ptype}'. Available: {sorted(_PLANT_REGISTRY)}"})
        try:
            plant = _PLANT_REGISTRY[ptype].from_config(cfg, backend=bk)
        except (ValueError, TypeError) as e:
            # `type` is shadowed by this function's parameter, so class name it is.
            return json.dumps({"error": f"{e.__class__.__name__}: {e}"})
    elif type:
        if type not in _PLANT_REGISTRY:
            return json.dumps({"error": f"Unknown plant type '{type}'. Available: {sorted(_PLANT_REGISTRY)}"})
        if not params:
            return json.dumps({"error": f"Provide 'params' for plant type '{type}'"})
        try:
            plant = _PLANT_REGISTRY[type].from_config({**params, "type": type}, backend=bk)
        except (ValueError, TypeError) as e:
            return json.dumps({"error": f"{e.__class__.__name__}: {e}"})
    else:
        return json.dumps({"error": "Provide either 'config_path' or both 'type' and 'params'"})

    _plant_store[name] = plant
    return json.dumps({"stored": name, **_plant_summary(plant)})


@server.tool()
def attach_plant(component_name: str, plant_name: str) -> str:
    """Attach a stored plant to a stored controller or estimator.

    This is what makes the general form runnable: SMC then gets ``f_x``/``g_x``
    derived from the plant, MPPI gets ``dynamics``/cost, and EKF/UKF get their
    process model. Linear components accept the attachment and ignore it.

    Args:
        component_name: Name of a stored controller or estimator.
        plant_name: Name of a stored plant (from ``create_plant``).
    """
    if component_name not in _store:
        return json.dumps({"error": f"No controller or estimator named '{component_name}'. Create one first."})
    if plant_name not in _plant_store:
        return json.dumps({"error": f"No plant named '{plant_name}'. Create one first."})

    component = _store[component_name]
    plant = _plant_store[plant_name]
    attach = getattr(component, "attach_plant", None)
    # Two kinds of consumer:
    #  * MPPI / EKF / UKF take the plant itself (dynamics, cost, process model);
    #  * SMC takes f_x / g_x as compute() arguments and has no attach_plant — for it
    #    the attachment is only the mapping controller_compute derives the model from.
    # A self-contained component (LQR's A/B, PID's gains, KF's C) needs neither.
    needs_model_terms = any(name in ("f_x", "g_x") for name in infer_contract(component).input_names)
    if not callable(attach) and not needs_model_terms:
        return json.dumps(
            {
                "error": (
                    f"'{component_name}' ({type(component).__name__}) does not accept a plant — "
                    "it is already self-contained (e.g. LQR carries its own A/B)."
                )
            }
        )
    if callable(attach):
        try:
            attach(plant)
        except (ValueError, TypeError) as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    _attached[component_name] = plant_name
    return json.dumps(
        {
            "attached": component_name,
            "plant": plant_name,
            "component_type": getattr(component, "_registry_name", type(component).__name__),
            "plant_type": _plant_summary(plant)["type"],
        }
    )


@server.tool()
def list_plant_types() -> str:
    """List all registered plant types available for creation."""
    return json.dumps({"plant_types": sorted(_PLANT_REGISTRY)})


@server.tool()
def list_plants() -> str:
    """List all created plant instances, with dims and what they are attached to."""
    return json.dumps({name: _plant_summary(plant) for name, plant in _plant_store.items()})


@server.tool()
def plant_model(
    name: str,
    state: list[float] | None = None,
    control: list[float] | None = None,
) -> str:
    """Report a plant's model: its discrete linearization, and optionally the true
    non-linear terms evaluated at a point.

    There is no ``get_dynamics`` on a Plant. The four distinct hooks are:

    - ``get_model()`` — discrete ``(A, B)``; **linearized** about the plant's operating
      point and Euler-discretized at ``dt``. This is what LQR/MPC use, and what this
      tool always returns. For a nonlinear plant it is an approximation.
    - ``dynamics(x, u)`` — the continuous-time vector field ``f(x, u)`` itself.
    - ``model_terms(x, n_u)`` — the control-affine pair ``(f_x, g_x)`` at ``x``; what
      SMC consumes.
    - ``control_matrix(x, u)`` — ``g(x, u) = ∂f/∂u``.

    Pass ``state`` to also get ``f`` and ``model_terms`` at that operating point —
    the exact nonlinear answer, as opposed to the linearization above.

    Args:
        name: Name of a stored plant.
        state: Optional operating point ``x`` (n_x,) to evaluate the nonlinear terms at.
        control: Optional ``u`` (n_u,) for ``dynamics``; defaults to zeros.
    """
    plant = _plant_store.get(name)
    if plant is None:
        return json.dumps({"error": f"No plant named '{name}'. Create one first."})
    try:
        A, B = plant.get_model()
    except (ValueError, TypeError) as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    summary = _plant_summary(plant)
    payload = {
        "name": name,
        "type": summary["type"],
        "dt": getattr(plant, "dt", None),
        "n_x": summary["n_x"],
        "n_u": summary["n_u"],
        "state": _flatten(plant.get_state()),
        "A": _to_nested(A),
        "B": _to_nested(B),
    }

    if state is not None:
        import numpy as np

        n_u = summary["n_u"] or 0
        # model_terms/control_matrix read `state.shape`, so coerce the JSON lists.
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        u = np.asarray(control if control is not None else [0.0] * int(n_u), dtype=np.float64).reshape(-1)
        try:
            f = plant.dynamics(x, u)
            f_x, g_x = plant.model_terms(x, n_u=int(n_u))
        except (ValueError, TypeError) as e:
            payload["nonlinear_error"] = f"{e.__class__.__name__}: {e}"
            return json.dumps(payload)
        payload["operating_point"] = {"x": _flatten(x), "u": _flatten(u)}
        payload["dynamics"] = _flatten(f)  # f(x, u): the true vector field
        payload["model_terms"] = {"f_x": _flatten(f_x), "g_x": _to_nested(g_x)}  # SMC's pair

    return json.dumps(payload)


def _flatten(value) -> list[float]:
    import numpy as np

    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _to_nested(value) -> list[list[float]]:
    import numpy as np

    arr = np.asarray(value, dtype=np.float64)
    return arr.tolist() if arr.ndim == 2 else arr.reshape(1, -1).tolist()
