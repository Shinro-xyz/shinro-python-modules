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
    """Report a plant's model — both the linear view and the nonlinear one — at one
    explicit evaluation point.

    There is no ``get_dynamics`` on a Plant; it exposes four hooks, and this tool
    reports all of them in one stable shape:

    - ``get_model()`` → the **linearization** ``(A, B)`` (what LQR/MPC use; an
      approximation for a nonlinear plant).
    - ``dynamics(x, u)`` → the vector field ``f(x, u)`` (what EKF/UKF/MPPI use).
    - ``model_terms(x)`` → the control-affine pair ``(f_x, g_x)`` (what SMC uses).
    - ``control_matrix(x, u)`` → ``g = ∂f/∂u``, used to *measure* whether the
      control-affine form is exact at this point.

    The evaluation point is always echoed, so nothing is implicit: ``state`` defaults
    to the plant's current state and ``control`` to ``plant.operating_control()``
    (the same operating point ``model_terms`` uses — hover speeds for a rotor plant),
    reported as ``evaluation.u_source``. ``evaluation.affine_exact`` answers the
    question the docs used to answer in prose: is ``f = f_x + g_x·u`` exact here, or
    only a tangent? ``affine_residual`` is ``‖f − (f_x + g_x·u)‖∞``.

    Args:
        name: Name of a stored plant.
        state: Operating point ``x`` (n_x,); defaults to the plant's current state.
        control: Control ``u`` (n_u,); defaults to the plant's operating control.
    """
    import numpy as np

    plant = _plant_store.get(name)
    if plant is None:
        return json.dumps({"error": f"No plant named '{name}'. Create one first."})
    try:
        A, B = plant.get_model()
    except (ValueError, TypeError, NotImplementedError) as e:
        return json.dumps({"error": f"{e.__class__.__name__}: {e}"})

    summary = _plant_summary(plant)
    A_arr = np.asarray(A, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    n_u = summary["n_u"] or (B_arr.shape[1] if B_arr.ndim == 2 else 0)

    # The evaluation point: explicit in the response, never a hidden default.
    x = np.asarray(plant.get_state() if state is None else state, dtype=np.float64).reshape(-1)
    u_op = plant.operating_control()
    if control is not None:
        u = np.asarray(control, dtype=np.float64).reshape(-1)
        u_source = "caller"
    elif u_op is not None:
        u = np.asarray(u_op, dtype=np.float64).reshape(-1)
        u_source = "operating_control"
    else:
        u = np.zeros(int(n_u))
        u_source = "zero"

    payload: dict = {
        "name": name,
        "type": summary["type"],
        "dt": getattr(plant, "dt", None),
        "n_x": summary["n_x"],
        "n_u": int(n_u),
        "state": _flatten(plant.get_state()),
        # At the plant's DEFAULT operating point — get_model() takes no x0/u0 portably.
        "linearization": {"A": _to_nested(A_arr), "B": _to_nested(B_arr)},
        "evaluation": {
            "x": _flatten(x),
            "u": _flatten(u),
            "u_source": u_source,
        },
    }

    try:
        f = np.asarray(plant.dynamics(x, u), dtype=np.float64).reshape(-1)
        f_x, g_x = plant.model_terms(x, n_u=int(n_u))
        f_x = np.asarray(f_x, dtype=np.float64).reshape(-1)
        g_x = np.asarray(g_x, dtype=np.float64)
        recon = f_x + g_x @ u
        residual = float(np.max(np.abs(recon - f))) if f.size else 0.0
        # Is g independent of u at this x? That is exactly "the control-affine form
        # is exact", measured rather than asserted.
        probe = u + 1e-3 * (1.0 + np.abs(u))
        g_probe = np.asarray(plant.control_matrix(x, probe), dtype=np.float64)
        affine_exact = bool(np.allclose(g_probe, g_x, rtol=1e-6, atol=1e-9))
    except (ValueError, TypeError, NotImplementedError) as e:
        payload["evaluation"]["error"] = f"{e.__class__.__name__}: {e}"
        return json.dumps(payload)

    payload["evaluation"].update(
        {
            "dynamics": _flatten(f),
            "model_terms": {"f_x": _flatten(f_x), "g_x": _to_nested(g_x)},
            "affine_exact": affine_exact,
            "affine_residual": residual,
        }
    )
    return json.dumps(payload)


def _flatten(value) -> list[float]:
    import numpy as np

    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _to_nested(value) -> list[list[float]]:
    import numpy as np

    arr = np.asarray(value, dtype=np.float64)
    return arr.tolist() if arr.ndim == 2 else arr.reshape(1, -1).tolist()
