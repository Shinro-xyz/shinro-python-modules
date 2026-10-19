"""Lifecycle tools: see what is stored, and evict what is not.

The server keeps every created component in a session-scoped store, and every traced
graph in the codegen server's store. Before these tools there was no way to enumerate
or remove them: a name could be overwritten, but a long-lived session accumulated
instances (and traced graphs, which hold a full copy of the graph) with no way out.

`delete_instance` refuses to delete a plant that components are still attached to —
those components would silently lose their model mid-session. Detach/delete them first.
"""

import json

from shinro.components import Controller, Plant, StateEstimator, TrajectoryGenerator
from shinro.mcp._common import _attached, _plant_store, _store, server


def _kind(obj) -> str:
    """Human-readable kind for a stored object."""
    if isinstance(obj, Plant):
        return "plant"
    if isinstance(obj, Controller):
        return "controller"
    if isinstance(obj, StateEstimator):
        return "estimator"
    if isinstance(obj, TrajectoryGenerator):
        return "trajectory"
    # Config-created trajectories are stored as sampled arrays (or a dict of
    # position/velocity/acceleration) rather than a generator object.
    return "trajectory"


@server.tool()
def list_instances() -> str:
    """List every stored controller, estimator, trajectory, and plant.

    One call to see the session's live state before deleting or re-creating anything:
    each entry reports its kind, registered type, and (for components) the plant it is
    attached to; plants report what is attached to them.
    """
    out: dict = {}
    for name, obj in _store.items():
        out[name] = {
            "kind": _kind(obj),
            "type": getattr(obj, "_registry_name", type(obj).__name__),
            "plant": _attached.get(name),
        }
    for name, plant in _plant_store.items():
        out[name] = {
            "kind": "plant",
            "type": getattr(plant, "_registry_name", type(plant).__name__),
            "attached_to": sorted(n for n, pname in _attached.items() if _plant_store.get(pname) is plant),
        }
    return json.dumps(out)


@server.tool()
def delete_instance(name: str) -> str:
    """Delete a stored controller, estimator, trajectory, or plant.

    Frees the instance and, for components, its plant attachment. Deleting a plant
    that other components are attached to is **refused** — those components would
    silently lose their model — so delete or re-attach the dependents first.

    Args:
        name: Name of the stored instance to remove.
    """
    if name in _plant_store:
        dependents = sorted(n for n, pname in _attached.items() if pname == name)
        if dependents:
            return json.dumps(
                {
                    "error": (
                        f"plant '{name}' is attached to {dependents}; "
                        "delete or re-attach those components first"
                    ),
                    "attached_to": dependents,
                }
            )
        del _plant_store[name]
        return json.dumps({"deleted": name, "kind": "plant"})

    if name not in _store:
        return json.dumps({"error": f"No controller, estimator, trajectory, or plant named '{name}'"})

    obj = _store.pop(name)
    _attached.pop(name, None)
    return json.dumps({"deleted": name, "kind": _kind(obj)})
