"""MCP server for shinro-python-modules — controller, estimator, trajectory, and system analysis tools.

This is the **component** server (``shinro-mcp``): designing and running
controllers, estimators, and trajectories, plus LTI analysis. The compile
pipeline lives on its own server — see ``shinro.mcp.codegen_server``
(``shinro-mcp-codegen``) — so a client that only designs components does not
carry the tracing/lowering/verification schemas in its context.

Tool implementations live in sibling modules by domain:
``_controllers.py``, ``_estimators.py``, ``_trajectories.py``, ``_analysis.py``.
Each registers its tools against ``_common.server`` at import time, so importing
this module is what populates the tool list.

This module stays the single import site for the component server: it re-exports
the tool functions and ``_store`` (used by ``tests/test_mcp_server.py``) and
provides ``main()``, which is the ``shinro-mcp`` console entry point.
"""

from shinro.mcp._analysis import (  # noqa: F401
    analyze_controllability,
    analyze_observability,
    balanced_truncation,
    gramian_continuous,
    gramian_discrete,
    gramian_finite,
    system_summary,
)
from shinro.mcp._common import (  # noqa: F401
    _attached,
    _from_list,
    _make_analyzer,
    _plant_store,
    _resolve_backend,
    _set_mpc_default_constraints,
    _store,
    _to_list,
    server,
)
from shinro.mcp._controllers import (  # noqa: F401
    controller_compute,
    controller_reset,
    create_controller,
    get_mpc_constraints,
    list_controller_types,
    list_controllers,
    set_mpc_constraints,
    set_pid_output_limits,
)
from shinro.mcp._estimators import (  # noqa: F401
    create_estimator,
    estimator_estimate,
    estimator_reset,
    list_estimator_types,
    list_estimators,
)
from shinro.mcp._lifecycle import (  # noqa: F401
    delete_instance,
    list_instances,
)
from shinro.mcp._plants import (  # noqa: F401
    attach_plant,
    create_plant,
    list_plant_types,
    list_plants,
    plant_model,
)
from shinro.mcp._simulation import (  # noqa: F401
    simulate_scenario,
)
from shinro.mcp._trajectories import (  # noqa: F401
    create_trajectory,
    list_trajectories,
    list_trajectory_types,
    trajectory_generate,
    trajectory_position_at,
)

__all__ = [
    # shared state / helpers
    "server",
    "_store",
    "_plant_store",
    "_attached",
    "_resolve_backend",
    "_to_list",
    "_from_list",
    "_set_mpc_default_constraints",
    "_make_analyzer",
    # controllers
    "create_controller",
    "controller_compute",
    "controller_reset",
    "set_mpc_constraints",
    "get_mpc_constraints",
    "set_pid_output_limits",
    "list_controller_types",
    "list_controllers",
    # estimators
    "create_estimator",
    "estimator_estimate",
    "estimator_reset",
    "list_estimator_types",
    "list_estimators",
    # trajectories
    "create_trajectory",
    "trajectory_generate",
    "trajectory_position_at",
    "list_trajectory_types",
    "list_trajectories",
    # system analysis
    "analyze_controllability",
    "analyze_observability",
    "gramian_continuous",
    "gramian_discrete",
    "gramian_finite",
    "system_summary",
    "balanced_truncation",
    # plants: the model seam for nonlinear controllers/estimators
    "create_plant",
    "attach_plant",
    "list_plant_types",
    "list_plants",
    "plant_model",
    # lifecycle
    "list_instances",
    "delete_instance",
    # design-time simulation
    "simulate_scenario",
]


def main() -> None:
    """Run the MCP server over stdio (entry point for ``shinro-mcp``)."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
