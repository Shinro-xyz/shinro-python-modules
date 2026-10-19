"""MCP server for shinro-python-modules — controller, estimator, trajectory, and system analysis tools.

Tool implementations live in sibling modules by domain:
``_controllers.py``, ``_estimators.py``, ``_trajectories.py``, ``_analysis.py``,
``_codegen.py``, ``_verification.py``. Each registers its tools against the
shared FastMCP instance from ``_common.py`` at import time, so importing this
module is what populates the tool list.

This module stays the single import site: it re-exports the tool functions and
``_store`` (used by ``tests/test_mcp_server.py``) and provides ``main()``, which
is the ``shinro-mcp`` console entry point.
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
from shinro.mcp._codegen import (  # noqa: F401
    build_composed_graph,
    check_component,
    generate_scenario,
    graph_summary,
    interpret_graph,
    list_codegen_ops,
    lower_graph,
    measure_graph,
    trace_component,
)
from shinro.mcp._common import (  # noqa: F401
    _from_list,
    _graph_store,
    _make_analyzer,
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
from shinro.mcp._trajectories import (  # noqa: F401
    create_trajectory,
    list_trajectories,
    list_trajectory_types,
    trajectory_generate,
    trajectory_position_at,
)
from shinro.mcp._verification import (  # noqa: F401
    build_kernel,
    run_gate_a,
    run_oracle,
    stamp_deployment,
    verify_deployment,
)

__all__ = [
    # shared state / helpers
    "server",
    "_store",
    "_graph_store",
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
    # codegen: trace -> compose -> inspect -> lower
    "list_codegen_ops",
    "check_component",
    "trace_component",
    "build_composed_graph",
    "graph_summary",
    "interpret_graph",
    "measure_graph",
    "lower_graph",
    "generate_scenario",
    # verification: gates, zig build, stamp, verify
    "run_gate_a",
    "build_kernel",
    "run_oracle",
    "stamp_deployment",
    "verify_deployment",
]


def main() -> None:
    """Run the MCP server over stdio (entry point for ``shinro-mcp``)."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
