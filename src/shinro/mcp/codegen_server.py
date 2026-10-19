"""MCP server for the shinro compile pipeline — trace, lower, verify.

The **codegen** server (``shinro-mcp-codegen``): the path from a component to a
deployable kernel. It is deliberately separate from the component server
(``shinro.mcp.server`` / ``shinro-mcp``) because the two serve different phases
of work, and an MCP client loads every tool schema of the servers it connects to
into the model's context on each turn. Splitting keeps a session that is only
designing a controller from carrying 14 tracing/lowering/verification schemas.

Tools (14), grouped by module:

- ``_codegen.py`` — ``list_codegen_ops``, ``check_component``,
  ``trace_component``, ``build_composed_graph``, ``graph_summary``,
  ``interpret_graph``, ``measure_graph``, ``lower_graph``, ``generate_scenario``
- ``_verification.py`` — ``run_gate_a``, ``build_kernel``, ``run_oracle``,
  ``stamp_deployment``, ``verify_deployment``

Both modules register against ``_common.codegen_server`` at import time, so
importing this module is what populates this server's tool list. Importing them
from the component server does *not* leak them there.

Run with the ``shinro-mcp-codegen`` console script, or
``python -m shinro.mcp.codegen_server``.
"""

from shinro.mcp._codegen import (  # noqa: F401
    build_composed_graph,
    check_component,
    delete_graph,
    generate_scenario,
    graph_summary,
    interpret_graph,
    list_codegen_ops,
    list_graphs,
    lower_graph,
    measure_graph,
    trace_component,
)
from shinro.mcp._common import _graph_store
from shinro.mcp._common import codegen_server as server
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
    "_graph_store",
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
    # graph lifecycle
    "list_graphs",
    "delete_graph",
    # verification: gates, zig build, stamp, verify
    "run_gate_a",
    "build_kernel",
    "run_oracle",
    "stamp_deployment",
    "verify_deployment",
]


def main() -> None:
    """Run the codegen MCP server over stdio (entry point for ``shinro-mcp-codegen``)."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
