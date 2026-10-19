"""shinro.mcp — Model Context Protocol servers for the shinro control suite.

Two servers, because the tool families serve different phases of work and an MCP
client loads every tool schema of a connected server into the model's context on
each turn:

- :mod:`shinro.mcp.server` — the **component** server (``shinro-mcp``): create and
  run controllers, estimators, and trajectories, plus LTI analysis. 25 tools.
- :mod:`shinro.mcp.codegen_server` — the **compile pipeline** server
  (``shinro-mcp-codegen``): trace a graph, lower it to Zig, run the verification
  gates, build and stamp the kernel. 14 tools.

Run either with its console script, or as a module::

    shinro-mcp                 # python -m shinro.mcp.server
    shinro-mcp-codegen         # python -m shinro.mcp.codegen_server

See ``docs/mcp_server.md`` for the client configuration and the tool reference.
"""
