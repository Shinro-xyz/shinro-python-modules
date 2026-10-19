"""MCP server for the experiment notebook — append-only JSONL run records.

The **notebook** server (``shinro-mcp-notebook``): recording what you tried and what
came out, so trials survive the session that produced them and can be compared later.

It is its own server for the same reason the other two are split: a session that never
records a run should not carry these schemas. It also needs neither the component nor
the graph store — a record is self-describing and persists to a file *you* name — so it
can run entirely on its own.

Tools (4):

- ``record_run`` — append one record; the server fills provenance (timestamp, version,
  git SHA, platform, cwd).
- ``list_runs`` — compact summaries, oldest first.
- ``get_run`` — one full record, by id or by name (most recent).
- ``compare_runs`` — rank the notebook by one metric (the "which should I keep?").

Run with the ``shinro-mcp-notebook`` console script, or
``python -m shinro.mcp.notebook_server``.
"""

from shinro.mcp._common import notebook_server as server
from shinro.mcp._notebook import (  # noqa: F401
    compare_runs,
    get_run,
    list_runs,
    record_run,
)

__all__ = [
    "server",
    "record_run",
    "list_runs",
    "get_run",
    "compare_runs",
]


def main() -> None:
    """Run the notebook MCP server over stdio (entry point for ``shinro-mcp-notebook``)."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
