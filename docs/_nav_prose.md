# Hand-maintained navigation for the PROSE pages.
#
# Format is mkdocs-literate-nav markdown (NOT yaml):
#     * [Title](path.md)
#     * Section title
#         * [Sub page](sub/page.md)
#
# Edit this file to add or reorder prose pages — it is safe from regeneration.
# scripts/gen_api.py reads it, appends the generated "API Reference" section, and
# writes the combined result to docs/SUMMARY.md (a build artifact — never edit
# SUMMARY.md or docs/_nav_api.yml by hand).

* [Home](index.md)
* [Quickstart](quickstart.md)
* [How it works](how-it-works.md)
* [Components](components.md)
* [Testing](testing.md)
* [MCP server](mcp_server.md)
* [Codegen](codegen.md)
