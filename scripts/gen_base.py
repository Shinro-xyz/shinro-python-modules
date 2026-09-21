"""Thin shim — the shipped KF+LQR graph now lives in ``shinro.codegen.recipes``.

Kept so ``from scripts.gen_base import build_base_graph`` keeps working for the
test suite and repo-internal tooling (``make zig-gen`` runs this file). New code
should import from ``shinro.codegen.recipes``.
"""

from shinro.codegen.recipes import (  # noqa: F401
    LQR_CONFIGS,
    build_base_graph,
)
from shinro.codegen.recipes import (
    main_lqr as main,
)

if __name__ == "__main__":
    main()
