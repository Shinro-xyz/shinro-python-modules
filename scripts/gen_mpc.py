"""Thin shim — the shipped KF+MPC graph now lives in ``shinro.codegen.recipes``.

Kept so ``from scripts.gen_mpc import build_mpc_composed_graph`` keeps working
for the test suite and repo-internal tooling (``make zig-mpc-gen`` runs this
file). New code should import from ``shinro.codegen.recipes``.
"""

from shinro.codegen.recipes import (  # noqa: F401
    DEFAULT_MPC_CONTROLLER,
    build_mpc_composed_graph,
)
from shinro.codegen.recipes import (
    main_mpc as main,
)

if __name__ == "__main__":
    main()
