"""Thin shim — the compile pipeline now lives in ``shinro.codegen.scenario_build``.

Kept so ``from scripts.build_scenario import ...`` keeps working for the test
suite and repo-internal tooling. New code should import from
``shinro.codegen.scenario_build``.
"""

from shinro.codegen.oracle import TOL_NON_QP, TOL_QP  # noqa: F401
from shinro.codegen.scenario_build import (  # noqa: F401
    EXIT_BUILD,
    EXIT_NO_ZIG,
    EXIT_OK,
    EXIT_ORACLE,
    EXIT_USAGE,
    BuildError,
    build_scenario,
    main,
)

if __name__ == "__main__":
    import sys

    sys.exit(main())
