"""Thin shim — the compile pipeline now lives in ``shinro.codegen.scenario_gen``.

Kept so ``from scripts.gen_scenario import ...`` keeps working for the test
suite and repo-internal tooling. New code should import from
``shinro.codegen.scenario_gen``.
"""

from shinro.codegen.scenario_gen import (  # noqa: F401
    _ALLOWED_OPTIMIZE,
    _COMPILE_KEYS,
    EXIT_OK,
    EXIT_UNTRACEABLE,
    EXIT_USAGE,
    _provenance,
    _sha256,
    _validate_compile,
    gen_scenario,
    load_scenario,
    main,
)

if __name__ == "__main__":
    import sys

    sys.exit(main())
