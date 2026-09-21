"""Thin shim — the component tracer now lives in ``shinro.codegen.component_cli``.

Kept so ``from scripts.trace_component import ...`` keeps working for the test
suite and repo-internal tooling. New code should import from
``shinro.codegen.component_cli`` (or use the ``shinro check`` / ``shinro trace``
verbs).
"""

from shinro.codegen.component_cli import (  # noqa: F401
    CONFIG_DIR,
    EXIT_OK,
    EXIT_ORACLE,
    EXIT_UNTRACEABLE,
    EXIT_USAGE,
    FACTORIES,
    REGISTRIES,
    _parse_pairs,
    cmd_check,
    cmd_contract,
    cmd_inventory,
    cmd_trace,
    load_component,
    main,
    parse_shape,
)

if __name__ == "__main__":
    import sys

    sys.exit(main())
