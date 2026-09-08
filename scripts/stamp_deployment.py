"""Thin shim — the deployment-record logic now lives in ``shinro.codegen.stamp``.

Kept so ``from scripts.stamp_deployment import ...`` keeps working for the test
suite and repo-internal tooling. New code should import from
``shinro.codegen.stamp``.
"""

from shinro.codegen.stamp import (  # noqa: F401
    SENTINEL,
    _master,
    _sha256_bytes,
    _sha256_file,
    _slot_from_configs,
    _slot_from_dir,
    main,
    stamp,
)

if __name__ == "__main__":
    main()
