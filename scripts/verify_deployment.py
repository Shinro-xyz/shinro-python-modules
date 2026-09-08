"""Thin shim — the deployment-verification logic now lives in ``shinro.codegen.verify``.

Kept so ``from scripts.verify_deployment import ...`` keeps working for the test
suite and repo-internal tooling. New code should import from
``shinro.codegen.verify``.
"""

from shinro.codegen.verify import (  # noqa: F401
    _check,
    _sha256_file,
    main,
    verify,
)

if __name__ == "__main__":
    main()
