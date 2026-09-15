"""Thin shim — kernel size metrics live in ``shinro.codegen.measure``.

Kept so the metric is runnable as a script (``make measure-kernels``); the
implementation is importable for tooling via ``shinro.codegen.measure``.
"""

from shinro.codegen.measure import main

if __name__ == "__main__":
    import sys

    sys.exit(main())
