"""``python -m shinro.cli`` — run the unified CLI in-process."""

from __future__ import annotations

import sys

from shinro.cli import main

if __name__ == "__main__":
    sys.exit(main())
