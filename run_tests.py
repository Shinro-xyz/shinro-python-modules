#!/usr/bin/env python3
"""Run the shinro-python-modules test suite.

Usage:
    python3 run_tests.py           # Full suite (skips slow horizon test)
    python3 run_tests.py --all     # Full suite including slow tests
    python3 run_tests.py --quick   # Unit tests only
    python3 run_tests.py --func    # Functional tests only
    python3 run_tests.py --int     # Integration suite (MuJoCo required, opt-in)
"""

import subprocess
import sys

BASE = ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"]

COMMANDS = {
    "--all":   BASE,
    "--quick": [
        "python3", "-m", "pytest",
        "tests/test_mcp_server.py",
        "-v", "--tb=short", "-k", "not test_very_large_horizon_mpc_times_out",
    ],
    "--func":  [
        "python3", "-m", "pytest",
        "tests/test_mcp_server_functional.py",
        "-v", "--tb=short",
    ],
    "--int": [
        "python3", "-m", "pytest",
        "tests/integration/",
        "-v", "--tb=short", "--override-ini=addopts=",
    ],
}

if __name__ == "__main__":
    flag = sys.argv[1] if len(sys.argv) > 1 else ""
    if flag in COMMANDS:
        cmd = COMMANDS[flag]
    else:
        cmd = BASE + ["-k", "not test_very_large_horizon_mpc_times_out"]
    sys.exit(subprocess.call(cmd))
