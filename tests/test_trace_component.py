"""Tests for scripts/trace_component.py — the standalone component tracer."""

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.trace_component import (
    EXIT_OK,
    EXIT_UNTRACEABLE,
    EXIT_USAGE,
    cmd_trace,
    parse_shape,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "src" / "shinro" / "configs"

LQR = str(CONFIG / "controllers" / "lqr_base.toml")
KF = str(CONFIG / "estimators" / "kalman_base.toml")
PID = str(REPO_ROOT / "tests" / "fixtures" / "configs" / "controllers" / "pid_arm.toml")
TRAJ = str(CONFIG / "trajectories" / "base_straight.toml")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/trace_component.py", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


class TestParseShape:
    def test_vector(self):
        assert parse_shape("3") == (3,)

    def test_matrix(self):
        assert parse_shape("3x1") == (3, 1)
        assert parse_shape("3x3") == (3, 3)

    def test_scalar(self):
        assert parse_shape("scalar") == ()

    def test_invalid(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_shape("banana")


class TestTraceOracle:
    def test_lqr(self):
        rc = cmd_trace(LQR, {"current_state": (3,), "target_state": (3,)}, {}, 20, 0)
        assert rc == EXIT_OK

    def test_kalman_with_state(self):
        rc = cmd_trace(
            KF,
            {"measurement": (3, 1), "control_input": (3, 1)},
            {"x_hat": (3, 1), "P": (3, 3)},
            20,
            0,
        )
        assert rc == EXIT_OK

    def test_pid_with_state(self):
        rc = cmd_trace(
            PID,
            {"current_state": (6,), "target_state": (6,)},
            {"_integral": (6,), "_prev_error": (6,), "_has_run": (6,)},
            20,
            0,
        )
        assert rc == EXIT_OK

    def test_missing_shapes_is_usage_error(self):
        rc = cmd_trace(LQR, {}, {}, 20, 0)
        assert rc == EXIT_USAGE

    def test_untraceable_component(self):
        # SMC uses .flatten() on a tracer — not traceable as-is.
        rc = cmd_trace(
            str(CONFIG / "controllers" / "smc.toml"),
            {"x": (3,), "f_x": (3,), "g_x": (3, 3)},
            {},
            5,
            0,
        )
        assert rc == EXIT_UNTRACEABLE

    def test_non_component_config_is_usage_error(self):
        # Trajectory from_config returns a schedule array, not a component.
        rc = cmd_trace(TRAJ, {"t": ()}, {}, 5, 0)
        assert rc == EXIT_USAGE


class TestCLI:
    def test_inventory(self):
        result = _run()
        assert result.returncode == EXIT_OK
        assert "controllers" in result.stdout
        assert "LQR" in result.stdout
        assert "KalmanFilter" in result.stdout

    def test_contract(self):
        result = _run("--list", LQR)
        assert result.returncode == EXIT_OK
        assert "LQR.compute(current_state, target_state)" in result.stdout

    def test_trace_cli(self):
        result = _run(LQR, "--shape", "current_state=3", "--shape", "target_state=3")
        assert result.returncode == EXIT_OK
        assert "bit-exact" in result.stdout
