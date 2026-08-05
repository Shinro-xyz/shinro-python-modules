"""Shared assertion helpers for the integration suite.

Tolerances live in the scenario TOML (``[scenario.tolerance]``) so the tests
stay declarative. These helpers unpack the ``StepRecord`` history and apply the
declared thresholds.
"""

import numpy as np

from tests.integration.helpers.scenario_runner import StepRecord


def tracking_error(records: list[StepRecord]) -> np.ndarray:
    """Per-step tracking error ``||reference - plant_state||``.

    Args:
        records: Run history from :func:`run_scenario`.

    Returns:
        Array of shape (n_steps,).
    """
    return np.array(
        [
            float(np.linalg.norm(np.asarray(r.reference) - np.asarray(r.plant_state)))
            for r in records
        ]
    )


def estimator_error(records: list[StepRecord]) -> np.ndarray:
    """Per-step estimator error ``||estimated - true_state||``.

    Args:
        records: Run history from :func:`run_scenario`.

    Returns:
        Array of shape (n_steps,).
    """
    return np.array(
        [
            float(np.linalg.norm(np.asarray(r.estimated) - np.asarray(r.true_state)))
            for r in records
        ]
    )


def assert_steady_state(records: list[StepRecord], tolerance: float, tail: int = 100) -> None:
    """Assert the mean tracking error over the last ``tail`` steps is bounded.

    Args:
        records: Run history.
        tolerance: Allowed mean tracking error.
        tail: Number of trailing steps to average.

    Raises:
        AssertionError: If the trailing mean error exceeds ``tolerance``.
    """
    errs = tracking_error(records)
    mean = float(np.mean(errs[-tail:]))
    assert mean <= tolerance, f"steady-state tracking error {mean:.4f} > tolerance {tolerance}"


def assert_estimator_recovery(records: list[StepRecord], tolerance: float, tail: int = 100) -> None:
    """Assert the estimator error stays bounded over the trailing steps.

    Args:
        records: Run history.
        tolerance: Allowed mean estimator error.
        tail: Number of trailing steps to average.

    Raises:
        AssertionError: If the trailing mean estimator error exceeds ``tolerance``.
    """
    errs = estimator_error(records)
    mean = float(np.mean(errs[-tail:]))
    assert mean <= tolerance, f"estimator error {mean:.4f} > tolerance {tolerance}"


def assert_finite_state(records: list[StepRecord]) -> None:
    """Assert every plant state in the history is finite.

    Args:
        records: Run history.

    Raises:
        AssertionError: If any recorded plant state contains NaN/Inf.
    """
    for r in records:
        assert np.all(np.isfinite(r.plant_state)), (
            f"non-finite plant state at t={r.t:.3f}: {r.plant_state}"
        )
