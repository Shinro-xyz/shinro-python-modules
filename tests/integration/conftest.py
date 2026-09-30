"""Shared fixtures for the full-loop integration test environment.

The integration suite requires MuJoCo; it is also excluded from the default
``make test`` run through the ``integration`` pytest marker (see
``pyproject.toml``).

This module intentionally performs **no MuJoCo-dependent imports at module
level**. The ``mujoco`` import is delayed into the :func:`mujoco_available`
fixture, so an environment without the optional extra skips cleanly at fixture
time instead of raising a collection-time import error.
"""

import pytest


@pytest.fixture(scope="session")
def mujoco_available():
    """Skip the suite when the optional ``[mujoco]`` extra is not installed."""
    pytest.importorskip("mujoco")
    return True
