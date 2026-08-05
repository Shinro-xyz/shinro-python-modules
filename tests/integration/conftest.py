"""Shared fixtures for the full-loop integration test environment.

The integration suite requires MuJoCo plus the LeKiwi MJCF/mesh assets. It is
also excluded from the default ``make test`` run through the ``integration``
pytest marker (see ``pyproject.toml``).

This module intentionally performs **no MuJoCo-dependent imports at module level**.
All imports of ``mujoco``, ``demos.helpers``, or other MuJoCo-backed modules are
delayed until fixture execution, and each fixture calls
:func:`pytest.importorskip("mujoco")` first so that environments without MuJoCo
skip cleanly instead of raising a collection-time import error.
"""

from pathlib import Path

import pytest

# LeKiwi repo layout constants (pure pathlib, no MuJoCo needed).
_LEKIWI_HOME = Path(__file__).parent.parent.parent
_MJCF_PATH = _LEKIWI_HOME / "lekiwi-sim" / "mjcf_lcmm_robot.xml"
_MESH_DIR = _LEKIWI_HOME / "lekiwi-sim" / "meshes"


@pytest.fixture(scope="session")
def mujoco_available():
    """Skip the suite if MuJoCo or the LeKiwi assets are missing.

    The :func:`pytest.importorskip` is inside the fixture body so it is only
    evaluated when a test actually needs this fixture. When the ``integration``
    marker deselects the tests, this fixture never runs and no import is attempted.
    """
    pytest.importorskip("mujoco")
    if not _MJCF_PATH.exists():
        pytest.skip(f"LeKiwi MJCF not found at {_MJCF_PATH}")
    if not _MESH_DIR.exists():
        pytest.skip("LeKiwi mesh assets directory missing")
    return True


@pytest.fixture(scope="session")
def lekiwi_xml(mujoco_available):
    """Raw LeKiwi MJCF document as a string."""
    return _MJCF_PATH.read_text()


@pytest.fixture(scope="session")
def lekiwi_assets(mujoco_available):
    """Dict of mesh filename -> bytes for ``xml_string`` loading."""
    from demos.helpers import load_model_assets

    return load_model_assets(_MESH_DIR)


@pytest.fixture(scope="session")
def lekiwi_xml_freejoint(mujoco_available, lekiwi_xml):
    """LeKiwi MJCF with the arm nested under the base and a free joint injected."""
    from demos.helpers import inject_free_joint

    return inject_free_joint(lekiwi_xml)
