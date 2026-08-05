"""Shared fixtures for the full-loop integration test environment.

The integration suite requires MuJoCo plus the LeKiwi MJCF/mesh assets. When
they are unavailable the whole suite is skipped via :func:`pytest.importorskip`.
The suite is also excluded from the default ``make test`` run through the
``integration`` pytest marker (see ``pyproject.toml``).
"""

from pathlib import Path

import pytest

from demos.helpers import inject_free_joint, load_model_assets
from lekiwi_sim import HERE, MJCF_PATH


@pytest.fixture(scope="session")
def mujoco_available():
    """Skip everything if MuJoCo or the LeKiwi assets are missing."""
    mujoco = pytest.importorskip("mujoco")
    mjcf = Path(MJCF_PATH)
    if not mjcf.exists():
        pytest.skip(f"LeKiwi MJCF not found at {MJCF_PATH}")
    if not (HERE / "lekiwi-sim" / "meshes").exists():
        pytest.skip("LeKiwi mesh assets directory missing")
    return mujoco


@pytest.fixture(scope="session")
def lekiwi_xml(mujoco_available):
    """Raw LeKiwi MJCF document as a string."""
    return Path(MJCF_PATH).read_text()


@pytest.fixture(scope="session")
def lekiwi_assets(mujoco_available):
    """Dict of mesh filename -> bytes for ``xml_string`` loading."""
    return load_model_assets(HERE / "lekiwi-sim" / "meshes")


@pytest.fixture(scope="session")
def lekiwi_xml_freejoint(mujoco_available, lekiwi_xml):
    """LeKiwi MJCF with the arm nested under the base and a free joint injected."""
    return inject_free_joint(lekiwi_xml)
