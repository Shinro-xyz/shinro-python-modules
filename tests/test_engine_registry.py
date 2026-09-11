"""Engine registry: [engine] manifest section resolves via _ENGINE_REGISTRY.

Stage 1 of the sim generalization: engine selection is TOML-driven like every
other component. Legacy manifests (top-level model/dt, no [engine]) still work
with a deprecation warning and produce identical simulations.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco optional extra not installed")

from shinro.factories.registry import _ENGINE_REGISTRY, register_engine
from shinro.physics_engine.mujoco import MuJoCoEngine, MuJoCoEngineConfig
from shinro.simulation.robotsim import RobotSim
from shinro.utils.config_resolver import resolve_config_path

FREE_ROBOT_XML = """
<mujoco model="free_base_test">
  <worldbody>
    <body name="chassis" pos="0 0 0.1">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.05" mass="5"/>
      <body name="wheel1" pos="0.12 0 0">
        <joint name="drive_1" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.09 0.02" mass="0.5"/>
      </body>
      <body name="wheel2" pos="-0.06 0.104 0">
        <joint name="drive_2" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.09 0.02" mass="0.5"/>
      </body>
      <body name="wheel3" pos="-0.06 -0.104 0">
        <joint name="drive_3" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.09 0.02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="drive_1" joint="drive_1"/>
    <motor name="drive_2" joint="drive_2"/>
    <motor name="drive_3" joint="drive_3"/>
  </actuator>
</mujoco>
"""

PLANTS_BLOCK = """
[joint_groups]
drive_joints = ["drive_1", "drive_2", "drive_3"]

[[plants]]
type = "HolonomicMobileRobot"
name = "base"
num_wheels = 3
radius_robots = 0.12
gamma = -1.57079632679
radius_wheels = 0.09
"""


@pytest.fixture
def xml_path(tmp_path):
    p = tmp_path / "free_robot.xml"
    p.write_text(FREE_ROBOT_XML)
    return str(p)


def _run(sim, n=10):
    for _ in range(n):
        sim.base.step(np.array([0.3, 0.1, 0.2]))
        sim.step()
    return sim.engine.data.qpos.copy(), np.asarray(sim.base.get_state())


class TestEngineRegistry:
    def test_mujoco_registered(self):
        assert _ENGINE_REGISTRY["mujoco"] is MuJoCoEngine
        assert MuJoCoEngine._registry_name == "mujoco"

    def test_unknown_engine_type_loud(self, tmp_path, xml_path):
        manifest = tmp_path / "bad.toml"
        manifest.write_text(f'[engine]\ntype = "gazebo"\nmodel = "{xml_path}"\n' + PLANTS_BLOCK)
        with pytest.raises(KeyError, match="Unknown engine type 'gazebo'"):
            RobotSim(str(manifest))

    def test_config_typo_loud(self, tmp_path, xml_path):
        manifest = tmp_path / "bad.toml"
        manifest.write_text(f'[engine]\ntype = "mujoco"\nmodle = "{xml_path}"\n' + PLANTS_BLOCK)
        with pytest.raises(ValueError, match="unknown key"):
            RobotSim(str(manifest))

    def test_missing_model_and_xml_string_loud(self):
        with pytest.raises(ValueError, match="one of 'model' or 'xml_string'"):
            MuJoCoEngine.from_config(MuJoCoEngineConfig())


class TestEngineManifestEquivalence:
    """[engine] section form ≡ legacy top-level form, bit-exact."""

    def test_equivalence(self, tmp_path, xml_path):
        legacy = tmp_path / "legacy.toml"
        legacy.write_text(f'model = "{xml_path}"\ndt = 0.02\n' + PLANTS_BLOCK)
        sectioned = tmp_path / "sectioned.toml"
        sectioned.write_text(f'[engine]\ntype = "mujoco"\nmodel = "{xml_path}"\ndt = 0.02\n' + PLANTS_BLOCK)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy_result = _run(RobotSim(str(legacy)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            sectioned_result = _run(RobotSim(str(sectioned)))  # no warning = no fallback taken

        assert np.array_equal(legacy_result[0], sectioned_result[0])
        assert np.array_equal(legacy_result[1], sectioned_result[1])

    def test_legacy_form_warns(self, tmp_path, xml_path):
        legacy = tmp_path / "legacy.toml"
        legacy.write_text(f'model = "{xml_path}"\ndt = 0.02\n' + PLANTS_BLOCK)
        with pytest.warns(DeprecationWarning, match="\\[engine\\]"):
            RobotSim(str(legacy))

    def test_shipped_manifest_still_builds(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sim = RobotSim(str(resolve_config_path("robot_config.toml")))
        assert isinstance(sim.engine, MuJoCoEngine)
        assert set(sim.plants) == {"arm", "base"}


def test_register_engine_enforces_config():
    """Engines without a frozen Config dataclass are rejected at registration."""
    with pytest.raises(TypeError, match="does not define a frozen"):

        @register_engine("test_no_config")
        class _NoConfig:
            pass
