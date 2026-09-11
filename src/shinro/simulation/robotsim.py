# FILE: simulation/robotsim.py
"""
Generic robot simulation factory. Reads a YAML config and creates the
engine, plants, and wiring automatically.

Usage:
    from shinro.simulation import RobotSim
    sim = RobotSim("robot_config.toml")
    sim.arm.step(twist)
    sim.base.step(velocity)
    sim.step()
    state = sim.get_state()
"""

import tomllib
from typing import Any

import mujoco
import numpy as np

from shinro.physics_engine.mujoco import MuJoCoEngine
from shinro.utils.config_resolver import resolve_config_path


class RobotSim:
    """
    Generic robot simulation factory. Reads a TOML config and creates the
    engine, plants, and wiring automatically.

    Usage:
        sim = RobotSim("robot_config.toml")
        sim.arm.step(twist)
        sim.base.step(velocity)
        sim.step()
        state = sim.get_state()
    """

    def __init__(self, config_path: str, xml_string: str = None, assets: dict = None):
        with open(resolve_config_path(config_path), "rb") as f:
            self.config = tomllib.load(f)

        dt = self.config.get("dt", 0.02)
        model_path = self.config.get("model", "")

        if xml_string is not None:
            self.engine = MuJoCoEngine(dt=dt, xml_string=xml_string, assets=assets)
        else:
            self.engine = MuJoCoEngine(model_path=model_path, dt=dt)

        joint_groups = self.config.get("joint_groups", {})

        self._plants = {}
        for plant_cfg in self.config.get("plants", []):
            ptype = plant_cfg["type"]
            pname = plant_cfg["name"]

            from shinro.factories.registry import _PLANT_REGISTRY

            cls = _PLANT_REGISTRY[ptype]
            plant_config = {**plant_cfg, "joint_groups": joint_groups, "engine": self.engine, "dt": dt}
            plant = cls.from_config(plant_config)

            self._plants[pname] = plant
            setattr(self, pname, plant)

    def reset(self):
        self.engine.reset()
        for name, plant in self._plants.items():
            if hasattr(plant, "state") and isinstance(plant.state, np.ndarray):
                plant.state = np.zeros_like(plant.state)

    def step(self):
        """Advance the world one tick, then let each plant reconcile.

        All physics runs through the engine; plants that self-integrate their
        own dynamics (e.g. an analytical wheeled base) implement
        ``Plant.post_engine_step`` to write their state back into the engine —
        nothing in this factory is specific to any robot.
        """
        self.engine.step()
        for plant in self._plants.values():
            plant.post_engine_step(self.engine)

    def get_state(self) -> dict:
        return self.engine.get_sensor_data()

    @property
    def plants(self) -> dict[str, Any]:
        """Read-only mapping of plant name → plant instance."""
        return dict(self._plants)

    def get_plant(self, name: str) -> Any:
        return self._plants.get(name)


def interactive_viewer(model_path: str):
    """Open an interactive MuJoCo viewer for manual inspection."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Interactive viewer opened. Close window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
