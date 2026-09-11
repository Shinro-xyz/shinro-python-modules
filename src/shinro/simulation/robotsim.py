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
import warnings
from typing import Any

import numpy as np

from shinro.factories.registry import _ENGINE_REGISTRY
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

        engine_cfg = self.config.get("engine")
        if engine_cfg is None:
            # Back-compat: legacy manifests declare model/dt at the top level.
            warnings.warn(
                "Sim manifest is missing an [engine] section — defaulting to "
                "'mujoco' with the top-level 'model'/'dt' keys. Add an explicit "
                "[engine] table (type, model, dt); this fallback will be removed.",
                DeprecationWarning,
                stacklevel=2,
            )
            engine_cfg = {"type": "mujoco", "model": self.config.get("model", ""), "dt": self.config.get("dt", 0.02)}
        if xml_string is not None:
            engine_cfg = {**engine_cfg, "xml_string": xml_string}

        engine_type = engine_cfg.get("type")
        if engine_type not in _ENGINE_REGISTRY:
            # Engines register on module import; load shinro.physics_engine.<type>
            # on demand so the mujoco import stays lazy (optional extra) and a
            # third-party engine lands by module name.
            import importlib

            try:
                importlib.import_module(f"shinro.physics_engine.{engine_type}")
            except ModuleNotFoundError:
                pass  # fall through to the registry error below
        if engine_type not in _ENGINE_REGISTRY:
            raise KeyError(f"Unknown engine type '{engine_type}'. Registered: {sorted(_ENGINE_REGISTRY)}")
        self.engine = _ENGINE_REGISTRY[engine_type].from_config(engine_cfg, assets=assets)
        dt = self.engine.dt

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
