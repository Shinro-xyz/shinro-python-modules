import tomllib
from factories.registry import _TRAJECTORY_REGISTRY
from utils.array_backend import ArrayBackend


class TrajectoryFactory:
    def __init__(self, config_path: str):
        with open(config_path, "rb") as f:
            self.config = tomllib.load(f)

    def create(self, backend: ArrayBackend = None):
        cls = _TRAJECTORY_REGISTRY[self.config["type"]]
        return cls.from_config(self.config, backend=backend)
