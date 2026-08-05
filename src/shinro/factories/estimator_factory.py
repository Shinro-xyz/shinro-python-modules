import tomllib

from shinro.factories.registry import _ESTIMATOR_REGISTRY
from shinro.utils.array_backend import ArrayBackend
from shinro.utils.config_resolver import resolve_config_path


class EstimatorFactory:
    def __init__(self, config_path: str):
        with open(resolve_config_path(config_path), "rb") as f:
            self.config = tomllib.load(f)

    def create(self, backend: ArrayBackend = None):
        cls = _ESTIMATOR_REGISTRY[self.config["type"]]
        return cls.from_config(self.config, backend=backend)
