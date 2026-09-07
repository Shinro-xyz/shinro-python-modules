import tomllib

from shinro.factories.registry import _ESTIMATOR_REGISTRY
from shinro.utils.array_backend import ArrayBackend
from shinro.utils.config_resolver import resolve_config_path


class EstimatorFactory:
    def __init__(self, config_path: str | None = None, config: dict | None = None):
        if (config_path is None) == (config is None):
            raise ValueError("EstimatorFactory requires exactly one of config_path or config.")
        if config_path is not None:
            with open(resolve_config_path(config_path), "rb") as f:
                config = tomllib.load(f)
        self.config = config

    def create(self, backend: ArrayBackend = None):
        cls = _ESTIMATOR_REGISTRY[self.config["type"]]
        return cls.from_config(self.config, backend=backend)
