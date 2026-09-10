import tomllib

from shinro.factories.registry import _CONTROLLER_REGISTRY
from shinro.utils.array_backend import ArrayBackend
from shinro.utils.config_resolver import resolve_config_path


class ControllerFactory:
    def __init__(self, config_path: str | None = None, config: dict | None = None):
        if (config_path is None) == (config is None):
            raise ValueError("ControllerFactory requires exactly one of config_path or config.")
        if config_path is not None:
            with open(resolve_config_path(config_path), "rb") as f:
                config = tomllib.load(f)
        self.config = config

    def create(self, backend: ArrayBackend = None, plant=None, derive_model: bool = False):
        """Construct the registered controller from the config.

        Args:
            backend: Array backend for the controller.
            plant: Optional plant for derived-value injection — ``dt`` is
                filled from ``plant.dt`` when the config omits it (loud error
                on disagreement) and, with ``derive_model``,
                ``A_dynamics``/``B_dynamics`` are derived when both are absent.
            derive_model: Derive the model from the plant (plant-only mode).
        """
        cls = _CONTROLLER_REGISTRY[self.config["type"]]
        if plant is not None:
            return cls.from_config(cls.load_config(self.config, plant=plant, derive_model=derive_model), backend=backend)
        return cls.from_config(self.config, backend=backend)
