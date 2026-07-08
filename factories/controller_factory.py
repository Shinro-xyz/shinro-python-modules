import yaml
from factories.registry import _CONTROLLER_REGISTRY


class ControllerFactory:
    """Creates Controller instances from YAML config files via the registry.

    Usage:
        ctrl = ControllerFactory("configs/controllers/lqr_base.yaml").create()
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def create(self):
        cls = _CONTROLLER_REGISTRY[self.config["type"]]
        return cls.from_config(self.config)