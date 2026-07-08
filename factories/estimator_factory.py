import yaml
from factories.registry import _ESTIMATOR_REGISTRY


class EstimatorFactory:
    """Creates StateEstimator instances from YAML config files via the registry.

    Usage:
        est = EstimatorFactory("configs/estimators/luenberger_base.yaml").create()
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def create(self):
        cls = _ESTIMATOR_REGISTRY[self.config["type"]]
        return cls.from_config(self.config)