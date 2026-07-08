_CONTROLLER_REGISTRY = {}
_ESTIMATOR_REGISTRY = {}
_TRAJECTORY_REGISTRY = {}


def register_controller(name):
    def decorator(cls):
        cls._registry_name = name
        _CONTROLLER_REGISTRY[name] = cls
        return cls
    return decorator


def register_estimator(name):
    def decorator(cls):
        cls._registry_name = name
        _ESTIMATOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_trajectory(name):
    def decorator(cls):
        cls._registry_name = name
        _TRAJECTORY_REGISTRY[name] = cls
        return cls
    return decorator