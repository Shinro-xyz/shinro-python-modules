_CONTROLLER_REGISTRY = {}
_ESTIMATOR_REGISTRY = {}
_TRAJECTORY_REGISTRY = {}
_PLANT_REGISTRY = {}
_PLANT_DETECTOR_REGISTRY = {}


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


def register_plant(name):
    def decorator(cls):
        cls._registry_name = name
        _PLANT_REGISTRY[name] = cls
        return cls
    return decorator


def register_plant_detector(plant_type):
    """Register a detector function that identifies a plant type from an MJCF XML tree.

    The detector receives an ``xml.etree.ElementTree.Element`` (the root of the MJCF
    document) and returns ``True`` if the XML matches the plant type.

    Detectors are non-exclusive — multiple can fire for the same XML (e.g., LeKiwi
    produces ArmRobot + HolonomicMobileRobot). The generator collects all matches
    and produces one ``[[plants]]`` entry per match.
    """
    def decorator(fn):
        _PLANT_DETECTOR_REGISTRY[plant_type] = fn
        return fn
    return decorator
