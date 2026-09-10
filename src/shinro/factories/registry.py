import warnings
from dataclasses import is_dataclass

_CONTROLLER_REGISTRY = {}
_ESTIMATOR_REGISTRY = {}
_TRAJECTORY_REGISTRY = {}
_PLANT_REGISTRY = {}
_PLANT_DETECTOR_REGISTRY = {}


def _check_config(cls, kind: str, name: str, *, strict: bool) -> None:
    """Require a frozen Config dataclass on a registered component.

    The dataclass's fields ARE the component's config schema, parsed strictly
    by ``ConfigDriven.parse_config`` (see shinro/components.py).

    Args:
        cls: The component class being registered.
        kind: Human-readable role ("Controller", "Estimator", ...).
        name: The registered name.
        strict: Raise when ``Config`` is missing; otherwise emit a UserWarning.
            Strict for every registry except controllers (warning-only until the
            onnx_rl / lerobot_diffusion adapters — both pending rewrite, their
            NN runtimes are untraceable — define Config).
    """
    if is_dataclass(getattr(cls, "Config", None)):
        return
    msg = (
        f"{kind} '{cls.__name__}' (registered as '{name}') does not define a frozen "
        f"Config dataclass — see LQRConfig in controllers/lqr.py for the pattern."
    )
    if strict:
        raise TypeError(msg)
    warnings.warn(msg + " This will become a hard error.", UserWarning, stacklevel=3)


def register_controller(name):
    def decorator(cls):
        # TODO: strict=True once onnx_rl / lerobot_diffusion adapters are rewritten.
        _check_config(cls, "Controller", name, strict=False)
        cls._registry_name = name
        _CONTROLLER_REGISTRY[name] = cls
        return cls

    return decorator


def register_estimator(name):
    def decorator(cls):
        _check_config(cls, "Estimator", name, strict=True)
        cls._registry_name = name
        _ESTIMATOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_trajectory(name):
    def decorator(cls):
        _check_config(cls, "Trajectory", name, strict=True)
        cls._registry_name = name
        _TRAJECTORY_REGISTRY[name] = cls
        return cls

    return decorator


def register_plant(name):
    def decorator(cls):
        _check_config(cls, "Plant", name, strict=True)
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
