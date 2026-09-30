import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass

_CONTROLLER_REGISTRY = {}
_ESTIMATOR_REGISTRY = {}
_TRAJECTORY_REGISTRY = {}
_PLANT_REGISTRY = {}
_PLANT_DETECTOR_REGISTRY = {}
_ENGINE_REGISTRY = {}
_PHYSICS_PRESET_REGISTRY: dict[str, Callable[[dict], "PhysicsModel"]] = {}


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

    Detectors are non-exclusive — multiple can fire for the same XML (e.g., a
    mobile manipulator produces ArmRobot + HolonomicMobileRobot). The generator
    collects all matches and produces one ``[[plants]]`` entry per match.
    """

    def decorator(fn):
        _PLANT_DETECTOR_REGISTRY[plant_type] = fn
        return fn

    return decorator


def register_engine(name):
    """Register a physics engine (MuJoCo, Box2D, Gazebo, ...) by name.

    Same contract as the component registries: a frozen ``Config`` dataclass
    declares the engine's TOML schema, parsed strictly at build time so
    manifest typos surface at parse, not inside the simulator backend.
    """

    def decorator(cls):
        _check_config(cls, "PhysicsEngine", name, strict=True)
        cls._registry_name = name
        _ENGINE_REGISTRY[name] = cls
        return cls

    return decorator


@dataclass(frozen=True)
class PhysicsModel:
    """Engine model produced by a physics preset.

    ``xml_string`` is an MJCF document as text and ``assets`` maps mesh
    filename -> bytes, exactly the pair :class:`~shinro.simulation.robotsim.RobotSim`
    forwards to the engine.
    """

    xml_string: str
    assets: dict[str, bytes] = field(default_factory=dict)


PhysicsPreset = Callable[[dict], PhysicsModel]


def register_physics_preset(name: str) -> Callable[[PhysicsPreset], PhysicsPreset]:
    """Register a factory for a scenario's ``[physics].preset = "<name>"``.

    shinro ships **no** presets: a preset is a robot-specific way to turn a
    ``[physics]`` section into an engine model (load an MJCF, rewrite its tree,
    bundle its mesh assets). Robot packages register the presets they need; the
    module must be imported before use, which the CLIs' ``--import MODULE`` does.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: PhysicsPreset) -> PhysicsPreset:
        if name in _PHYSICS_PRESET_REGISTRY:
            raise ValueError(f"physics preset '{name}' is already registered")
        _PHYSICS_PRESET_REGISTRY[name] = fn
        return fn

    return decorator


def available_physics_presets() -> list[str]:
    """Return the registered physics preset names, sorted."""
    return sorted(_PHYSICS_PRESET_REGISTRY)


def resolve_physics_preset(name: str) -> PhysicsPreset:
    """Look up a registered physics preset by name.

    Raises:
        ValueError: If ``name`` is not registered, naming the available presets
            and pointing at the import step.
    """
    try:
        return _PHYSICS_PRESET_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown [physics].preset '{name}'. Registered: {available_physics_presets()}. "
            "A preset is provided by a robot package and must be imported before use "
            "(the CLI's --import MODULE does this)."
        ) from None
