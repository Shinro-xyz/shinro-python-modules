from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type checkers / IDEs only — never runs at import
    from .robotsim import RobotSim

__all__ = ["RobotSim"]


def __getattr__(name):
    """Lazy attribute access (PEP 562).

    ``RobotSim`` imports mujoco at module load, which is an optional extra.
    Deferring the import means ``import shinro.simulation.runner`` (and
    plant-only scenarios) work on a minimal install without mujoco; the
    physics engine only loads when someone actually builds a ``RobotSim``.
    """
    if name == "RobotSim":
        from .robotsim import RobotSim

        return RobotSim
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
