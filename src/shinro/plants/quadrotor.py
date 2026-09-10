from dataclasses import dataclass

from shinro.components import Plant
from shinro.factories.registry import register_plant


@dataclass(frozen=True)
class QuadrotorConfig:
    """Strict TOML schema for :class:`Quadrotor`.

    Placeholder schema documenting the intended 12D/4D parameter surface (see
    ``configs/plants/quadrotor.toml``); construction raises NotImplementedError.
    """

    mass: float = 0.5
    radius: float = 0.1
    inertia: list[float] | None = None
    thrust_coeff: float = 1.0
    torque_coeff: float = 0.1
    dt: float = 0.01
    g: float = 9.81
    name: str = "quadrotor"


@register_plant("Quadrotor")
class Quadrotor(Plant):
    """Quadrotor — follows HolonomicMobileRobot pattern.

    State: 12D (pose + twist)
    Control: 4D (thrust + body torques) — higher-level abstraction TBD

    TODO: implement standalone dynamics + MuJoCo engine mode
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Quadrotor plant is a placeholder — not yet implemented")

    def get_state(self):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def step(self, u):
        raise NotImplementedError

    def physics_engine(self, engine):
        raise NotImplementedError

    Config = QuadrotorConfig

    @classmethod
    def from_config(cls, config, backend=None):
        raise NotImplementedError
