from shinro.components import Plant
from shinro.factories.registry import register_plant


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

    @classmethod
    def from_config(cls, config, backend=None):
        raise NotImplementedError
