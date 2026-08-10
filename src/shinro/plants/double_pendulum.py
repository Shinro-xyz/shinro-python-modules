import numpy as np

from shinro.components import Plant
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.factories.registry import register_plant

@register_plant( "DoublePendulum")
class DoublePendulum(Plant):
    def __init__(self) -> None:
        pass

    def get_model(self, *args: Any, **kwargs: Any) -> Any:
        return super().get_model(*args, **kwargs)

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        return super().get_state(*args, **kwargs)

    def dynamics(self, state: Any, control: Any) -> Any:
        return super().dynamics(state, control)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return super().step(*args, **kwargs)

    
    