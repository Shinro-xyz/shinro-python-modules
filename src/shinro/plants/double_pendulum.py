import numpy as np
from shinro.components import Plant
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.factories.registry import register_plant

@register_plant("DoublePendulum")
class DoublePendulum(Plant):
    def __init__(self,mass_top:float, mass_bottom:float, length_top:float, length_bottom:float,dt:float):
        self.m1=mass_top
        self.m2=mass_bottom
        self.l1=length_top
        self.l2=length_bottom
        self.dt=dt

    def dynamics(self, state, control):
        