from dataclasses import dataclass

from shinro.components import Plant
from shinro.factories.registry import register_plant
import numpy as np
from shinro.utils.array_backend import ArrayBackend, NumpyBackend

@dataclass(frozen=True)
class QuadrotorConfig:
    """Strict TOML schema for :class:`Quadrotor`.

    Placeholder schema documenting the intended 12D/4D parameter surface (see
    ``samples/plants/quadrotor.toml``); construction raises NotImplementedError.
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

    def __init__(self,mass: float,radius: float,inertia: list[float],dt:float,g:float=9.81,thrust_coeff: float=1.0,torque_coeff: float=0.1, backend: ArrayBackend | None= None):
            self.m= mass
            self.I=inertia
            self.dt=dt
            self.r=radius
            self.g=g
            self.k=thrust_coeff
            self.b=torque_coeff
            self.input_dim=4
            self.bk= backend or NumpyBackend()

    def _rotation_matrix(self, roll, pitch, yaw):
        R_x=np.array(
            [[1,0,0],
            [0,np.cos(roll), -np.sin(roll)],
            [0,np.sin(roll),np.cos(roll)]])
        R_y= self.bk.array(
            [[self.bk.cos(pitch),0,self.bk.sin(pitch)],
            [0,1,0],
            [-self.bk.sin(pitch), 0, self.bk.cos(pitch)]])
        R_z= np.array(
            [[np.cos(yaw),-np.sin(yaw),0],
            [np.sin(yaw), np.cos(yaw),0],
            [0,0,1]]
        )
        return R_z@R_y@R_x

    def _angular_transformation_matrix(self,roll, pitch,yaw):
        T=np.array(
            [[1, np.sin(roll)*np.tan(pitch), np.cos(roll)*np.tan(pitch)],
            [0, np.cos(roll),-np.sin(roll)],
            [0,np.sin(roll)/np.cos(pitch), np.cos(roll)/np.cos(pitch)]]
        )
        return T
    def get_dynamics(self, state, control, bk=None):
        