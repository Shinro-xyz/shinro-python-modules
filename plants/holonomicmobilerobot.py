from typing import Optional
from components import Plant
from factories.registry import register_plant
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_plant("HolonomicMobileRobot")
class HolonomicMobileRobot(Plant):
    def __init__(
        self,
        num_wheels: int,
        radius_robots: float,
        gamma: float,
        radius_wheels: float,
        dt: float,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.n = num_wheels
        self.R = radius_robots
        self.gamma = gamma
        self.r = radius_wheels
        self.dt = dt
        self.state = self.bk.zeros(3)
        self.A_kinematics, self.A_pinv_kin = self._build_kinematics()
        self._engine = None

    def physics_engine(self, engine):
        self._engine = engine
        if engine is not None:
            self.bk = engine.backend

    def _build_kinematics(self):
        theta_per_wheel = 2 * np.pi / self.n
        angle_list = [i * theta_per_wheel + self.gamma for i in range(self.n)]
        sin_list = np.sin(angle_list)
        cos_list = np.cos(angle_list)
        A_kin = np.column_stack((sin_list, -cos_list, np.full_like(sin_list, -self.R)))
        return self.bk.from_numpy(A_kin), self.bk.from_numpy(np.linalg.pinv(A_kin))

    def _rot_matrix(self, theta):
        c = self.bk.cos(theta)
        s = self.bk.sin(theta)
        return self.bk.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    def step(self, u_world):
        theta = self.state[2]
        rot_matrix = self._rot_matrix(theta)
        u_body = rot_matrix @ u_world
        wheel_speeds = (1.0 / self.r) * self.A_kinematics @ u_body
        self.state = self.state + u_world * self.dt
        if self._engine is not None:
            self._target_wheel_delta = wheel_speeds * self.dt
        return wheel_speeds

    def set_pose(self, x: float, y: float, theta: float):
        self.state = self.bk.array([x, y, theta])

    def get_state(self):
        return self.bk.copy(self.state)

    def get_model(self):
        A = self.bk.eye(3)
        B = self.dt * self.bk.eye(3)
        return A, B

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        plant = cls(
            num_wheels=config["num_wheels"],
            radius_robots=config["radius_robots"],
            gamma=config["gamma"],
            radius_wheels=config["radius_wheels"],
            dt=config["dt"],
            backend=bk,
        )
        engine = config.get("engine")
        if engine is not None:
            plant.physics_engine(engine)
        return plant
