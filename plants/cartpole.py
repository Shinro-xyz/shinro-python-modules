from typing import Optional
from components import Plant, PhysicsEngine
from factories.registry import register_plant, register_plant_detector
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_plant("CartPole")
class CartPole(Plant):
    def __init__(
        self,
        cart_mass: float = 0.5,
        pole_mass: float = 0.1,
        pole_length: float = 0.5,
        damping: float = 0.0,
        gravity: float = 9.81,
        dt: float = 0.01,
        track_limits: Optional[tuple] = None,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.M = cart_mass
        self.m = pole_mass
        self.l = pole_length
        self.b = damping
        self.g = gravity
        self.dt = dt
        self.track_limits = track_limits
        self.state = self.bk.zeros(4)
        self._engine = None

    def physics_engine(self, engine: Optional[PhysicsEngine]):
        self._engine = engine
        if engine is not None:
            self.bk = engine.backend
            self.state = self.bk.zeros(4)
        else:
            self.bk = NumpyBackend()
            self.state = self.bk.zeros(4)

    def get_state(self):
        if self._engine is not None:
            x = self._engine.get_joint_qpos("slider")
            x_dot = self._engine.get_joint_vel("slider")
            theta = self._engine.get_joint_qpos("hinge")
            theta_dot = self._engine.get_joint_vel("hinge")
            return self.bk.array([x, x_dot, theta, theta_dot])
        return self.bk.copy(self.state)

    def get_model(self):
        M, m, l, g, b = self.M, self.m, self.l, self.g, self.b
        A = self.bk.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -m * g / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (M + m) * g / (M * l), -b / (M * l**2)],
        ])
        B = self.bk.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * l)]])
        return A, B

    def _compute_accels(self, x, theta, x_dot, theta_dot, F):
        M, m, l, g, b = self.M, self.m, self.l, self.g, self.b
        sin_theta = self.bk.sin(theta)
        cos_theta = self.bk.cos(theta)
        denom = l - m * l * cos_theta**2 / (M + m)
        theta_ddot = (g * sin_theta - cos_theta * (F + m * l * theta_dot**2 * sin_theta) / (M + m)) / denom
        x_ddot = (F + m * l * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)) / (M + m)
        return x_ddot, theta_ddot

    def dynamics(self, state, control):
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
        F = control[0] if hasattr(control, '__len__') else control
        x_ddot, theta_ddot = self._compute_accels(x, theta, x_dot, theta_dot, F)
        return self.bk.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def step(self, u):
        if self._engine is not None:
            self._engine.set_joint_ctrl("slider", u[0] if hasattr(u, '__len__') else u)
            self._engine.step()
            self.state = self.get_state()
            return self.state

        x, x_dot, theta, theta_dot = self.state[0], self.state[1], self.state[2], self.state[3]
        F = u[0] if hasattr(u, '__len__') else u
        x_ddot, theta_ddot = self._compute_accels(x, theta, x_dot, theta_dot, F)
        theta_dot_new = theta_dot + theta_ddot * self.dt
        x_dot_new = x_dot + x_ddot * self.dt
        theta_new = theta + theta_dot_new * self.dt
        x_new = x + x_dot_new * self.dt
        self.state = self.bk.array([x_new, x_dot_new, theta_new, theta_dot_new])
        if self.track_limits is not None:
            self.state = self.bk.array([
                self.bk.clip(x_new, self.track_limits[0], self.track_limits[1]),
                x_dot_new, theta_new, theta_dot_new,
            ])
        return self.state

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        track_limits = None
        if "track_limits" in config:
            tl = config["track_limits"]
            track_limits = (tl[0], tl[1])
        plant = cls(
            cart_mass=config.get("cart_mass", 0.5),
            pole_mass=config.get("pole_mass", 0.1),
            pole_length=config.get("pole_length", 0.5),
            damping=config.get("damping", 0.0),
            gravity=config.get("gravity", 9.81),
            dt=config.get("dt", 0.01),
            track_limits=track_limits,
            backend=bk,
        )
        engine = config.get("engine")
        if engine is not None:
            plant.physics_engine(engine)
        return plant


@register_plant_detector("CartPole")
def detect_cartpole(xml_root):
    """Detect CartPole: 1 slide joint + 1 hinge joint + 1 motor actuator."""
    joints = xml_root.findall('.//joint')
    actuators = xml_root.findall('.//actuator/*')
    if len(joints) != 2 or len(actuators) != 1:
        return False
    types = [j.get('type') for j in joints]
    return 'slide' in types and 'hinge' in types and actuators[0].tag == 'motor'
