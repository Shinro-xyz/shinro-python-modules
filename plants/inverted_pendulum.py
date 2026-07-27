from typing import Optional
from components import Plant, PhysicsEngine
from factories.registry import register_plant, register_plant_detector
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_plant("InvertedPendulum")
class InvertedPendulum(Plant):
    def __init__(
        self,
        mass: float = 0.1,
        length: float = 0.5,
        damping: float = 0.0,
        gravity: float = 9.81,
        dt: float = 0.01,
        state_bounds: Optional[tuple] = None,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.dt = dt
        self.state_bounds = state_bounds
        self.state = self.bk.zeros(2)
        self._engine = None

    def physics_engine(self, engine: Optional[PhysicsEngine]):
        self._engine = engine
        if engine is not None:
            self.bk = engine.backend
            self.state = self.bk.zeros(2)
        else:
            self.bk = NumpyBackend()
            self.state = self.bk.zeros(2)

    def get_state(self):
        if self._engine is not None:
            qpos = self._engine.get_joint_qpos("hinge")
            qvel = self._engine.get_joint_vel("hinge")
            return self.bk.array([qpos, qvel])
        return self.bk.copy(self.state)

    def get_model(self):
        g, l, b, m = self.g, self.l, self.b, self.m
        A = self.bk.array([[0.0, 1.0], [g / l, -b / (m * l**2)]])
        B = self.bk.array([[0.0], [1.0 / (m * l**2)]])
        return A, B

    def dynamics(self, state, control):
        theta, theta_dot = state[0], state[1]
        tau = control[0] if hasattr(control, '__len__') else control
        theta_ddot = (self.g / self.l) * self.bk.sin(theta) + tau / (self.m * self.l**2) - (self.b / (self.m * self.l**2)) * theta_dot
        return self.bk.array([theta_dot, theta_ddot])

    def step(self, u):
        if self._engine is not None:
            self._engine.set_joint_ctrl("hinge", u[0] if hasattr(u, '__len__') else u)
            self._engine.step()
            self.state = self.get_state()
            return self.state

        theta, theta_dot = self.state[0], self.state[1]
        tau = u[0] if hasattr(u, '__len__') else u
        theta_ddot = (self.g / self.l) * self.bk.sin(theta) + tau / (self.m * self.l**2) - (self.b / (self.m * self.l**2)) * theta_dot
        theta_dot_new = theta_dot + theta_ddot * self.dt
        theta_new = theta + theta_dot_new * self.dt
        self.state = self.bk.array([theta_new, theta_dot_new])
        if self.state_bounds is not None:
            self.state = self.bk.clip(self.state, self.state_bounds[0], self.state_bounds[1])
        return self.state

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        state_bounds = None
        if "state_bounds" in config:
            sb = config["state_bounds"]
            state_bounds = (bk.array(sb.get("min", [-3.14, -10.0])), bk.array(sb.get("max", [3.14, 10.0])))
        plant = cls(
            mass=config.get("mass", 0.1),
            length=config.get("length", 0.5),
            damping=config.get("damping", 0.0),
            gravity=config.get("gravity", 9.81),
            dt=config.get("dt", 0.01),
            state_bounds=state_bounds,
            backend=bk,
        )
        engine = config.get("engine")
        if engine is not None:
            plant.physics_engine(engine)
        return plant


@register_plant_detector("InvertedPendulum")
def detect_inverted_pendulum(xml_root):
    """Detect InvertedPendulum: exactly 1 hinge joint + 1 motor actuator."""
    joints = xml_root.findall('.//joint')
    actuators = xml_root.findall('.//actuator/*')
    if len(joints) != 1 or len(actuators) != 1:
        return False
    return joints[0].get('type') == 'hinge' and actuators[0].tag == 'motor'
