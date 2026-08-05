from typing import Optional
from components import Plant, PhysicsEngine
from factories.registry import register_plant, register_plant_detector
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_plant("InvertedPendulum")
class InvertedPendulum(Plant):
    """2D inverted pendulum with standalone analytical dynamics and optional MuJoCo engine.

    Models a simple pendulum with a point mass at the end of a massless rod,
    hinged at the origin. The state is the angle from upright :math:`[\\theta,
    \\dot{\\theta}]` and the control is torque at the pivot :math:`[\\tau]`.

    Supports two modes:

    1. **Standalone** — integrates the analytical dynamics using semi-implicit
       Euler (velocity updated before position).
    2. **Physics engine** — attaches a MuJoCo engine for mesh-accurate
       simulation.

    The continuous-time dynamics are:

    .. math::

        \\ddot{\\theta} = \\frac{g}{l} \\sin(\\theta)
        + \\frac{\\tau}{m l^2} - \\frac{b}{m l^2} \\dot{\\theta}

    The linearized model around the upright equilibrium
    :math:`(\\theta=0, \\dot{\\theta}=0)` is:

    .. math::

        A = \\begin{bmatrix} 0 & 1 \\\\ g/l & -b/(m l^2) \\end{bmatrix},
        \\quad
        B = \\begin{bmatrix} 0 \\\\ 1/(m l^2) \\end{bmatrix}

    Args:
        mass: Mass of the pendulum bob (kg).
        length: Length of the pendulum rod (m).
        damping: Linear damping coefficient at the pivot (Nms/rad).
        gravity: Gravitational acceleration (m/s^2).
        dt: Time step in seconds.
        state_bounds: Optional (min, max) bounds for state clipping.
            Each is an array of shape (2,).
        backend: Array backend. Defaults to NumpyBackend.
    """

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
        """Attach or detach a physics engine.

        When attached, the backend is inherited from the engine and the
        state is reset to zeros. When detached, the backend reverts to
        NumpyBackend.

        Args:
            engine: PhysicsEngine instance or None to detach.
        """
        self._engine = engine
        if engine is not None:
            self.bk = engine.backend
            self.state = self.bk.zeros(2)
        else:
            self.bk = NumpyBackend()
            self.state = self.bk.zeros(2)

    def get_state(self):
        """Get the current state :math:`[\\theta, \\dot{\\theta}]`.

        When a physics engine is attached, reads joint position and velocity
        from the engine. Otherwise returns a copy of the internal state.

        Returns:
            State vector (2,) — [theta, theta_dot].
        """
        if self._engine is not None:
            qpos = self._engine.get_joint_qpos("hinge")
            qvel = self._engine.get_joint_vel("hinge")
            return self.bk.array([qpos, qvel])
        return self.bk.copy(self.state)

    def get_model(self):
        """Get the linearized discrete-time state-space model.

        Returns:
            Tuple of (A, B) where A is (2, 2) and B is (2, 1).
        """
        g, l, b, m = self.g, self.l, self.b, self.m
        A = self.bk.array([[0.0, 1.0], [g / l, -b / (m * l**2)]])
        B = self.bk.array([[0.0], [1.0 / (m * l**2)]])
        return A, B

    def dynamics(self, state, control):
        """Continuous-time dynamics :math:`\\dot{x} = f(x, u)`.

        Args:
            state: State vector (2,) — [theta, theta_dot].
            control: Control vector (1,) or scalar — [tau].

        Returns:
            Time derivative of the state (2,) — [theta_dot, theta_ddot].
        """
        theta, theta_dot = state[0], state[1]
        tau = control[0] if hasattr(control, '__len__') else control
        theta_ddot = (self.g / self.l) * self.bk.sin(theta) + tau / (self.m * self.l**2) - (self.b / (self.m * self.l**2)) * theta_dot
        return self.bk.array([theta_dot, theta_ddot])

    def step(self, u):
        """Execute one control step.

        When a physics engine is attached, sets the torque actuator and
        advances the engine. Otherwise integrates the analytical dynamics
        using semi-implicit Euler.

        Args:
            u: Control input (1,) or scalar — torque at pivot (Nm).

        Returns:
            New state vector (2,) — [theta, theta_dot].
        """
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
        """Create an InvertedPendulum from a TOML config dict.

        Config fields:
            mass: Pendulum bob mass (kg).
            length: Pendulum rod length (m).
            damping: Linear damping coefficient.
            gravity: Gravitational acceleration (m/s^2).
            dt: Time step.
            state_bounds: Optional dict with ``min`` and ``max`` lists.
            engine: Optional PhysicsEngine instance to attach.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            InvertedPendulum instance.
        """
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
    """Detect InvertedPendulum from an MJCF XML tree.

    Matches XMLs with exactly 1 hinge joint and 1 motor actuator.

    Args:
        xml_root: Root element of the parsed MJCF XML.

    Returns:
        True if the XML matches the InvertedPendulum pattern.
    """
    joints = xml_root.findall('.//joint')
    actuators = xml_root.findall('.//actuator/*')
    if len(joints) != 1 or len(actuators) != 1:
        return False
    return joints[0].get('type') == 'hinge' and actuators[0].tag == 'motor'
