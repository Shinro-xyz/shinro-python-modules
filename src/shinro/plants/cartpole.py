

from shinro.components import PhysicsEngine, Plant
from shinro.factories.registry import register_plant, register_plant_detector
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


@register_plant("CartPole")
class CartPole(Plant):
    """4D cart-pole system with standalone analytical dynamics and optional MuJoCo engine.

    Models a cart on a frictionless track with a pole hinged on top. The
    state is :math:`[x, \\dot{x}, \\theta, \\dot{\\theta}]` (cart position,
    cart velocity, pole angle from upright, pole angular velocity) and the
    control is horizontal force on the cart :math:`[F]`.

    Supports two modes:

    1. **Standalone** — integrates the coupled analytical dynamics using
       semi-implicit Euler (velocities updated before positions).
    2. **Physics engine** — attaches a MuJoCo engine for mesh-accurate
       simulation.

    The equations of motion for the coupled system are:

    .. math::

        \\ddot{\\theta} = \\frac{g \\sin\\theta - \\cos\\theta
        \\left(\\frac{F + m l \\dot{\\theta}^2 \\sin\\theta}{M + m}\\right)}
        {l - \\frac{m l \\cos^2\\theta}{M + m}}

        \\ddot{x} = \\frac{F + m l \\left(\\dot{\\theta}^2 \\sin\\theta -
        \\ddot{\\theta} \\cos\\theta\\right)}{M + m}

    The linearized model around the upright equilibrium
    :math:`(x=0, \\dot{x}=0, \\theta=0, \\dot{\\theta}=0)` is:

    .. math::

        A = \\begin{bmatrix}
        0 & 1 & 0 & 0 \\\\
        0 & 0 & -mg/M & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        0 & 0 & (M+m)g/(Ml) & -b/(M l^2)
        \\end{bmatrix},
        \\quad
        B = \\begin{bmatrix} 0 \\\\ 1/M \\\\ 0 \\\\ -1/(M l) \\end{bmatrix}

    Args:
        cart_mass: Mass of the cart (kg).
        pole_mass: Mass of the pole (kg).
        pole_length: Length of the pole (m).
        damping: Linear damping coefficient at the pole hinge (Nms/rad).
        gravity: Gravitational acceleration (m/s^2).
        dt: Time step in seconds.
        track_limits: Optional (min, max) bounds for cart position (m).
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        cart_mass: float = 0.5,
        pole_mass: float = 0.1,
        pole_length: float = 0.5,
        damping: float = 0.0,
        gravity: float = 9.81,
        dt: float = 0.01,
        track_limits: tuple | None = None,
        backend: ArrayBackend | None = None,
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

    def physics_engine(self, engine: PhysicsEngine | None):
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
            self.state = self.bk.zeros(4)
        else:
            self.bk = NumpyBackend()
            self.state = self.bk.zeros(4)

    def get_state(self):
        """Get the current state :math:`[x, \\dot{x}, \\theta, \\dot{\\theta}]`.

        When a physics engine is attached, reads joint positions and
        velocities from the engine. Otherwise returns a copy of the
        internal state.

        Returns:
            State vector (4,) — [x, x_dot, theta, theta_dot].
        """
        if self._engine is not None:
            x = self._engine.get_joint_qpos("slider")
            x_dot = self._engine.get_joint_vel("slider")
            theta = self._engine.get_joint_qpos("hinge")
            theta_dot = self._engine.get_joint_vel("hinge")
            return self.bk.array([x, x_dot, theta, theta_dot])
        return self.bk.copy(self.state)

    def get_model(self):
        """Get the linearized discrete-time state-space model.

        Returns:
            Tuple of (A, B) where A is (4, 4) and B is (4, 1).
        """
        M, m, pole_len, g, b = self.M, self.m, self.l, self.g, self.b
        A = self.bk.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -m * g / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (M + m) * g / (M * pole_len), -b / (M * pole_len**2)],
        ])
        B = self.bk.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * pole_len)]])
        return A, B

    def _compute_accels(self, x, theta, x_dot, theta_dot, F):
        """Compute the accelerations from the equations of motion.

        Solves the coupled 2x2 system for :math:`\\ddot{x}` and
        :math:`\\ddot{\\theta}` given the current state and control.

        Args:
            x: Cart position (m).
            theta: Pole angle (rad).
            x_dot: Cart velocity (m/s).
            theta_dot: Pole angular velocity (rad/s).
            F: Horizontal force on cart (N).

        Returns:
            Tuple of (x_ddot, theta_ddot).
        """
        M, m, pole_len, g, _ = self.M, self.m, self.l, self.g, self.b
        sin_theta = self.bk.sin(theta)
        cos_theta = self.bk.cos(theta)
        denom = pole_len - m * pole_len * cos_theta**2 / (M + m)
        theta_ddot = (g * sin_theta - cos_theta * (F + m * pole_len * theta_dot**2 * sin_theta) / (M + m)) / denom
        x_ddot = (F + m * pole_len * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)) / (M + m)
        return x_ddot, theta_ddot

    def dynamics(self, state, control):
        """Continuous-time dynamics :math:`\\dot{x} = f(x, u)`.

        Args:
            state: State vector (4,) — [x, x_dot, theta, theta_dot].
            control: Control vector (1,) or scalar — [F].

        Returns:
            Time derivative of the state (4,) — [x_dot, x_ddot, theta_dot, theta_ddot].
        """
        x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
        F = control[0] if hasattr(control, '__len__') else control
        x_ddot, theta_ddot = self._compute_accels(x, theta, x_dot, theta_dot, F)
        return self.bk.stack([x_dot, x_ddot, theta_dot, theta_ddot])

    def step(self, u):
        """Execute one control step.

        When a physics engine is attached, sets the cart force actuator and
        advances the engine. Otherwise integrates the coupled analytical
        dynamics using semi-implicit Euler.

        Args:
            u: Control input (1,) or scalar — horizontal force on cart (N).

        Returns:
            New state vector (4,) — [x, x_dot, theta, theta_dot].
        """
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
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a CartPole from a TOML config dict.

        Config fields:
            cart_mass: Mass of the cart (kg).
            pole_mass: Mass of the pole (kg).
            pole_length: Length of the pole (m).
            damping: Linear damping coefficient.
            gravity: Gravitational acceleration (m/s^2).
            dt: Time step.
            track_limits: Optional list of [min, max] for cart position.
            engine: Optional PhysicsEngine instance to attach.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            CartPole instance.
        """
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
    """Detect CartPole from an MJCF XML tree.

    Matches XMLs with 1 slide joint, 1 hinge joint, and 1 motor actuator.

    Args:
        xml_root: Root element of the parsed MJCF XML.

    Returns:
        True if the XML matches the CartPole pattern.
    """
    joints = xml_root.findall('.//joint')
    actuators = xml_root.findall('.//actuator/*')
    if len(joints) != 2 or len(actuators) != 1:
        return False
    types = [j.get('type') for j in joints]
    return 'slide' in types and 'hinge' in types and actuators[0].tag == 'motor'
