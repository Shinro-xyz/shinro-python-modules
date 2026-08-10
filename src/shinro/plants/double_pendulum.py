from shinro.components import PhysicsEngine, Plant
from shinro.factories.registry import register_plant, register_plant_detector
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.utils.linearization import linearize_plant


@register_plant("DoublePendulum")
class DoublePendulum(Plant):
    """4D planar double pendulum with standalone analytical dynamics and optional MuJoCo engine.

    Models two point masses at the ends of two massless rods, hinged in
    series. The state is :math:`[\\theta_1, \\theta_2, \\omega_1, \\omega_2]`
    (angle of each rod from upright and its angular velocity) and the control
    is joint torques :math:`[\\tau_1, \\tau_2]` applied at each hinge.

    Supports two modes:

    1. **Standalone** — integrates the analytical dynamics using semi-implicit
       Euler (velocities updated before positions).
    2. **Physics engine** — attaches a MuJoCo engine for mesh-accurate
       simulation (expects an MJCF model with ``hinge_1``/``hinge_2`` joints
       and ``torque_1``/``torque_2`` motor actuators).

    The equations of motion are the standard double pendulum manipulator form:

    .. math::

        M(\\theta) \\ddot{\\theta} + C(\\theta, \\dot{\\theta})
        \\dot{\\theta} + G(\\theta) = \\tau

    where :math:`M` is the mass matrix, :math:`C` the Coriolis matrix, and
    :math:`G` the gravity vector. The angular acceleration follows from
    :math:`\\ddot{\\theta} = M^{-1}(\\tau - C\\dot{\\theta} - G)`.

    Args:
        mass_top: Mass of the top pendulum bob (kg).
        mass_bottom: Mass of the bottom pendulum bob (kg).
        length_top: Length of the top rod (m).
        length_bottom: Length of the bottom rod (m).
        dt: Time step in seconds.
        g: Gravitational acceleration (m/s^2).
        state_bounds: Optional (min, max) bounds for state clipping.
            Each is an array of shape (4,).
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        mass_top: float = 0.1,
        mass_bottom: float = 0.1,
        length_top: float = 0.5,
        length_bottom: float = 0.5,
        dt: float = 0.01,
        g: float = 9.81,
        state_bounds: tuple | None = None,
        backend: ArrayBackend | None = None,
    ):
        if mass_top <= 0 or mass_bottom <= 0:
            raise ValueError("Pendulum masses must be positive.")
        if length_top <= 0 or length_bottom <= 0:
            raise ValueError("Pendulum lengths must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.bk = backend or NumpyBackend()
        self.m1 = mass_top
        self.m2 = mass_bottom
        self.l1 = length_top
        self.l2 = length_bottom
        self.dt = dt
        self.g = g
        self.input_dim = 2
        self.state_bounds = state_bounds
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

    def _make_mass_matrix(self, diff_theta):
        """Build the 2x2 mass matrix :math:`M(\\theta)`.

        Args:
            diff_theta: Angle difference :math:`\\theta_1 - \\theta_2`.

        Returns:
            Mass matrix M (2, 2).
        """
        M = self.bk.zeros((2, 2))
        M[0, 0] = (self.m1 + self.m2) * self.l1**2
        M[0, 1] = self.m2 * self.l1 * self.l2 * self.bk.cos(diff_theta)
        M[1, 0] = M[0, 1]
        M[1, 1] = self.m2 * self.l2**2
        return M

    def _make_coriolis_matrix(self, diff_theta, angular_velocities):
        """Build the 2x2 Coriolis matrix :math:`C(\\theta, \\dot{\\theta})`.

        Args:
            diff_theta: Angle difference :math:`\\theta_1 - \\theta_2`.
            angular_velocities: Vector (2,) — [omega_1, omega_2].

        Returns:
            Coriolis matrix C (2, 2).
        """
        C = self.bk.zeros((2, 2))
        w_1, w_2 = angular_velocities[0], angular_velocities[1]
        C[0, 0] = self.m2 * self.l1 * self.l2 * self.bk.sin(diff_theta) * w_2
        C[0, 1] = C[0, 0]
        C[1, 0] = -self.m2 * self.l2 * self.l1 * self.bk.sin(diff_theta) * w_1
        return C

    def _make_gravity_vector(self, theta_angles):
        """Build the gravity vector :math:`G(\\theta)`.

        Args:
            theta_angles: Vector (2,) — [theta_1, theta_2].

        Returns:
            Gravity vector G (2,).
        """
        theta_1, theta_2 = theta_angles[0], theta_angles[1]
        G = self.bk.zeros(2)
        G[0] = (self.m1 + self.m2) * self.g * self.l1 * self.bk.sin(theta_1)
        G[1] = self.m2 * self.g * self.l2 * self.bk.sin(theta_2)
        return G

    def dynamics(self, state, control):
        """Continuous-time dynamics :math:`\\dot{x} = f(x, u)`.

        State ordering is :math:`[\\theta_1, \\theta_2, \\omega_1, \\omega_2]`
        and control is :math:`[\\tau_1, \\tau_2]`. The angular acceleration is
        solved from the manipulator equation
        :math:`\\ddot{\\theta} = M^{-1}(\\tau - C\\dot{\\theta} - G)`.

        Args:
            state: State vector (4,) — [theta_1, theta_2, omega_1, omega_2].
            control: Control vector (2,) or scalar — [tau_1, tau_2].

        Returns:
            Time derivative of the state (4,) —
            [omega_1, omega_2, theta_1_ddot, theta_2_ddot].
        """
        theta_1, theta_2, omega_1, omega_2 = state[0], state[1], state[2], state[3]
        diff_theta = theta_1 - theta_2
        omega = self.bk.array([omega_1, omega_2])
        M = self._make_mass_matrix(diff_theta)
        C = self._make_coriolis_matrix(diff_theta, omega)
        G = self._make_gravity_vector(self.bk.array([theta_1, theta_2]))

        tau = control if hasattr(control, '__len__') else self.bk.array([control, 0.0])
        b = self.bk.array(tau) - C @ omega - G
        thetaddot = self.bk.solve(M, b)

        return self.bk.stack([omega_1, omega_2, thetaddot[0], thetaddot[1]])

    def get_model(self, x0=None, u0=None, eps=1e-6):
        """Get the linearized state-space model around an operating point.

        Linearizes the continuous-time dynamics :math:`f(x, u) = \\dot{x}`
        around ``(x0, u0)`` using central finite differences via
        :func:`shinro.utils.linearization.linearize_plant`. When ``x0``/``u0``
        are omitted, defaults to the rest equilibrium
        :math:`(\\theta=0, \\dot{\\theta}=0)` with zero control.

        Args:
            x0: Operating point state (4,) —
                [theta_1, theta_2, omega_1, omega_2]. Defaults to zeros.
            u0: Operating point control (2,) — [tau_1, tau_2]. Defaults to zeros.
            eps: Step size for finite differences.

        Returns:
            Tuple of (A, B) where A = ∂f/∂x is (4, 4) and B = ∂f/∂u is (4, 2).
        """
        return linearize_plant(self, x0, u0, eps=eps)

    def get_state(self):
        """Get the current state :math:`[\\theta_1, \\theta_2, \\omega_1, \\omega_2]`.

        When a physics engine is attached, reads joint positions and
        velocities from the engine. Otherwise returns a copy of the
        internal state.

        Returns:
            State vector (4,) — [theta_1, theta_2, omega_1, omega_2].
        """
        if self._engine is not None:
            qpos_1 = self._engine.get_joint_qpos("hinge_1")
            qpos_2 = self._engine.get_joint_qpos("hinge_2")
            qvel_1 = self._engine.get_joint_vel("hinge_1")
            qvel_2 = self._engine.get_joint_vel("hinge_2")
            return self.bk.array([qpos_1, qpos_2, qvel_1, qvel_2])
        return self.bk.copy(self.state)

    def step(self, u):
        """Execute one control step.

        When a physics engine is attached, sets the torque actuators and
        advances the engine. Otherwise integrates the analytical dynamics
        using semi-implicit Euler (velocities updated before positions).

        Args:
            u: Control input (2,) or scalar — [tau_1, tau_2] joint torques (Nm).

        Returns:
            New state vector (4,) — [theta_1, theta_2, omega_1, omega_2].
        """
        if self._engine is not None:
            self._engine.set_joint_ctrl("torque_1", u[0] if hasattr(u, '__len__') else u)
            self._engine.set_joint_ctrl("torque_2", u[1] if hasattr(u, '__len__') else 0.0)
            self._engine.step()
            self.state = self.get_state()
            return self.state

        xdot = self.dynamics(self.state, u)
        omega_1_new = self.state[2] + xdot[2] * self.dt
        omega_2_new = self.state[3] + xdot[3] * self.dt
        theta_1_new = self.state[0] + omega_1_new * self.dt
        theta_2_new = self.state[1] + omega_2_new * self.dt
        self.state = self.bk.array([theta_1_new, theta_2_new, omega_1_new, omega_2_new])
        if self.state_bounds is not None:
            self.state = self.bk.clip(self.state, self.state_bounds[0], self.state_bounds[1])
        return self.state

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a DoublePendulum from a TOML config dict.

        Config fields:
            mass_top: Top pendulum bob mass (kg).
            mass_bottom: Bottom pendulum bob mass (kg).
            length_top: Top rod length (m).
            length_bottom: Bottom rod length (m).
            dt: Time step.
            g: Gravitational acceleration (m/s^2).
            state_bounds: Optional dict with ``min`` and ``max`` lists.
            engine: Optional PhysicsEngine instance to attach.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            DoublePendulum instance.
        """
        bk = backend or NumpyBackend()
        state_bounds = None
        if "state_bounds" in config:
            sb = config["state_bounds"]
            state_bounds = (bk.array(sb.get("min", [-3.14, -3.14, -10.0, -10.0])),
                            bk.array(sb.get("max", [3.14, 3.14, 10.0, 10.0])))
        plant = cls(
            mass_top=config.get("mass_top", 0.1),
            mass_bottom=config.get("mass_bottom", 0.1),
            length_top=config.get("length_top", 0.5),
            length_bottom=config.get("length_bottom", 0.5),
            dt=config.get("dt", 0.01),
            g=config.get("g", 9.81),
            state_bounds=state_bounds,
            backend=bk,
        )
        engine = config.get("engine")
        if engine is not None:
            plant.physics_engine(engine)
        return plant


@register_plant_detector("DoublePendulum")
def detect_double_pendulum(xml_root):
    """Detect DoublePendulum from an MJCF XML tree.

    Matches XMLs with exactly 2 hinge joints and 2 motor actuators.

    Args:
        xml_root: Root element of the parsed MJCF XML.

    Returns:
        True if the XML matches the DoublePendulum pattern.
    """
    joints = xml_root.findall('.//joint')
    actuators = xml_root.findall('.//actuator/*')
    if len(joints) != 2 or len(actuators) != 2:
        return False
    types = [j.get('type') for j in joints]
    return types.count('hinge') == 2 and all(a.tag == 'motor' for a in actuators)
