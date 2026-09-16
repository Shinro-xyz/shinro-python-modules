from dataclasses import dataclass

from shinro.components import PhysicsEngine, Plant
from shinro.factories.registry import register_plant, register_plant_detector
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.utils.batching import as_batch, as_vector, column, control_batch
from shinro.utils.config_spec import BoundsConfig, strict_from_dict, strip_runtime_keys
from shinro.utils.linearization import discretize_euler, linearize_plant


@dataclass(frozen=True)
class DoublePendulumConfig:
    """Strict TOML schema for :class:`DoublePendulum`.

    The plant TOML is the single source of physics truth: every field has a
    default. ``engine`` is runtime-injected by sim-backed builds (RobotSim),
    never authored in TOML.
    """

    mass_top: float = 0.1
    mass_bottom: float = 0.1
    length_top: float = 0.5
    length_bottom: float = 0.5
    dt: float = 0.01
    g: float = 9.81
    state_bounds: dict | None = None
    name: str = "double_pendulum"


@register_plant("DoublePendulum")
class DoublePendulum(Plant):
    """4D planar double pendulum with standalone analytical dynamics and optional MuJoCo engine.

    Models two point masses at the ends of two massless rods, hinged in
    series. The state is :math:`[\\theta_1, \\theta_2, \\omega_1, \\omega_2]`
    (angle of each rod from the downward vertical and its angular velocity)
    and the control is joint torques :math:`[\\tau_1, \\tau_2]` applied at
    each hinge. The rest equilibrium :math:`\\theta=0` is stable (hanging
    down).

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

    def dynamics(self, state, control, bk=None):
        """Continuous-time dynamics :math:`\\dot{x} = f(x, u)`, batch-capable.

        State ordering is :math:`[\\theta_1, \\theta_2, \\omega_1, \\omega_2]`
        and control is :math:`[\\tau_1, \\tau_2]`. The angular acceleration
        solves the manipulator equation
        :math:`\\ddot{\\theta} = M^{-1}(\\tau - C\\dot{\\theta} - G)`, with

        .. math::

            M = \\begin{bmatrix} (m_1 + m_2) l_1^2 & m_2 l_1 l_2 \\cos\\Delta \\
            m_2 l_1 l_2 \\cos\\Delta & m_2 l_2^2 \\end{bmatrix},
            \\quad \\Delta = \\theta_1 - \\theta_2,

        :math:`C\\dot{\\theta} = [m_2 l_1 l_2 \\sin\\Delta\\,\\omega_2^2,
        -m_2 l_1 l_2 \\sin\\Delta\\,\\omega_1^2]`, and
        :math:`G = [(m_1 + m_2) g l_1 \\sin\\theta_1,
        m_2 g l_2 \\sin\\theta_2]`.

        The 2x2 solve is written in closed form (Cramer's rule) rather than
        ``bk.solve``: on the batched path ``M`` would be rank-3 (``(N, 2, 2)``),
        which the graph backend — strictly 2-D — does not represent. The
        entries are therefore ``(N, 1)`` columns, and the determinant
        :math:`m_2 l_1^2 l_2^2 (m_1 + m_2 \\sin^2\\Delta)` is strictly
        positive for positive masses, so the division is always safe.

        Args:
            state: State (4,) — [theta_1, theta_2, omega_1, omega_2] — or a
                batch (N, 4).
            control: Control (2,), batch (N, 2), or scalar — [tau_1, tau_2].
            bk: Backend to evaluate with. Defaults to the plant's backend.

        Returns:
            Time derivative with the rank of ``state`` —
            [omega_1, omega_2, theta_1_ddot, theta_2_ddot].
        """
        bk = self.bk if bk is None else bk
        x, single = as_batch(bk, state)
        u = control_batch(bk, control, 2)
        theta_1 = column(bk, x, 0)
        theta_2 = column(bk, x, 1)
        omega_1 = column(bk, x, 2)
        omega_2 = column(bk, x, 3)
        tau_1 = column(bk, u, 0)
        tau_2 = column(bk, u, 1)

        diff = theta_1 - theta_2
        sin_diff = bk.sin(diff)
        # Mass matrix entries; only m12 is state-dependent.
        m11 = (self.m1 + self.m2) * self.l1**2
        m12 = self.m2 * self.l1 * self.l2 * bk.cos(diff)
        m22 = self.m2 * self.l2**2
        # b = tau - C omega - G, one (N, 1) column per joint.
        b1 = (
            tau_1
            - self.m2 * self.l1 * self.l2 * sin_diff * omega_2 * omega_2
            - (self.m1 + self.m2) * self.g * self.l1 * bk.sin(theta_1)
        )
        b2 = (
            tau_2
            + self.m2 * self.l1 * self.l2 * sin_diff * omega_1 * omega_1
            - self.m2 * self.g * self.l2 * bk.sin(theta_2)
        )
        # theta_ddot = M^-1 b, 2x2 closed form.
        det = m11 * m22 - m12 * m12
        theta_1_ddot = (b1 * m22 - m12 * b2) / det
        theta_2_ddot = (m11 * b2 - b1 * m12) / det

        f = bk.stack([
            bk.ravel(omega_1),
            bk.ravel(omega_2),
            bk.ravel(theta_1_ddot),
            bk.ravel(theta_2_ddot),
        ]).T
        return as_vector(bk, f, single)

    def get_model(self, x0=None, u0=None, eps=1e-6):
        """Get the discrete-time state-space model around an operating point.

        Linearizes the continuous-time dynamics :math:`f(x, u) = \\dot{x}`
        around ``(x0, u0)`` using central finite differences via
        :func:`shinro.utils.linearization.linearize_plant`, then
        Euler-discretizes at the plant's ``dt``. When ``x0``/``u0`` are
        omitted, defaults to the rest equilibrium
        :math:`(\\theta=0, \\dot{\\theta}=0)` with zero control.

        Args:
            x0: Operating point state (4,) —
                [theta_1, theta_2, omega_1, omega_2]. Defaults to zeros.
            u0: Operating point control (2,) — [tau_1, tau_2]. Defaults to zeros.
            eps: Step size for finite differences.

        Returns:
            Tuple of (A, B) where A = I + dt·∂f/∂x is (4, 4) and
            B = dt·∂f/∂u is (4, 2).
        """
        A_c, B_c = linearize_plant(self, x0, u0, eps=eps)
        return discretize_euler(A_c, B_c, self.dt, backend=self.bk)

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

    Config = DoublePendulumConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a DoublePendulum from a TOML config dict or :class:`DoublePendulumConfig`.

        Config fields:
            mass_top: Top bob mass (kg).
            mass_bottom: Bottom bob mass (kg).
            length_top: Top rod length (m).
            length_bottom: Bottom rod length (m).
            dt: Time step.
            g: Gravitational acceleration (m/s^2).
            state_bounds: Optional dict with ``min`` and ``max`` lists.

        Args:
            config: TOML config dict (may carry a runtime-injected ``engine``)
                or DoublePendulumConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            DoublePendulum instance.
        """
        bk = backend or NumpyBackend()
        clean, runtime = (
            strip_runtime_keys(config, ("engine", "joint_groups")) if isinstance(config, dict) else (config, {})
        )
        cfg = cls.parse_config(clean)
        state_bounds = None
        if cfg.state_bounds is not None:
            sb = strict_from_dict(BoundsConfig, cfg.state_bounds, "DoublePendulum.state_bounds")
            state_bounds = (
                bk.array(sb.min if sb.min is not None else [-3.14, -3.14, -10.0, -10.0]),
                bk.array(sb.max if sb.max is not None else [3.14, 3.14, 10.0, 10.0]),
            )
        plant = cls(
            mass_top=cfg.mass_top,
            mass_bottom=cfg.mass_bottom,
            length_top=cfg.length_top,
            length_bottom=cfg.length_bottom,
            dt=cfg.dt,
            g=cfg.g,
            state_bounds=state_bounds,
            backend=bk,
        )
        engine = runtime.get("engine")
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
