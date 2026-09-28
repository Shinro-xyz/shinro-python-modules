from dataclasses import dataclass

import numpy as np

from shinro.components import PhysicsEngine, Plant
from shinro.factories.registry import register_plant
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.utils.batching import as_batch, as_vector, column, control_batch
from shinro.utils.config_spec import BoundsConfig, strict_from_dict, strict_from_list, strip_runtime_keys
from shinro.utils.linearization import discretize_euler, linearize_plant


@dataclass(frozen=True)
class RotorConfig:
    """One rotor in the :class:`Quadrotor` mixer table.

    Position is in the body frame — ``x`` forward, ``y`` left (metres) — and
    ``spin`` is the rotation sense seen from above: ``+1`` counter-clockwise,
    ``-1`` clockwise. The spin sign fixes the rotor's yaw reaction torque.
    """

    x: float
    y: float
    spin: float = 1.0


@dataclass(frozen=True)
class QuadrotorConfig:
    """Strict TOML schema for :class:`Quadrotor`.

    The plant TOML is the single source of physics truth: every field has a
    default, so a minimal config is just ``type`` + ``name``. ``engine`` is
    runtime-injected by sim-backed builds (RobotSim), never authored in TOML.

    ``inertia`` carries the body-frame principal moments of inertia
    :math:`[I_{xx}, I_{yy}, I_{zz}]` (kg·m²). ``None`` selects the built-in
    default used by :meth:`Quadrotor.__init__`.

    ``rotors`` is an optional explicit mixer table of ``[[rotors]]`` entries
    (``x``, ``y``, ``spin``). ``None`` selects the built-in ``+`` layout built
    from ``radius``; supplying it overrides the numbering/placement entirely.
    """

    mass: float = 0.5
    radius: float = 0.1
    inertia: list[float] | None = None
    thrust_coeff: float = 1.0
    torque_coeff: float = 0.1
    dt: float = 0.01
    g: float = 9.81
    rotors: list[dict] | None = None
    state_bounds: dict | None = None
    name: str = "quadrotor"


@register_plant("Quadrotor")
class Quadrotor(Plant):
    """12-state quadrotor with four rotor-speed inputs.

    The state is :math:`[x, y, z, \\phi, \\theta, \\psi, \\dot{x}, \\dot{y},
    \\dot{z}, p, q, r]` — position, Euler angles (roll/pitch/yaw), world-frame
    linear velocity, and body-frame angular velocity. The control is the four
    rotor angular speeds :math:`[\\omega_1, \\omega_2, \\omega_3, \\omega_4]`.

    Rotor placement is a configurable mixer. Each rotor sits at body
    :math:`(x_i, y_i)` and has a spin sign :math:`\\sigma_i` (:math:`+1`
    counter-clockwise seen from above, :math:`-1` clockwise). Rotor
    :math:`i` contributes thrust :math:`k\\omega_i^2` along body :math:`+z`,
    so its moment about the body axes is :math:`k\\omega_i^2 (y_i, -x_i, 0)`,
    and its yaw reaction is :math:`-b\\sigma_i\\omega_i^2` about :math:`z`:

    .. math::

        F = k \\sum_i \\omega_i^2, \\quad
        \\tau_x = k \\sum_i y_i \\omega_i^2, \\quad
        \\tau_y = -k \\sum_i x_i \\omega_i^2, \\quad
        \\tau_z = -b \\sum_i \\sigma_i \\omega_i^2.

    The default mixer (no ``rotors`` config) is the ``+`` layout
    :math:`[(\\ell,0), (0,\\ell), (-\\ell,0), (0,-\\ell)]` with spins
    :math:`[+,-,+,-]`, which gives the familiar
    :math:`\\tau_x = k\\ell(\\omega_2^2 - \\omega_4^2)`,
    :math:`\\tau_y = k\\ell(\\omega_3^2 - \\omega_1^2)`,
    :math:`\\tau_z = b(\\omega_2^2 + \\omega_4^2 - \\omega_1^2 - \\omega_3^2)`.
    A different numbering/placement is just a different table; see
    :class:`RotorConfig` and ``samples/plants/quadrotor.toml``.

    The translational dynamics apply the body thrust along body :math:`+z`
    through the rotation matrix :math:`R = R_z(\\psi) R_y(\\theta) R_x(\\phi)`
    and cancel gravity:

    .. math::

        \\ddot{\\mathbf{p}} = \\frac{1}{m} R \\begin{bmatrix}0\\\\0\\\\F\\end{bmatrix}
        - \\begin{bmatrix}0\\\\0\\\\g\\end{bmatrix}.

    The rotational dynamics are Euler's rigid-body equations with the
    diagonal inertia :math:`I = \\mathrm{diag}(I_{xx}, I_{yy}, I_{zz})`,

    .. math::

        \\dot{\\boldsymbol{\\omega}} = I^{-1}
        \\left(\\boldsymbol{\\tau} - \\boldsymbol{\\omega} \\times I\\boldsymbol{\\omega}\\right),

    and the Euler angles evolve through the body-rate kinematics
    :math:`\\dot{\\boldsymbol{\\eta}} = T(\\phi, \\theta)\\,\\boldsymbol{\\omega}`.

    Supports standalone analytical integration (explicit-Euler ``step``) and
    :meth:`dynamics` for finite-difference linearization and batched MPPI
    rollouts. MuJoCo engine mode is **not yet implemented**: attaching an engine
    only wires the backend, and ``step`` refuses to run while one is attached.

    Args:
        mass: Total mass (kg).
        radius: Rotor-to-center arm length :math:`\\ell` (m).
        inertia: Body-frame principal moments ``[I_xx, I_yy, I_zz]`` (kg·m²).
            Defaults to ``[0.005, 0.005, 0.01]``.
        dt: Integration step (s).
        g: Gravitational acceleration (m/s²).
        thrust_coeff: Rotor thrust coefficient :math:`k` (N·s²/rad²).
        torque_coeff: Rotor drag/yaw coefficient :math:`b` (N·m·s²/rad²).
        rotors: Optional mixer table of ``(x, y, spin)`` tuples, one per rotor
            in control order. ``None`` selects the default ``+`` layout built
            from ``radius``.
        state_bounds: Optional ``(min, max)`` arrays of shape (12,) clipping the
            state after each ``step``.
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        mass: float = 0.5,
        radius: float = 0.1,
        inertia: list[float] | None = None,
        dt: float = 0.01,
        g: float = 9.81,
        thrust_coeff: float = 1.0,
        torque_coeff: float = 0.1,
        rotors: list[tuple[float, float, float]] | None = None,
        state_bounds: tuple | None = None,
        backend: ArrayBackend | None = None,
    ):
        if mass <= 0:
            raise ValueError("Quadrotor mass must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        inertia = [0.005, 0.005, 0.01] if inertia is None else list(inertia)
        if len(inertia) != 3 or any(i <= 0 for i in inertia):
            raise ValueError("inertia must be three positive principal moments [I_xx, I_yy, I_zz].")
        if rotors is None:
            if radius <= 0:
                raise ValueError("Quadrotor radius (rotor arm) must be positive.")
            table: tuple[tuple[float, float, float], ...] = (
                (radius, 0.0, 1.0),
                (0.0, radius, -1.0),
                (-radius, 0.0, 1.0),
                (0.0, -radius, -1.0),
            )
        else:
            table = tuple(rotors)
        if len(table) != 4:
            raise ValueError("Quadrotor requires exactly four rotors in the mixer table.")

        self.bk = backend or NumpyBackend()
        self.m = mass
        self.r = radius
        self.I = inertia
        self.dt = dt
        self.g = g
        self.k = thrust_coeff
        self.b = torque_coeff
        self.rotors = table
        self.input_dim = len(table)
        self.state_bounds = state_bounds
        self.state = self.bk.zeros(12)
        self._engine = None

    def physics_engine(self, engine: PhysicsEngine | None):
        """Attach or detach a physics engine.

        Engine-backed simulation is not yet implemented for the quadrotor; this
        only stores the engine and inherits its backend so a future MJCF mapping
        can reuse the same wiring. ``step`` refuses to run while an engine is
        attached, so the analytic model is never silently bypassed.

        Args:
            engine: PhysicsEngine instance or None to detach.
        """
        self._engine = engine
        self.bk = engine.backend if engine is not None else NumpyBackend()

    def _rotation_matrix(self, roll, pitch, yaw):
        """Body-to-world rotation matrix :math:`R_z R_y R_x` (scalar reference).

        Returns the (3, 3) matrix used to rotate the body thrust direction into
        the world frame. Kept as a scalar reference for the closed-form
        components written out in :meth:`dynamics`; inputs must be Python
        floats (or numpy scalars), not traced batches.

        Args:
            roll: Roll angle (rad).
            pitch: Pitch angle (rad).
            yaw: Yaw angle (rad).

        Returns:
            Rotation matrix (3, 3) as a numpy array.
        """
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        return np.array([
            [c_p * c_y, s_r * s_p * c_y - c_r * s_y, c_r * s_p * c_y + s_r * s_y],
            [c_p * s_y, s_r * s_p * s_y + c_r * c_y, c_r * s_p * s_y - s_r * c_y],
            [-s_p, s_r * c_p, c_r * c_p],
        ])

    def _angular_transformation_matrix(self, roll, pitch):
        """Euler-rate map :math:`T(\\phi, \\theta)` (scalar reference).

        Maps body angular velocity to Euler-angle rates,
        :math:`\\dot{\\boldsymbol{\\eta}} = T \\boldsymbol{\\omega}`. Kept as a
        scalar reference for the closed-form components written out in
        :meth:`dynamics`; inputs must be Python floats (or numpy scalars), not
        traced batches.

        Args:
            roll: Roll angle (rad).
            pitch: Pitch angle (rad).

        Returns:
            Transformation matrix (3, 3) as a numpy array.
        """
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        return np.array([
            [1.0, s_r * s_p / c_p, c_r * s_p / c_p],
            [0.0, c_r, -s_r],
            [0.0, s_r / c_p, c_r / c_p],
        ])

    def dynamics(self, state, control, bk=None):
        """Continuous-time dynamics :math:`\\dot{x} = f(x, u)`, batch-capable.

        State ordering is :math:`[x, y, z, \\phi, \\theta, \\psi, \\dot{x},
        \\dot{y}, \\dot{z}, p, q, r]` and control is the four rotor speeds
        :math:`[\\omega_1, \\omega_2, \\omega_3, \\omega_4]`. The returned
        derivative is :math:`[\\dot{x}, \\dot{y}, \\dot{z}, \\dot{\\phi},
        \\dot{\\theta}, \\dot{\\psi}, \\ddot{x}, \\ddot{y}, \\ddot{z},
        \\dot{p}, \\dot{q}, \\dot{r}]`.

        The rotation and Euler-rate terms are expanded into their closed-form
        components rather than applied as 3x3 matrix products: on the batched
        path the matrices would be rank-3 ``(N, 3, 3)``, which the graph backend
        (strictly 2-D) does not represent.

        Args:
            state: State (12,) — [position, Euler angles, world velocity, body
                rates] — or a batch (N, 12).
            control: Control (4,), batch (N, 4), or scalar — rotor angular
                speeds.
            bk: Backend to evaluate with. Defaults to the plant's backend.

        Returns:
            Time derivative with the rank of ``state``, (12,) or (N, 12).
        """
        bk = self.bk if bk is None else bk
        x, single = as_batch(bk, state)
        u = control_batch(bk, control, 4)

        roll = column(bk, x, 3)
        pitch = column(bk, x, 4)
        yaw = column(bk, x, 5)
        vx = column(bk, x, 6)
        vy = column(bk, x, 7)
        vz = column(bk, x, 8)
        p = column(bk, x, 9)
        q = column(bk, x, 10)
        r = column(bk, x, 11)

        # Rotor mixer (configurable table) -> collective thrust + body torques.
        # Rotor i at body (x_i, y_i) with spin sigma_i contributes thrust
        # k*w_i^2 along +z: moment k*w_i^2*(y_i, -x_i, 0) and yaw reaction
        # -b*sigma_i*w_i^2. Python-level unroll over the (static) table keeps
        # this trace-safe with a fixed graph cost.
        thrust = 0.0
        tau_roll = 0.0
        tau_pitch = 0.0
        tau_yaw = 0.0
        for i, (rx, ry, spin) in enumerate(self.rotors):
            wi = column(bk, u, i)
            sq = wi * wi
            thrust = thrust + self.k * sq
            tau_roll = tau_roll + self.k * ry * sq
            tau_pitch = tau_pitch - self.k * rx * sq
            tau_yaw = tau_yaw - self.b * spin * sq

        # Euler's rigid-body equations, diagonal inertia.
        i_xx, i_yy, i_zz = self.I[0], self.I[1], self.I[2]
        p_dot = (tau_roll + q * r * (i_yy - i_zz)) / i_xx
        q_dot = (tau_pitch + r * p * (i_zz - i_xx)) / i_yy
        r_dot = (tau_yaw + p * q * (i_xx - i_yy)) / i_zz

        # Euler-angle kinematics eta_dot = T(roll, pitch) @ [p, q, r].
        sin_r, cos_r = bk.sin(roll), bk.cos(roll)
        sin_p, cos_p = bk.sin(pitch), bk.cos(pitch)
        sin_y, cos_y = bk.sin(yaw), bk.cos(yaw)
        tan_p = sin_p / cos_p
        roll_dot = p + q * sin_r * tan_p + r * cos_r * tan_p
        pitch_dot = q * cos_r - r * sin_r
        # Divide by cos(pitch) directly rather than multiplying by a reciprocal:
        # Tracer has no reflected division, so ``1.0 / cos_p`` would not trace.
        yaw_dot = (q * sin_r + r * cos_r) / cos_p

        # World-frame acceleration: R @ [0, 0, F] / m - [0, 0, g].
        accel_x = thrust * (sin_p * cos_r * cos_y + sin_r * sin_y) / self.m
        accel_y = thrust * (sin_p * cos_r * sin_y - sin_r * cos_y) / self.m
        accel_z = thrust * (cos_p * cos_r) / self.m - self.g

        f = bk.stack([
            bk.ravel(vx),
            bk.ravel(vy),
            bk.ravel(vz),
            bk.ravel(roll_dot),
            bk.ravel(pitch_dot),
            bk.ravel(yaw_dot),
            bk.ravel(accel_x),
            bk.ravel(accel_y),
            bk.ravel(accel_z),
            bk.ravel(p_dot),
            bk.ravel(q_dot),
            bk.ravel(r_dot),
        ]).T
        return as_vector(bk, f, single)

    def get_state(self):
        """Get the current state.

        Returns:
            State vector (12,) — [x, y, z, roll, pitch, yaw, vx, vy, vz,
            p, q, r].
        """
        return self.bk.copy(self.state)

    def get_model(self, x0=None, u0=None, eps=1e-6):
        """Get the discrete-time state-space model around an operating point.

        Linearizes the continuous-time dynamics :math:`f(x, u) = \\dot{x}`
        around ``(x0, u0)`` using central finite differences via
        :func:`shinro.utils.linearization.linearize_plant`, then
        Euler-discretizes at the plant's ``dt``. When ``x0`` is omitted it
        defaults to the level, zero-velocity state; when ``u0`` is omitted it
        defaults to the rotor speed that balances gravity,
        :math:`\\omega_{\\text{hover}} = \\sqrt{mg/(4k)}`, so the model is
        linearized about the true hover equilibrium (``u0 = 0`` would give
        ``B = 0`` because the rotor forces are quadratic in speed).

        Args:
            x0: Operating point state (12,).
                Defaults to zeros (level attitude, zero velocity).
            u0: Operating point control (4,) — rotor speeds. Defaults to the
                hover rotor speed :math:`\\sqrt{mg/(4k)}` on all four rotors.
            eps: Step size for finite differences.

        Returns:
            Tuple of (A, B) where A = I + dt·∂f/∂x is (12, 12) and
            B = dt·∂f/∂u is (12, 4).
        """
        if u0 is None:
            w_hover = (self.m * self.g / (4.0 * self.k)) ** 0.5
            u0 = self.bk.array([w_hover, w_hover, w_hover, w_hover])
        A_c, B_c = linearize_plant(self, x0, u0, eps=eps)
        return discretize_euler(A_c, B_c, self.dt, backend=self.bk)

    def step(self, u):
        """Execute one control step with explicit Euler integration.

        Velocities (world linear + body angular) are updated first, then
        positions and Euler angles are advanced with the updated velocities.

        Args:
            u: Control input (4,) or scalar — rotor angular speeds.

        Returns:
            New state vector (12,).

        Raises:
            NotImplementedError: If a physics engine is attached (MuJoCo engine
                mode is not implemented yet).
        """
        if self._engine is not None:
            raise NotImplementedError(
                "Quadrotor MuJoCo engine mode is not implemented; detach the engine "
                "to use the analytic step."
            )

        xdot = self.dynamics(self.state, u)
        vel = self.state[6:9] + xdot[6:9] * self.dt
        rates = self.state[9:12] + xdot[9:12] * self.dt
        pos = self.state[0:3] + vel * self.dt
        euler = self.state[3:6] + xdot[3:6] * self.dt
        self.state = self.bk.hstack([pos, euler, vel, rates])
        if self.state_bounds is not None:
            self.state = self.bk.clip(self.state, self.state_bounds[0], self.state_bounds[1])
        return self.state

    Config = QuadrotorConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a Quadrotor from a TOML config dict or :class:`QuadrotorConfig`.

        Config fields:
            mass: Total mass (kg).
            radius: Rotor arm length (m) — used for the default ``+`` mixer.
            inertia: Body-frame principal moments (kg·m²).
            thrust_coeff: Rotor thrust coefficient.
            torque_coeff: Rotor drag/yaw coefficient.
            dt: Time step (s).
            g: Gravitational acceleration (m/s²).
            rotors: Optional ``[[rotors]]`` mixer table (``x``, ``y``, ``spin``
                per entry); overrides the default layout.
            state_bounds: Optional dict with ``min`` and ``max`` lists.

        Args:
            config: TOML config dict (may carry a runtime-injected ``engine``)
                or QuadrotorConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Quadrotor instance.
        """
        bk = backend or NumpyBackend()
        clean, runtime = (
            strip_runtime_keys(config, ("engine", "joint_groups")) if isinstance(config, dict) else (config, {})
        )
        cfg = cls.parse_config(clean)
        state_bounds = None
        if cfg.state_bounds is not None:
            sb = strict_from_dict(BoundsConfig, cfg.state_bounds, "Quadrotor.state_bounds")
            state_bounds = (
                bk.array(sb.min if sb.min is not None else [
                    -10.0, -10.0, -10.0, -3.14, -1.57, -3.14, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
                ]),
                bk.array(sb.max if sb.max is not None else [
                    10.0, 10.0, 10.0, 3.14, 1.57, 3.14, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                ]),
            )
        rotors = None
        if cfg.rotors is not None:
            parsed = strict_from_list(RotorConfig, cfg.rotors, "Quadrotor.rotors")
            rotors = [(r.x, r.y, r.spin) for r in parsed]
        plant = cls(
            mass=cfg.mass,
            radius=cfg.radius,
            inertia=cfg.inertia,
            dt=cfg.dt,
            g=cfg.g,
            thrust_coeff=cfg.thrust_coeff,
            torque_coeff=cfg.torque_coeff,
            rotors=rotors,
            state_bounds=state_bounds,
            backend=bk,
        )
        engine = runtime.get("engine")
        if engine is not None:
            plant.physics_engine(engine)
        return plant
