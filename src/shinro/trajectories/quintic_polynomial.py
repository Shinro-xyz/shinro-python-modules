
from dataclasses import dataclass

import numpy as np

from shinro.components import ConfigDriven, TrajectoryGenerator
from shinro.factories.registry import register_trajectory
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.utils.config_spec import strict_from_list


@dataclass(frozen=True)
class QuinticSegmentConfig:
    """One ``[[segments]]`` entry for the ``quintic_segments`` trajectory."""

    duration: float
    start: list[float]
    end: list[float]
    start_vel: list[float] | None = None
    end_vel: list[float] | None = None
    start_acc: list[float] | None = None
    end_acc: list[float] | None = None


@dataclass(frozen=True)
class QuinticSegmentsConfig:
    """Strict TOML schema for the ``quintic_segments`` trajectory."""

    dt: float
    segments: list[dict]
    name: str = "quintic_segments"


@dataclass(frozen=True)
class WaypointConfig:
    """One ``[[waypoints]]`` entry for the ``waypoints`` trajectory."""

    duration: float
    position: list[float]


@dataclass(frozen=True)
class WaypointsConfig:
    """Strict TOML schema for the ``waypoints`` trajectory."""

    dt: float
    waypoints: list[dict]
    name: str = "waypoints"


@dataclass(frozen=True)
class PhaseConfig:
    """One ``[[phases]]`` entry for the ``phase_list`` trajectory.

    ``signals`` maps signal names to per-step setpoints. A name matching a
    plant in the sim manifest is routed to ``plant.step(...)``; any other
    name is an actuator passthrough declared via the scenario's ``[signals]``
    table (see :func:`shinro.simulation.runner.iter_phase_schedule`).
    """

    duration: float
    signals: dict[str, list[float]]


@dataclass(frozen=True)
class PhasesConfig:
    """Strict TOML schema for the ``phase_list`` trajectory."""

    dt: float
    phases: list[dict]
    name: str = "phase_list"


@register_trajectory("quintic_segments")
class QuinticPolynomial(TrajectoryGenerator):
    """5th-order polynomial trajectory generator.

    Generates smooth point-to-point trajectories using a quintic polynomial:

    .. math::

        p(t) = a_5 t^5 + a_4 t^4 + a_3 t^3 + a_2 t^2 + a_1 t + a_0

    Enforces position, velocity, AND acceleration constraints at both start
    and end (6 boundary conditions → 6 coefficients). When all velocities
    and accelerations are zero (rest-to-rest), this reduces to the
    minimum-jerk trajectory:

    .. math::

        p(s) = p_0 + (p_f - p_0)(10s^3 - 15s^4 + 6s^5), \\quad s = t/T

    Solves a 6x6 Vandermonde-like linear system via ``bk.solve()``.
    Supports arbitrary N-dimensional positions — the right-hand side is
    stacked as (6, N) and solved once.

    Args:
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(self, backend: ArrayBackend | None = None):
        self.bk = backend or NumpyBackend()

    def generate(
        self,
        start_position,
        end_position,
        duration: float,
        start_vel=None,
        end_vel=None,
        start_acc=None,
        end_acc=None,
    ):
        """Compute quintic polynomial coefficients by solving the 6x6 system.

        Solves :math:`M c = b` where:

        .. math::

            M = \\begin{bmatrix}
            0 & 0 & 0 & 0 & 0 & 1 \\\\
            T^5 & T^4 & T^3 & T^2 & T & 1 \\\\
            0 & 0 & 0 & 0 & 1 & 0 \\\\
            5T^4 & 4T^3 & 3T^2 & 2T & 1 & 0 \\\\
            0 & 0 & 0 & 2 & 0 & 0 \\\\
            20T^3 & 12T^2 & 6T & 2 & 0 & 0
            \\end{bmatrix}, \\quad
            b = \\begin{bmatrix} p_0 \\\\ p_f \\\\ v_0 \\\\ v_f \\\\ a_0 \\\\ a_f \\end{bmatrix}

        Any boundary condition set to None defaults to zero (rest-to-rest).

        Args:
            start_position: Initial position vector (N,).
            end_position: Final position vector (N,).
            duration: Total trajectory time in seconds.
            start_vel: Initial velocity vector (N,). Defaults to zeros.
            end_vel: Final velocity vector (N,). Defaults to zeros.
            start_acc: Initial acceleration vector (N,). Defaults to zeros.
            end_acc: Final acceleration vector (N,). Defaults to zeros.
        """
        start_vel = self.bk.zeros_like(start_position) if start_vel is None else start_vel
        start_acc = self.bk.zeros_like(start_position) if start_acc is None else start_acc
        end_vel = self.bk.zeros_like(end_position) if end_vel is None else end_vel
        end_acc = self.bk.zeros_like(end_position) if end_acc is None else end_acc

        self.T = duration
        T = duration
        M = self.bk.array([
            [0, 0, 0, 0, 0, 1],
            [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
            [0, 0, 0, 0, 1, 0],
            [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0],
        ])

        b = [start_position, end_position, start_vel, end_vel, start_acc, end_acc]

        coeff_vectors = self.bk.solve(M, b)

        self.A = coeff_vectors[0]
        self.B = coeff_vectors[1]
        self.C = coeff_vectors[2]
        self.D = coeff_vectors[3]
        self.E = coeff_vectors[4]
        self.F = coeff_vectors[5]

    def position_at(self, t: float):
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds (clipped to [0, T]).

        Returns:
            Tuple of (position, velocity, acceleration) arrays, each of
            shape matching the input dimensions (N,).
        """
        t = self.bk.clip(t, 0, self.T)
        pos = self.A * t ** 5 + self.B * t ** 4 + self.C * t ** 3 + self.D * t ** 2 + self.E * t + self.F
        vel = 5 * self.A * t ** 4 + 4 * self.B * t ** 3 + 3 * self.C * t ** 2 + 2 * self.D * t + self.E
        acc = 20 * self.A * t ** 3 + 12 * self.B * t ** 2 + 6 * self.C * t + 2 * self.D
        return pos, vel, acc

    Config = QuinticSegmentsConfig


@register_trajectory("quintic_segments")
class QuinticPolynomialConfigAdapter(ConfigDriven):
    """Adapter so ``from_config`` uses the ``generate()`` + ``position_at()`` API.

    Registered as ``"quintic_segments"`` in the trajectory registry.
    """

    Config = QuinticSegmentsConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a waypoint schedule from a TOML config dict or :class:`QuinticSegmentsConfig`.

        Config fields:
            dt: Time step.
            segments: List of segment dicts, each with:
                - duration: Segment duration (s).
                - start: Start position list.
                - end: End position list.
                - start_vel: Optional start velocity (default: zeros).
                - end_vel: Optional end velocity (default: zeros).
                - start_acc: Optional start acceleration (default: zeros).
                - end_acc: Optional end acceleration (default: zeros).

        Args:
            config: TOML config dict or QuinticSegmentsConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, N) with position waypoints.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        segs = strict_from_list(QuinticSegmentConfig, cfg.segments, "quintic_segments.segment")
        schedule = []
        for seg in segs:
            n_steps = int(np.round(seg.duration / cfg.dt))
            p0 = bk.array(seg.start)
            pf = bk.array(seg.end)
            start_vel = bk.array(seg.start_vel if seg.start_vel is not None else [0.0] * len(seg.start))
            end_vel = bk.array(seg.end_vel if seg.end_vel is not None else [0.0] * len(seg.end))
            start_acc = bk.array(seg.start_acc if seg.start_acc is not None else [0.0] * len(seg.start))
            end_acc = bk.array(seg.end_acc if seg.end_acc is not None else [0.0] * len(seg.end))
            traj = QuinticPolynomial(backend=bk)
            traj.generate(p0, pf, seg.duration, start_vel, end_vel, start_acc, end_acc)
            for k in range(n_steps):
                t = k * cfg.dt
                pos, _, _ = traj.position_at(t)
                schedule.append(pos)
        return bk.array(schedule)


@register_trajectory("waypoints")
class WaypointSchedule(ConfigDriven):
    """Simple waypoint schedule — constant position per segment.

    Returns a flat array of position waypoints, one per time step.
    """

    Config = WaypointsConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a waypoint schedule from a TOML config dict or :class:`WaypointsConfig`.

        Config fields:
            dt: Time step.
            waypoints: List of waypoint dicts, each with:
                - duration: How long to hold this position (s).
                - position: Position list.

        Args:
            config: TOML config dict or WaypointsConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, N) with position waypoints.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        wps = strict_from_list(WaypointConfig, cfg.waypoints, "waypoints.waypoint")
        schedule = []
        for wp in wps:
            n_steps = int(np.round(wp.duration / cfg.dt))
            schedule.extend([bk.array(wp.position)] * n_steps)
        return bk.array(schedule)


@register_trajectory("phase_list")
class PhaseSchedule(ConfigDriven):
    """Multi-signal phase schedule for multi-plant feedforward sequences.

    Returns a dict mapping signal name → per-step setpoints. Signal names
    are declared by the manifest: a key matching a plant name is routed to
    ``plant.step(...)``; any other key is an actuator passthrough declared in
    the scenario's ``[signals]`` table. See
    :func:`shinro.simulation.runner.iter_phase_schedule`.
    """

    Config = PhasesConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a phase schedule from a TOML config dict or :class:`PhasesConfig`.

        Config fields:
            dt: Time step.
            phases: List of ``[[phases]]`` entries, each with:
                - duration: Phase duration (s).
                - signals: Dict of signal name → per-step setpoint (list of
                  floats; single-scalar signals like a gripper use a
                  one-element list, e.g. ``jaw = [0.5]``).

        Args:
            config: TOML config dict or PhasesConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Dict mapping signal name → array of shape (total_steps, N) (or
            (total_steps, 1) for scalar signals).
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        phases = strict_from_list(PhaseConfig, cfg.phases, "phase_list.phase")
        schedules: dict[str, list] = {}
        for phase in phases:
            n_steps = int(np.round(phase.duration / cfg.dt))
            for name, setpoint in phase.signals.items():
                seq = schedules.setdefault(name, [])
                arr = bk.array(setpoint)
                for _ in range(n_steps):
                    seq.append(bk.copy(arr))
        missing = {name for name in schedules if len(schedules[name]) != max(len(v) for v in schedules.values())}
        if missing:
            raise ValueError(f"phase_list: signal(s) {sorted(missing)} appear in fewer phases than others — every signal must be declared in every phase")
        return {name: bk.array(seq) for name, seq in schedules.items()}
