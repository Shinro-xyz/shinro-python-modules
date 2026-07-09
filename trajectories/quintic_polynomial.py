from typing import Optional
from components import TrajectoryGenerator
from factories.registry import register_trajectory
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


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

    def __init__(self, backend: Optional[ArrayBackend] = None):
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


@register_trajectory("quintic_segments")
class QuinticPolynomialConfigAdapter:
    """Adapter so ``from_config`` uses the ``generate()`` + ``position_at()`` API.

    Registered as ``"quintic_segments"`` in the trajectory registry.
    """

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create a waypoint schedule from a TOML config dict.

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
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, N) with position waypoints.
        """
        bk = backend or NumpyBackend()
        dt = config["dt"]
        schedule = []
        for seg in config["segments"]:
            n_steps = int(np.round(seg["duration"] / dt))
            p0 = bk.array(seg["start"])
            pf = bk.array(seg["end"])
            T = seg["duration"]
            start_vel = bk.array(seg.get("start_vel", [0.0, 0.0, 0.0]))
            end_vel = bk.array(seg.get("end_vel", [0.0, 0.0, 0.0]))
            start_acc = bk.array(seg.get("start_acc", [0.0, 0.0, 0.0]))
            end_acc = bk.array(seg.get("end_acc", [0.0, 0.0, 0.0]))
            traj = QuinticPolynomial(backend=bk)
            traj.generate(p0, pf, T, start_vel, end_vel, start_acc, end_acc)
            for k in range(n_steps):
                t = k * dt
                pos, _, _ = traj.position_at(t)
                schedule.append(pos)
        return bk.array(schedule)


@register_trajectory("waypoints")
class WaypointSchedule:
    """Simple waypoint schedule — constant position per segment.

    Returns a flat array of position waypoints, one per time step.
    """

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create a waypoint schedule from a TOML config dict.

        Config fields:
            dt: Time step.
            waypoints: List of waypoint dicts, each with:
                - duration: How long to hold this position (s).
                - position: Position list.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, N) with position waypoints.
        """
        bk = backend or NumpyBackend()
        dt = config["dt"]
        schedule = []
        for wp in config["waypoints"]:
            n_steps = int(np.round(wp["duration"] / dt))
            schedule.extend([bk.array(wp["position"])] * n_steps)
        return bk.array(schedule)


@register_trajectory("phase_list")
class PhaseSchedule:
    """Multi-signal phase schedule for pick-and-place sequences.

    Returns a dict with ``"arm"``, ``"base"``, and ``"jaw"`` arrays, each
    containing the per-step setpoint for that subsystem.
    """

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create a phase schedule from a TOML config dict.

        Config fields:
            dt: Time step.
            phases: List of phase dicts, each with:
                - duration: Phase duration (s).
                - arm: Arm velocity twist list (6,).
                - base: Base velocity list (3,).
                - jaw: Jaw position (float).

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Dict with keys ``"arm"``, ``"base"``, ``"jaw"``, each an array
            of shape (total_steps, N).
        """
        bk = backend or NumpyBackend()
        dt = config["dt"]
        arm_sched = []
        base_sched = []
        jaw_sched = []
        for phase in config["phases"]:
            n_steps = int(np.round(phase["duration"] / dt))
            arm = bk.array(phase["arm"])
            base = bk.array(phase["base"])
            jaw = float(phase["jaw"])
            for _ in range(n_steps):
                arm_sched.append(bk.copy(arm))
                base_sched.append(bk.copy(base))
                jaw_sched.append(jaw)
        return {"arm": bk.array(arm_sched), "base": bk.array(base_sched), "jaw": bk.array(jaw_sched)}
