from typing import Optional
from components import TrajectoryGenerator
from factories.registry import register_trajectory
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_trajectory("cubic_segments")
class CubicPolynomial(TrajectoryGenerator):
    """3rd-order polynomial trajectory generator.

    Generates smooth point-to-point trajectories using a cubic polynomial:

    .. math::

        p(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3

    The coefficients are computed in **closed form** (no matrix solve) from
    boundary conditions on position and velocity at both ends. Acceleration
    is continuous but NOT constrained at boundaries (use QuinticPolynomial
    if acceleration constraints are needed).

    Supports arbitrary N-dimensional positions via element-wise operations.

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
        start_vel,
        end_vel,
    ):
        """Compute cubic polynomial coefficients from boundary conditions.

        Closed-form solution:

        .. math::

            a_0 &= p_0 \\\\
            a_1 &= v_0 \\\\
            a_2 &= \\frac{3\\Delta p - T(2v_0 + v_f)}{T^2} \\\\
            a_3 &= \\frac{-2\\Delta p + T(v_0 + v_f)}{T^3}

        where :math:`\\Delta p = p_f - p_0` and :math:`T =` duration.

        Args:
            start_position: Initial position vector (N,).
            end_position: Final position vector (N,).
            duration: Total trajectory time in seconds.
            start_vel: Initial velocity vector (N,).
            end_vel: Final velocity vector (N,).
        """
        self.a0 = start_position
        self.a1 = start_vel
        a2_numerator = 3 * (end_position - start_position) - duration * (2 * start_vel + end_vel)
        self.a2 = a2_numerator / (duration ** 2)

        a3_numerator = -2 * (end_position - start_position) + duration * (start_vel + end_vel)
        self.a3 = a3_numerator / (duration ** 3)

        self.duration = duration

    def position_at(self, t: float):
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds (clipped to [0, duration]).

        Returns:
            Tuple of (position, velocity, acceleration) arrays, each of
            shape matching the input dimensions (N,).
        """
        t = self.bk.clip(t, 0, self.duration)
        pos = self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3
        vel = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2
        acc = 2 * self.a2 + 6 * self.a3 * t
        return pos, vel, acc

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
            traj = cls(backend=bk)
            traj.generate(p0, pf, T, start_vel, end_vel)
            for k in range(n_steps):
                t = k * dt
                pos, _, _ = traj.position_at(t)
                schedule.append(pos)
        return bk.array(schedule)
