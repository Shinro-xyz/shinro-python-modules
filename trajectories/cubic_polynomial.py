import numpy as np
from components import TrajectoryGenerator
from factories.registry import register_trajectory


@register_trajectory("cubic_segments")
class CubicPolynomial(TrajectoryGenerator):
    """3rd-order polynomial trajectory generator with position and velocity continuity.

    Generates smooth point-to-point trajectories using a cubic polynomial:
        p(t) = a₀ + a₁t + a₂t² + a₃t³

    The coefficients are computed in closed form from boundary conditions on
    position and velocity at both ends. Acceleration is continuous but NOT
    constrained at boundaries (use QuinticPolynomial if acceleration constraints
    are needed).

    Supports arbitrary N-dimensional positions via numpy broadcasting — one
    solve handles all dimensions simultaneously.

    Usage:
        traj = CubicPolynomial()
        traj.generate(
            start_position=np.array([0.0, 0.0, 0.0]),
            end_position=np.array([1.0, 0.5, 0.3]),
            duration=2.0,
            start_vel=np.zeros(3),
            end_vel=np.zeros(3),
        )
        pos, vel, acc = traj.position_at(t=1.0)
    """

    def generate(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        duration: float,
        start_vel: np.ndarray,
        end_vel: np.ndarray,
    ):
        """Compute cubic polynomial coefficients from boundary conditions.

        Solves the closed-form cubic coefficients:
            a₀ = p₀
            a₁ = v₀
            a₂ = (3Δp - T(2v₀ + v_f)) / T²
            a₃ = (-2Δp + T(v₀ + v_f)) / T³

        where Δp = p_f - p₀ and T = duration.

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
            Tuple of (position, velocity, acceleration) arrays, each of shape
            matching the input dimensions (N,).
        """
        t = np.clip(t, 0, self.duration)
        pos = self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3
        vel = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2
        acc = 2 * self.a2 + 6 * self.a3 * t

        return pos, vel, acc

    @classmethod
    def from_config(cls, config):
        dt = config["dt"]
        schedule = []
        for seg in config["segments"]:
            n_steps = int(np.round(seg["duration"] / dt))
            p0 = np.array(seg["start"])
            pf = np.array(seg["end"])
            T = seg["duration"]
            a0 = p0
            a2 = 3.0 * (pf - p0) / (T * T)
            a3 = -2.0 * (pf - p0) / (T * T * T)
            for k in range(n_steps):
                t = k * dt
                schedule.append(a0 + a2 * t * t + a3 * t * t * t)
        return np.array(schedule)

    