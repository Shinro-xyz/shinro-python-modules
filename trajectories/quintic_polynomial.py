import numpy as np
from typing import Optional
from components import TrajectoryGenerator
from factories.registry import register_trajectory


@register_trajectory("quintic_segments")
class QuinticPolynomial(TrajectoryGenerator):
    """5th-order polynomial trajectory generator with full boundary condition control.

    Generates smooth point-to-point trajectories using a quintic polynomial:
        p(t) = a₅t⁵ + a₄t⁴ + a₃t³ + a₂t² + a₁t + a₀

    Enforces position, velocity, AND acceleration constraints at both start
    and end (6 boundary conditions → 6 coefficients). When all velocities and
    accelerations are zero (rest-to-rest), this reduces to the minimum-jerk
    trajectory: p(s) = p₀ + (p_f - p₀)(10s³ - 15s⁴ + 6s⁵), s = t/T.

    Uses np.linalg.solve with a 6×6 Vandermonde-like matrix. Supports
    arbitrary N-dimensional positions — pass b as (6, N) and solve once.

    Usage:
        traj = QuinticPolynomial()
        traj.generate(
            start_position=np.array([0.0, 0.0, 0.0]),
            end_position=np.array([1.0, 0.5, 0.3]),
            duration=2.0,
        )
        pos, vel, acc = traj.position_at(t=1.0)

        # With custom boundary conditions:
        traj.generate(
            start_position=np.array([0.0, 0.0]),
            end_position=np.array([1.0, 0.0]),
            duration=1.0,
            start_vel=np.array([0.5, 0.0]),
            end_vel=np.array([0.0, 0.0]),
            start_acc=np.array([0.1, 0.0]),
            end_acc=np.array([0.0, 0.0]),
        )
    """

    def generate(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        duration: float,
        start_vel: Optional[np.ndarray] = None,
        end_vel: Optional[np.ndarray] = None,
        start_acc: Optional[np.ndarray] = None,
        end_acc: Optional[np.ndarray] = None,
    ):
        """Compute quintic polynomial coefficients by solving the 6×6 linear system.

        Solves M @ c = b where:
            M = [[0,   0,   0,   0,   0,   1],      # p(0)   = a₀
                 [T⁵,  T⁴,  T³,  T²,  T,   1],      # p(T)   = a₅T⁵ + ... + a₀
                 [0,   0,   0,   0,   1,   0],      # v(0)   = a₁
                 [5T⁴, 4T³, 3T², 2T,  1,   0],      # v(T)   = 5a₅T⁴ + ... + a₁
                 [0,   0,   0,   2,   0,   0],      # a(0)   = 2a₂
                 [20T³,12T², 6T,  2,   0,   0]]     # a(T)   = 20a₅T³ + ... + 2a₂

            b = [p₀, p_f, v₀, v_f, a₀, a_f]

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
        start_vel = np.zeros_like(start_position) if start_vel is None else start_vel
        start_acc = np.zeros_like(start_position) if start_acc is None else start_acc
        end_vel = np.zeros_like(end_position) if end_vel is None else end_vel
        end_acc = np.zeros_like(end_position) if end_acc is None else end_acc

        self.T = duration
        T = duration
        M = np.array([
            [0, 0, 0, 0, 0, 1],
            [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
            [0, 0, 0, 0, 1, 0],
            [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
            [0, 0, 0, 2, 0, 0],
            [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0],
        ])

        b = [start_position, end_position, start_vel, end_vel, start_acc, end_acc]

        coeff_vectors = np.linalg.solve(M, b)

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
            Tuple of (position, velocity, acceleration) arrays, each of shape
            matching the input dimensions (N,).
        """
        t = np.clip(t, 0, self.T)
        pos = self.A * t ** 5 + self.B * t ** 4 + self.C * t ** 3 + self.D * t ** 2 + self.E * t + self.F
        vel = 5 * self.A * t ** 4 + 4 * self.B * t ** 3 + 3 * self.C * t ** 2 + 2 * self.D * t + self.E
        acc = 20 * self.A * t ** 3 + 12 * self.B * t ** 2 + 6 * self.C * t + 2 * self.D

        return pos, vel, acc


@register_trajectory("quintic_segments")
class QuinticPolynomialConfigAdapter:
    """Adapter so from_config uses the generate() + position_at() API."""
    @classmethod
    def from_config(cls, config):
        dt = config["dt"]
        schedule = []
        default_vel = np.array([0.0, 0.0, 0.0])
        default_acc = np.array([0.0, 0.0, 0.0])
        for seg in config["segments"]:
            n_steps = int(np.round(seg["duration"] / dt))
            p0 = np.array(seg["start"])
            pf = np.array(seg["end"])
            T = seg["duration"]
            start_vel = np.array(seg.get("start_vel", default_vel))
            end_vel = np.array(seg.get("end_vel", default_vel))
            start_acc = np.array(seg.get("start_acc", default_acc))
            end_acc = np.array(seg.get("end_acc", default_acc))
            traj = QuinticPolynomial()
            traj.generate(p0, pf, T, start_vel, end_vel, start_acc, end_acc)
            for k in range(n_steps):
                t = k * dt
                pos, _, _ = traj.position_at(t)
                schedule.append(pos)
        return np.array(schedule)


@register_trajectory("waypoints")
class WaypointSchedule:
    @classmethod
    def from_config(cls, config):
        dt = config["dt"]
        schedule = []
        for wp in config["waypoints"]:
            n_steps = int(np.round(wp["duration"] / dt))
            schedule.extend([np.array(wp["position"])] * n_steps)
        return np.array(schedule)


@register_trajectory("phase_list")
class PhaseSchedule:
    @classmethod
    def from_config(cls, config):
        dt = config["dt"]
        arm_sched = []
        base_sched = []
        jaw_sched = []
        for phase in config["phases"]:
            n_steps = int(np.round(phase["duration"] / dt))
            arm = np.array(phase["arm"])
            base = np.array(phase["base"])
            jaw = float(phase["jaw"])
            for _ in range(n_steps):
                arm_sched.append(arm.copy())
                base_sched.append(base.copy())
                jaw_sched.append(jaw)
        return {"arm": np.array(arm_sched), "base": np.array(base_sched), "jaw": np.array(jaw_sched)}