"""Minimum-snap (7th-order) polynomial trajectory generator."""

from dataclasses import dataclass
from typing import Any

from shinro.components import TrajectoryGenerator
from shinro.factories.registry import register_trajectory
from shinro.trajectories.sampling import sample_schedule
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


@dataclass(frozen=True)
class MinSnapConfig:
    """Strict TOML schema for the ``min_snap`` trajectory.

    Fields:
        dt: Sampling period (s) of the emitted waypoint schedule.
        start: Initial position (list of floats).
        end: Final position (list of floats).
        duration: Trajectory duration :math:`T` (s).
        start_vel, end_vel, start_acc, end_acc, start_jerk, end_jerk: Optional
            boundary derivatives. All default to zero — the rest-to-rest
            minimum-snap trajectory.
        derivatives: Also emit the reference velocity/acceleration. When true,
            ``from_config`` returns a ``{"position", "velocity",
            "acceleration"}`` dict of ``(steps, N)`` arrays.
        name: Registered trajectory name (validated against the ``type`` key).
    """

    dt: float
    start: list[float]
    end: list[float]
    duration: float
    start_vel: list[float] | None = None
    end_vel: list[float] | None = None
    start_acc: list[float] | None = None
    end_acc: list[float] | None = None
    start_jerk: list[float] | None = None
    end_jerk: list[float] | None = None
    derivatives: bool = False
    name: str = "min_snap"


@register_trajectory("min_snap")
class MinSnapPolynomial(TrajectoryGenerator):
    r"""7th-order polynomial trajectory, minimum snap when rest-to-rest.

    .. math::

        p(t) = a_7 t^7 + a_6 t^6 + a_5 t^5 + a_4 t^4 + a_3 t^3 + a_2 t^2
            + a_1 t + a_0

    Eight boundary conditions — position, velocity, acceleration, and **jerk**
    at both ends — fix the eight coefficients with one 8x8 ``bk.solve``. With
    the free boundary derivatives zero this minimises
    :math:`\int (p''')^2\,\mathrm{d}t` (snap is :math:`p''''`), the 7th-order
    sibling of the minimum-jerk quintic:

    .. math::

        p(s) = p_0 + (p_f - p_0)\,(35 s^4 - 84 s^5 + 70 s^6 - 20 s^7),
        \qquad s = t / T

    Jerk is only a boundary condition: ``position_at`` returns
    ``(position, velocity, acceleration)`` like every other generator (the
    sampler does not carry a jerk channel).

    Args:
        backend: Array backend. Defaults to NumpyBackend.
    """

    Config = MinSnapConfig

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
        start_jerk=None,
        end_jerk=None,
    ):
        """Compute the 7th-order coefficients by solving the 8x8 system.

        Any boundary condition set to ``None`` defaults to zero (rest-to-rest).

        Args:
            start_position: Initial position vector (N,).
            end_position: Final position vector (N,).
            duration: Total trajectory time in seconds.
            start_vel, end_vel, start_acc, end_acc, start_jerk, end_jerk:
                Boundary derivatives (N,) or ``None`` for zero.

        Raises:
            ValueError: If ``duration`` is not positive.
        """
        if duration is None or duration <= 0:
            raise ValueError(f"MinSnapPolynomial: duration must be positive, got {duration}")
        bk = self.bk
        start_vel = bk.zeros_like(start_position) if start_vel is None else start_vel
        end_vel = bk.zeros_like(end_position) if end_vel is None else end_vel
        start_acc = bk.zeros_like(start_position) if start_acc is None else start_acc
        end_acc = bk.zeros_like(end_position) if end_acc is None else end_acc
        start_jerk = bk.zeros_like(start_position) if start_jerk is None else start_jerk
        end_jerk = bk.zeros_like(end_position) if end_jerk is None else end_jerk

        self.T = duration
        T = duration
        # Columns are [a_7, a_6, a_5, a_4, a_3, a_2, a_1, a_0]; rows are
        # p, p', p'', p''' evaluated at 0 and T.
        M = bk.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1],
                [T**7, T**6, T**5, T**4, T**3, T**2, T, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [7 * T**6, 6 * T**5, 5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                [0, 0, 0, 0, 0, 2, 0, 0],
                [42 * T**5, 30 * T**4, 20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
                [0, 0, 0, 0, 6, 0, 0, 0],
                [210 * T**4, 120 * T**3, 60 * T**2, 24 * T, 6, 0, 0, 0],
            ]
        )
        rhs = [start_position, end_position, start_vel, end_vel, start_acc, end_acc, start_jerk, end_jerk]
        coeffs = bk.solve(M, rhs)
        self.coeffs = [coeffs[i] for i in range(8)]  # a7, a6, ..., a0

    def position_at(self, t: float):
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds (clipped to :math:`[0, T]`).

        Returns:
            Tuple of (position, velocity, acceleration) arrays, each shape
            ``(N,)``.
        """
        t = self.bk.clip(t, 0, self.T)
        c7, c6, c5, c4, c3, c2, c1, c0 = self.coeffs
        pos = c7 * t**7 + c6 * t**6 + c5 * t**5 + c4 * t**4 + c3 * t**3 + c2 * t**2 + c1 * t + c0
        vel = 7 * c7 * t**6 + 6 * c6 * t**5 + 5 * c5 * t**4 + 4 * c4 * t**3 + 3 * c3 * t**2 + 2 * c2 * t + c1
        acc = 42 * c7 * t**5 + 30 * c6 * t**4 + 20 * c5 * t**3 + 12 * c4 * t**2 + 6 * c3 * t + 2 * c2
        return pos, vel, acc

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None) -> Any:
        """Create a waypoint schedule from a TOML config dict or :class:`MinSnapConfig`.

        Config fields:
            dt: Sampling period (s).
            start: Initial position (list of floats).
            end: Final position (list of floats).
            duration: Trajectory duration (s).
            start_vel, end_vel, start_acc, end_acc, start_jerk, end_jerk:
                Optional boundary derivatives (default: zeros).
            derivatives: Emit the velocity/acceleration dict instead of positions.

        Args:
            config: TOML config dict or MinSnapConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, N) with position waypoints, or — when
            ``derivatives`` is true — a dict of ``(total_steps, N)`` arrays
            under ``"position"``, ``"velocity"``, and ``"acceleration"``.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)

        def _vec(value):
            return None if value is None else bk.array(value)

        traj = cls(backend=bk)
        traj.generate(
            bk.array(cfg.start),
            bk.array(cfg.end),
            cfg.duration,
            _vec(cfg.start_vel),
            _vec(cfg.end_vel),
            _vec(cfg.start_acc),
            _vec(cfg.end_acc),
            _vec(cfg.start_jerk),
            _vec(cfg.end_jerk),
        )
        sampled = sample_schedule(traj, cfg.dt, order=2 if cfg.derivatives else 0)
        return sampled if cfg.derivatives else sampled["position"]
