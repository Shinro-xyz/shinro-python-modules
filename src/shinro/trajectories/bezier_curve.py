"""Bézier curve trajectory generator."""

import math
from dataclasses import dataclass
from typing import Any

from shinro.components import TrajectoryGenerator
from shinro.factories.registry import register_trajectory
from shinro.trajectories.sampling import sample_schedule
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


@dataclass(frozen=True)
class BezierConfig:
    """Strict TOML schema for the ``bezier`` trajectory.

    Fields:
        dt: Sampling period (s) of the emitted waypoint schedule.
        duration: Trajectory duration :math:`T` (s).
        control_points: Ordered control points; each is a list of floats of
            the same length (the spatial dimension). The curve degree is
            ``len(control_points) - 1`` and the two ends are interpolated
            exactly.
        derivatives: Also emit the reference velocity/acceleration. When true,
            ``from_config`` returns a ``{"position", "velocity",
            "acceleration"}`` dict of ``(steps, d)`` arrays instead of the
            position schedule.
        name: Registered trajectory name (validated against the ``type`` key).
    """

    dt: float
    duration: float
    control_points: list[list[float]]
    derivatives: bool = False
    name: str = "bezier"


@register_trajectory("bezier")
class BezierCurve(TrajectoryGenerator):
    r"""Bézier trajectory over a list of control points.

    With :math:`n =` (number of control points) :math:`- 1` and
    :math:`s = t / T`:

    .. math::

        B(t) = \sum_{i=0}^{n} \binom{n}{i}\, s^i\, (1 - s)^{n-i}\, P_i

    The endpoints are interpolated (:math:`B(0) = P_0`, :math:`B(T) = P_n`),
    and the velocity / acceleration are themselves Bézier curves over the
    scaled control-point differences:

    .. math::

        \dot{B}(t) &= \frac{n}{T} \sum_{i=0}^{n-1} \binom{n-1}{i}\, s^i\,
            (1 - s)^{n-1-i}\, (P_{i+1} - P_i), \\
        \ddot{B}(t) &= \frac{n\,(n-1)}{T^2} \sum_{j=0}^{n-2} \binom{n-2}{j}\,
            s^j\, (1 - s)^{n-2-j}\, (P_{j+2} - 2 P_{j+1} + P_j).

    The boundary velocities are :math:`\frac{n}{T}(P_1 - P_0)` and
    :math:`\frac{n}{T}(P_n - P_{n-1})` — **not** zero unless the end points
    are duplicated in the control list.

    Args:
        control_points: Ordered control points, each shape ``(d,)``. The
            curve degree is implied by the list length (a 4-point list is a
            cubic).
        backend: Array backend. Defaults to NumpyBackend.
    """

    Config = BezierConfig

    def __init__(self, control_points, backend: ArrayBackend | None = None):
        self.bk = backend or NumpyBackend()
        self.P = control_points

    def _bernstein(self, points, coeffs, s):
        r"""Evaluate :math:`\sum_i \binom{n}{i} s^i (1-s)^{n-i} \mathrm{points}_i`.

        Args:
            points: Control block of shape ``(n + 1, d)``.
            coeffs: Binomial coefficients :math:`\binom{n}{i}`, shape ``(n + 1,)``.
            s: Normalised time in ``[0, 1]``.

        Returns:
            Curve point of shape ``(d,)``.
        """
        n = len(points) - 1
        k = self.bk.array(range(n + 1))
        w = coeffs * (s**k) * ((1.0 - s) ** (n - k))
        return w @ points

    def generate(self, start_position=None, end_position=None, duration: float | None = None):
        """Compute the Bernstein blocks for position, velocity, acceleration.

        ``start_position`` / ``end_position`` are informational only — the
        curve's ends are the first and last control points, and supplying
        conflicting values is an error.

        Args:
            start_position: Optional expected start; must equal ``P[0]``.
            end_position: Optional expected end; must equal ``P[-1]``.
            duration: Trajectory duration :math:`T` (s).

        Raises:
            ValueError: If the control list has fewer than 2 points, ragged
                point dimensions, a non-positive duration, or declared
                start/end that disagree with the control list ends.
        """
        if duration is None or duration <= 0:
            raise ValueError(f"BezierCurve: duration must be positive, got {duration}")
        if len(self.P) < 2:
            raise ValueError("BezierCurve: needs at least 2 control points")
        dims = {len(p) for p in self.P}
        if len(dims) != 1:
            raise ValueError(
                f"BezierCurve: all control points must have the same dimension, got {sorted(dims)}"
            )

        P = self.bk.array(self.P)
        n = len(self.P) - 1
        self.T = duration

        if start_position is not None and not bool(
            self.bk.allclose(self.bk.array(start_position), P[0])
        ):
            raise ValueError(
                "BezierCurve: start_position disagrees with the first control point "
                "(a Bezier curve interpolates its control-list ends exactly)"
            )
        if end_position is not None and not bool(
            self.bk.allclose(self.bk.array(end_position), P[-1])
        ):
            raise ValueError(
                "BezierCurve: end_position disagrees with the last control point "
                "(a Bezier curve interpolates its control-list ends exactly)"
            )
        self.start_position = P[0]
        self.end_position = P[-1]

        self._points = P
        self._bc0 = self.bk.array([math.comb(n, i) for i in range(n + 1)])

        # Degree n-1 velocity curve on scaled first differences (P_{i+1} - P_i).
        m = len(self.P)
        d1 = self.bk.slice_(P, 1, m) - self.bk.slice_(P, 0, m - 1)
        self._vel_points = d1 * (n / self.T)
        self._bc1 = self.bk.array([math.comb(n - 1, i) for i in range(n)])

        # Degree n-2 acceleration curve on scaled second differences.
        self._has_acc = n >= 2
        if self._has_acc:
            d2 = self.bk.slice_(d1, 1, n) - self.bk.slice_(d1, 0, n - 1)
            self._acc_points = d2 * (n * (n - 1) / (self.T * self.T))
            self._bc2 = self.bk.array([math.comb(n - 2, j) for j in range(n - 1)])

    def position_at(self, t: float):
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds (clipped to :math:`[0, T]`).

        Returns:
            Tuple of (position, velocity, acceleration) arrays, each of shape
            ``(d,)``. Acceleration is exact zeros for a degree-1 curve.
        """
        t = self.bk.clip(t, 0, self.T)
        s = t / self.T
        pos = self._bernstein(self._points, self._bc0, s)
        vel = self._bernstein(self._vel_points, self._bc1, s)
        acc = (
            self._bernstein(self._acc_points, self._bc2, s)
            if self._has_acc
            else self.bk.zeros_like(pos)
        )
        return pos, vel, acc

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None) -> Any:
        """Create a waypoint schedule from a TOML config dict or :class:`BezierConfig`.

        Config fields:
            dt: Sampling period (s).
            duration: Trajectory duration (s).
            control_points: Ordered control points (list of float lists).

        Args:
            config: TOML config dict or BezierConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, d) with position waypoints, or — when
            ``derivatives`` is true — a dict of ``(total_steps, d)`` arrays
            under ``"position"``, ``"velocity"``, and ``"acceleration"``.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        traj = cls(cfg.control_points, backend=bk)
        traj.generate(duration=cfg.duration)
        sampled = sample_schedule(traj, cfg.dt, order=2 if cfg.derivatives else 0)
        return sampled if cfg.derivatives else sampled["position"]
