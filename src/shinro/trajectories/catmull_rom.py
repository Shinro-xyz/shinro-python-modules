"""Catmull-Rom (C1 cubic Hermite) trajectory generator."""

from dataclasses import dataclass
from typing import Any

from shinro.components import TrajectoryGenerator
from shinro.factories.registry import register_trajectory
from shinro.trajectories.cubic_polynomial import CubicPolynomial
from shinro.trajectories.sampling import sample_schedule
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.utils.config_spec import strict_from_list


@dataclass(frozen=True)
class CatmullRomWaypointConfig:
    """One ``[[waypoints]]`` entry for the ``catmull_rom`` trajectory.

    ``duration`` is the travel time from the *previous* waypoint to this one
    (the first entry's duration is the time from ``start``).
    """

    duration: float
    position: list[float]


@dataclass(frozen=True)
class CatmullRomConfig:
    """Strict TOML schema for the ``catmull_rom`` trajectory.

    Fields:
        dt: Sampling period (s) of the emitted waypoint schedule.
        start: Position of waypoint 0.
        waypoints: Ordered ``[[waypoints]]`` entries, each a position reached
            after ``duration`` seconds from the previous waypoint.
        endpoint_tangent: ``"zero"`` (rest-to-rest ends, default) or
            ``"one_sided"`` (classic Catmull-Rom one-sided end tangent).
        derivatives: Also emit the reference velocity/acceleration. When true,
            ``from_config`` returns a ``{"position", "velocity",
            "acceleration"}`` dict of ``(steps, d)`` arrays instead of the
            position schedule.
        name: Registered trajectory name (validated against the ``type`` key).
    """

    dt: float
    start: list[float]
    waypoints: list[dict]
    endpoint_tangent: str = "zero"
    derivatives: bool = False
    name: str = "catmull_rom"


@register_trajectory("catmull_rom")
class CatmullRom(TrajectoryGenerator):
    r"""C1 Catmull-Rom spline through a start point and ordered waypoints.

    Each hop is a cubic Hermite segment (reusing
    :class:`~shinro.trajectories.CubicPolynomial`) whose boundary velocities are
    the Catmull-Rom tangents

    .. math::

        \dot{P}_i = \frac{P_{i+1} - P_{i-1}}{d_{i-1} + d_i}

    at interior waypoints, where :math:`d_i` is the duration of hop
    :math:`i \to i+1`. Each tangent is the end velocity of one segment and the
    start velocity of the next, so the concatenated curve is C1 and **passes
    through** every waypoint without stopping — contrast ``cubic_segments``,
    whose per-segment defaults bring the curve to rest at each waypoint.

    Endpoint velocities default to zero (rest-to-rest start/stop);
    ``endpoint_tangent="one_sided"`` uses :math:`(P_1 - P_0)/d_0` and
    :math:`(P_n - P_{n-1})/d_{n-1}` — the classic one-sided Catmull-Rom ends.

    Args:
        backend: Array backend. Defaults to NumpyBackend.
    """

    Config = CatmullRomConfig

    def __init__(self, backend: ArrayBackend | None = None):
        self.bk = backend or NumpyBackend()

    def _tangents(self, points, durations, endpoint_tangent: str):
        """Catmull-Rom boundary velocities for every waypoint.

        Args:
            points: List of ``(d,)`` backend arrays (waypoint 0 first).
            durations: Hop durations; ``durations[i]`` spans points[i] to
                points[i+1].
            endpoint_tangent: ``"zero"`` or ``"one_sided"``.

        Returns:
            List of ``(d,)`` velocity arrays, one per waypoint.

        Raises:
            ValueError: On an unknown ``endpoint_tangent`` value.
        """
        bk = self.bk
        n = len(durations)
        v = [bk.zeros_like(points[0]) for _ in range(n + 1)]
        for i in range(1, n):
            v[i] = (points[i + 1] - points[i - 1]) / (durations[i - 1] + durations[i])

        if endpoint_tangent == "zero":
            v[0] = bk.zeros_like(points[0])
            v[n] = bk.zeros_like(points[0])
        elif endpoint_tangent == "one_sided":
            v[0] = (points[1] - points[0]) / durations[0]
            v[n] = (points[n] - points[n - 1]) / durations[n - 1]
        else:
            raise ValueError(
                f"CatmullRom: endpoint_tangent must be 'zero' or 'one_sided', got {endpoint_tangent!r}"
            )
        return v

    def generate(self, waypoints, durations, endpoint_tangent: str = "zero"):
        """Build the C1 segments from waypoints and hop durations.

        Args:
            waypoints: Sequence of n+1 positions (each shape ``(d,)``),
                ``waypoints[0]`` the start.
            durations: n hop durations (s); ``durations[i]`` is the time from
                ``waypoints[i]`` to ``waypoints[i+1]``.
            endpoint_tangent: ``"zero"`` or ``"one_sided"``.

        Raises:
            ValueError: On fewer than 2 waypoints, ragged dimensions, a
                duration count that disagrees with the waypoint count, a
                non-positive duration, or an unknown ``endpoint_tangent``.
        """
        if len(waypoints) < 2:
            raise ValueError("CatmullRom: needs at least 2 waypoints (start + 1)")
        dims = {len(p) for p in waypoints}
        if len(dims) != 1:
            raise ValueError(
                f"CatmullRom: all waypoints must have the same dimension, got {sorted(dims)}"
            )
        n = len(durations)
        if n != len(waypoints) - 1:
            raise ValueError(
                f"CatmullRom: expected {len(waypoints) - 1} hop durations for "
                f"{len(waypoints)} waypoints, got {n}"
            )
        if any(d <= 0 for d in durations):
            raise ValueError("CatmullRom: every hop duration must be positive")

        points = [self.bk.array(p) for p in waypoints]
        tangents = self._tangents(points, durations, endpoint_tangent)

        self._segments = []
        t0 = 0.0
        for i in range(n):
            seg = CubicPolynomial(backend=self.bk)
            seg.generate(points[i], points[i + 1], durations[i], tangents[i], tangents[i + 1])
            self._segments.append((t0, durations[i], seg))
            t0 += durations[i]
        self.T = t0

    def position_at(self, t: float):
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds (clipped to :math:`[0, T]`).

        Returns:
            Tuple of (position, velocity, acceleration) arrays, each of shape
            ``(d,)``.
        """
        t = self.bk.clip(t, 0, self.T)
        for start, duration, seg in self._segments:
            if t <= start + duration:
                return seg.position_at(t - start)
        start, _, seg = self._segments[-1]
        return seg.position_at(t - start)

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None) -> Any:
        """Create a waypoint schedule from a TOML config dict or :class:`CatmullRomConfig`.

        Config fields:
            dt: Sampling period (s).
            start: Position of waypoint 0.
            waypoints: Ordered ``[[waypoints]]`` entries, each with a
                ``duration`` (s from the previous waypoint) and a ``position``.
            endpoint_tangent: ``"zero"`` (default) or ``"one_sided"``.

        Args:
            config: TOML config dict or CatmullRomConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, d) with position waypoints, or — when
            ``derivatives`` is true — a dict of ``(total_steps, d)`` arrays
            under ``"position"``, ``"velocity"``, and ``"acceleration"``.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        wps = strict_from_list(CatmullRomWaypointConfig, cfg.waypoints, "catmull_rom.waypoint")
        waypoints = [cfg.start] + [wp.position for wp in wps]
        durations = [wp.duration for wp in wps]
        traj = cls(backend=bk)
        traj.generate(waypoints, durations, endpoint_tangent=cfg.endpoint_tangent)
        sampled = sample_schedule(traj, cfg.dt, order=2 if cfg.derivatives else 0)
        return sampled if cfg.derivatives else sampled["position"]
