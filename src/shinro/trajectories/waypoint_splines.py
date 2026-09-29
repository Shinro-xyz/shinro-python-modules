"""Akima and natural cubic-spline waypoint trajectories.

Both pass through an ordered waypoint list at C1 (Akima) / C2 (natural cubic
spline) and are built from cubic Hermite hops, reusing
:class:`~shinro.trajectories.CubicPolynomial` — only the waypoint tangents
differ (Akima's local four-slope estimate vs the tridiagonal second-derivative
solve).
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from shinro.components import TrajectoryGenerator
from shinro.factories.registry import register_trajectory
from shinro.trajectories.cubic_polynomial import CubicPolynomial
from shinro.trajectories.sampling import sample_schedule
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.utils.config_spec import strict_from_list


@dataclass(frozen=True)
class WaypointEntry:
    """One ``[[waypoints]]`` entry for the waypoint-spline trajectories.

    ``duration`` is the travel time from the *previous* waypoint (from
    ``start`` for the first entry).
    """

    duration: float
    position: list[float]


@dataclass(frozen=True)
class AkimaConfig:
    """Strict TOML schema for the ``akima`` trajectory.

    Fields:
        dt: Sampling period (s) of the emitted waypoint schedule.
        start: Position of waypoint 0.
        waypoints: Ordered ``[[waypoints]]`` entries, each a position reached
            after ``duration`` seconds from the previous waypoint.
        derivatives: Also emit the reference velocity/acceleration. When true,
            ``from_config`` returns a ``{"position", "velocity",
            "acceleration"}`` dict of ``(steps, d)`` arrays.
        name: Registered trajectory name (validated against the ``type`` key).
    """

    dt: float
    start: list[float]
    waypoints: list[dict]
    derivatives: bool = False
    name: str = "akima"


@dataclass(frozen=True)
class CubicSplineConfig:
    """Strict TOML schema for the ``cubic_spline`` trajectory.

    Fields:
        dt: Sampling period (s) of the emitted waypoint schedule.
        start: Position of waypoint 0.
        waypoints: Ordered ``[[waypoints]]`` entries, each a position reached
            after ``duration`` seconds from the previous waypoint.
        derivatives: Also emit the reference velocity/acceleration. When true,
            ``from_config`` returns a ``{"position", "velocity",
            "acceleration"}`` dict of ``(steps, d)`` arrays.
        name: Registered trajectory name (validated against the ``type`` key).
    """

    dt: float
    start: list[float]
    waypoints: list[dict]
    derivatives: bool = False
    name: str = "cubic_spline"


class _WaypointSpline(TrajectoryGenerator):
    """Shared plumbing for a waypoint spline built from cubic Hermite hops.

    Subclasses implement :meth:`_tangents` (one velocity per waypoint); this
    class validates the waypoints, builds one :class:`CubicPolynomial` per hop,
    and evaluates the concatenated curve.
    """

    def __init__(self, backend: ArrayBackend | None = None):
        self.bk = backend or NumpyBackend()

    @abstractmethod
    def _tangents(self, points, durations) -> list[Any]:
        """Return one velocity per waypoint (subclass-specific)."""
        ...

    def generate(self, waypoints, durations):
        """Build the C1 segments from waypoints and hop durations.

        Args:
            waypoints: Sequence of n+1 positions (each shape ``(d,)``),
                ``waypoints[0]`` the start.
            durations: n hop durations (s); ``durations[i]`` is the time from
                ``waypoints[i]`` to ``waypoints[i+1]``.

        Raises:
            ValueError: On fewer than 2 waypoints, ragged dimensions, a
                duration count that disagrees with the waypoint count, or a
                non-positive duration.
        """
        name = type(self).__name__
        if len(waypoints) < 2:
            raise ValueError(f"{name}: needs at least 2 waypoints (start + 1)")
        dims = {len(p) for p in waypoints}
        if len(dims) != 1:
            raise ValueError(f"{name}: all waypoints must have the same dimension, got {sorted(dims)}")
        n = len(durations)
        if n != len(waypoints) - 1:
            raise ValueError(
                f"{name}: expected {len(waypoints) - 1} hop durations for "
                f"{len(waypoints)} waypoints, got {n}"
            )
        if any(d <= 0 for d in durations):
            raise ValueError(f"{name}: every hop duration must be positive")

        points = [self.bk.array(p) for p in waypoints]
        tangents = self._tangents(points, durations)

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
        """Create a waypoint schedule from a TOML config dict.

        Args:
            config: TOML config dict (or the class's ``Config`` instance).
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, d) with position waypoints, or — when
            ``derivatives`` is true — a dict of ``(total_steps, d)`` arrays
            under ``"position"``, ``"velocity"``, and ``"acceleration"``.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        label = getattr(cls, "_registry_name", cls.__name__)
        wps = strict_from_list(WaypointEntry, cfg.waypoints, f"{label}.waypoint")
        waypoints = [cfg.start] + [wp.position for wp in wps]
        durations = [wp.duration for wp in wps]
        traj = cls(backend=bk)
        traj.generate(waypoints, durations)
        sampled = sample_schedule(traj, cfg.dt, order=2 if cfg.derivatives else 0)
        return sampled if cfg.derivatives else sampled["position"]


@register_trajectory("akima")
class Akima(_WaypointSpline):
    r"""Akima spline — overshoot-resistant C1 waypoint interpolation.

    The tangent at interior waypoint :math:`i` is Akima's local four-slope
    estimate over the neighbouring interval slopes :math:`s_{i-2..i+1}`:

    .. math::

        \dot{P}_i = \frac{|s_{i+1} - s_i|\, s_{i-1}
            + |s_{i-1} - s_{i-2}|\, s_i}{|s_{i+1} - s_i| + |s_{i-1} - s_{i-2}|}

    (the average of :math:`s_{i-1}` and :math:`s_i` where the denominator
    vanishes, applied elementwise). End tangents use linearly extrapolated end
    slopes, so the curve is defined by a **local** stencil and — unlike
    :class:`~shinro.trajectories.CatmullRom` — is designed not to overshoot
    steep changes. Each hop is a cubic Hermite segment
    (:class:`~shinro.trajectories.CubicPolynomial`), giving C1 continuity.

    Config is ``start`` + ``[[waypoints]]{duration, position}``, like
    ``catmull_rom``.

    Args:
        backend: Array backend. Defaults to NumpyBackend.
    """

    Config = AkimaConfig

    def _tangents(self, points, durations):
        bk = self.bk
        n = len(durations)
        if n == 1:
            slope = (points[1] - points[0]) / durations[0]
            return [slope, slope]

        s = [(points[i + 1] - points[i]) / durations[i] for i in range(n)]
        s_m1 = 2.0 * s[0] - s[1]
        s_m2 = 2.0 * s_m1 - s[0]
        s_n = 2.0 * s[n - 1] - s[n - 2]
        s_n1 = 2.0 * s_n - s[n - 1]
        ext = [s_m2, s_m1, *s, s_n, s_n1]  # ext[j] == s_{j-2}

        # ext[j] == s_{j-2}, so the window at waypoint i is the four slopes
        # s_{i-2}, s_{i-1}, s_i, s_{i+1} that Akima's rule consumes.
        tangents = []
        for i in range(n + 1):
            s_im2, s_im1, s_i, s_ip1 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
            w_right = bk.abs(s_ip1 - s_i)  # |s_{i+1} - s_i|
            w_left = bk.abs(s_im1 - s_im2)  # |s_{i-1} - s_{i-2}|
            weight = w_right + w_left
            safe = bk.where(weight > 0.0, weight, 1.0)
            # Akima: curvature-weighted average of the adjacent secants s_{i-1}, s_i
            tangents.append(
                bk.where(weight > 0.0, (w_right * s_im1 + w_left * s_i) / safe, (s_im1 + s_i) / 2.0)
            )
        return tangents


@register_trajectory("cubic_spline")
class CubicSpline(_WaypointSpline):
    r"""Natural cubic spline — C2 waypoint interpolation.

    The interior second derivatives :math:`M_i = S''(P_i)` solve the
    tridiagonal system

    .. math::

        h_{i-1} M_{i-1} + 2(h_{i-1} + h_i) M_i + h_i M_{i+1}
            = 6\left(\frac{P_{i+1} - P_i}{h_i} - \frac{P_i - P_{i-1}}{h_{i-1}}\right)

    with the **natural** end conditions :math:`M_0 = M_n = 0` (zero boundary
    acceleration). The waypoint velocities are then

    .. math::

        \dot{P}_i = \frac{P_{i+1} - P_i}{h_i} - \frac{h_i (2 M_i + M_{i+1})}{6}

    and each hop is a cubic Hermite segment
    (:class:`~shinro.trajectories.CubicPolynomial`), which reproduces the
    spline segment exactly — so the concatenated curve is C2. Unlike Akima it
    is **global** (one ``bk.solve``) and can overshoot steep changes.

    Config is ``start`` + ``[[waypoints]]{duration, position}``, like
    ``catmull_rom``.

    Args:
        backend: Array backend. Defaults to NumpyBackend.
    """

    Config = CubicSplineConfig

    def _tangents(self, points, durations):
        bk = self.bk
        n = len(durations)
        if n == 1:
            slope = (points[1] - points[0]) / durations[0]
            return [slope, slope]

        rows = []
        rhs = []
        for i in range(1, n):
            row = [0.0] * (n - 1)
            row[i - 1] = 2.0 * (durations[i - 1] + durations[i])
            if i - 1 >= 1:
                row[i - 2] = durations[i - 1]
            if i + 1 <= n - 1:
                row[i] = durations[i]
            rows.append(row)
            rhs.append(
                6.0
                * (
                    (points[i + 1] - points[i]) / durations[i]
                    - (points[i] - points[i - 1]) / durations[i - 1]
                )
            )
        interior = bk.solve(bk.array(rows), rhs)  # (n-1, d): M_1 .. M_{n-1}
        zero = bk.zeros_like(points[0])
        m = [zero] + [interior[i] for i in range(n - 1)] + [zero]

        tangents = []
        for k in range(n + 1):
            if k == 0:
                slope = (points[1] - points[0]) / durations[0]
                tangents.append(slope - durations[0] * (2.0 * m[0] + m[1]) / 6.0)
            elif k == n:
                slope = (points[n] - points[n - 1]) / durations[n - 1]
                tangents.append(slope + durations[n - 1] * (m[n - 1] + 2.0 * m[n]) / 6.0)
            else:
                slope = (points[k + 1] - points[k]) / durations[k]
                tangents.append(slope - durations[k] * (2.0 * m[k] + m[k + 1]) / 6.0)
        return tangents
