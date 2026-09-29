"""B-spline trajectory generator (Cox–de Boor recursion)."""

from collections.abc import Sequence
from dataclasses import dataclass

from shinro.components import TrajectoryGenerator
from shinro.factories.registry import register_trajectory
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


@dataclass(frozen=True)
class BSplineConfig:
    """Strict TOML schema for the ``bspline`` trajectory.

    Fields:
        dt: Sampling period (s) of the emitted waypoint schedule.
        duration: Trajectory duration :math:`T` (s).
        degree: Polynomial degree :math:`p` of the curve.
        control_points: Ordered control points; each is a list of floats of
            the same length (the spatial dimension).
        knots: Knot vector. Must be nondecreasing and hold exactly
            ``len(control_points) + degree + 1`` knots. A clamped vector
            (first/last knot repeated ``degree + 1`` times) interpolates
            ``control_points[0]`` and ``control_points[-1]`` exactly.
        name: Registered trajectory name (validated against the ``type`` key).
    """

    dt: float
    duration: float
    degree: int
    control_points: list[list[float]]
    knots: list[float]
    name: str = "bspline"


@register_trajectory("bspline")
class BSpline(TrajectoryGenerator):
    r"""B-spline trajectory over a knot vector and control polygon.

    .. math::

        C(t) = \sum_{i=0}^{n} N_{i,p}(t)\, P_i, \qquad n = m - p - 1

    with :math:`N_{i,p}` the Cox–de Boor recursion over the knot vector
    :math:`\{u_i\}` and :math:`m =` len(knots) :math:`= n + p + 2`.

    Derivatives reuse the same recursion by degree reduction: the :math:`k`-th
    derivative curve is the B-spline of degree :math:`p - k` over the
    differentiated control polygon

    .. math::

        Q_i = p\,\frac{P_{i+1} - P_i}{u_{i+p+1} - u_{i+1}}

    with the first and last knot dropped (applied :math:`k` times,
    :meth:`_bspline_derv`). ``generate`` precomputes the first and second
    derivative curves; :meth:`position_at` returns ``(pos, vel, acc)`` like
    :class:`~shinro.trajectories.BezierCurve`.

    Args:
        degree_polynomial: Polynomial degree :math:`p` of the curve.
        control_points: Ordered control points, each shape ``(d,)``.
        time_knot_vector: Knot vector, exactly ``len(control_points) +
            degree_polynomial + 1`` nondecreasing knots.
        backend: Array backend. Defaults to NumpyBackend.
    """

    Config = BSplineConfig

    def __init__(
        self,
        degree_polynomial: int,
        control_points: Sequence[Sequence[float]],
        time_knot_vector: Sequence[float],
        backend: ArrayBackend | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.P = control_points
        self.knots = time_knot_vector
        self.p = degree_polynomial
        self.T: float | None = None

    # cox de boor recursion definition
    def _bspline_basis(self, i: int, t: float, p: int, knots=None):
        # i-> basis index,
        # t-> evaluation point,
        # p-> polynomial degree
        # knots-> knot vector (defaults to self.knots; derivative curves pass
        # their reduced vector)
        if knots is None:
            knots = self.knots
        if p == 0:
            if knots[i] <= t < knots[i + 1]:
                return 1.0

            if t == knots[-1] and knots[i] <= t <= knots[i + 1]:
                return 1.0

            return 0.0

        left = 0.0
        denom = knots[i + p] - knots[i]
        if denom > 0:
            left = (t - knots[i]) / denom * self._bspline_basis(i, t, p - 1, knots)

        right = 0.0
        denom = knots[i + p + 1] - knots[i + 1]
        if denom > 0:
            right = (knots[i + p + 1] - t) / denom * self._bspline_basis(i + 1, t, p - 1, knots)

        return left + right

    # get all the basis functions
    def _all_basis(self, t: float, p: int, knots=None):
        if knots is None:
            knots = self.knots
        n = len(knots) - p - 2
        return self.bk.array([self._bspline_basis(i, t, p, knots) for i in range(n + 1)])

    def _bspline_derv(self, knots, k: int, control_points, p: int, P):
        """k-th derivative curve, as a B-spline of degree ``p - k``.

        Cox–de Boor degree reduction: one differentiation step maps a
        degree-``p`` curve to a degree-``p - 1`` curve on the scaled control
        polygon differences :math:`Q_i = p (P_{i+1} - P_i) / (u_{i+p+1} -
        u_{i+1})`, with one knot chopped from each end of the vector. The
        returned ``(points, degree, knots)`` triple is directly evaluable with
        the same :meth:`_bspline_basis` / ``_all_basis`` recursion.

        Args:
            knots: Knot vector of the current (partly differentiated) curve.
            k: Remaining derivative order.
            control_points: Control polygon of the current curve (carried
                through the recursion; equals ``P`` after the first step).
            p: Current polynomial degree.
            P: Current control points (one differentiation per step).

        Returns:
            Tuple ``(points, degree, knots)`` of the derivative curve. For
            ``k > p`` the derivative is identically zero: a degree-0 curve on
            a zero control polygon.
        """
        t_start = knots[p]
        t_end = knots[-p - 1] if p > 0 else knots[-1]

        if k == 0:
            return self.bk.array(P), p, self.bk.array(knots)

        if k > p:
            # k-th derivative of a degree-p curve vanishes; keep the caller's
            # evaluation machinery working on a zero control polygon.
            pts = self.bk.array(P)
            zero_pts = self.bk.zeros_like(pts)
            # Length len(P) + degree(0) + 1, so the basis count is len(P).
            padded_knots = [t_start] * (len(P) - 1) + [t_end] * 2
            return zero_pts, 0, self.bk.array(padded_knots)

        # First derivative: degree p-1 curve over the scaled differences Q_i.
        pts = self.bk.array(P)
        Q = []
        for i in range(len(P) - 1):
            denom = knots[i + p + 1] - knots[i + 1]
            if denom > 0:
                Q.append((p / denom) * (pts[i + 1] - pts[i]))
            else:
                # Degenerate knot span: the corresponding basis is identically
                # zero, so the control point contributes nothing.
                Q.append(pts[i + 1] * 0.0)

        return self._bspline_derv(knots[1:-1], k - 1, Q, p - 1, Q)

    def _eval_curve(self, points, p: int, knots, t: float):
        """Evaluate the curve ``(points, p, knots)`` at time ``t`` (clipped to [0, T])."""
        t = self.bk.clip(t, 0, self.T)
        n = len(knots) - p - 2
        N = self.bk.array([self._bspline_basis(i, t, p, knots) for i in range(n + 1)])
        return N @ points

    def generate(self, duration: float, start_position=None, end_position=None):
        """Validate the curve data and precompute the derivative curves.

        ``start_position`` / ``end_position`` are informational only — the
        curve's ends are interpolated only by a *clamped* knot vector, so no
        agreement check is implied.

        Args:
            duration: Trajectory duration :math:`T` (s).
            start_position: Unused (see above).
            end_position: Unused (see above).

        Raises:
            ValueError: On a non-positive duration, fewer than 2 control
                points, ragged point dimensions, a negative degree, a knot
                count that disagrees with the control polygon, or a
                non-monotone knot vector.
        """
        if duration is None or duration <= 0:
            raise ValueError(f"BSpline: duration must be positive, got {duration}")
        if self.p < 0:
            raise ValueError(f"BSpline: degree must be non-negative, got {self.p}")
        if len(self.P) < 2:
            raise ValueError("BSpline: needs at least 2 control points")
        dims = {len(p) for p in self.P}
        if len(dims) != 1:
            raise ValueError(
                f"BSpline: all control points must have the same dimension, got {sorted(dims)}"
            )
        m = len(self.P)
        if m + self.p + 1 != len(self.knots):
            raise ValueError(
                f"BSpline: knot vector must hold len(control_points) + degree + 1 = "
                f"{m + self.p + 1} knots, got {len(self.knots)}"
            )
        if any(self.knots[i] > self.knots[i + 1] for i in range(len(self.knots) - 1)):
            raise ValueError("BSpline: knot vector must be nondecreasing")

        self.T = duration
        self._points = self.bk.array(self.P)
        self._knots = self.knots

        Q1, p1, k1 = self._bspline_derv(self.knots, 1, self.P, self.p, self.P)
        self._vel_curve = (self.bk.array(Q1), p1, self.bk.array(k1))
        Q2, p2, k2 = self._bspline_derv(self.knots, 2, self.P, self.p, self.P)
        self._acc_curve = (self.bk.array(Q2), p2, self.bk.array(k2))

    def position_at(self, t: float):
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds (clipped to :math:`[0, T]`).

        Returns:
            Tuple of (position, velocity, acceleration) arrays, each of shape
            ``(d,)``. Velocity and acceleration are exact zeros for curves of
            degree < 1 / < 2 respectively.
        """
        pos = self._eval_curve(self._points, self.p, self._knots, t)
        vel_points, vel_p, vel_knots = self._vel_curve
        vel = self._eval_curve(vel_points, vel_p, vel_knots, t)
        acc_points, acc_p, acc_knots = self._acc_curve
        acc = self._eval_curve(acc_points, acc_p, acc_knots, t)
        return pos, vel, acc

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a waypoint schedule from a TOML config dict or :class:`BSplineConfig`.

        Config fields:
            dt: Sampling period (s).
            duration: Trajectory duration (s).
            degree: Polynomial degree of the curve.
            control_points: Ordered control points (list of float lists).
            knots: Knot vector (list of floats).

        Args:
            config: TOML config dict or BSplineConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, d) with position waypoints.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        traj = cls(cfg.degree, cfg.control_points, cfg.knots, backend=bk)
        traj.generate(duration=cfg.duration)
        n_steps = round(cfg.duration / cfg.dt)
        schedule = [traj.position_at(step * cfg.dt)[0] for step in range(n_steps)]
        return bk.array(schedule)
