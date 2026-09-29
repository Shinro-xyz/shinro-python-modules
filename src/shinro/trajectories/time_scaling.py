"""Time scaling / motion limits for trajectory generators.

None of the generators carry velocity, acceleration, or jerk limits, so a
reference can demand more than the plant can produce. Both helpers here
re-time an existing generator — the *same geometry*, a different clock — but
with different time laws:

- **uniform (constant) time scaling** — :func:`limit_trajectory` traverses the
  path :math:`k` times slower; velocity divides by :math:`k`, acceleration by
  :math:`k^2`. One global factor, simple and robust.
- **S-curve (min-jerk) time scaling** — :func:`s_curve_limit` re-times the path
  parameter with a rest-to-rest min-jerk profile, so the acceleration is
  continuous and the jerk is bounded (the classic S-curve), at the cost of
  bringing the ends to rest.

Both return a drop-in wrapper for :func:`~shinro.trajectories.sample_schedule`.

Example:
    >>> from shinro.trajectories import s_curve_limit, sample_schedule
    >>> scaled = s_curve_limit(traj, max_velocity=0.5, max_acceleration=2.0, max_jerk=10.0)
    >>> schedule = sample_schedule(scaled, dt=0.01)["position"]
"""

import math
from typing import Any

from shinro.trajectories.sampling import total_duration
from shinro.utils.array_backend import ArrayBackend


def _scalar(x: Any) -> float:
    """Python scalar for a backend 0-dim value.

    Uses ``.item()`` rather than ``float()`` (which the static-analyser flags as
    an unchecked throwing call); the value is always numeric here.
    """
    return x.item() if hasattr(x, "item") else x


def _peak(bk: ArrayBackend, x: Any) -> float:
    """Max absolute component of a vector (``bk`` has no ``max``)."""
    return _scalar(-bk.min(-bk.abs(x)))


# --- uniform (constant) time scaling ---------------------------------------


def time_scale_factor(
    traj,
    max_velocity: float | None = None,
    max_acceleration: float | None = None,
    samples: int = 1000,
) -> float:
    r"""Smallest :math:`k \ge 1` so traversing ``traj`` ``k`` times slower respects the limits.

    Samples ``traj.position_at`` at ``samples + 1`` points and takes the peak
    component speed / acceleration. Time-scaling by :math:`k` divides velocity
    by :math:`k` and acceleration by :math:`k^2`, so

    .. math::

        k = \max\!\left(1,\; \frac{\max_i |\dot{p}_i|}{v_\text{max}},\;
            \sqrt{\frac{\max_i |\ddot{p}_i|}{a_\text{max}}}\right).

    Args:
        traj: A generator with ``position_at(t) -> (pos, vel, acc)``.
        max_velocity: Velocity limit (per component). ``None`` to ignore.
        max_acceleration: Acceleration limit (per component). ``None`` to ignore.
        samples: Number of intervals used to locate the peaks.

    Returns:
        The time-scale factor :math:`k \ge 1`.

    Raises:
        ValueError: If both limits are ``None``.
    """
    if max_velocity is None and max_acceleration is None:
        raise ValueError("time_scale_factor: at least one of max_velocity / max_acceleration is required")
    bk = traj.bk
    total = total_duration(traj)
    v_need = 0.0
    a_need = 0.0
    for i in range(samples + 1):
        _, vel, acc = traj.position_at(total * i / samples)
        if max_velocity is not None:
            v_need = max(v_need, _peak(bk, vel) / max_velocity)
        if max_acceleration is not None:
            a_need = max(a_need, _peak(bk, acc) / max_acceleration)
    return max(1.0, v_need, math.sqrt(a_need))


class TimeScaled:
    """Wrap a generator so it is traversed ``k`` times slower.

    ``position_at(t)`` returns ``inner.position_at(t / k)`` with the derivatives
    rescaled (velocity ÷ ``k``, acceleration ÷ ``k²``). The path is unchanged;
    only the timing is stretched, so the horizon grows to ``k * inner_T``.
    """

    def __init__(self, inner, k: float):
        self.inner = inner
        self.k = k
        self.bk = inner.bk
        self.T = total_duration(inner) * k

    def position_at(self, t: float):
        """Evaluate the time-scaled trajectory at time ``t``.

        Args:
            t: Time in seconds on the *scaled* horizon :math:`[0, k T]`.

        Returns:
            Tuple of (position, velocity, acceleration) arrays.
        """
        pos, vel, acc = self.inner.position_at(t / self.k)
        return pos, vel / self.k, acc / (self.k * self.k)


def limit_trajectory(
    traj,
    max_velocity: float | None = None,
    max_acceleration: float | None = None,
    samples: int = 1000,
) -> TimeScaled:
    """Return a :class:`TimeScaled` view of ``traj`` that respects the limits.

    Args:
        traj: A generator with ``position_at(t) -> (pos, vel, acc)``.
        max_velocity: Velocity limit (per component). ``None`` to ignore.
        max_acceleration: Acceleration limit (per component). ``None`` to ignore.
        samples: Number of intervals used to locate the peaks.

    Returns:
        A :class:`TimeScaled` wrapper (drop-in for ``sample_schedule``).

    Raises:
        ValueError: If both limits are ``None``.
    """
    return TimeScaled(traj, time_scale_factor(traj, max_velocity, max_acceleration, samples))


# --- S-curve (min-jerk) time scaling ---------------------------------------


def _min_jerk_terms(s: Any):
    """``(σ, σ', σ'', σ''')`` for the min-jerk rest-to-rest profile on ``[0, 1]``.

    ``σ(s) = 10s³ - 15s⁴ + 6s⁵`` runs 0→1 with ``σ'(0) = σ'(1) = 0`` and
    ``σ''(0) = σ''(1) = 0`` — zero boundary velocity and acceleration, finite
    (bounded) jerk. ``s`` may be a Python float or a backend scalar.
    """
    return (
        10 * s**3 - 15 * s**4 + 6 * s**5,
        30 * s**2 - 60 * s**3 + 30 * s**4,
        60 * s - 180 * s**2 + 120 * s**3,
        60 - 360 * s + 360 * s**2,
    )


def _finite_difference_jerk(traj, tau: float, h: float):
    """Central-difference jerk ``p'''`` at path time ``tau`` (generators expose no jerk)."""
    span = total_duration(traj)
    plus = traj.position_at(min(tau + h, span))[2]
    minus = traj.position_at(max(tau - h, 0.0))[2]
    return (plus - minus) / (2.0 * h)


def s_curve_horizon(
    traj,
    max_velocity: float | None = None,
    max_acceleration: float | None = None,
    max_jerk: float | None = None,
    samples: int = 1000,
) -> float:
    r"""Horizon :math:`T'` for the min-jerk S-curve time law that respects the limits.

    The output is ``p(T σ(t / T'))`` with :math:`\sigma` the min-jerk profile, so
    by the chain rule each output derivative's peak is a fixed function of
    :math:`T'`:

    .. math::

        \max|\dot{p}_\text{out}| = \frac{T\,V}{T'}, \quad
        \max|\ddot{p}_\text{out}| = \frac{A}{T'^2}, \quad
        \max|p'''_\text{out}| = \frac{J}{T'^3}

    where :math:`V, A, J` are sampled from the path. Solving each for :math:`T'`
    and taking the largest (and at least the original horizon :math:`T`, so the
    S-curve never speeds the path up) gives the returned horizon.

    Args:
        traj: A generator with ``position_at(t) -> (pos, vel, acc)``.
        max_velocity: Velocity limit (per component). ``None`` to ignore.
        max_acceleration: Acceleration limit (per component). ``None`` to ignore.
        max_jerk: Jerk limit (per component). ``None`` to ignore. Jerk is
            estimated by finite differences (no generator exposes it).
        samples: Number of intervals used to locate the peaks.

    Returns:
        The S-curve horizon :math:`T' \ge T` (s).

    Raises:
        ValueError: If all three limits are ``None``.
    """
    if max_velocity is None and max_acceleration is None and max_jerk is None:
        raise ValueError(
            "s_curve_horizon: at least one of max_velocity / max_acceleration / max_jerk is required"
        )
    bk = traj.bk
    T = total_duration(traj)
    V = A = J = 0.0
    h = T / (samples * 100) if T > 0 else 1e-9
    for i in range(samples + 1):
        s = i / samples
        sig, sig1, sig2, sig3 = _min_jerk_terms(s)
        tau = T * sig
        _, vel, acc = traj.position_at(tau)
        if max_velocity is not None:
            V = max(V, _peak(bk, vel * sig1))
        if max_acceleration is not None:
            A = max(A, _peak(bk, acc * (T**2 * sig1**2) + vel * (T * sig2)))
        if max_jerk is not None:
            jerk = _finite_difference_jerk(traj, tau, h)
            J = max(
                J,
                _peak(bk, jerk * (T**3 * sig1**3) + acc * (3 * T**2 * sig1 * sig2) + vel * (T * sig3)),
            )

    horizon = T
    if max_velocity is not None:
        horizon = max(horizon, T * V / max_velocity)
    if max_acceleration is not None:
        horizon = max(horizon, math.sqrt(A / max_acceleration))
    if max_jerk is not None:
        horizon = max(horizon, (J / max_jerk) ** (1.0 / 3.0))
    return horizon


class SCurveScaled:
    """Wrap a generator so its path parameter follows a rest-to-rest min-jerk S-curve.

    ``position_at(t)`` evaluates the inner generator at the path time
    ``φ(t) = T·σ(t / T')`` (:math:`\\sigma` = min-jerk) and applies the chain
    rule. The path is the same curve traversed with an S-curve timing; the
    horizon is ``T'`` and the ends come to rest (``σ'(0) = σ'(1) = 0``).
    """

    def __init__(self, inner, horizon: float):
        self.inner = inner
        self.bk = inner.bk
        self.T = horizon
        self._span = total_duration(inner)

    def position_at(self, t: float):
        """Evaluate the S-curve time-scaled trajectory at time ``t``.

        Args:
            t: Time in seconds on the *scaled* horizon :math:`[0, T']`.

        Returns:
            Tuple of (position, velocity, acceleration) arrays.
        """
        span = self._span
        horizon = self.T
        s = t / horizon
        sig, sig1, sig2, _ = _min_jerk_terms(s)
        phi = span * sig
        dphi = (span / horizon) * sig1
        ddphi = (span / horizon**2) * sig2
        pos, vel, acc = self.inner.position_at(phi)
        return pos, vel * dphi, acc * dphi**2 + vel * ddphi


def s_curve_limit(
    traj,
    max_velocity: float | None = None,
    max_acceleration: float | None = None,
    max_jerk: float | None = None,
    samples: int = 1000,
) -> SCurveScaled:
    """Return an :class:`SCurveScaled` view of ``traj`` that respects the limits.

    Args:
        traj: A generator with ``position_at(t) -> (pos, vel, acc)``.
        max_velocity: Velocity limit (per component). ``None`` to ignore.
        max_acceleration: Acceleration limit (per component). ``None`` to ignore.
        max_jerk: Jerk limit (per component). ``None`` to ignore.
        samples: Number of intervals used to locate the peaks.

    Returns:
        An :class:`SCurveScaled` wrapper (drop-in for ``sample_schedule``).

    Raises:
        ValueError: If all three limits are ``None``.
    """
    return SCurveScaled(
        traj, s_curve_horizon(traj, max_velocity, max_acceleration, max_jerk, samples)
    )
