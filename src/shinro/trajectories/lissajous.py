"""Lissajous (harmonic) trajectory generator."""

import math
from dataclasses import dataclass

from shinro.components import TrajectoryGenerator
from shinro.factories.registry import register_trajectory
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


@dataclass(frozen=True)
class LissajousConfig:
    """Strict TOML schema for the ``lissajous`` trajectory.

    Fields:
        dt: Sampling period (s) of the emitted waypoint schedule.
        start: One extremum of the oscillation, :math:`p_0 + A` (one entry
            per axis).
        end: The opposite extremum, :math:`p_0 - A` (one entry per axis).
        duration: Trajectory duration :math:`T` (s).
        k: Per-axis harmonic indices. Axis ``i`` oscillates at
            :math:`\\omega_i = (2 k_i + 1) \\pi / T`.
        R: Optional ``N x N`` rotation matrix orienting the oscillation frame
            relative to the world frame. Defaults to the identity.
        name: Registered trajectory name (validated against the ``type`` key).
    """

    dt: float
    start: list[float]
    end: list[float]
    duration: float
    k: list[float]
    R: list[list[float]] | None = None
    name: str = "lissajous"


@register_trajectory("lissajous")
class Lissajous(TrajectoryGenerator):
    r"""Lissajous trajectory generator.

    Drives every axis through a cosine at an odd-harmonic frequency, giving a
    smooth zero-velocity rest-to-rest motion that interpolates ``start`` and
    ``end`` while tracing a Lissajous figure around their midpoint:

    .. math::

        p(t) = p_0 + R\,\bigl(A_\text{local} \odot \cos(\omega t)\bigr),
        \qquad
        \omega_i = \frac{(2 k_i + 1)\pi}{T}

    with center :math:`p_0 = (p_f + p_0^\text{start}) / 2`, world amplitude
    :math:`A = (\text{start} - \text{end}) / 2`, and local amplitude
    :math:`A_\text{local} = R^\top A`.

    The per-axis frequencies act on the **local** frame, so ``R`` (a rotation
    matrix, world-from-local) orients the figure — mixing axes — while
    :math:`p(0) = p_0 + A = \text{start}` and :math:`p(T) = p_0 - A =
    \text{end}` still hold because :math:`R R^\top = I`. The odd harmonics make
    both boundary velocities zero:

    .. math::

        \dot{p}(t) &= -R\,\bigl(A_\text{local} \odot \omega \odot \sin(\omega t)\bigr), \\
        \ddot{p}(t) &= -R\,\bigl(A_\text{local} \odot \omega^2 \odot \cos(\omega t)\bigr).

    Supports arbitrary N-dimensional positions: ``k`` has one entry per axis
    and ``R`` is ``N x N`` (or omitted for the identity).

    Args:
        k: Per-axis harmonic indices.
        R: Optional ``N x N`` rotation matrix orienting the oscillation frame.
            Defaults to the identity.
        backend: Array backend. Defaults to NumpyBackend.
    """

    Config = LissajousConfig

    def __init__(self, k, R=None, backend: ArrayBackend | None = None):
        self.bk = backend or NumpyBackend()
        self.k = k
        self.R = self.bk.array(R) if R is not None else None

    def generate(self, start_position, end_position, duration: float):
        """Compute the center, amplitude, and per-axis frequencies.

        Args:
            start_position: One extremum of the oscillation, :math:`p_0 + A`
                (N,).
            end_position: The opposite extremum, :math:`p_0 - A` (N,).
            duration: Trajectory duration :math:`T` (s).

        Raises:
            ValueError: If ``k`` does not have one entry per axis, or ``R`` is
                not an ``N x N`` rotation matrix.
        """
        self.p0 = (start_position + end_position) / 2.0
        self.A = (start_position - end_position) / 2.0
        self.T = duration

        n = len(self.p0)
        if len(self.k) != n:
            raise ValueError(
                f"Lissajous: k has {len(self.k)} entries but the positions have {n} axes"
            )
        R = self.bk.eye(n) if self.R is None else self.R
        if tuple(R.shape) != (n, n):
            raise ValueError(f"Lissajous: R must be ({n}, {n}), got {tuple(R.shape)}")
        if not self.bk.allclose(R.T @ R, self.bk.eye(n)):
            raise ValueError("Lissajous: R must be a rotation matrix (orthogonal)")
        self.R = R

        self.A_local = R.T @ self.A
        self.omegas = (2.0 * self.bk.array(self.k) + 1.0) * math.pi / self.T

    def position_at(self, t: float):
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds (clipped to :math:`[0, T]`).

        Returns:
            Tuple of (position, velocity, acceleration) arrays, each of shape
            ``(N,)``.
        """
        t = self.bk.clip(t, 0, self.T)
        wt = self.omegas * t
        cos_wt = self.bk.cos(wt)
        sin_wt = self.bk.sin(wt)
        pos = self.p0 + self.R @ (self.A_local * cos_wt)
        vel = -(self.R @ (self.A_local * self.omegas * sin_wt))
        acc = -(self.R @ (self.A_local * self.omegas**2 * cos_wt))
        return pos, vel, acc

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a waypoint schedule from a TOML config dict or :class:`LissajousConfig`.

        Config fields:
            dt: Sampling period (s).
            start: One extremum of the oscillation (list of floats).
            end: The opposite extremum (list of floats).
            duration: Trajectory duration (s).
            k: Per-axis harmonic indices.
            R: Optional rotation matrix (nested list) orienting the figure.
                Defaults to the identity.

        Args:
            config: TOML config dict or LissajousConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, N) with position waypoints.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        traj = cls(cfg.k, R=cfg.R, backend=bk)
        traj.generate(bk.array(cfg.start), bk.array(cfg.end), cfg.duration)
        n_steps = round(cfg.duration / cfg.dt)
        schedule = [traj.position_at(step * cfg.dt)[0] for step in range(n_steps)]
        return bk.array(schedule)
