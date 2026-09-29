"""Circular / helical arc trajectory generator."""

import math
from dataclasses import dataclass
from typing import Any

from shinro.components import TrajectoryGenerator
from shinro.factories.registry import register_trajectory
from shinro.trajectories.sampling import sample_schedule
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


@dataclass(frozen=True)
class CircularArcConfig:
    """Strict TOML schema for the ``circular_arc`` trajectory.

    Fields:
        dt: Sampling period (s) of the emitted waypoint schedule.
        center: Circle centre (3,).
        start: A point on the circle (3,); sets the radius and start angle.
            Must be perpendicular to ``normal`` about the centre.
        duration: Trajectory duration :math:`T` (s).
        sweep_angle: Signed arc angle (radians). The end point is ``start``
            rotated about ``normal`` by this angle.
        normal: Arc-plane normal (3,); defaults to ``[0, 0, 1]`` (xy-plane).
        pitch: Helix pitch — axial advance along ``normal`` per full turn
            (:math:`2\\pi` of sweep). ``0`` (default) is a planar arc.
        derivatives: Also emit the reference velocity/acceleration. When true,
            ``from_config`` returns a ``{"position", "velocity",
            "acceleration"}`` dict of ``(steps, N)`` arrays.
        name: Registered trajectory name (validated against the ``type`` key).
    """

    dt: float
    center: list[float]
    start: list[float]
    duration: float
    sweep_angle: float
    normal: list[float] | None = None
    pitch: float = 0.0
    derivatives: bool = False
    name: str = "circular_arc"


@register_trajectory("circular_arc")
class CircularArc(TrajectoryGenerator):
    r"""Circular arc (optionally helical) around a centre.

    With :math:`r = \text{start} - \text{center}`, unit normal :math:`\hat{n}`
    (:math:`r \perp \hat{n}`), angular rate :math:`\omega =
    \theta_\text{sweep}/T`, and :math:`\theta(t) = \omega t`:

    .. math::

        p(t) &= c + r\cos\theta + (\hat{n} \times r)\sin\theta
            + \hat{n}\,\frac{\text{pitch}}{2\pi}\,\theta, \\
        \dot{p}(t) &= \omega\bigl(-r\sin\theta + (\hat{n} \times r)\cos\theta
            \bigr) + \hat{n}\,\frac{\omega\,\text{pitch}}{2\pi}, \\
        \ddot{p}(t) &= -\omega^2\bigl(r\cos\theta
            + (\hat{n} \times r)\sin\theta\bigr).

    The speed is constant :math:`\omega r` (plus the axial component) and the
    acceleration points at the centre — an exact circle, no solve.

    .. math::

        \text{start} = p(0), \qquad
        p(T) = \text{start} + \hat{n}\,\frac{\text{pitch}}{2\pi}\,\theta_\text{sweep}

    Args:
        backend: Array backend. Defaults to NumpyBackend.
    """

    Config = CircularArcConfig

    def __init__(self, backend: ArrayBackend | None = None):
        self.bk = backend or NumpyBackend()

    def generate(
        self,
        center,
        start,
        duration: float,
        sweep_angle: float,
        normal=None,
        pitch: float = 0.0,
    ):
        """Resolve the circle geometry (radius, plane, angular rate).

        Args:
            center: Circle centre (3,).
            start: A point on the circle (3,), perpendicular to ``normal``
                about the centre.
            duration: Trajectory duration :math:`T` (s).
            sweep_angle: Signed arc angle (radians).
            normal: Arc-plane normal (3,). Defaults to ``[0, 0, 1]``.
            pitch: Axial advance along ``normal`` per full turn.

        Raises:
            ValueError: On a non-positive duration, a non-3D centre/start/
                normal, a zero-length normal, ``start == center``, or a
                ``start - center`` that is not perpendicular to ``normal``.
        """
        if duration is None or duration <= 0:
            raise ValueError(f"CircularArc: duration must be positive, got {duration}")
        bk = self.bk
        center = bk.array(center)
        start = bk.array(start)
        n = bk.array([0.0, 0.0, 1.0]) if normal is None else bk.array(normal)
        for label, vec in (("center", center), ("start", start), ("normal", n)):
            if tuple(vec.shape) != (3,):
                raise ValueError(f"CircularArc: {label} must be 3-D, got shape {tuple(vec.shape)}")

        n_norm = bk.norm(n)
        if bk.allclose(n_norm, bk.zeros_like(n_norm)):
            raise ValueError("CircularArc: normal must be non-zero")
        n_unit = n / n_norm

        r_vec = start - center
        radius = bk.norm(r_vec)
        if bk.allclose(radius, bk.zeros_like(radius)):
            raise ValueError("CircularArc: start must differ from center")
        # r_vec must lie in the arc plane: |r x n_hat| == |r|.
        if not bk.allclose(bk.norm(bk.cross(r_vec, n_unit)), radius):
            raise ValueError("CircularArc: start - center must be perpendicular to normal")

        self.center = center
        self.r_vec = r_vec
        self.perp_dir = bk.cross(n_unit, r_vec)  # in-plane, leads r_vec by 90 deg
        self.normal = n_unit
        self.omega = sweep_angle / duration
        self.pitch = pitch
        self.T = duration

    def position_at(self, t: float):
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Time in seconds (clipped to :math:`[0, T]`).

        Returns:
            Tuple of (position, velocity, acceleration) arrays, each shape
            ``(3,)``.
        """
        t = self.bk.clip(t, 0, self.T)
        theta = self.omega * t
        c = self.bk.cos(theta)
        s = self.bk.sin(theta)
        rot = self.r_vec * c + self.perp_dir * s
        axial = self.normal * (self.pitch * theta / (2.0 * math.pi))
        pos = self.center + rot + axial
        vel = (-self.r_vec * s + self.perp_dir * c) * self.omega + self.normal * (
            self.pitch * self.omega / (2.0 * math.pi)
        )
        acc = -rot * (self.omega**2)
        return pos, vel, acc

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None) -> Any:
        """Create a waypoint schedule from a TOML config dict or :class:`CircularArcConfig`.

        Config fields:
            dt: Sampling period (s).
            center: Circle centre (3,).
            start: A point on the circle (3,).
            duration: Trajectory duration (s).
            sweep_angle: Signed arc angle (radians).
            normal: Arc-plane normal (3,); defaults to ``[0, 0, 1]``.
            pitch: Helix pitch (axial advance per full turn).
            derivatives: Emit the velocity/acceleration dict instead of positions.

        Args:
            config: TOML config dict or CircularArcConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            Array of shape (total_steps, 3) with position waypoints, or — when
            ``derivatives`` is true — a dict of ``(total_steps, 3)`` arrays
            under ``"position"``, ``"velocity"``, and ``"acceleration"``.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        traj = cls(backend=bk)
        traj.generate(
            cfg.center,
            cfg.start,
            cfg.duration,
            cfg.sweep_angle,
            normal=cfg.normal,
            pitch=cfg.pitch,
        )
        sampled = sample_schedule(traj, cfg.dt, order=2 if cfg.derivatives else 0)
        return sampled if cfg.derivatives else sampled["position"]
