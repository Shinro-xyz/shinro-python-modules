"""Sliding Mode Controller — robust nonlinear control with chattering suppression.

Computes the control law:

.. math::

    u = (c^T g(x))^{-1} \\left( -c^T f(x) - k_1 |s|^\\alpha \\, \\text{smooth}(s) - k_2 s \\right)

where :math:`s = c^T x` is the sliding surface. Supports multiple boundary-layer
smoothers (sat, tanh, sigmoid) to suppress chattering.

Scope — a single sliding surface, for every plant shape:
    ``c`` is one row, so ``s = c^T x`` is a scalar and the law controls one
    error combination. This covers the whole ``(n_x, n_u)`` rectangle:
    ``n_u == 1`` divides directly (:meth:`SlidingModeController._scalar_control`),
    ``n_u > 1`` takes the minimum-norm closed form of the underdetermined
    system ``(c^T g) u = num`` (:meth:`SlidingModeController._vector_control`),
    and both behave the same when evaluated and when lowered to a compiled
    kernel. **Multi-surface SMC is deliberately out of scope** — a surface
    *matrix* ``C`` (``m > 1`` surfaces solved jointly) is not implemented.
    Field practice is single-surface, or decentralized (one instance per axis,
    which this class already supports by instantiation), and for coupled MIMO
    plants the min-norm solve would not respect actuator limits anyway (that is
    an allocation problem, not a control-law one). The gate for revisiting it
    is the off-diagonal/diagonal ratio of ``C·g(x)`` at the operating point;
    see the 2026-09-14 lab note for the full decision record.

Usage:
    # In samples/controllers/smc.toml:
    #   type = "SMC"
    #   c = [1.0, 2.0]
    #   k1 = 10.0
    #   phi = 0.1
    #   k2 = 1.0
    #   smoother = "tanh"
    #   alpha = 0.5
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from shinro.components import Controller
from shinro.factories.registry import register_controller
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


def _as_float(value: Any, field: str) -> float:
    """Coerce a config scalar to float, naming the field when it fails.

    Config values arrive from TOML (or hand-written dicts), so a typo is a
    user error worth naming: a bare ``float("abc")`` reports only "could not
    convert string to float", not which field or config was wrong.

    Args:
        value: The raw config value.
        field: The config field name, for the error message.

    Returns:
        The value as a ``float``.

    Raises:
        ValueError: If the value is not numeric.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SMCConfig.{field} must be a number, got {value!r}") from exc


@dataclass(frozen=True)
class SMCConfig:
    """Strict TOML schema for :class:`SlidingModeController`.

    ``c`` must form a Hurwitz polynomial (validated at construction).
    ``dt`` is accepted for TOML compatibility but unused by the control law.
    """

    c: list[float]
    k1: float
    phi: float = 0.0
    k2: float = 0.0
    smoother: str = "sat"
    alpha: float = 0.0
    controllability_eps: float = 1e-12
    dt: float | None = None
    name: str = "smc"


@register_controller("SMC")
class SlidingModeController(Controller):
    """Sliding Mode Controller for nonlinear systems.

    Implements the equivalent control approach with a switching term and
    optional boundary-layer smoothing. The sliding surface coefficients
    ``c`` must form a Hurwitz polynomial.

    One surface per instance: ``c`` is a single surface vector, so the class is
    instantiated once per controlled axis (a bank of instances is the
    multi-axis pattern). Multi-surface SMC — a surface matrix solved jointly —
    is intentionally not implemented; see the module docstring's "Scope" note.

    Args:
        c: Sliding surface coefficients (n,). The polynomial
            ``c[0] + c[1] p + ... + c[n-1] p^{n-1}`` must be Hurwitz.
        k1: Discontinuous (switching) gain — drives the state to the surface.
        phi: Boundary layer thickness for chattering suppression. If 0,
            uses sign (pure switching).
        k2: Linear (proportional) gain on the sliding variable.
        smoother: Boundary-layer smoothing function. One of ``"sat"``,
            ``"tanh"``, or ``"sigmoid"``.
        alpha: Fractional power exponent for the switching term
            :math:`|s|^\\alpha`. 0 gives sign-only; 1 gives linear.
        controllability_eps: Near-zero threshold on the controllability
            denominator: :math:`|c^T g(x)|` for a scalar input, and its
            multi-input generalization :math:`\\|c^T g(x)\\|` (the two agree
            when ``n_u == 1``). **A deployment design parameter, not a
            numerical constant** — set it above the smallest value the plant
            can legitimately produce (the law amplifies :math:`1/c^T g`, so
            command saturation and chattering arrive long before the
            arithmetic floor). The 1e-12 default only protects the division
            itself, for plants whose ``g`` is well-conditioned everywhere.
            Live numpy calls raise ``RuntimeError`` below it; the lowered
            (compiled) graph instead emits a fail-safe zero command and
            reports a ``healthy`` flag, since a straight-line kernel cannot
            raise.
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        c,
        k1: float,
        phi: float = 0.0,
        k2: float = 0.0,
        smoother: str = "sat",
        alpha: float = 0.0,
        controllability_eps: float = 1e-12,
        backend: ArrayBackend | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.c = self.bk.array(c).flatten()
        self.k1 = _as_float(k1, "k1")
        self.k2 = _as_float(k2, "k2")
        self.phi = _as_float(phi, "phi")
        self.alpha = _as_float(alpha, "alpha")
        self.controllability_eps = _as_float(controllability_eps, "controllability_eps")

        SMOOTHERS = self._dict_boundaries()

        if smoother not in SMOOTHERS:
            raise ValueError(f"Unknown smoother '{smoother}'. Options: {list(SMOOTHERS)}")

        if not self._is_hurwitz():
            raise ValueError(
                "Sliding surface coefficients are not Hurwitz. "
                "The polynomial c1 + c2 p + ... + cn p^{n-1} must have "
                "all roots with negative real parts."
            )

        self._smoother = SMOOTHERS[smoother]
        self._smoother_name = smoother

    @property
    def n(self) -> int:
        """Number of sliding surface coefficients (state dimension)."""
        return len(self.c)

    def _sat(self, s):
        """Saturation boundary layer — clips s/phi to [-1, 1]."""
        return self.bk.clip(s / self.phi, -1.0, 1.0)

    def _tanh(self, s):
        """Hyperbolic tangent boundary layer."""
        return self.bk.tanh(s / self.phi)

    def _sigmoid(self, s):
        """Sigmoid-like boundary layer: s / (|s| + phi)."""
        return s / (self.bk.abs(s) + self.phi)

    def _dict_boundaries(self):
        """Map smoother name to its implementation."""
        return {
            "sat": self._sat,
            "tanh": self._tanh,
            "sigmoid": self._sigmoid,
        }

    def _is_hurwitz(self):
        """Check that the sliding surface polynomial has all roots with negative real parts."""
        c_np = self.bk.to_numpy(self.c)
        poly = c_np[::-1]
        roots = np.roots(poly)
        return all(np.real(r) < 0 for r in roots)

    def compute(self, x, f_x, g_x):
        """Compute the sliding mode control action.

        Evaluates :math:`u = (c^T g)^{-1} ( -c^T f - k_1 |s|^\\alpha \\, \\text{smooth}(s) - k_2 s )`.

        For scalar input (``c^T g`` is scalar), uses direct division. For
        vector input the equivalent-control equation ``(c^T g) u = num`` is
        underdetermined (one surface, ``n_u`` unknowns), so the law takes its
        minimum-norm solution ``u = cg^T (cg cg^T)^{-1} num`` — the same
        point ``np.linalg.lstsq`` returned, but written as matmul/transpose/
        divide so it traces and lowers.

        The computation is written so it traces as well as it evaluates: the
        ``u``/``n_u`` branch keys off ``g_x``'s *shape* (known at trace time),
        the dot products use column-vector form (the tracer has no 1D @ 1D
        contraction), and the near-zero ``c^T g`` guard is an exception only
        when ``g_x`` is concrete — a graph cannot raise, so the traced path
        emits a fail-safe instead (see :meth:`_scalar_control`).

        Args:
            x: Current state vector (n,).
            f_x: Drift dynamics :math:`f(x)` evaluated at x (n,).
            g_x: Control matrix :math:`g(x)` evaluated at x (n, n_u).

        Returns:
            Control input vector (n_u,).

        Raises:
            RuntimeError: If :math:`c^T g(x)` is near-zero and ``g_x`` is
                concrete (eager backends only; the traced path cannot raise).
        """
        x = self.bk.ravel(self.bk.array(x))
        f_x = self.bk.ravel(self.bk.array(f_x))
        g_x = self.bk.array(g_x)
        if len(g_x.shape) == 1:
            # 1-D g_x is a single control column; normalize to (n, 1).
            g_x = self.bk.reshape(g_x, (g_x.shape[0], 1))

        n = self.n
        # Column-vector contractions: (n,) @ (n, 1) -> (1,) and
        # (n,) @ (n, n_u) -> (n_u,). numpy would collapse 1D @ 1D to a 0-d
        # scalar, which the tracer rejects (no 0-d propagation) — same numbers,
        # a uniform (1,)/(n_u,) shape on every backend.
        s = self.c @ self.bk.reshape(x, (n, 1))
        cf = self.c @ self.bk.reshape(f_x, (n, 1))
        cg = self.bk.ravel(self.c @ g_x)

        if self.phi > 0:
            smooth_s = self._smoother(s)
        else:
            smooth_s = self.bk.sign(s)

        s_dot_desired = -self.k1 * self.bk.abs(s) ** self.alpha * smooth_s - self.k2 * s
        num = s_dot_desired - cf

        if len(g_x.shape) >= 2 and g_x.shape[-1] > 1:
            return self._vector_control(num, cg)
        return self._scalar_control(num, cg)

    def _scalar_control(self, num, cg):
        """Single-input law: ``u = (s_dot_desired - c^T f) / (c^T g)``.

        The guard behaves differently per backend by design:

        - **Concrete (numpy/torch)**: ``c^T g`` below ``controllability_eps``
          raises ``RuntimeError`` — the developer-facing signal.
        - **Traced**: a graph cannot raise (the compiled VM is straight-line;
          a Zig panic crossing the C ABI aborts the host process). Instead the
          guard becomes data: ``u`` is forced to the fail-safe zero command
          and a ``healthy`` flag is published as an auxiliary output port
          (:meth:`ArrayBackend.emit_named_output`), so the host can run its own
          fault policy inside the same tick. Zero-command is the floor, not a
          safety guarantee — for an open-loop-unstable plant the host policy is
          the real fail-safe.

        Args:
            num: ``s_dot_desired - c^T f`` (1,).
            cg: ``c^T g`` (1,).

        Returns:
            Control input (1,).

        Raises:
            RuntimeError: If ``|c^T g| < controllability_eps`` and ``cg`` is
                concrete.
        """
        concrete = self.bk.to_numpy(cg)
        if isinstance(concrete, np.ndarray):
            cg_val = np.asarray(concrete).reshape(-1)[0]
            if abs(cg_val) < self.controllability_eps:
                raise RuntimeError("c^T g(x) is near-zero — loss of controllability")
            return num / cg

        # Traced: the guard is data, not an exception (see the class docstring
        # above). u_raw may compute inf/nan here; `where` discards it.
        u_raw = num / cg
        fault = self.bk.abs(cg) < self.controllability_eps
        self.bk.emit_named_output("healthy", 1.0 - fault)
        return self.bk.where(fault, self.bk.zeros_like(u_raw), u_raw)

    def _vector_control(self, num, cg):
        """Multi-input law: minimum-norm solution of ``(c^T g) u = num``.

        ``c^T g`` is a ``1 x n_u`` row, so the equation is underdetermined and
        ``(c^T g)^{-1}`` does not exist. Writing ``A = c^T g``, the
        minimum-norm exact solution is the pseudo-inverse ``A^+ b``, which for
        a single full-rank row collapses to ``A^T (A A^T)^{-1} b`` — the same
        answer ``np.linalg.lstsq`` gives, but built from matmul/transpose/
        divide so both the eager backends and the traced/lowered graph share
        one implementation.

        The guard mirrors :meth:`_scalar_control` (see that docstring for why
        the concrete and traced backends diverge): it tests ``‖c^T g‖``
        rather than its square, so ``controllability_eps`` means the same
        magnitude here as ``|c^T g|`` does on the scalar branch (``n_u == 1``
        makes them identical).

        Args:
            num: ``s_dot_desired - c^T f`` (1,).
            cg: ``c^T g`` (n_u,).

        Returns:
            Control input (n_u,).

        Raises:
            RuntimeError: If ``‖c^T g‖ < controllability_eps`` and ``cg``
                is concrete.
        """
        n_u = cg.shape[0]
        # Make the row explicit so A^T (A A^T)^-1 is expressible with matmul
        # and transpose alone (the tracer has no 1-D pseudo-inverse op).
        cg_row = self.bk.reshape(cg, (1, n_u))
        denom = cg_row @ cg_row.T  # (1,1) — ||cg||^2, the Gram scalar
        norm = denom**0.5  # (1,1) — ||cg||; equals |c^T g| when n_u == 1
        num_col = self.bk.reshape(num, (1, 1))  # (1,) -> (1,1), rank-aligned

        concrete = self.bk.to_numpy(norm)
        if isinstance(concrete, np.ndarray):
            # Guard before dividing: on the invalid input the division would
            # only produce inf/nan (and a numpy divide warning) before the raise.
            norm_val = np.asarray(concrete).reshape(-1)[0]
            if norm_val < self.controllability_eps:
                raise RuntimeError("c^T g(x) is near-zero — loss of controllability")
            return self.bk.ravel(cg_row.T @ (num_col / denom))

        # Traced: the guard is data, not an exception (see _scalar_control).
        # Apply it to the (1,1) scalar factor rather than to u: zeroing scale
        # zeroes u = cg^T scale, and `where`'s operands stay the same shape.
        scale = num_col / denom  # (1,1) — the A^T b factor; may be inf/nan
        fault = norm < self.controllability_eps
        self.bk.emit_named_output("healthy", 1.0 - self.bk.ravel(fault))
        safe = self.bk.where(fault, self.bk.zeros_like(scale), scale)
        return self.bk.ravel(cg_row.T @ safe)

    def reset(self):
        """No internal state to reset for SMC."""

    Config = SMCConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an SMC controller from a TOML config dict or :class:`SMCConfig`.

        Config fields:
            c: List of sliding surface coefficients (n,).
            k1: Discontinuous (switching) gain.
            phi: Boundary layer thickness (default 0.0).
            k2: Linear gain on sliding variable (default 0.0).
            smoother: Smoothing function — ``"sat"``, ``"tanh"``, or
                ``"sigmoid"`` (default ``"sat"``).
            alpha: Fractional power exponent (default 0.0).
            controllability_eps: Near-zero :math:`|c^T g|` threshold,
                plant-scaled (default 1e-12 = arithmetic-only guard); see
                :class:`SlidingModeController` for how to choose it.

        Args:
            config: TOML config dict or SMCConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            SlidingModeController instance.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        return cls(
            c=bk.array(cfg.c),
            k1=cfg.k1,
            phi=cfg.phi,
            k2=cfg.k2,
            smoother=cfg.smoother,
            alpha=cfg.alpha,
            controllability_eps=cfg.controllability_eps,
            backend=bk,
        )
