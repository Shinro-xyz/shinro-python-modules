"""Sliding Mode Controller — robust nonlinear control with chattering suppression.

Computes the control law:

.. math::

    u = (c^T g(x))^{-1} \\left( -c^T f(x) - k_1 |s|^\\alpha \\, \\text{smooth}(s) - k_2 s \\right)

where :math:`s = c^T x` is the sliding surface. Supports multiple boundary-layer
smoothers (sat, tanh, sigmoid) to suppress chattering.

Usage:
    # In configs/controllers/smc.toml:
    #   type = "SMC"
    #   c = [1.0, 2.0]
    #   k1 = 10.0
    #   phi = 0.1
    #   k2 = 1.0
    #   smoother = "tanh"
    #   alpha = 0.5
"""

import numpy as np

from components import Controller
from factories.registry import register_controller
from utils.array_backend import ArrayBackend, NumpyBackend


@register_controller("SMC")
class SlidingModeController(Controller):
    """Sliding Mode Controller for nonlinear systems.

    Implements the equivalent control approach with a switching term and
    optional boundary-layer smoothing. The sliding surface coefficients
    ``c`` must form a Hurwitz polynomial.

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
        backend: ArrayBackend | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.c = self.bk.array(c).flatten()
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.phi = float(phi)
        self.alpha = float(alpha)

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
        vector input, solves the least-squares problem.

        Args:
            x: Current state vector (n,).
            f_x: Drift dynamics :math:`f(x)` evaluated at x (n,).
            g_x: Control matrix :math:`g(x)` evaluated at x (n, n_u).

        Returns:
            Control input vector (n_u,).

        Raises:
            RuntimeError: If :math:`c^T g(x)` is near-zero for scalar input.
        """
        x = self.bk.array(x).flatten()
        f_x = self.bk.array(f_x).flatten()
        g_x = self.bk.array(g_x)

        s = self.c @ x
        cf = self.c @ f_x
        cg = self.c @ g_x

        if self.phi > 0:
            smooth_s = self._smoother(s)
        else:
            smooth_s = self.bk.sign(s)

        s_dot_desired = -self.k1 * self.bk.abs(s) ** self.alpha * smooth_s - self.k2 * s

        cg_flat = self.bk.ravel(cg)
        cg_size = self.bk.to_numpy(cg_flat).size
        if cg_size == 1:
            cg_val = float(self.bk.to_numpy(cg_flat)[0])
            if abs(cg_val) < 1e-12:
                raise RuntimeError("c^T g(x) is near-zero — loss of controllability")
            u = self.bk.array([(s_dot_desired - cf) / cg_val])
        else:
            cg_np = self.bk.to_numpy(cg_flat).reshape(1, -1)
            rhs_np = np.array([float(self.bk.to_numpy(s_dot_desired - cf))])
            u_np, _, _, _ = np.linalg.lstsq(cg_np, rhs_np, rcond=None)
            u = self.bk.from_numpy(u_np.flatten())

        return u

    def reset(self):
        """No internal state to reset for SMC."""

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an SMC controller from a TOML config dict.

        Config fields:
            c: List of sliding surface coefficients (n,).
            k1: Discontinuous (switching) gain.
            phi: Boundary layer thickness (default 0.0).
            k2: Linear gain on sliding variable (default 0.0).
            smoother: Smoothing function — ``"sat"``, ``"tanh"``, or
                ``"sigmoid"`` (default ``"sat"``).
            alpha: Fractional power exponent (default 0.0).

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            SlidingModeController instance.
        """
        bk = backend or NumpyBackend()
        return cls(
            c=bk.array(config["c"]),
            k1=config["k1"],
            phi=config.get("phi", 0.0),
            k2=config.get("k2", 0.0),
            smoother=config.get("smoother", "sat"),
            alpha=config.get("alpha", 0.0),
            backend=bk,
        )
