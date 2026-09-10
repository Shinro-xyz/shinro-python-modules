from dataclasses import dataclass
from typing import Any

from scipy.linalg import solve_discrete_are

from shinro.components import Controller
from shinro.factories.registry import register_controller
from shinro.utils.array_backend import ArrayBackend, NumpyBackend, parse_matrix
from shinro.utils.config_spec import strict_from_dict


@dataclass(frozen=True)
class LQRConfig:
    """Strict TOML schema for :class:`LQR`.

    ``dt`` / ``A_dynamics`` / ``B_dynamics`` are optional: scenario builds
    inject them from the plant (see :mod:`shinro.utils.linearization`);
    standalone use must supply ``B_dynamics`` or ``dt``.
    """

    state_cost: list[float] | list[list[float]]
    control_cost: list[float] | list[list[float]]
    dt: float | None = None
    A_dynamics: Any = None
    B_dynamics: Any = None
    name: str = "lqr"


@register_controller("LQR")
class LQR(Controller):
    """Linear Quadratic Regulator for discrete-time systems.

    Computes the optimal state-feedback control law :math:`u = -K (x - x_t)`
    that minimizes:

    .. math::

        J = \\sum_k \\left( x_k^T Q x_k + u_k^T R u_k \\right)

    The gain K is computed once via the Discrete Algebraic Riccati Equation
    (DARE) and applied online as a single matrix-vector multiply.

    The DARE solve uses scipy (numpy-only) since there is no equivalent in
    PyTorch. The conversion is handled transparently via ``bk.to_numpy`` /
    ``bk.from_numpy``.

    Args:
        state_cost_matrix: Q — penalizes state deviation (n_x, n_x).
        control_cost_matrix: R — penalizes control effort (n_u, n_u).
        dynamics_state_matrix: A — discrete-time state transition (n_x, n_x).
        dynamics_control_matrix: B — control input matrix (n_x, n_u).
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        state_cost_matrix,
        control_cost_matrix,
        dynamics_state_matrix,
        dynamics_control_matrix,
        backend: ArrayBackend | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.A = dynamics_state_matrix
        self.B = dynamics_control_matrix
        self.Q = state_cost_matrix
        self.R = control_cost_matrix
        self.gain_calculation()

    def gain_calculation(self):
        """Solve DARE and compute the optimal LQR gain matrix K.

        Solves :math:`P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q`
        via ``scipy.linalg.solve_discrete_are``, then computes:

        .. math::

            K = (R + B^T P B)^{-1} B^T P A

        The gain K is stored as ``self.K`` and used in ``compute()``.
        """
        A_np = self.bk.to_numpy(self.A)
        B_np = self.bk.to_numpy(self.B)
        P_np = solve_discrete_are(A_np, B_np, self.bk.to_numpy(self.Q), self.bk.to_numpy(self.R))
        P = self.bk.from_numpy(P_np)
        self.K = self.bk.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

    def compute(self, current_state, target_state: Any | None = None):
        """Compute the optimal control input :math:`u = -K (x - x_t)`.

        Args:
            current_state: Current state vector (n_x,).
            target_state: Desired state vector (n_x,). Defaults to zeros.

        Returns:
            Control input vector (n_u,).
        """
        if target_state is None:
            target_state = self.bk.zeros_like(current_state)
        error = target_state - current_state
        return self.K @ error

    def reset(self):
        """No internal state to reset for LQR."""

    Config = LQRConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an LQR controller from a TOML config dict or :class:`LQRConfig`.

        Config fields:
            state_cost: Diagonal Q weights (n_x,) or full Q matrix (n_x, n_x).
            control_cost: Diagonal R weights (n_u,) or full R matrix (n_u, n_u).
            dt: Time step — used to set B = dt * I unless B_dynamics is given.
            A_dynamics: Optional full A matrix (n_x, n_x). Defaults to I.
            B_dynamics: Optional full B matrix (n_x, n_u). Defaults to dt * I.

        Args:
            config: TOML config dict or LQRConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            LQR instance.
        """
        bk = backend or NumpyBackend()
        cfg = strict_from_dict(LQRConfig, config, "LQR") if isinstance(config, dict) else config
        Q = parse_matrix(bk, cfg.state_cost)
        n = Q.shape[0]
        A = bk.array(cfg.A_dynamics) if cfg.A_dynamics is not None else bk.eye(n)
        if cfg.B_dynamics is not None:
            B = bk.array(cfg.B_dynamics)
        elif cfg.dt is not None:
            B = cfg.dt * bk.eye(n)
        else:
            raise ValueError("LQR: no B_dynamics and no dt — standalone use requires one of them")
        return cls(
            state_cost_matrix=Q,
            control_cost_matrix=parse_matrix(bk, cfg.control_cost),
            dynamics_state_matrix=A,
            dynamics_control_matrix=B,
            backend=bk,
        )
