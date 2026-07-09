from typing import Optional
from components import Controller
from scipy.linalg import solve_discrete_are
from factories.registry import register_controller
from utils.array_backend import ArrayBackend, NumpyBackend


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
        backend: Optional[ArrayBackend] = None,
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

    def compute(self, current_state, target_state: Optional = None):
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

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create an LQR controller from a TOML config dict.

        Config fields:
            state_cost: List of diagonal Q weights (n_x,).
            control_cost: List of diagonal R weights (n_u,).
            dt: Time step — used to set B = dt * I.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            LQR instance.
        """
        bk = backend or NumpyBackend()
        n = len(config["state_cost"])
        return cls(
            state_cost_matrix=bk.diag(config["state_cost"]),
            control_cost_matrix=bk.diag(config["control_cost"]),
            dynamics_state_matrix=bk.eye(n),
            dynamics_control_matrix=config["dt"] * bk.eye(n),
            backend=bk,
        )
