import numpy as np
from typing import Optional
from components import Controller
from scipy.linalg import solve_discrete_are
from factories.registry import register_controller


@register_controller("LQR")
class LQR(Controller):
    """Linear Quadratic Regulator (LQR) for discrete-time systems.

    Computes the optimal state-feedback control law u = -K @ x that minimizes:
        J = Σ (x_k^T Q x_k + u_k^T R u_k)

    The gain K is computed offline via the Discrete Algebraic Riccati Equation
    (DARE) and applied online as a simple matrix-vector multiply.

    Usage:
        lqr = LQR(Q=np.diag([10, 10, 1]), R=np.diag([0.1, 0.1]),
                  A=np.eye(3), B=0.02 * np.eye(3))
        u = lqr.compute(current_state, target_state)
    """

    def __init__(
        self,
        state_cost_matrix: np.ndarray,
        control_cost_matrix: np.ndarray,
        dynamics_state_matrix: np.ndarray,
        dynamics_control_matrix: np.ndarray,
    ):
        """Initialize the LQR controller and compute the optimal gain.

        Args:
            state_cost_matrix: Q — penalizes state deviation (n_x, n_x).
            control_cost_matrix: R — penalizes control effort (n_u, n_u).
            dynamics_state_matrix: A — discrete-time state transition (n_x, n_x).
            dynamics_control_matrix: B — control input matrix (n_x, n_u).
        """
        self.A = dynamics_state_matrix
        self.B = dynamics_control_matrix
        self.Q = state_cost_matrix
        self.R = control_cost_matrix
        self.gain_calculation()

    def gain_calculation(self):
        """Solve DARE and compute the optimal LQR gain matrix K.

        Solves P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
        then computes K = (R + B^T P B)^{-1} B^T P A.

        The gain K is stored as self.K and used in compute().
        """
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

    def compute(self, current_state: np.ndarray, target_state: Optional[np.ndarray] = None):
        """Compute the optimal control input u = -K @ (x - x_target).

        Args:
            current_state: Current state vector (n_x,).
            target_state: Desired state vector (n_x,). Defaults to zeros.

        Returns:
            Control input vector (n_u,).
        """
        if target_state is None:
            target_state = np.zeros_like(current_state)
        error = target_state - current_state
        return self.K @ error

    def reset(self):
        """Reset the controller state. No internal state for LQR."""
        pass

    @classmethod
    def from_config(cls, config):
        n = len(config["state_cost"])
        return cls(
            state_cost_matrix=np.diag(config["state_cost"]),
            control_cost_matrix=np.diag(config["control_cost"]),
            dynamics_state_matrix=np.eye(n),
            dynamics_control_matrix=config["dt"] * np.eye(n),
        )
    