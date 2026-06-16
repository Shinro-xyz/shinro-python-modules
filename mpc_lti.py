import numpy as np
from scipy.sparse import block_diag
import osqp
from scipy import sparse


class MPC_LTI:
    """
    Linear Time-Invariant Model Predictive Control (MPC) solver.
    """
    def __init__(
        self,
        horizon: int,
        control_cost_matrix: np.ndarray,
        state_cost_matrix: np.ndarray,
        A_dynamics: np.ndarray,
        B_dynamics: np.ndarray,
        terminal_cost: np.ndarray,
    ):
        """
        Initialize the MPC solver with dynamics and cost parameters.

        Args:
            horizon: Prediction horizon length.
            control_cost_matrix: Cost matrix R for control effort.
            state_cost_matrix: Cost matrix Q for state deviation.
            A_dynamics: State transition matrix.
            B_dynamics: Control input matrix.
            terminal_cost: Terminal state cost matrix P.
        """
        self.N = horizon
        self.Q = state_cost_matrix
        self.R = control_cost_matrix
        self.A = A_dynamics
        self.B = B_dynamics
        self.P = terminal_cost

    def constraints(self, constraint_matrix: np.ndarray, upper_bounds: np.ndarray, lower_bounds: np.ndarray):
        """
        Set the constraints for the optimization problem.

        Args:
            constraint_matrix: Matrix defining linear constraints.
            upper_bounds: Upper limit for the constraints.
            lower_bounds: Lower limit for the constraints.
        """
        self.A_constraints = sparse.csc_matrix(constraint_matrix)
        self.lcons = lower_bounds
        self.ucons = upper_bounds

    def _mpc_dynamics_matrices(self):
        """
        Precompute the MPC dynamics matrices T_bar and S_bar.
        """
        self.n = self.A.shape[0]  # state dim
        self.m = self.B.shape[1]  # control dim

        T_list = []
        for n_step in range(self.N):
            A_new = np.linalg.matrix_power(self.A, n_step)
            T_list.append(A_new)
        self.T_bar = np.vstack(T_list)

        self.S_bar = np.zeros((self.N * self.n, self.N * self.m))
        for i in range(self.N):
            for j in range(i + 1):
                self.S_bar[i * self.n : (i + 1) * self.n, j * self.m : (j + 1) * self.m] = (
                    np.linalg.matrix_power(self.A, i - j) @ self.B
                )

    def _mpc_cost_matrices(self):
        """
        Precompute the quadratic cost matrices H and F.
        """
        # Q_bar: block diagonal of Q (with P at the end for terminal cost)
        Q_bar = np.zeros((self.N * self.n, self.N * self.n))
        for i in range(self.N - 1):
            Q_bar[i * self.n : (i + 1) * self.n, i * self.n : (i + 1) * self.n] = self.Q
        Q_bar[(self.N - 1) * self.n : self.N * self.n, (self.N - 1) * self.n : self.N * self.n] = (
            self.P
        )  # terminal cost

        # R_bar: block diagonal of R
        R_bar = np.kron(np.eye(self.N), self.R)
        self.H = 2 * (R_bar + self.S_bar.T @ Q_bar @ self.S_bar)
        self.F = 2 * (self.T_bar @ Q_bar @ self.S_bar)

    def solve(self, x0):
        """
        Solve the MPC optimization problem for a given initial state.

        Args:
            x0: Initial state vector.

        Returns:
            The optimal control action for the current step.
        """
        prob = osqp.OSQP()
        q = self.F.T @ x0
        prob.setup(
            sparse.csc_matrix(self.H), q.flatten(), self.A_constraints, self.lcons, self.ucons, warm_starting=True
        )
        res = prob.solve()

        # solve the problem
        if res.info.status == "solved":
            z_optimal = res.x  # optimal sequence of control outputs
            ctrl = z_optimal[: self.m]
        else:
            print("osqp could not find a solution")

        return ctrl

