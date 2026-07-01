import numpy as np
from scipy.sparse import block_diag
import osqp
from scipy import sparse
from components import Controller


class MPC_LTI(Controller):
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

        self._mpc_dynamics_matrices()
        self._mpc_cost_matrices()

    def constraints(self, constraint_matrix: np.ndarray, upper_bounds: np.ndarray, lower_bounds: np.ndarray):
        """
        Set the constraints for the optimization problem.

        Args:
            constraint_matrix: Matrix defining linear constraints (per time step).
            upper_bounds: Upper limit for the constraints (per time step).
            lower_bounds: Lower limit for the constraints (per time step).
        """
        # Tile constraints across the horizon
        self.A_constraints = sparse.csc_matrix(
            block_diag([constraint_matrix] * self.N)
        )
        self.lcons = np.tile(lower_bounds, self.N)
        self.ucons = np.tile(upper_bounds, self.N)

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
        self.F = 2 * (self.T_bar.T @ Q_bar @ self.S_bar)

    def compute(self, x0):
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
            sparse.csc_matrix(self.H), q.flatten(), self.A_constraints, self.lcons, self.ucons,
            warm_starting=True, verbose=False,
        )
        res = prob.solve()

        # solve the problem
        if res.info.status == "solved":
            z_optimal = res.x  # optimal sequence of control outputs
            ctrl = z_optimal[: self.m]
        else:
            print("osqp could not find a solution")

        return ctrl


class MPC_LTI_DeltaU(MPC_LTI):
    """
    MPC with Δu (control rate) regularization via state augmentation.

    Augments the state to [x; u_prev] so the control variable becomes
    Δu = u_k - u_{k-1}. The cost penalizes Δu^T S Δu, smoothing chatter.

    Usage:
        mpc = MPC_LTI_DeltaU(
            horizon=15,
            control_cost_matrix=R,      # penalizes u_prev in augmented state
            state_cost_matrix=Q,        # penalizes x
            A_dynamics=A,
            B_dynamics=B,
            terminal_cost=P,
            delta_u_penalty=S,          # penalizes Δu (the actual control variable)
        )
        mpc.constraints(...)
        u = mpc.compute(x0, u_prev=last_u)
    """
    def __init__(self, delta_u_penalty: np.ndarray, **kwargs):
        self.S_delta = delta_u_penalty
        super().__init__(**kwargs)

    def _augment_dynamics(self):
        """Augment state to [x; u_prev], control becomes Δu."""
        n, m = self.n, self.m
        R_orig = self.R.copy()

        # Augmented dynamics: z_{k+1} = [A B; 0 I] z_k + [B; I] Δu_k
        self.A = np.block([
            [self.A, self.B],
            [np.zeros((m, n)), np.eye(m)]
        ])
        self.B = np.vstack([self.B, np.eye(m)])

        # Augmented state cost: penalize x with Q, u_prev with R
        Q_aug = np.zeros((n + m, n + m))
        Q_aug[:n, :n] = self.Q
        Q_aug[n:, n:] = R_orig
        self.Q = Q_aug

        # Control cost is now S (Δu penalty)
        self.R = self.S_delta

        # Terminal cost: penalize x with P, u with R
        P_aug = np.zeros((n + m, n + m))
        P_aug[:n, :n] = self.P
        P_aug[n:, n:] = R_orig
        self.P = P_aug

        self.n = n + m  # augmented state dim

    def _mpc_dynamics_matrices(self):
        """Override: augment dynamics before computing T_bar, S_bar."""
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self._augment_dynamics()
        # Now call the parent's matrix computation with augmented dims
        super()._mpc_dynamics_matrices()

    def compute(self, x0, u_prev=None):
        """
        Solve MPC with Δu regularization.

        Args:
            x0: Original (non-augmented) state vector.
            u_prev: Previous control input. Required.

        Returns:
            The optimal control action for the current step.
        """
        if u_prev is None:
            u_prev = np.zeros(self.m)
        # Augment state: [x; u_prev]
        x0_aug = np.concatenate([x0, u_prev])
        return super().compute(x0_aug)

