import numpy as np
from typing import Optional
from scipy.sparse import block_diag
import osqp
from scipy import sparse
from components import Controller
from factories.registry import register_controller


class MPC_LTI(Controller):
    """Linear Time-Invariant Model Predictive Control with OSQP.

    Solves the constrained quadratic program:
        min_U   Σ (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N
        s.t.    x_{k+1} = A x_k + B u_k
                F u_k ≤ b_upper,  F u_k ≥ b_lower

    Precomputes the dense QP matrices H and F offline. At each call to
    compute(), solves the QP with OSQP and returns the first control action.

    Usage:
        mpc = MPC_LTI(
            horizon=15,
            control_cost_matrix=R,
            state_cost_matrix=Q,
            A_dynamics=A,
            B_dynamics=B,
            terminal_cost=P,
        )
        mpc.constraints(F, b_upper, b_lower)
        u = mpc.compute(x0)
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
        """Initialize the MPC solver with dynamics and cost parameters.

        Args:
            horizon: Prediction horizon length N.
            control_cost_matrix: R — control effort cost (n_u, n_u).
            state_cost_matrix: Q — state deviation cost (n_x, n_x).
            A_dynamics: State transition matrix (n_x, n_x).
            B_dynamics: Control input matrix (n_x, n_u).
            terminal_cost: P — terminal state cost (n_x, n_x).
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
        """Set linear constraints on the control sequence.

        Defines F u_k ≤ b_upper and F u_k ≥ b_lower for each step k,
        tiled across the horizon.

        Args:
            constraint_matrix: Per-step constraint matrix F (n_c, n_u).
            upper_bounds: Upper bound vector (n_c,).
            lower_bounds: Lower bound vector (n_c,).
        """
        self.A_constraints = sparse.csc_matrix(
            block_diag([constraint_matrix] * self.N)
        )
        self.lcons = np.tile(lower_bounds, self.N)
        self.ucons = np.tile(upper_bounds, self.N)

    def _mpc_dynamics_matrices(self):
        """Precompute the lifted dynamics matrices T_bar and S_bar.

        T_bar maps x0 to the predicted state sequence:
            X = T_bar @ x0 + S_bar @ U

        where X = [x_1, x_2, ..., x_N]^T and U = [u_0, ..., u_{N-1}]^T.
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
                self.S_bar[i * self.n: (i + 1) * self.n, j * self.m: (j + 1) * self.m] = (
                    np.linalg.matrix_power(self.A, i - j) @ self.B
                )

    def _mpc_cost_matrices(self):
        """Precompute the quadratic cost matrices H and F.

        The QP objective is: 0.5 * U^T H U + x0^T F^T U
        where H = 2(R_bar + S_bar^T Q_bar S_bar)
              F = 2 T_bar^T Q_bar S_bar
        """
        Q_bar = np.zeros((self.N * self.n, self.N * self.n))
        for i in range(self.N - 1):
            Q_bar[i * self.n: (i + 1) * self.n, i * self.n: (i + 1) * self.n] = self.Q
        Q_bar[(self.N - 1) * self.n: self.N * self.n, (self.N - 1) * self.n: self.N * self.n] = self.P

        R_bar = np.kron(np.eye(self.N), self.R)
        self.H = 2 * (R_bar + self.S_bar.T @ Q_bar @ self.S_bar)
        self.F = 2 * (self.T_bar.T @ Q_bar @ self.S_bar)

    def compute(self, x0):
        """Solve the MPC QP for a given initial state.

        Args:
            x0: Initial state vector (n_x,).

        Returns:
            Optimal first control action (n_u,).
        """
        prob = osqp.OSQP()
        q = self.F.T @ x0
        prob.setup(
            sparse.csc_matrix(self.H), q.flatten(), self.A_constraints, self.lcons, self.ucons,
            warm_starting=True, verbose=False,
        )
        res = prob.solve()

        if res.info.status == "solved":
            z_optimal = res.x
            ctrl = z_optimal[:self.m]
        else:
            print("osqp could not find a solution")
            ctrl = np.zeros(self.m)

        return ctrl


@register_controller("MPC_DeltaU")
class MPC_LTI_DeltaU(MPC_LTI):
    """MPC with Δu (control rate) regularization via state augmentation.

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
        """Initialize MPC with Δu regularization.

        Args:
            delta_u_penalty: S — cost matrix for Δu (n_u, n_u).
            **kwargs: Passed to MPC_LTI.__init__.
        """
        self.S_delta = delta_u_penalty
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        n = len(config["state_cost"])
        ctrl = cls(
            delta_u_penalty=np.diag(config["delta_u_penalty"]),
            horizon=config["horizon"],
            control_cost_matrix=np.diag(config["control_cost"]),
            state_cost_matrix=np.diag(config["state_cost"]),
            A_dynamics=np.eye(n),
            B_dynamics=config["dt"] * np.eye(n),
            terminal_cost=np.diag(config["state_cost"]),
        )
        if "constraints" in config:
            F = np.vstack([np.eye(n), -np.eye(n)])
            ctrl.constraints(F, config["constraints"]["upper"], config["constraints"]["lower"])
        return ctrl


    def _augment_dynamics(self):
        """Augment state to [x; u_prev], control becomes Δu.

        New dynamics:
            z_{k+1} = [A B; 0 I] z_k + [B; I] Δu_k
        where z = [x; u_prev].
        """
        n, m = self.n, self.m
        R_orig = self.R.copy()

        self.A = np.block([
            [self.A, self.B],
            [np.zeros((m, n)), np.eye(m)]
        ])
        self.B = np.vstack([self.B, np.eye(m)])

        Q_aug = np.zeros((n + m, n + m))
        Q_aug[:n, :n] = self.Q
        Q_aug[n:, n:] = R_orig
        self.Q = Q_aug

        self.R = self.S_delta

        P_aug = np.zeros((n + m, n + m))
        P_aug[:n, :n] = self.P
        P_aug[n:, n:] = R_orig
        self.P = P_aug

        self.n = n + m

    def _mpc_dynamics_matrices(self):
        """Override: augment dynamics before computing T_bar, S_bar."""
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self._augment_dynamics()
        super()._mpc_dynamics_matrices()

    def compute(self, x0, u_prev: Optional[np.ndarray] = None):
        """Solve MPC with Δu regularization.

        Args:
            x0: Original (non-augmented) state vector (n_x,).
            u_prev: Previous control input (n_u,). Required for first call.

        Returns:
            Optimal control action (n_u,).
        """
        if u_prev is None:
            u_prev = np.zeros(self.m)
        x0_aug = np.concatenate([x0, u_prev])
        return super().compute(x0_aug)


@register_controller("MPC_LTI")
class MPC_LTI_Base(MPC_LTI):
    @classmethod
    def from_config(cls, config):
        n = len(config["state_cost"])
        ctrl = cls(
            horizon=config["horizon"],
            control_cost_matrix=np.diag(config["control_cost"]),
            state_cost_matrix=np.diag(config["state_cost"]),
            A_dynamics=np.eye(n),
            B_dynamics=config["dt"] * np.eye(n),
            terminal_cost=np.diag(config["state_cost"]),
        )
        if "constraints" in config:
            F = np.vstack([np.eye(n), -np.eye(n)])
            ctrl.constraints(F, config["constraints"]["upper"], config["constraints"]["lower"])
        return ctrl

