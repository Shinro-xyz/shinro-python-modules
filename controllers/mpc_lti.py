from typing import Optional, Any
import osqp
from scipy import sparse
from components import Controller
from factories.registry import register_controller
from utils.array_backend import ArrayBackend, NumpyBackend, parse_matrix


class MPC_LTI(Controller):
    """Linear Time-Invariant Model Predictive Control with OSQP.

    Solves the constrained quadratic program:

    .. math::

        \\min_U \\quad \\sum_{k=0}^{N-1} \\left( x_k^T Q x_k + u_k^T R u_k \\right)
        + x_N^T P x_N

    subject to:

    .. math::

        x_{k+1} = A x_k + B u_k, \\quad F u_k \\leq b_{\\text{upper}},
        \\quad F u_k \\geq b_{\\text{lower}}

    Precomputes the dense QP matrices H and F offline using lifted dynamics.
    At each call to ``compute()``, solves the QP with OSQP and returns the
    first control action.

    The lifted matrix construction uses ``bk.xxx`` calls and is backend-agnostic.
    The OSQP solver itself is C-based and always uses numpy — the per-step
    ``compute()`` converts via ``bk.to_numpy`` / ``bk.from_numpy``.

    Args:
        horizon: Prediction horizon length N.
        control_cost_matrix: R — control effort cost (n_u, n_u).
        state_cost_matrix: Q — state deviation cost (n_x, n_x).
        A_dynamics: State transition matrix (n_x, n_x).
        B_dynamics: Control input matrix (n_x, n_u).
        terminal_cost: P — terminal state cost (n_x, n_x).
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        horizon: int,
        control_cost_matrix,
        state_cost_matrix,
        A_dynamics,
        B_dynamics,
        terminal_cost,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.N = horizon
        self.Q = state_cost_matrix
        self.R = control_cost_matrix
        self.A = A_dynamics
        self.B = B_dynamics
        self.P = terminal_cost

        self._mpc_dynamics_matrices()
        self._mpc_cost_matrices()

    def constraints(self, constraint_matrix, upper_bounds, lower_bounds):
        """Set linear constraints on the control sequence.

        Defines :math:`F u_k \\leq b_{\\text{upper}}` and
        :math:`F u_k \\geq b_{\\text{lower}}` for each step k,
        tiled across the horizon.

        Args:
            constraint_matrix: Per-step constraint matrix F (n_c, n_u).
            upper_bounds: Upper bound vector (n_c,).
            lower_bounds: Lower bound vector (n_c,).
        """
        self.A_constraints = sparse.csc_matrix(
            sparse.block_diag([sparse.coo_array(constraint_matrix)] * self.N)
        )
        self.lcons = self.bk.tile(lower_bounds, self.N)
        self.ucons = self.bk.tile(upper_bounds, self.N)

    def _mpc_dynamics_matrices(self):
        """Precompute the lifted dynamics matrices T_bar and S_bar.

        The predicted state sequence is:

        .. math::

            X = T_{\\text{bar}} x_0 + S_{\\text{bar}} U

        where :math:`X = [x_1, x_2, \\ldots, x_N]^T` and
        :math:`U = [u_0, \\ldots, u_{N-1}]^T`.

        T_bar has shape (N*n_x, n_x), S_bar has shape (N*n_x, N*n_u).
        """
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]

        T_list = []
        for n_step in range(self.N):
            A_new = self.bk.matrix_power(self.A, n_step)
            T_list.append(A_new)
        self.T_bar = self.bk.vstack(T_list)

        self.S_bar = self.bk.zeros((self.N * self.n, self.N * self.m))
        for i in range(self.N):
            for j in range(i + 1):
                self.S_bar[i * self.n: (i + 1) * self.n, j * self.m: (j + 1) * self.m] = (
                    self.bk.matrix_power(self.A, i - j) @ self.B
                )

    def _mpc_cost_matrices(self):
        """Precompute the quadratic cost matrices H and F.

        The QP objective is:

        .. math::

            \\frac{1}{2} U^T H U + x_0^T F^T U

        where:

        .. math::

            H = 2(R_{\\text{bar}} + S_{\\text{bar}}^T Q_{\\text{bar}} S_{\\text{bar}})
            F = 2 T_{\\text{bar}}^T Q_{\\text{bar}} S_{\\text{bar}}
        """
        Q_bar = self.bk.zeros((self.N * self.n, self.N * self.n))
        for i in range(self.N - 1):
            Q_bar[i * self.n: (i + 1) * self.n, i * self.n: (i + 1) * self.n] = self.Q
        Q_bar[(self.N - 1) * self.n: self.N * self.n, (self.N - 1) * self.n: self.N * self.n] = self.P

        R_bar = self.bk.kron(self.bk.eye(self.N), self.R)
        self.H = 2 * (R_bar + self.S_bar.T @ Q_bar @ self.S_bar)
        self.F = 2 * (self.T_bar.T @ Q_bar @ self.S_bar)

    def compute(self, x0):
        """Solve the MPC QP for a given initial state.

        Converts x0 to numpy, runs OSQP, converts the result back to the
        backend's native type.

        Args:
            x0: Initial state vector (n_x,).

        Returns:
            Optimal first control action (n_u,).
        """
        x0_np = self.bk.to_numpy(x0)
        F_np = self.bk.to_numpy(self.F)
        q = F_np.T @ x0_np
        prob = osqp.OSQP()
        prob.setup(
            sparse.csc_matrix(self.bk.to_numpy(self.H)), q.flatten(),
            self.A_constraints, self.bk.to_numpy(self.lcons), self.bk.to_numpy(self.ucons),
            warm_starting=True, verbose=False,
        )
        res = prob.solve()

        if res.info.status == "solved":
            z_optimal = res.x
            ctrl = self.bk.from_numpy(z_optimal[:self.m])
        else:
            print("osqp could not find a solution")
            ctrl = self.bk.zeros(self.m)

        return ctrl


@register_controller("MPC_DeltaU")
class MPC_LTI_DeltaU(MPC_LTI):
    """MPC with :math:`\\Delta u` (control rate) regularization.

    Augments the state to :math:`[x; u_{\\text{prev}}]` so the control
    variable becomes :math:`\\Delta u = u_k - u_{k-1}`. The cost penalizes
    :math:`\\Delta u^T S \\Delta u`, smoothing chatter.

    The augmented dynamics are:

    .. math::

        z_{k+1} = \\begin{bmatrix} A & B \\\\ 0 & I \\end{bmatrix} z_k
        + \\begin{bmatrix} B \\\\ I \\end{bmatrix} \\Delta u_k

    where :math:`z = [x; u_{\\text{prev}}]`.

    Args:
        delta_u_penalty: S — cost matrix for :math:`\\Delta u` (n_u, n_u).
        **kwargs: Passed to MPC_LTI.__init__.
    """

    def __init__(self, delta_u_penalty, backend: Optional[ArrayBackend] = None, **kwargs):
        self.bk = backend or NumpyBackend()
        self.S_delta = delta_u_penalty
        super().__init__(backend=self.bk, **kwargs)

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create an MPC_DeltaU controller from a TOML config dict.

        Config fields:
            delta_u_penalty: Diagonal S weights (n_u,) or full S matrix (n_u, n_u).
            horizon: Prediction horizon.
            state_cost: Diagonal Q weights (n_x,) or full Q matrix (n_x, n_x).
            control_cost: Diagonal R weights (n_u,) or full R matrix (n_u, n_u).
            dt: Time step.
            A_dynamics: Optional full A matrix (n_x, n_x). Defaults to I.
            B_dynamics: Optional full B matrix (n_x, n_u). Defaults to dt * I.
            constraints: Optional dict with ``upper`` and ``lower`` bound lists.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            MPC_LTI_DeltaU instance.
        """
        bk = backend or NumpyBackend()
        Q = parse_matrix(bk, config["state_cost"])
        n = Q.shape[0]
        ctrl = cls(
            delta_u_penalty=parse_matrix(bk, config["delta_u_penalty"]),
            horizon=config["horizon"],
            control_cost_matrix=parse_matrix(bk, config["control_cost"]),
            state_cost_matrix=Q,
            A_dynamics=bk.array(config.get("A_dynamics", bk.eye(n))),
            B_dynamics=bk.array(config.get("B_dynamics", config["dt"] * bk.eye(n))),
            terminal_cost=parse_matrix(bk, config.get("terminal_cost", config["state_cost"])),
            backend=bk,
        )
        if "constraints" in config:
            if "matrix" in config["constraints"]:
                F = bk.array(config["constraints"]["matrix"])
            else:
                F = bk.vstack([bk.eye(n), -bk.eye(n)])
            ctrl.constraints(F, config["constraints"]["upper"], config["constraints"]["lower"])
        return ctrl

    def _augment_dynamics(self):
        """Augment state to [x; u_prev], control becomes Delta u.

        Builds the augmented matrices A_aug, B_aug, Q_aug, P_aug, and
        sets R = S_delta (the Delta u penalty).
        """
        n, m = self.n, self.m
        R_orig = self.bk.copy(self.R)

        self.A = self.bk.block([
            [self.A, self.B],
            [self.bk.zeros((m, n)), self.bk.eye(m)]
        ])
        self.B = self.bk.vstack([self.B, self.bk.eye(m)])

        Q_aug = self.bk.zeros((n + m, n + m))
        Q_aug[:n, :n] = self.Q
        Q_aug[n:, n:] = R_orig
        self.Q = Q_aug

        self.R = self.S_delta

        P_aug = self.bk.zeros((n + m, n + m))
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

    def compute(self, x0, u_prev: Optional[Any] = None):
        """Solve MPC with :math:`\\Delta u` regularization.

        Augments the state with the previous control input before solving.

        Args:
            x0: Original (non-augmented) state vector (n_x,).
            u_prev: Previous control input (n_u,). Defaults to zeros.

        Returns:
            Optimal control action (n_u,).
        """
        if u_prev is None:
            u_prev = self.bk.zeros(self.m)
        x0_aug = self.bk.hstack([x0, u_prev])
        return super().compute(x0_aug)


@register_controller("MPC_LTI")
class MPC_LTI_Base(MPC_LTI):
    """Thin wrapper around MPC_LTI with a ``from_config`` classmethod.

    Registered as ``"MPC_LTI"`` in the controller registry. Uses default
    dynamics :math:`A = I, B = dt \\cdot I`.
    """

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create an MPC_LTI controller from a TOML config dict.

        Config fields:
            horizon: Prediction horizon.
            state_cost: Diagonal Q weights (n_x,) or full Q matrix (n_x, n_x).
            control_cost: Diagonal R weights (n_u,) or full R matrix (n_u, n_u).
            dt: Time step.
            A_dynamics: Optional full A matrix (n_x, n_x). Defaults to I.
            B_dynamics: Optional full B matrix (n_x, n_u). Defaults to dt * I.
            constraints: Optional dict with ``upper`` and ``lower`` bound lists.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            MPC_LTI instance.
        """
        bk = backend or NumpyBackend()
        Q = parse_matrix(bk, config["state_cost"])
        n = Q.shape[0]
        ctrl = cls(
            horizon=config["horizon"],
            control_cost_matrix=parse_matrix(bk, config["control_cost"]),
            state_cost_matrix=Q,
            A_dynamics=bk.array(config.get("A_dynamics", bk.eye(n))),
            B_dynamics=bk.array(config.get("B_dynamics", config["dt"] * bk.eye(n))),
            terminal_cost=parse_matrix(bk, config.get("terminal_cost", config["state_cost"])),
            backend=bk,
        )
        if "constraints" in config:
            if "matrix" in config["constraints"]:
                F = bk.array(config["constraints"]["matrix"])
            else:
                F = bk.vstack([bk.eye(n), -bk.eye(n)])
            ctrl.constraints(F, config["constraints"]["upper"], config["constraints"]["lower"])
        return ctrl
