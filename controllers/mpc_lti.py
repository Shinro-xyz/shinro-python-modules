from typing import Optional
from scipy.sparse import block_diag
import osqp
from scipy import sparse
from components import Controller
from factories.registry import register_controller
from utils.array_backend import ArrayBackend, NumpyBackend


class MPC_LTI(Controller):
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
        self.A_constraints = sparse.csc_matrix(
            block_diag([constraint_matrix] * self.N)
        )
        self.lcons = self.bk.tile(lower_bounds, self.N)
        self.ucons = self.bk.tile(upper_bounds, self.N)

    def _mpc_dynamics_matrices(self):
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
        Q_bar = self.bk.zeros((self.N * self.n, self.N * self.n))
        for i in range(self.N - 1):
            Q_bar[i * self.n: (i + 1) * self.n, i * self.n: (i + 1) * self.n] = self.Q
        Q_bar[(self.N - 1) * self.n: self.N * self.n, (self.N - 1) * self.n: self.N * self.n] = self.P

        R_bar = self.bk.kron(self.bk.eye(self.N), self.R)
        self.H = 2 * (R_bar + self.S_bar.T @ Q_bar @ self.S_bar)
        self.F = 2 * (self.T_bar.T @ Q_bar @ self.S_bar)

    def compute(self, x0):
        x0_np = self.bk.to_numpy(x0)
        q = self.F.T @ x0_np
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
    def __init__(self, delta_u_penalty, backend: Optional[ArrayBackend] = None, **kwargs):
        self.bk = backend or NumpyBackend()
        self.S_delta = delta_u_penalty
        super().__init__(backend=self.bk, **kwargs)

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        n = len(config["state_cost"])
        ctrl = cls(
            delta_u_penalty=bk.diag(config["delta_u_penalty"]),
            horizon=config["horizon"],
            control_cost_matrix=bk.diag(config["control_cost"]),
            state_cost_matrix=bk.diag(config["state_cost"]),
            A_dynamics=bk.eye(n),
            B_dynamics=config["dt"] * bk.eye(n),
            terminal_cost=bk.diag(config["state_cost"]),
            backend=bk,
        )
        if "constraints" in config:
            F = bk.vstack([bk.eye(n), -bk.eye(n)])
            ctrl.constraints(F, config["constraints"]["upper"], config["constraints"]["lower"])
        return ctrl

    def _augment_dynamics(self):
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
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self._augment_dynamics()
        super()._mpc_dynamics_matrices()

    def compute(self, x0, u_prev: Optional = None):
        if u_prev is None:
            u_prev = self.bk.zeros(self.m)
        x0_aug = self.bk.hstack([x0, u_prev])
        return super().compute(x0_aug)


@register_controller("MPC_LTI")
class MPC_LTI_Base(MPC_LTI):
    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        n = len(config["state_cost"])
        ctrl = cls(
            horizon=config["horizon"],
            control_cost_matrix=bk.diag(config["control_cost"]),
            state_cost_matrix=bk.diag(config["state_cost"]),
            A_dynamics=bk.eye(n),
            B_dynamics=config["dt"] * bk.eye(n),
            terminal_cost=bk.diag(config["state_cost"]),
            backend=bk,
        )
        if "constraints" in config:
            F = bk.vstack([bk.eye(n), -bk.eye(n)])
            ctrl.constraints(F, config["constraints"]["upper"], config["constraints"]["lower"])
        return ctrl
