from typing import Optional
from components import Controller
from scipy.linalg import solve_discrete_are
from factories.registry import register_controller
from utils.array_backend import ArrayBackend, NumpyBackend


@register_controller("LQR")
class LQR(Controller):
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
        A_np = self.bk.to_numpy(self.A)
        B_np = self.bk.to_numpy(self.B)
        P_np = solve_discrete_are(A_np, B_np, self.bk.to_numpy(self.Q), self.bk.to_numpy(self.R))
        P = self.bk.from_numpy(P_np)
        self.K = self.bk.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

    def compute(self, current_state, target_state: Optional = None):
        if target_state is None:
            target_state = self.bk.zeros_like(current_state)
        error = target_state - current_state
        return self.K @ error

    def reset(self):
        pass

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        n = len(config["state_cost"])
        return cls(
            state_cost_matrix=bk.diag(config["state_cost"]),
            control_cost_matrix=bk.diag(config["control_cost"]),
            dynamics_state_matrix=bk.eye(n),
            dynamics_control_matrix=config["dt"] * bk.eye(n),
            backend=bk,
        )
