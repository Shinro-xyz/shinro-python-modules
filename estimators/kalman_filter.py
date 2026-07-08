from typing import Optional
from components import StateEstimator
from factories.registry import register_estimator
from utils.array_backend import ArrayBackend, NumpyBackend


@register_estimator("KalmanFilter")
class KalmanFilter(StateEstimator):
    def __init__(
        self,
        A,
        B,
        Q,
        R,
        C: Optional = None,
        D: Optional = None,
        x0: Optional = None,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.C = self.bk.eye(A.shape[0]) if C is None else C
        self.D = self.bk.zeros((self.C.shape[0], B.shape[1])) if D is None else D

        self.x_hat = self.bk.zeros((A.shape[0], 1)) if x0 is None else self.bk.copy(x0)
        self.P = self.bk.eye(A.shape[0]) * 0.1

    def estimate(self, measurement, control_input):
        x_pred = self.A @ self.x_hat + self.B @ control_input
        self.P = self.A @ self.P @ self.A.T + self.Q

        S = self.C @ self.P @ self.C.T + self.R
        K_gain = self.P @ self.C.T @ self.bk.inv(S)

        y_pred = self.C @ x_pred + self.D @ control_input
        innovations = measurement - y_pred

        self.x_hat = x_pred + K_gain @ innovations
        self.P = (self.bk.eye(self.A.shape[0]) - K_gain @ self.C) @ self.P

        return self.x_hat

    def reset(self, x0: Optional = None):
        self.x_hat = self.bk.zeros((self.A.shape[0], 1)) if x0 is None else self.bk.copy(x0)
        self.P = self.bk.eye(self.A.shape[0]) * 0.1

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        n = len(config["process_noise"])
        return cls(
            A=bk.eye(n),
            B=config["dt"] * bk.eye(n),
            Q=bk.diag(config["process_noise"]),
            R=bk.diag(config["measurement_noise"]),
            C=bk.eye(n),
            D=bk.zeros((n, n)),
            x0=bk.zeros((n, 1)),
            backend=bk,
        )
