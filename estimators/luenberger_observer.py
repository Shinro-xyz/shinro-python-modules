from typing import Optional
from components import StateEstimator
from factories.registry import register_estimator
from utils.array_backend import ArrayBackend, NumpyBackend


@register_estimator("LuenbergerObserver")
class LuenbergerObserver(StateEstimator):
    def __init__(
        self,
        A,
        B,
        observer_gain,
        C: Optional = None,
        D: Optional = None,
        x0: Optional = None,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.A = A
        self.B = B
        self.C = self.bk.eye(A.shape[0]) if C is None else C
        self.D = self.bk.zeros((self.C.shape[0], B.shape[1])) if D is None else D
        self.L = observer_gain
        self.x_hat = self.bk.zeros((A.shape[0], 1)) if x0 is None else self.bk.copy(x0)

    def estimate(self, measurement, control_input):
        x_pred = self.A @ self.x_hat + self.B @ control_input
        innovations = measurement - (self.C @ x_pred + self.D @ control_input)
        self.x_hat = x_pred + self.L @ innovations
        return self.x_hat

    def reset(self, x0: Optional = None):
        self.x_hat = self.bk.zeros((self.A.shape[0], 1)) if x0 is None else self.bk.copy(x0)

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        gain = bk.diag(config["observer_gain"])
        n = gain.shape[0]
        return cls(
            A=bk.eye(n),
            B=config["dt"] * bk.eye(n),
            observer_gain=gain,
            C=bk.eye(n),
            D=bk.zeros((n, n)),
            x0=bk.zeros((n, 1)),
            backend=bk,
        )
