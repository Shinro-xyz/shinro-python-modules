import numpy as np
from typing import Optional
from components import StateEstimator
from factories.registry import register_estimator


@register_estimator("LuenbergerObserver")
class LuenbergerObserver(StateEstimator):
    """Luenberger observer for deterministic linear state estimation.

    Implements the discrete-time observer dynamics:
        x̂ₖ₊₁ = A x̂ₖ + B uₖ + L (yₖ − C x̂ₖ − D uₖ)

    where L is the observer gain chosen to place the eigenvalues of (A − LC)
    inside the unit circle for stable estimation. Unlike the Kalman filter,
    the Luenberger observer uses a fixed gain and does not assume noise
    statistics.

    Usage:
        observer = LuenbergerObserver(
            A=np.eye(3), B=0.02 * np.eye(3),
            observer_gain=np.diag([0.8, 0.8, 0.8]),
            C=np.eye(3),
        )
        x_hat = observer.estimate(measurement=y, control_input=u)
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        observer_gain: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """Initialize the Luenberger observer with system matrices and gain.

        Args:
            A: State transition matrix (n x n).
            B: Control input matrix (n x m).
            observer_gain: Observer gain matrix L (n x p). Must place
                eigenvalues of (A - LC) inside the unit circle.
            C: Output matrix (p x n). Defaults to identity.
            D: Feedthrough matrix (p x m). Defaults to zeros.
            x0: Initial state estimate (n x 1). Defaults to zeros.
        """
        self.A = A
        self.B = B
        self.C = np.eye(A.shape[0]) if C is None else C
        self.D = np.zeros((self.C.shape[0], B.shape[1])) if D is None else D
        self.L = observer_gain
        self.x_hat = np.zeros((A.shape[0], 1)) if x0 is None else x0.copy()

    def estimate(self, measurement: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """Perform one step of state estimation.

        Computes the predicted state from the dynamics, calculates the
        innovation (measurement residual), and corrects the prediction
        using the observer gain:
            x̂ₖ₊₁ = A x̂ₖ + B uₖ + L (yₖ − C(A x̂ₖ + B uₖ) − D uₖ)

        Args:
            measurement: Output measurement yₖ (p x 1).
            control_input: Control input uₖ (m x 1).

        Returns:
            Updated state estimate x̂ₖ₊₁ (n x 1).
        """
        x_pred = self.A @ self.x_hat + self.B @ control_input
        innovations = measurement - (self.C @ x_pred + self.D @ control_input)
        self.x_hat = x_pred + self.L @ innovations
        return self.x_hat

    def reset(self, x0: Optional[np.ndarray] = None):
        self.x_hat = np.zeros((self.A.shape[0], 1)) if x0 is None else x0.copy()

    @classmethod
    def from_config(cls, config):
        gain = np.diag(config["observer_gain"])
        n = gain.shape[0]
        return cls(
            A=np.eye(n),
            B=config["dt"] * np.eye(n),
            observer_gain=gain,
            C=np.eye(n),
            D=np.zeros((n, n)),
            x0=np.zeros((n, 1)),
        )