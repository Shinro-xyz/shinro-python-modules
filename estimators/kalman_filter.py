from typing import Optional
from components import StateEstimator
from factories.registry import register_estimator
from utils.array_backend import ArrayBackend, NumpyBackend


@register_estimator("KalmanFilter")
class KalmanFilter(StateEstimator):
    """Discrete-time linear Kalman filter for optimal state estimation.

    Implements the predict-update cycle for a system of the form:

    .. math::

        x_{k+1} &= A x_k + B u_k + w_k, \\quad w_k \\sim \\mathcal{N}(0, Q) \\\\
        y_k &= C x_k + D u_k + v_k, \\quad v_k \\sim \\mathcal{N}(0, R)

    Tracks the posterior state estimate :math:`\\hat{x}` and error covariance
    :math:`P` through the standard Kalman filter equations.

    Uses column vectors :math:`(n, 1)` throughout (not flat :math:`(n,)`).

    Args:
        A: State transition matrix (n_x, n_x).
        B: Control input matrix (n_x, n_u).
        Q: Process noise covariance (n_x, n_x).
        R: Measurement noise covariance (n_y, n_y).
        C: Observation matrix (n_y, n_x). Defaults to identity.
        D: Feedthrough matrix (n_y, n_u). Defaults to zeros.
        x0: Initial state estimate (n_x, 1). Defaults to zeros.
        backend: Array backend. Defaults to NumpyBackend.
    """

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
        """Run one predict-update cycle and return the posterior state estimate.

        Implements the standard Kalman filter equations:

        1. Predict:
           :math:`x_{\\text{pred}} = A \\hat{x} + B u`
           :math:`P_{\\text{pred}} = A P A^T + Q`

        2. Update:
           :math:`K = P_{\\text{pred}} C^T (C P_{\\text{pred}} C^T + R)^{-1}`
           :math:`\\hat{x} = x_{\\text{pred}} + K (y - C x_{\\text{pred}} - D u)`
           :math:`P = (I - K C) P_{\\text{pred}}`

        Args:
            measurement: Observation vector (n_y, 1) from sensors.
            control_input: Control vector (n_u, 1) applied at this step.

        Returns:
            Posterior state estimate :math:`\\hat{x}` (n_x, 1).
        """
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
        """Reset the filter to its initial state.

        Args:
            x0: Initial state estimate (n_x, 1). Defaults to zeros.
        """
        self.x_hat = self.bk.zeros((self.A.shape[0], 1)) if x0 is None else self.bk.copy(x0)
        self.P = self.bk.eye(self.A.shape[0]) * 0.1

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create a Kalman filter from a TOML config dict.

        Config fields:
            process_noise: List of diagonal Q weights (n_x,).
            measurement_noise: List of diagonal R weights (n_y,).
            dt: Time step — used to set B = dt * I.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            KalmanFilter instance.
        """
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
