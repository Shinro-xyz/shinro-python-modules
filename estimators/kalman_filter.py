import numpy as np
from typing import Optional
from components import StateEstimator
from factories.registry import register_estimator


@register_estimator("KalmanFilter")
class KalmanFilter(StateEstimator):
    """Discrete-time linear Kalman filter for optimal state estimation.

    Implements the predict-update cycle for a system of the form:
        x_{k+1} = A @ x_k + B @ u_k + w_k,   w_k ~ N(0, Q)
        y_k     = C @ x_k + D @ u_k + v_k,   v_k ~ N(0, R)

    Tracks the posterior state estimate (x_hat) and error covariance (P)
    through the standard Kalman filter equations.

    Usage:
        kf = KalmanFilter(
            A=np.eye(3), B=0.02 * np.eye(3),
            Q=0.01 * np.eye(3), R=0.05 * np.eye(3),
            C=np.eye(3),
        )
        x_hat = kf.estimate(measurement=y, control_input=u)
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """Initialize Kalman filter with system matrices and initial state.

        Args:
            A: State transition matrix (n_x, n_x).
            B: Control input matrix (n_x, n_u).
            Q: Process noise covariance (n_x, n_x).
            R: Measurement noise covariance (n_y, n_y).
            C: Observation matrix (n_y, n_x). Defaults to identity.
            D: Feedthrough matrix (n_y, n_u). Defaults to zeros.
            x0: Initial state estimate (n_x, 1). Defaults to zeros.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.C = np.eye(A.shape[0]) if C is None else C
        self.D = np.zeros((self.C.shape[0], B.shape[1])) if D is None else D

        self.x_hat = np.zeros((A.shape[0], 1)) if x0 is None else x0.copy()
        self.P = np.eye(A.shape[0]) * 0.1

    def estimate(self, measurement: np.ndarray, control_input: np.ndarray):
        """Run one predict-update cycle and return the posterior state estimate.

        Implements the standard Kalman filter equations:
        1. Predict:  x_pred = A @ x_hat + B @ u
                     P_pred = A @ P @ A^T + Q
        2. Update:   K = P_pred @ C^T @ (C @ P_pred @ C^T + R)^{-1}
                     x_hat = x_pred + K @ (y - C @ x_pred - D @ u)
                     P = (I - K @ C) @ P_pred

        Args:
            measurement: Observation vector (n_y, 1) from sensors.
            control_input: Control vector (n_u, 1) applied at this step.

        Returns:
            Posterior state estimate x_hat (n_x, 1) after incorporating the
            measurement.
        """
        # Predict
        x_pred = self.A @ self.x_hat + self.B @ control_input
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Kalman gain
        S = self.C @ self.P @ self.C.T + self.R
        K_gain = self.P @ self.C.T @ np.linalg.inv(S)

        # Update
        y_pred = self.C @ x_pred + self.D @ control_input
        innovations = measurement - y_pred

        self.x_hat = x_pred + K_gain @ innovations
        self.P = (np.eye(self.A.shape[0]) - K_gain @ self.C) @ self.P

        return self.x_hat

    def reset(self, x0: Optional[np.ndarray] = None):
        self.x_hat = np.zeros((self.A.shape[0], 1)) if x0 is None else x0.copy()
        self.P = np.eye(self.A.shape[0]) * 0.1

    @classmethod
    def from_config(cls, config):
        n = len(config["process_noise"])
        return cls(
            A=np.eye(n),
            B=config["dt"] * np.eye(n),
            Q=np.diag(config["process_noise"]),
            R=np.diag(config["measurement_noise"]),
            C=np.eye(n),
            D=np.zeros((n, n)),
            x0=np.zeros((n, 1)),
        )