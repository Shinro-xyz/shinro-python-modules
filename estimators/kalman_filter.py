import numpy as np
from components import StateEstimator

class KalmanFilter(StateEstimator):
    """Discrete-time linear Kalman filter for state estimation.

    Implements the predict-update cycle for a system of the form:
        x_{k+1} = A @ x_k + B @ u_k + w_k,   w_k ~ N(0, Q)
        y_k     = C @ x_k + D @ u_k + v_k,   v_k ~ N(0, R)

    Tracks the posterior state estimate (x_hat) and error covariance (P).
    """

    def __init__(self, A:np.ndarray, B:np.ndarray, Q:np.ndarray, R: np.ndarray,C:np.ndarray| None=None, D:np.ndarray | None=None, x0:np.ndarray | None=None):
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
        self.A=A
        self.B=B
        self.Q=Q
        self.R=R

        self.C= np.eye(A.shape[0]) if C is None else C
        self.D= np.zeros((self.C.shape[0],B.shape[1])) if D is None else D

        self.x_hat=np.zeros((A.shape[0],1)) if x0 is None else x0.copy()
        self.P=np.eye(A.shape[0])*0.1

    def estimate(self, measurement: np.ndarray, control_input: np.ndarray):
        """Run one predict-update cycle and return the posterior state estimate.

        Args:
            measurement: Observation vector (n_y, 1) from sensors.
            control_input: Control vector (n_u, 1) applied at this step.

        Returns:
            Posterior state estimate x_hat (n_x, 1) after incorporating the
            measurement.
        """
        # step 1: predict
        x_pred=self.A@self.x_hat+self.B@control_input
        self.P= self.A@self.P@self.A.T+self.Q

        #step 2: Kalman calculations
        S= self.C@self.P@self.C.T+self.R
        K_gain=self.P@self.C.T@np.linalg.inv(S)

        #step 3: Updates
        y_pred=self.C@x_pred+self.D@control_input
        innovations=measurement-y_pred

        self.x_hat=x_pred+K_gain@innovations
        self.P=(np.eye(self.A.shape[0])-K_gain@self.C)@self.P

        return self.x_hat

    def reset(self, x0:np.ndarray | None=None):
        """Reset the filter to an initial state.

        Args:
            x0: New initial state estimate (n_x, 1). Defaults to zeros.
        """
        self.x_hat=np.zeros((self.A.shape[0],1)) if x0 is None else x0.copy()
        self.P=np.eye(self.A.shape[0])*0.1
        