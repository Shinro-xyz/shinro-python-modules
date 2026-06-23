import numpy as np
from components import StateEstimator

class KalmanFilter(StateEstimator):
    def __init__(self, A:np.ndarray, B:np.ndarray, Q:np.ndarray, R: np.ndarray,C:np.ndarray| None=None, D:np.ndarray | None=None, x0:np.ndarray | None=None):
        self.A=A
        self.B=B
        self.Q=Q
        self.R=R

        self.C= np.eye(A.shape[0]) if C is None else C
        self.D= np.zeros((self.C.shape[0],B.shape[1])) if D is None else D

        self.x_hat=np.zeros((A.shape[0],1)) if x0 is None else x0.copy()
        self.P=np.eye(A.shape[0])*0.1

    def estimate(self, measurement: np.ndarray, control_input: np.ndarray):
        x_pred=
        