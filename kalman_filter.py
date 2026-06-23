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
        self.x_hat=np.zeros((self.A.shape[0],1)) if x0 is None else x0.copy()
        self.P=np.eye(self.A.shape[0])*0.1
        