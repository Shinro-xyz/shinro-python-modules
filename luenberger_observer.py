import numpy as np
from components import StateEstimator

class LuenbergerObserver(StateEstimator):
    def __init__(self, A:np.ndarray, B:np.ndarray, observer_gain:np.ndarray,C:np.ndarray | None=None, D: np.ndarray | None=None, x_hat:np.ndarray | None=None, 
    ):
        self.A=A
        self.B=B
        if C is None:
            self.C=np.eye(A.shape[0])
        else:
            self.C=C

        if D is None:
            self.D=np.zeros((self.C.shape[0], B.shape[1]))

        else: 
            self.D=D

        self.L= observer_gain

    def estimate(self, measurement: np.ndarray, control_input: np.ndarray):
        