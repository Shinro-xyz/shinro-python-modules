import numpy as np
from components import StateEstimator

class KalmanFilter(StateEstimator):
    def __init__(self, A:np.ndarray, B=np.ndarray