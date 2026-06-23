import numpy as np
from components import StateEstimator

class LuenBergerObserver(StateEstimator):
    def __init__(self, A:np.ndarray, B:np.ndarray, C:np.ndarray | None=None, D: np.ndarray | None=None, L