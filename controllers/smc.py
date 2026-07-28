## sliding mode controller, SMC implementation with state space form

## todo for agent: make it backend agnostic and toml configuration, registry patterns

from components import Controller
import numpy as np

class SlidingModeController(Controller):
    def __init__(
        self,
        c: np.ndarray,
        k1: float,
        phi :float=0.0,
        k2: float=0.0,
        smoother: str="sat",
    ):
        self.c=np.asarray(c, dtype=float).flatten()
        self.k1=float(k1)
        self.k2=float(k2)
        self.phi=float(phi)
    
    #State dimensions    
    @property
    def n(self)->int:
        return len(self.c)

    def _sat(self,s:np.ndarray):
        return np.clip(s/self.phi,-1,1)

    def _hyperbolic_tangent(self,s:np.ndarray):
        return np.tanh(s/self.phi)

    def _sigmoid(self,s:np.ndarray):
        return s/(np.linalg.norm)
    