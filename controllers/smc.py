## sliding mode controller, SMC implementation with state space form

## todo for agent:
# 1. make it backend agnostic and
# 2. toml configuration,
# 3. registry patterns

from re import X

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

        SMOOTHERS= self._dict_boundaries()

        if smoother not in SMOOTHERS:
            raise ValueError(f"Unknown smoother '{smoother}'. Options: {list(SMOOTHERS)}")

        if not self._is_hurwitz():
                    raise ValueError(
                        "Sliding surface coefficients are not Hurwitz. "
                        "The polynomial c1 + c2 p + ... + cn p^{n-1} must have "
                        "all roots with negative real parts."
                    )

        self._smoother = SMOOTHERS[smoother]
        self._smoother_name = smoother

    #State dimensions
    @property
    def n(self)->int:
        return len(self.c)

    def _sat(self,s:np.ndarray):
        return np.clip(s/self.phi,-1,1)

    def _tanh(self,s:np.ndarray):
        return np.tanh(s/self.phi)

    def _sigmoid(self,s:np.ndarray):
        return s/(np.abs(s)+self.phi)

    def _dict_boundaries(self):
        SMOOTHER={"sat":self._sat,
                "tanh":self._tanh,
                "sigmoid":self._sigmoid}

        return SMOOTHER

    def _is_hurwitz(self):
        poly=np.zeros(self.n+1)
        poly[-1]=1.0
        for i, ci in enumerate(self.c):
            poly[-(i + 2)] = ci
        roots = np.roots(poly)
        return all(np.real(r) < 0 for r in roots)

    def compute(
        self,
        x:np.ndarray,
        f_x:np.ndarray,
        g_x:np.ndarray
    ):
        x=np.asarray(x).flatten()
        f_x=np.asarray(f_x).flatten()
        g_x=np.asarray(g_x)

        # calculating the sliding surface
        s=self.c@x

        # calulating the sdot
        s_f=self.c@f_x
        s_g=self.c@g_x
        
        
        
    