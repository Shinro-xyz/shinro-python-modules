from typing import Optional

from components import Controller
from factories.registry import register_controller
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_controller("SMC")
class SlidingModeController(Controller):
    def __init__(
        self,
        c,
        k1: float,
        phi: float = 0.0,
        k2: float = 0.0,
        smoother: str = "sat",
        alpha: float = 0.0,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.c = self.bk.array(c).flatten()
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.phi = float(phi)
        self.alpha = float(alpha)

        SMOOTHERS = self._dict_boundaries()

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

    @property
    def n(self) -> int:
        return len(self.c)

    def _sat(self, s):
        return self.bk.clip(s / self.phi, -1.0, 1.0)

    def _tanh(self, s):
        return self.bk.tanh(s / self.phi)

    def _sigmoid(self, s):
        return s / (self.bk.abs(s) + self.phi)

    def _dict_boundaries(self):
        return {
            "sat": self._sat,
            "tanh": self._tanh,
            "sigmoid": self._sigmoid,
        }

    def _is_hurwitz(self):
        c_np = self.bk.to_numpy(self.c)
        poly = np.zeros(self.n + 1)
        poly[-1] = 1.0
        for i, ci in enumerate(c_np):
            poly[-(i + 2)] = ci
        roots = np.roots(poly)
        return all(np.real(r) < 0 for r in roots)

    def compute(self, x, f_x, g_x):
        x = self.bk.array(x).flatten()
        f_x = self.bk.array(f_x).flatten()
        g_x = self.bk.array(g_x)

        s = self.c @ x
        cf = self.c @ f_x
        cg = self.c @ g_x

        if self.phi > 0:
            smooth_s = self._smoother(s)
        else:
            smooth_s = self.bk.sign(s)

        s_dot_desired = -self.k1 * self.bk.abs(s) ** self.alpha * smooth_s

        cg_flat = self.bk.ravel(cg)
        if cg_flat.size == 1:
            cg_val = float(self.bk.to_numpy(cg_flat)[0])
            if abs(cg_val) < 1e-12:
                raise RuntimeError("c^T g(x) is near-zero — loss of controllability")
            u = self.bk.array([(s_dot_desired - cf) / cg_val])
        else:
            cg_np = self.bk.to_numpy(cg_flat).reshape(1, -1)
            rhs_np = np.array([float(self.bk.to_numpy(s_dot_desired - cf))])
            u_np, _, _, _ = np.linalg.lstsq(cg_np, rhs_np, rcond=None)
            u = self.bk.from_numpy(u_np.flatten())

        return u

    def reset(self):
        pass

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        return cls(
            c=bk.array(config["c"]),
            k1=config["k1"],
            phi=config.get("phi", 0.0),
            k2=config.get("k2", 0.0),
            smoother=config.get("smoother", "sat"),
            alpha=config.get("alpha", 0.0),
            backend=bk,
        )
