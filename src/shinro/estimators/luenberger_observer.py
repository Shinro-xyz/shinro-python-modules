from dataclasses import dataclass
from typing import Any

from shinro.components import StateEstimator
from shinro.factories.registry import register_estimator
from shinro.utils.array_backend import ArrayBackend, NumpyBackend, parse_matrix


@dataclass(frozen=True)
class LuenbergerObserverConfig:
    """Strict TOML schema for :class:`LuenbergerObserver`.

    ``dt`` / ``A_dynamics`` / ``B_dynamics`` are optional: scenario builds
    inject them from the plant (see :mod:`shinro.utils.linearization`);
    standalone use must supply ``B_dynamics`` or ``dt``.
    """

    observer_gain: list[float] | list[list[float]]
    dt: float | None = None
    A_dynamics: Any = None
    B_dynamics: Any = None
    C: Any = None
    D: Any = None
    name: str = "luenberger"


@register_estimator("LuenbergerObserver")
class LuenbergerObserver(StateEstimator):
    """Luenberger observer for deterministic linear state estimation.

    Implements the discrete-time observer dynamics:

    .. math::

        \\hat{x}_{k+1} = A \\hat{x}_k + B u_k + L (y_k - C \\hat{x}_k - D u_k)

    where L is the observer gain chosen to place the eigenvalues of
    :math:`(A - LC)` inside the unit circle for stable estimation.

    Unlike the Kalman filter, the Luenberger observer uses a fixed gain
    and does not assume noise statistics. No matrix inverses are needed
    at runtime — just three matrix-vector multiplies.

    Uses column vectors :math:`(n, 1)` throughout (not flat :math:`(n,)`).

    Args:
        A: State transition matrix (n_x, n_x).
        B: Control input matrix (n_x, n_u).
        observer_gain: Observer gain matrix L (n_x, n_y). Must place
            eigenvalues of (A - LC) inside the unit circle.
        C: Output matrix (n_y, n_x). Defaults to identity.
        D: Feedthrough matrix (n_y, n_u). Defaults to zeros.
        x0: Initial state estimate (n_x, 1). Defaults to zeros.
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        A,
        B,
        observer_gain,
        C: Any | None = None,
        D: Any | None = None,
        x0: Any | None = None,
        backend: ArrayBackend | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.A = A
        self.B = B
        self.C = self.bk.eye(A.shape[0]) if C is None else C
        self.D = self.bk.zeros((self.C.shape[0], B.shape[1])) if D is None else D
        self.L = observer_gain
        self.x_hat = self.bk.zeros((A.shape[0], 1)) if x0 is None else self.bk.copy(x0)

    def estimate(self, measurement, control_input):
        """Perform one step of state estimation.

        Computes the predicted state from the dynamics, calculates the
        innovation (measurement residual), and corrects the prediction
        using the observer gain:

        .. math::

            \\hat{x}_{k+1} = A \\hat{x}_k + B u_k + L (y_k - C (A \\hat{x}_k + B u_k) - D u_k)

        Args:
            measurement: Output measurement :math:`y_k` (n_y, 1).
            control_input: Control input :math:`u_k` (n_u, 1).

        Returns:
            Updated state estimate :math:`\\hat{x}_{k+1}` (n_x, 1).
        """
        x_pred = self.A @ self.x_hat + self.B @ control_input
        innovations = measurement - (self.C @ x_pred + self.D @ control_input)
        self.x_hat = x_pred + self.L @ innovations
        return self.x_hat

    def reset(self, x0: Any | None = None):
        """Reset the observer to its initial state.

        Args:
            x0: Initial state estimate (n_x, 1). Defaults to zeros.
        """
        self.x_hat = self.bk.zeros((self.A.shape[0], 1)) if x0 is None else self.bk.copy(x0)

    Config = LuenbergerObserverConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a Luenberger observer from a TOML config dict or :class:`LuenbergerObserverConfig`.

        Config fields:
            observer_gain: Diagonal gain weights (n_x,) or full gain matrix (n_x, n_y).
            dt: Time step — used to set B = dt * I unless B_dynamics is given.
            A_dynamics: Optional full A matrix (n_x, n_x). Defaults to I.
            B_dynamics: Optional full B matrix (n_x, n_u). Defaults to dt * I.
            C: Optional full C matrix (n_y, n_x). Defaults to I.
            D: Optional full D matrix (n_y, n_u). Defaults to zeros.

        Args:
            config: TOML config dict or LuenbergerObserverConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            LuenbergerObserver instance.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        gain = parse_matrix(bk, cfg.observer_gain)
        n = gain.shape[0]
        A = bk.array(cfg.A_dynamics) if cfg.A_dynamics is not None else bk.eye(n)
        if cfg.B_dynamics is not None:
            B = bk.array(cfg.B_dynamics)
        elif cfg.dt is not None:
            B = cfg.dt * bk.eye(n)
        else:
            raise ValueError("LuenbergerObserver: no B_dynamics and no dt — standalone use requires one of them")
        return cls(
            A=A,
            B=B,
            observer_gain=gain,
            C=bk.array(cfg.C) if cfg.C is not None else bk.eye(n),
            D=bk.array(cfg.D) if cfg.D is not None else bk.zeros((n, B.shape[1])),
            x0=bk.zeros((n, 1)),
            backend=bk,
        )
