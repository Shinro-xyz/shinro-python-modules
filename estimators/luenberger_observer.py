from typing import Optional
from components import StateEstimator
from factories.registry import register_estimator
from utils.array_backend import ArrayBackend, NumpyBackend


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
        C: Optional = None,
        D: Optional = None,
        x0: Optional = None,
        backend: Optional[ArrayBackend] = None,
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

    def reset(self, x0: Optional = None):
        """Reset the observer to its initial state.

        Args:
            x0: Initial state estimate (n_x, 1). Defaults to zeros.
        """
        self.x_hat = self.bk.zeros((self.A.shape[0], 1)) if x0 is None else self.bk.copy(x0)

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create a Luenberger observer from a TOML config dict.

        Config fields:
            observer_gain: List of diagonal gain weights (n_x,).
            dt: Time step — used to set B = dt * I.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            LuenbergerObserver instance.
        """
        bk = backend or NumpyBackend()
        gain = bk.diag(config["observer_gain"])
        n = gain.shape[0]
        return cls(
            A=bk.eye(n),
            B=config["dt"] * bk.eye(n),
            observer_gain=gain,
            C=bk.eye(n),
            D=bk.zeros((n, n)),
            x0=bk.zeros((n, 1)),
            backend=bk,
        )
