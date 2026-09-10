from dataclasses import dataclass
from typing import Any

from shinro.components import StateEstimator
from shinro.factories.registry import register_estimator
from shinro.utils.array_backend import ArrayBackend, NumpyBackend, parse_matrix


@dataclass(frozen=True)
class KalmanFilterConfig:
    """Strict TOML schema for :class:`KalmanFilter`.

    ``dt`` / ``A_dynamics`` / ``B_dynamics`` are optional: scenario builds
    inject them from the plant (see :mod:`shinro.utils.linearization`);
    standalone use must supply ``B_dynamics`` or ``dt``.
    """

    process_noise: list[float] | list[list[float]]
    measurement_noise: list[float] | list[list[float]]
    dt: float | None = None
    A_dynamics: Any = None
    B_dynamics: Any = None
    C: Any = None
    D: Any = None
    name: str = "kalman"


@register_estimator("KalmanFilter")
class KalmanFilter(StateEstimator):
    """Discrete-time linear Kalman filter for optimal state estimation.

    Implements the predict-update cycle for a system of the form:

    .. math::

        x_{k+1} &= A x_k + B u_k + w_k, \\quad w_k \\sim \\mathcal{N}(0, Q) \\\\
        y_k &= C x_k + D u_k + v_k, \\quad v_k \\sim \\mathcal{N}(0, R)

    Tracks the posterior state estimate :math:`\\hat{x}` and error covariance
    :math:`P` through the standard Kalman filter equations.

    Uses column vectors :math:`(n, 1)` throughout (not flat :math:`(n,)`).

    Args:
        A: State transition matrix (n_x, n_x).
        B: Control input matrix (n_x, n_u).
        Q: Process noise covariance (n_x, n_x).
        R: Measurement noise covariance (n_y, n_y).
        C: Observation matrix (n_y, n_x). Defaults to identity.
        D: Feedthrough matrix (n_y, n_u). Defaults to zeros.
        x0: Initial state estimate (n_x, 1). Defaults to zeros.
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        A,
        B,
        Q,
        R,
        C: Any | None = None,
        D: Any | None = None,
        x0: Any | None = None,
        backend: ArrayBackend | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.C = self.bk.eye(A.shape[0]) if C is None else C
        self.D = self.bk.zeros((self.C.shape[0], B.shape[1])) if D is None else D

        self.x_hat = self.bk.zeros((A.shape[0], 1)) if x0 is None else self.bk.copy(x0)
        self.P = self.bk.eye(A.shape[0]) * 0.1

    def estimate(self, measurement, control_input):
        """Run one predict-update cycle and return the posterior state estimate.

        Implements the standard Kalman filter equations:

        1. Predict:
           :math:`x_{\\text{pred}} = A \\hat{x} + B u`
           :math:`P_{\\text{pred}} = A P A^T + Q`

        2. Update:
           :math:`K = P_{\\text{pred}} C^T (C P_{\\text{pred}} C^T + R)^{-1}`
           :math:`\\hat{x} = x_{\\text{pred}} + K (y - C x_{\\text{pred}} - D u)`
           :math:`P = (I - K C) P_{\\text{pred}}`

        Args:
            measurement: Observation vector (n_y, 1) from sensors.
            control_input: Control vector (n_u, 1) applied at this step.

        Returns:
            Posterior state estimate :math:`\\hat{x}` (n_x, 1).
        """
        x_pred = self.A @ self.x_hat + self.B @ control_input
        self.P = self.A @ self.P @ self.A.T + self.Q

        S = self.C @ self.P @ self.C.T + self.R
        K_gain = self.P @ self.C.T @ self.bk.inv(S)

        y_pred = self.C @ x_pred + self.D @ control_input
        innovations = measurement - y_pred

        self.x_hat = x_pred + K_gain @ innovations
        self.P = (self.bk.eye(self.A.shape[0]) - K_gain @ self.C) @ self.P

        return self.x_hat

    def reset(self, x0: Any | None = None):
        """Reset the filter to its initial state.

        Args:
            x0: Initial state estimate (n_x, 1). Defaults to zeros.
        """
        self.x_hat = self.bk.zeros((self.A.shape[0], 1)) if x0 is None else self.bk.copy(x0)
        self.P = self.bk.eye(self.A.shape[0]) * 0.1

    Config = KalmanFilterConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a Kalman filter from a TOML config dict or :class:`KalmanFilterConfig`.

        Config fields:
            process_noise: Diagonal Q weights (n_x,) or full Q matrix (n_x, n_x).
            measurement_noise: Diagonal R weights (n_y,) or full R matrix (n_y, n_y).
            dt: Time step — used to set B = dt * I unless B_dynamics is given.
            A_dynamics: Optional full A matrix (n_x, n_x). Defaults to I.
            B_dynamics: Optional full B matrix (n_x, n_u). Defaults to dt * I.
            C: Optional full C matrix (n_y, n_x). Defaults to I.
            D: Optional full D matrix (n_y, n_u). Defaults to zeros.

        Args:
            config: TOML config dict or KalmanFilterConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            KalmanFilter instance.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        Q = parse_matrix(bk, cfg.process_noise)
        n = Q.shape[0]
        R = parse_matrix(bk, cfg.measurement_noise)
        n_y = R.shape[0]
        A = bk.array(cfg.A_dynamics) if cfg.A_dynamics is not None else bk.eye(n)
        if cfg.B_dynamics is not None:
            B = bk.array(cfg.B_dynamics)
        elif cfg.dt is not None:
            B = cfg.dt * bk.eye(n)
        else:
            raise ValueError("KalmanFilter: no B_dynamics and no dt — standalone use requires one of them")
        return cls(
            A=A,
            B=B,
            Q=Q,
            R=R,
            C=bk.array(cfg.C) if cfg.C is not None else bk.eye(n),
            D=bk.array(cfg.D) if cfg.D is not None else bk.zeros((n_y, B.shape[1])),
            x0=bk.zeros((n, 1)),
            backend=bk,
        )
