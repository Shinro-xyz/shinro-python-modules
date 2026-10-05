"""Extended Kalman filter for nonlinear state estimation.

Linearizes a nonlinear process and measurement model about the current
estimate and applies the Kalman predict-update cycle to the local model.
Unlike :class:`~shinro.estimators.kalman_filter.KalmanFilter`, the process
model is a continuous-time derivative ``f(x, u) -> dx/dt`` (the same contract
as :meth:`shinro.components.Plant.dynamics`), Euler-discretized at ``dt``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shinro.components import StateEstimator
from shinro.factories.registry import register_estimator
from shinro.utils.array_backend import ArrayBackend, NumpyBackend, parse_matrix


@dataclass(frozen=True)
class ExtendedKalmanFilterConfig:
    """Strict TOML schema for :class:`ExtendedKalmanFilter`.

    ``dynamics_fn`` / ``measurement_fn`` are callables and cannot be
    serialized to TOML: :meth:`ExtendedKalmanFilter.from_config` leaves them
    ``None`` and the caller injects them after construction. ``dt`` is
    optional — scenario builds inject it from the plant.
    """

    process_noise: list[float] | list[list[float]]
    measurement_noise: list[float] | list[list[float]]
    dt: float | None = None
    initial_state: list[float] | None = None
    name: str = "ekf"


@register_estimator("ExtendedKalmanFilter")
class ExtendedKalmanFilter(StateEstimator):
    """Extended Kalman filter for nonlinear state estimation.

    Estimates the state of a continuous-time nonlinear system

    .. math::

        \\dot{x} = f(x, u), \\qquad y = h(x)

    by linearizing :math:`f` and :math:`h` about the current estimate and
    running the standard Kalman predict-update cycle on the local model.

    **Conventions.** Everything is a flat vector — state :math:`(n_x,)`,
    control :math:`(n_u,)`, measurement :math:`(n_y,)` — matching the rest of
    the control stack. ``dynamics_fn(x, u) -> dx/dt`` is *continuous-time*:
    the filter Euler-discretizes it at ``dt``, so the discrete process
    Jacobian is :math:`F = I + dt \\cdot \\partial f/\\partial x`. The control
    Jacobian :math:`\\partial f/\\partial u` is never needed — the EKF assumes
    the applied input is known exactly, so it does not enter the covariance.

    **Predict** (local linearization about :math:`\\hat{x}`):

    .. math::

        x^- &= \\hat{x} + dt \\cdot f(\\hat{x}, u) \\\\
        F &= I + dt \\cdot \\frac{\\partial f}{\\partial x}\\bigg|_{\\hat{x}} \\\\
        P^- &= F P F^T + Q

    **Update** (linearization about :math:`x^-`):

    .. math::

        H &= \\frac{\\partial h}{\\partial x}\\bigg|_{x^-} \\\\
        S &= H P^- H^T + R \\\\
        K &= P^- H^T S^{-1} \\\\
        \\hat{x} &= x^- + K \\,(y - h(x^-)) \\\\
        P &= (I - K H) P^-

    The two Jacobians are computed by the backend's numerical Jacobian
    (:meth:`~shinro.utils.array_backend.ArrayBackend.jacobian` — central finite
    differences on numpy, autograd on torch), so ``dynamics_fn`` and
    ``measurement_fn`` are the only model inputs the filter needs.

    Args:
        dynamics_fn: Continuous-time process model ``f(x, u) -> dx/dt``.
            Receives and returns flat backend arrays. May be ``None`` only
            when built via :meth:`from_config`, to be injected later.
        measurement_fn: Measurement model ``h(x) -> y``. Receives and returns
            flat backend arrays. Same ``None`` rule as ``dynamics_fn``.
        dt: Integration / sample time step in seconds.
        Q: Process noise covariance (n_x, n_x).
        R: Measurement noise covariance (n_y, n_y).
        x0: Initial state estimate (n_x,). Defaults to zeros.
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        dynamics_fn: Callable[..., Any] | None,
        measurement_fn: Callable[..., Any] | None,
        dt: float,
        Q: Any,
        R: Any,
        x0: Any | None = None,
        backend: ArrayBackend | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.dynamics_fn = dynamics_fn
        self.measurement_fn = measurement_fn
        self.dt = dt
        self.Q = Q
        self.R = R
        self.n_x = Q.shape[0]
        self.n_y = R.shape[0]
        self.x_hat = self.bk.zeros(self.n_x) if x0 is None else self._flat(self.bk.copy(x0), self.n_x)
        self.P = self.bk.eye(self.n_x) * 0.1

    def _flat(self, x, n):
        """Coerce an ``(n,)`` or ``(n, 1)`` vector to flat ``(n,)``.

        Guards the covariance math against the silent ``(n,) - (n, 1)``
        broadcast that would otherwise turn a column-vector measurement into
        an ``(n, n)`` innovation.
        """
        return self.bk.reshape(x, (n,))

    def _callables(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        """Return ``(dynamics_fn, measurement_fn)``, raising if either is unset."""
        if self.dynamics_fn is None or self.measurement_fn is None:
            raise RuntimeError(
                "dynamics_fn and measurement_fn must be set before calling estimate(). "
                "A filter built with from_config starts with them None — inject them "
                "(ekf.dynamics_fn = ..., ekf.measurement_fn = ...) or construct directly."
            )
        return self.dynamics_fn, self.measurement_fn

    def _dynamics_jacobian(self, x, u):
        """Return :math:`F = \\partial f/\\partial x` at ``(x, u)``, shape (n_x, n_x)."""
        dynamics_fn, _ = self._callables()
        return self.bk.jacobian(lambda x_: self._flat(dynamics_fn(x_, u), self.n_x), x)

    def _measurement_jacobian(self, x):
        """Return :math:`H = \\partial h/\\partial x` at ``x``, shape (n_y, n_x)."""
        _, measurement_fn = self._callables()
        return self.bk.jacobian(lambda x_: self._flat(measurement_fn(x_), self.n_y), x)

    def estimate(self, measurement, control_input):
        """Run one predict-update cycle and return the posterior state estimate.

        See the class docstring for the equations. The Jacobians are the local
        linearization; the covariance update uses the same simple
        :math:`(I - KH)P^-` form as
        :class:`~shinro.estimators.kalman_filter.KalmanFilter` (not the Joseph form).

        Args:
            measurement: Observation vector :math:`y` (n_y,).
            control_input: Control vector :math:`u` applied at this step (n_u,).

        Returns:
            Posterior state estimate :math:`\\hat{x}` (n_x,).

        Raises:
            RuntimeError: If ``dynamics_fn`` or ``measurement_fn`` is unset
                (a filter built with :meth:`from_config` starts with both
                ``None`` and they must be injected first).
        """
        dynamics_fn, measurement_fn = self._callables()

        x = self._flat(self.x_hat, self.n_x)
        u = self._flat(control_input, control_input.shape[0])
        z = self._flat(measurement, self.n_y)

        # --- predict: Euler-discretize the continuous-time process model ---
        f = self._flat(dynamics_fn(x, u), self.n_x)
        x_pred = x + self.dt * f
        F = self.bk.eye(self.n_x) + self.dt * self._dynamics_jacobian(x, u)
        P_pred = F @ self.P @ F.T + self.Q

        # --- update: linearize the measurement model about the prediction ---
        H = self._measurement_jacobian(x_pred)
        z_pred = self._flat(measurement_fn(x_pred), self.n_y)
        S = H @ P_pred @ H.T + self.R
        K_gain = P_pred @ H.T @ self.bk.inv(S)

        self.x_hat = x_pred + K_gain @ (z - z_pred)
        self.P = (self.bk.eye(self.n_x) - K_gain @ H) @ P_pred
        return self.x_hat

    def reset(self, x0: Any | None = None):
        """Reset the filter to its initial state.

        Args:
            x0: Initial state estimate (n_x,). Defaults to zeros.
        """
        self.x_hat = self.bk.zeros(self.n_x) if x0 is None else self._flat(self.bk.copy(x0), self.n_x)
        self.P = self.bk.eye(self.n_x) * 0.1

    Config = ExtendedKalmanFilterConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an extended Kalman filter from a TOML config dict or :class:`ExtendedKalmanFilterConfig`.

        Config fields:
            process_noise: Diagonal Q weights (n_x,) or full Q matrix (n_x, n_x).
            measurement_noise: Diagonal R weights (n_y,) or full R matrix (n_y, n_y).
            dt: Time step. Required at runtime; injected from the plant in
                scenario builds.
            initial_state: Optional initial state estimate (n_x,). Defaults to zeros.

        The ``dynamics_fn`` / ``measurement_fn`` callables cannot be serialized
        to TOML. They are created as ``None`` and must be injected after
        construction:

        .. code-block:: python

            ekf = ExtendedKalmanFilter.from_config(config)
            ekf.dynamics_fn = my_f
            ekf.measurement_fn = my_h

        Args:
            config: TOML config dict or ExtendedKalmanFilterConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            ExtendedKalmanFilter instance.

        Raises:
            ValueError: If ``dt`` is None (standalone use requires it).
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        if cfg.dt is None:
            raise ValueError(
                "ExtendedKalmanFilter: dt is required — omit it only in scenario "
                "builds, where the plant's dt is injected"
            )
        Q = parse_matrix(bk, cfg.process_noise)
        R = parse_matrix(bk, cfg.measurement_noise)
        x0 = None if cfg.initial_state is None else bk.array(cfg.initial_state)
        return cls(
            dynamics_fn=None,
            measurement_fn=None,
            dt=cfg.dt,
            Q=Q,
            R=R,
            x0=x0,
            backend=bk,
        )
