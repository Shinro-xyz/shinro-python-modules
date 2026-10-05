"""Extended Kalman filter for nonlinear state estimation.

Linearizes a nonlinear process and measurement model about the current
estimate and applies the Kalman predict-update cycle to the local model.
Unlike :class:`~shinro.estimators.kalman_filter.KalmanFilter`, the process
model is a continuous-time derivative ``f(x, u) -> dx/dt`` (the same contract
as :meth:`shinro.components.Plant.dynamics`), Euler-discretized at ``dt``.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

from shinro.components import StateEstimator
from shinro.factories.registry import register_estimator
from shinro.utils.array_backend import ArrayBackend, NumpyBackend, parse_matrix


@cache
def _accepts_backend(fn: Any) -> bool:
    """Whether a model callable takes a backend (a ``bk`` parameter or ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / C callables expose no signature
        return False
    if "bk" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _call_model(fn: Any, *args: Any, bk: Any) -> Any:
    """Call a model callable, passing the backend when its signature accepts one.

    ``Plant.dynamics`` (and any model written for the compiled path) takes an
    optional ``bk`` — that is how a traced component routes the plant's ops
    through its :class:`~shinro.codegen.trace_backend.TraceBackend` instead of the
    plant's concrete backend. A plain user lambda takes only the model arguments.
    Passing ``bk`` only when accepted keeps both conventions working.
    """
    if _accepts_backend(fn):
        return fn(*args, bk=bk)
    return fn(*args)


@dataclass(frozen=True)
class ExtendedKalmanFilterConfig:
    """Strict TOML schema for :class:`ExtendedKalmanFilter`.

    ``dynamics_fn`` / ``measurement_fn`` are callables and cannot be
    serialized to TOML: :meth:`ExtendedKalmanFilter.from_config` leaves them
    ``None`` and the caller injects them after construction. ``dt`` is
    optional — scenario builds inject it from the plant. ``measurement_matrix``
    (a linear ``C``) is TOML-serializable and supplies ``h``/``H`` without a
    callable, which is how a compiled scenario gets a measurement model.
    """

    process_noise: list[float] | list[list[float]]
    measurement_noise: list[float] | list[list[float]]
    dt: float | None = None
    initial_state: list[float] | None = None
    measurement_matrix: list[list[float]] | None = None
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
    differences on numpy, autograd on torch; on the tracing backend it lowers as
    graph nodes via :meth:`~shinro.codegen.trace_backend.TraceBackend.jacobian`),
    so ``dynamics_fn`` and ``measurement_fn`` are the only model inputs the filter
    needs. A **linear** measurement model may be given as ``measurement_matrix``
    instead, which yields the exact ``H = C`` with no differentiation.

    A model callable may accept an optional ``bk`` (as
    :meth:`shinro.components.Plant.dynamics` does). The filter passes its own
    backend when the signature accepts one, which is what lets a traced filter
    route the plant's ops through its ``TraceBackend``; plain two-argument
    callables keep working.

    Args:
        dynamics_fn: Continuous-time process model ``f(x, u) -> dx/dt``.
            Receives and returns flat backend arrays. May be ``None`` only
            when built via :meth:`from_config`, to be injected later.
        measurement_fn: Measurement model ``h(x) -> y``. Receives and returns
            flat backend arrays. May be ``None`` when ``measurement_matrix``
            supplies a linear model instead, or (via :meth:`from_config`) to be
            injected later.
        dt: Integration / sample time step in seconds.
        Q: Process noise covariance (n_x, n_x).
        R: Measurement noise covariance (n_y, n_y).
        x0: Initial state estimate (n_x,). Defaults to zeros.
        backend: Array backend. Defaults to NumpyBackend.
        measurement_matrix: Optional linear measurement matrix ``C`` (n_y, n_x),
            making ``h(x) = C x`` and the measurement Jacobian exactly ``C`` — no
            callable and no finite differences needed.
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
        measurement_matrix: Any | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.dynamics_fn = dynamics_fn
        self.measurement_fn = measurement_fn
        self.dt = dt
        self.Q = Q
        self.R = R
        self.n_x = Q.shape[0]
        self.n_y = R.shape[0]
        self.C = None if measurement_matrix is None else self.bk.array(measurement_matrix)
        if self.C is not None and tuple(self.C.shape) != (self.n_y, self.n_x):
            raise ValueError(
                f"measurement_matrix shape {tuple(self.C.shape)} does not match R's {self.n_y} "
                f"measurements x Q's {self.n_x} states"
            )
        self.x_hat = self.bk.zeros(self.n_x) if x0 is None else self._flat(self.bk.copy(x0), self.n_x)
        self.P = self.bk.eye(self.n_x) * 0.1

    def _flat(self, x, n):
        """Coerce an ``(n,)`` or ``(n, 1)`` vector to flat ``(n,)``.

        Guards the covariance math against the silent ``(n,) - (n, 1)``
        broadcast that would otherwise turn a column-vector measurement into
        an ``(n, n)`` innovation.
        """
        return self.bk.reshape(x, (n,))

    _MISSING_MODEL = (
        "dynamics_fn and measurement_fn (or a measurement_matrix) must be set before calling "
        "estimate(). A filter built with from_config starts with them None — inject them "
        "(ekf.dynamics_fn = ..., ekf.measurement_fn = ...), give the config a measurement_matrix, "
        "or construct directly."
    )

    def _dynamics_fn(self) -> Callable[..., Any]:
        """The process model, raising a clear error when it was never injected."""
        if self.dynamics_fn is None:
            raise RuntimeError(self._MISSING_MODEL)
        return self.dynamics_fn

    def _measurement_fn(self) -> Callable[..., Any]:
        """The measurement callable, raising when neither it nor a ``C`` was supplied."""
        if self.measurement_fn is None:
            raise RuntimeError(self._MISSING_MODEL)
        return self.measurement_fn

    def _flat_or_batch(self, out):
        """Normalize a model result: flatten a lone vector, pass a probe batch through.

        A finite-difference probe calls the model with a leading probe axis, so a
        batch-capable model returns ``(n_probes, m)`` and must not be squeezed; a
        single evaluation returns ``(m,)`` (or ``(m, 1)``) and is flattened.
        """
        if len(out.shape) == 1:
            return out
        if len(out.shape) == 2 and out.shape[1] == 1:
            return self.bk.ravel(out)
        return out

    def _dynamics(self, x, u):
        """Evaluate :math:`f(x, u)` through the filter's backend."""
        return self._flat(_call_model(self._dynamics_fn(), x, u, bk=self.bk), self.n_x)

    def _measurement(self, x):
        """Evaluate :math:`h(x)` — the linear ``C`` when configured, else the callable."""
        if self.C is not None:
            return self.C @ x
        return self._flat(_call_model(self._measurement_fn(), x, bk=self.bk), self.n_y)

    def _dynamics_jacobian(self, x, u):
        """Return :math:`F = \\partial f/\\partial x` at ``(x, u)``, shape (n_x, n_x)."""
        dynamics_fn = self._dynamics_fn()
        return self.bk.jacobian(lambda x_: self._flat_or_batch(_call_model(dynamics_fn, x_, u, bk=self.bk)), x)

    def _measurement_jacobian(self, x):
        """Return :math:`H = \\partial h/\\partial x` at ``x``, shape (n_y, n_x).

        With a ``measurement_matrix`` this is exactly ``C`` — no callable, no
        differentiation (and no graph nodes when lowered).
        """
        if self.C is not None:
            return self.C
        measurement_fn = self._measurement_fn()
        return self.bk.jacobian(lambda x_: self._flat_or_batch(_call_model(measurement_fn, x_, bk=self.bk)), x)

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
            RuntimeError: If the model is unset — ``dynamics_fn`` missing, or
                neither ``measurement_fn`` nor ``measurement_matrix`` supplied
                (a filter built with :meth:`from_config` starts with both
                callables ``None`` and they must be injected first).
        """
        dynamics_fn = self._dynamics_fn()  # validate up front, before any partial work

        x = self._flat(self.x_hat, self.n_x)
        u = self._flat(control_input, control_input.shape[0])
        z = self._flat(measurement, self.n_y)

        # --- predict: Euler-discretize the continuous-time process model ---
        f = self._flat(_call_model(dynamics_fn, x, u, bk=self.bk), self.n_x)
        x_pred = x + self.dt * f
        F = self.bk.eye(self.n_x) + self.dt * self._dynamics_jacobian(x, u)
        P_pred = F @ self.P @ F.T + self.Q

        # --- update: linearize the measurement model about the prediction ---
        H = self._measurement_jacobian(x_pred)
        z_pred = self._measurement(x_pred)
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

    def attach_plant(self, plant: Any) -> None:
        """Inject the plant's process model into a filter that has none.

        The compiled/served path hands the EKF the plant so its ``dynamics_fn``
        is the plant's model — written with the rank helpers and an optional
        ``bk``, so it traces and lowers — instead of a hand-wired lambda. Only an
        **unset** ``dynamics_fn`` is filled, so an explicit injection wins.

        The measurement side is not touched: plants carry no measurement model
        today, so a compiled EKF takes ``h`` from its config's ``measurement_matrix``.

        Args:
            plant: The plant whose :meth:`~shinro.components.Plant.dynamics` to use.

        Raises:
            ValueError: If the plant exposes no dynamics and none was injected.
        """
        if self.dynamics_fn is not None:
            return
        dynamics = getattr(plant, "dynamics", None)
        if dynamics is None or not callable(dynamics):
            raise ValueError(
                f"attach_plant: {type(plant).__name__} exposes no dynamics(), and the filter "
                f"has no dynamics_fn — inject one explicitly (ekf.dynamics_fn = ...)."
            )
        self.dynamics_fn = dynamics

    Config = ExtendedKalmanFilterConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an extended Kalman filter from a TOML config dict or :class:`ExtendedKalmanFilterConfig`.

        Config fields:
            process_noise: Diagonal Q weights (n_x,) or full Q matrix (n_x, n_x).
            measurement_noise: Diagonal R weights (n_y,) or full R matrix (n_y, n_y).
            dt: Time step. Required at runtime; injected from the plant in
                scenario builds.
            measurement_matrix: Optional linear measurement matrix ``C`` (n_y, n_x);
                when set, ``h(x) = C x`` and ``H = C`` exactly, with no callable.
            initial_state: Optional initial state estimate (n_x,). Defaults to zeros.

        The ``dynamics_fn`` / ``measurement_fn`` callables cannot be serialized
        to TOML. They are created as ``None`` and must be injected after
        construction:

        .. code-block:: python

            ekf = ExtendedKalmanFilter.from_config(config)
            ekf.dynamics_fn = my_f
            ekf.measurement_fn = my_h

        A config may instead carry a linear ``measurement_matrix`` (``C``), in which
        case ``measurement_fn`` is never needed — the compiled path takes this route,
        since a plant has no measurement callable to inject.

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
            measurement_matrix=cfg.measurement_matrix,
        )
