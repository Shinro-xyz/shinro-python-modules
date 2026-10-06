from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shinro.components import StateEstimator
from shinro.estimators.model_call import accepts_backend, call_model
from shinro.factories.registry import register_estimator
from shinro.utils.array_backend import ArrayBackend, NumpyBackend, parse_matrix


@dataclass(frozen=True)
class UnscentedKFConfig:
    """Strict TOML schema for :class:`UnscentedKF`.

    ``dynamics_fn`` / ``measurement_fn`` are callables and cannot be serialized
    to TOML: :meth:`UnscentedKF.from_config` leaves them ``None`` and the caller
    injects them after construction. ``dt`` is optional — scenario builds inject
    it from the plant. ``measurement_matrix`` (a linear ``C``) is
    TOML-serializable and supplies ``h`` without a callable, which is how a
    config-only filter gets a measurement model.
    """

    process_noise: list[float] | list[list[float]]
    measurement_noise: list[float] | list[list[float]]
    dt: float | None = None
    initial_state: list[float] | None = None
    measurement_matrix: list[list[float]] | None = None
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    name: str = "ukf"


@register_estimator("UnscentedKF")
class UnscentedKF(StateEstimator):
    def __init__(
        self,
        dynamics_fn: Callable[...,Any]| None,
        measurement_fn: Callable[...,Any]| None,
        dt: float,
        alpha: float,
        beta:float,
        kappa:float,
        Q: Any,
        R:Any,
        x0: Any | None= None,
        backend: ArrayBackend | None= None,
        measurement_matrix: Any | None = None,
        batched: bool | None = None,
        ):
            self.bk= backend or NumpyBackend()
            self.dynamics_fn= dynamics_fn
            self.measurement_fn=measurement_fn
            self.batched= batched
            self.dt= dt
            self.alpha=alpha
            self.beta= beta
            self.kappa=kappa
            self.Q=Q
            self.R=R

            self.nx=Q.shape[0] #process noise
            self.ny= R.shape[0] #measurement noise

            #optional linear measurement model: h(x) = C x, no callable needed
            self.C= None if measurement_matrix is None else self.bk.array(measurement_matrix)
            if self.C is not None and tuple(self.C.shape) != (self.ny, self.nx):
                raise ValueError(
                    f"measurement_matrix shape {tuple(self.C.shape)} does not match R's {self.ny} "
                    f"measurements x Q's {self.nx} states"
                )

            # initializing the kalman algo values
            self.x_hat= self.bk.zeros(self.nx) if x0 is None else self.bk.ravel(self.bk.copy(x0))
            self.P=self.bk.eye(self.nx)*0.1

            #calculate the weights in the start
            self._weights_sigmapoints()

    def _sigma_points(self, x, P):
        n=self.nx
        L_covar=self.bk.cholesky((self.lamb+n)*P) #cholesky factorization (n,n)

        #sigma points as rows, built by concat (no index assignment) so the same
        #construction stays traceable once a cholesky VM op exists
        x_row=self.bk.reshape(x,(1,n))
        perturb=self.bk.concat([L_covar.T, -L_covar.T], axis=0) #(2n,n): +cols then -cols
        return self.bk.concat([x_row, x_row+perturb], axis=0)   #(2n+1,n)

    def _weights_sigmapoints(self):
        # lambda must be defined here: the weights are computed in __init__,
        # before any sigma-point pass could set it.
        self.lamb= self.alpha**2*(self.nx+self.kappa)-self.nx # find the lambda

        #one weight per sigma point: the mean plus 2n spread points
        w_mean=self.bk.zeros((2*self.nx+1,))
        w_covar=self.bk.zeros((2*self.nx+1,))

        w_mean[0]=self.lamb/(self.nx+self.lamb)
        w_covar[0]=w_mean[0]+(1-self.alpha**2+self.beta)
        for i in range(1, 2*self.nx+1):
            w_covar[i]=1/(2*(self.nx+self.lamb))
            w_mean[i]=1/(2*(self.nx+self.lamb))

        self.w_mean=w_mean
        self.w_covar=w_covar

    def _weighted_mean(self, points):
        """Weighted mean of a stack of points, shape (L, d) -> (d,).

        Expressed as a matmul ``w(1,L) @ points(L,d)`` rather than a reduction,
        so it stays inside the VM's op set when the filter is lowered. The
        weights are baked constants: they are reshaped on the array (numpy /
        torch), which the tracer lifts to a ``const`` node — routing a raw array
        through ``bk.reshape`` would hit the trace backend and cannot infer ``-1``.
        """
        w=self.w_mean.reshape(1,self.w_mean.shape[0])
        return self.bk.ravel(w @ points)

    def _weighted_cov(self, points, mean):
        """Weighted covariance of a stack of points about ``mean``, (L, d) -> (d, d).

        The mean is lifted to a ``(1, d)`` row before subtracting: the tracer
        requires matching ranks (no rank-2 minus rank-1 broadcast), unlike numpy.
        """
        w=self.w_covar.reshape(self.w_covar.shape[0],1)
        d=points-self.bk.reshape(mean,(1,points.shape[1]))
        return (w*d).T @ d

    def _cross_cov(self, x_points, x_mean, z_points, z_mean):
        """Weighted cross-covariance between two point sets, (d_x, d_z).

        Both means are lifted to ``(1, d)`` rows for the same reason as
        :meth:`_weighted_cov`.
        """
        w=self.w_covar.reshape(self.w_covar.shape[0],1)
        dx=x_points-self.bk.reshape(x_mean,(1,x_points.shape[1]))
        dz=z_points-self.bk.reshape(z_mean,(1,z_points.shape[1]))
        return (w*dx).T @ dz

    def _dynamics(self, x, u):
        """Evaluate the process model, raising when none was supplied."""
        if self.dynamics_fn is None:
            raise RuntimeError("UnscentedKF: dynamics_fn must be set before calling estimate()")
        return call_model(self.dynamics_fn, x, u, bk=self.bk)

    def _measurement(self, x):
        """Evaluate :math:`h(x)` — the linear ``C`` when configured, else the callable."""
        if self.C is not None:
            return self.C @ x
        if self.measurement_fn is None:
            raise RuntimeError(
                "UnscentedKF: measurement_fn or measurement_matrix must be set before calling estimate()"
            )
        return call_model(self.measurement_fn, x, bk=self.bk)

    def _batchable(self, fn):
        """Whether a model can consume the whole sigma-point batch in one call.

        Plants and compiled models accept the ``(N, n)`` batch contract (and a
        ``bk``); a plain user lambda takes only ``(n,)``, so it is evaluated row
        by row. ``self.batched`` forces the choice when set (``False`` is the
        escape hatch for a ``bk``-aware model that is nonetheless not
        batch-capable).
        """
        if self.batched is not None:
            return self.batched
        return fn is not None and accepts_backend(fn)

    def _dynamics_batch(self, chi, u):
        """Evaluate :math:`f` over every sigma point, ``(L, n) -> (L, n)``.

        One batched call when the model is batch-capable, else a row-by-row
        fallback so a single-point lambda keeps working. Batched is what lets a
        trace lower as a single ``f`` subgraph instead of ``L`` copies.
        """
        fn=self.dynamics_fn
        if fn is None:
            raise RuntimeError("UnscentedKF: dynamics_fn must be set before calling estimate()")
        if self._batchable(fn):
            return self.bk.reshape(call_model(fn, chi, u, bk=self.bk), (chi.shape[0], self.nx))
        rows=[self.bk.ravel(call_model(fn, chi[i], u, bk=self.bk)) for i in range(chi.shape[0])]
        return self.bk.stack(rows)

    def _measurement_batch(self, chi):
        """Evaluate :math:`h` over every sigma point, ``(L, n) -> (L, n_y)``."""
        if self.C is not None:
            return chi @ self.C.T
        fn=self.measurement_fn
        if fn is None:
            raise RuntimeError(
                "UnscentedKF: measurement_fn or measurement_matrix must be set before calling estimate()"
            )
        if self._batchable(fn):
            return self.bk.reshape(call_model(fn, chi, bk=self.bk), (chi.shape[0], self.ny))
        rows=[self.bk.ravel(call_model(fn, chi[i], bk=self.bk)) for i in range(chi.shape[0])]
        return self.bk.stack(rows)

    def estimate(self, measurement,control_input):
        #flatten so a (n_u,1)/(n_y,1) column cannot broadcast into an (n,n) step
        u=self.bk.ravel(control_input)
        z=self.bk.ravel(measurement)

        # --- predict: push the whole sigma-point batch through the model ---
        #dynamics_fn follows the EKF contract: continuous dx/dt, Euler-stepped
        chi=self._sigma_points(self.x_hat, self.P)
        chi_pred=chi+self.dt*self._dynamics_batch(chi, u)

        x_pred=self._weighted_mean(chi_pred)
        P_pred=self._weighted_cov(chi_pred, x_pred)+self.Q

        # --- update: redraw sigma points around the prediction ---
        chi_upd=self._sigma_points(x_pred, P_pred)
        z_sigma=self._measurement_batch(chi_upd)

        z_pred=self._weighted_mean(z_sigma)
        S=self._weighted_cov(z_sigma, z_pred)+self.R
        Pxz=self._cross_cov(chi_upd, x_pred, z_sigma, z_pred)

        K_gain=Pxz @ self.bk.inv(S)
        self.x_hat=x_pred+K_gain @ (z-z_pred)
        self.P=P_pred-K_gain @ S @ K_gain.T
        return self.x_hat

    def reset(self, x0: Any | None = None):
        """Reset the filter to its initial state.

        Clears the posterior estimate and covariance back to their construction
        values (:math:`\\hat{x}=0`, :math:`P=0.1\\,I`) so the filter can be reused
        on a new run. Mirrors :meth:`ExtendedKalmanFilter.reset`.

        Args:
            x0: Initial state estimate (n_x,). Defaults to zeros.
        """
        self.x_hat= self.bk.zeros(self.nx) if x0 is None else self.bk.ravel(self.bk.copy(x0))
        self.P=self.bk.eye(self.nx)*0.1

    def attach_plant(self, plant: Any) -> None:
        """Inject the plant's process model into a filter that has none.

        The compiled/served path hands the UKF the plant so its ``dynamics_fn``
        is the plant's model — written with the rank helpers and an optional
        ``bk``, so it traces and lowers — instead of a hand-wired lambda. Only an
        **unset** ``dynamics_fn`` is filled, so an explicit injection wins.

        The measurement side is not touched: plants carry no measurement model
        today, so a config-only UKF takes ``h`` from its ``measurement_matrix``.

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
                f"has no dynamics_fn — inject one explicitly (ukf.dynamics_fn = ...)."
            )
        self.dynamics_fn = dynamics

    Config = UnscentedKFConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an unscented Kalman filter from a TOML config dict or :class:`UnscentedKFConfig`.

        The ``dynamics_fn`` / ``measurement_fn`` callables cannot be serialized
        to TOML: they are created as ``None`` and must be injected after
        construction (``ukf.dynamics_fn = ...``), or filled from a plant. A
        config may instead carry a linear ``measurement_matrix`` (``C``), in which
        case ``measurement_fn`` is never needed.

        Args:
            config: TOML config dict or UnscentedKFConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            UnscentedKF instance.

        Raises:
            ValueError: If ``dt`` is None (standalone use requires it).
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        if cfg.dt is None:
            raise ValueError(
                "UnscentedKF: dt is required — omit it only in scenario builds, "
                "where the plant's dt is injected"
            )
        Q = parse_matrix(bk, cfg.process_noise)
        R = parse_matrix(bk, cfg.measurement_noise)
        x0 = None if cfg.initial_state is None else bk.array(cfg.initial_state)
        return cls(
            dynamics_fn=None,
            measurement_fn=None,
            dt=cfg.dt,
            alpha=cfg.alpha,
            beta=cfg.beta,
            kappa=cfg.kappa,
            Q=Q,
            R=R,
            x0=x0,
            backend=bk,
            measurement_matrix=cfg.measurement_matrix,
        )
