from typing import Any
from shinro.components import StateEstimator
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.factories.registry import register_estimator

@register_estimator("ExtendedKalmanFilter")
class ExtendedKalmanFilter(StateEstimator):
    def __init__(self,dynamics_fn:Any,measurement_fn: Any, dt: float, Q, R, x0, backend: ArrayBackend | None=None):
        self.dynamics_fn=dynamics_fn
        self.measurement_fn=measurement_fn
        self.dt=dt
        self.bk= backend if backend is not None else NumpyBackend
        self.Q=Q
        self.R=R
        self.x0=x0 #feed a starting point for estimation

        # define x_hat, P_hat
        self.x_hat=self.bk.copy(x0)
        self.P=self.bk.eye(x0.shape[0])*0.1

    # finding the jacobian of dynamics, f(x,u)
    def _dynamics_jacobian(self,x,u):
        # assuming col vector shape, (n_x,1)
       return self.bk.jacobian(lambda x1: self.dynamics_fn(x1,u),x)

    # finding the jacobian of measurement function, h(x)
    def _measurement_jacobian(self,x):
        return self.bk.jacobian(self.measurement_fn, x)

    def estimate(self,measurement,control_input):
        # predict loop
        x_pred= self.dynamics_fn(self.x_hat,control_input)
        F= self._dynamics_jacobian(self.x_hat, control_input)
        self.P= F@self.P@F.T+self.Q

        #update loop
        pred_measurement= self.measurement_fn(self.x_hat)
        H= self._measurement_jacobian(self.x_hat)
        meas_innov= measurement-pred_measurement
        S=H@self.P@H.T+self.R
        K_k= self.P@H.T@self.bk.inv(S)

        self.x_hat= self.x_hat+K_k@meas_innov
        self.P=(self.bk.eye(self.P.shape[0])-K_k@H)@self.P
        return self.x_hat

    def reset(self):
        self.x_hat=self.bk.zeros(self.x0.shape[0])
        self.P=self.bk.eye(self.x0.shape[0])*0.1