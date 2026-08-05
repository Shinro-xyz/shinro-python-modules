from shinro.components import Controller
import numpy as np

# Model Predictive Path Integral (MPPI) controller
# optimal control algorithm, sampling based, information theoretic

class MPPI_Controller(Controller):
    def __init__(self,
        dynamics_fn,
        cost_fn,
        num_samples:int,
        temperature:float,
        dt: float,
        horizon:int,
        noise_sigma:np.ndarray,
        u_min,
        u_max,
    ):
        self.dynamics_fn=dynamics_fn
        self.cost_fn=cost_fn
        self.N=horizon
        self.K=num_samples
        self.dt=dt
        self.lam=temperature

        if noise_sigma is None:
            noise_sigma=np.array([0.5])

        self.noise_sigma=np.asarray(noise_sigma)
        self.