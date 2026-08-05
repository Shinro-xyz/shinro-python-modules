import numpy as np

from shinro.components import Controller

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
        self.N=num_samples
        self.K=horizon
        self.dt=dt
        self.lam=temperature

        if noise_sigma is None:
            noise_sigma=np.array([0.5])

        self.noise_sigma=np.asarray(noise_sigma)
        self.u_min=np.asarray(u_min) if u_min is not None else None
        self.u_max=np.asarray(u_max) if u_max is not None else None

        # finding the necessary shapes for making the controller
        self.D_u= len(self.noise_sigma)

        #nominal control sequence over K steps horizon: (K, D_u) shape

        self.u=np.zeros((self.K, self.D_u))

    def compute(self, x0: np.ndarray):

        D_x=len(x0)
        # generating gaussian noise
        epsilon=np.random.normal(loc=0.0,scale=self.noise_sigma, size=(self.N,self.K,self.D_u))

        # adding perturbations to nominal control--> N,K,D_u
        v=np.expand_dims(self.u, axis=0)+epsilon

        # control limits
        if self.u_min is not None or self.u_max is not None:
            v=np.clip(v,self.u_min,self.u_max)

        #parallel N rollouts
        x_current=np.tile(x0,(self.N,1))
        costs= np.zeros(self.N)

        # stepping forward in time through K prediction horizon
        for k in range(self.K):
            u_k=v[:,k,:]
            costs+=self.cost_fn(x_current,u_k)
            inv_var_weighted_u=self.u[k]/(self.noise_sigma**2)
            control_penalty=self.lam*np.sum(inv_var_weighted_u*epsilon[:,k,:],axis=1)
            costs+=control_penalty

            #advance dynamics forward
            x_current=self.dynamics_fn(x_current,u_k,self.dt)

        #terminal costs
        costs+=self.cost_fn(x_current,np.zeros((self.N,self.D_u)))

        # softmax weighing
        beta= np.min(costs)
        softmax_w=np.exp(-(costs-beta)/self.lam)
        softmax_w/=np.sum(softmax_w)

        #update u_k for all rollouts
        