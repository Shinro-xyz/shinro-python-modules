import numpy as np
from shinro.components import Plant
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.factories.registry import register_plant
from shinro.utils.linearization import linearize_plant

@register_plant("DoublePendulum")
class DoublePendulum(Plant):
    def __init__(self,mass_top:float, mass_bottom:float, length_top:float, length_bottom:float,dt:float, g:float=9.81):
        self.m1=mass_top
        self.m2=mass_bottom
        self.l1=length_top
        self.l2=length_bottom
        self.dt=dt
        self.g:float=g
        self.input_dim=2

    def _make_mass_matrix(self, diff_theta):
        M=np.zeros((2,2))
        m1,m2=self.m1, self.m2
        l1,l2=self.l1,self.l2
        M[0,0]= (m1+m2)*l1**2
        M[0,1]= m2*l1*l2*np.cos(diff_theta)
        M[1,0]= M[0,1]
        M[1,1]=m2*l2**2
        self.M=M
        return M

    def _make_coriolis_matrix(self,diff_theta,angular_velocities):
        C=np.zeros((2,2))
        w_1,w_2=angular_velocities[0], angular_velocities[1]
        m1,m2=self.m1, self.m2
        l1,l2=self.l1,self.l2
        C[0,0]=m2*l1*l2*np.sin(diff_theta)*w_2
        C[0,1]=C[0,0]
        C[1,0]=-m2*l2*l1*np.sin(diff_theta)*w_1
        self.C=C
        return C
    
    def _make_gravity_vector(self, theta_angles):
        m1,m2=self.m1, self.m2
        l1,l2=self.l1,self.l2
        g=self.g
        theta_1,theta_2=theta_angles[0], theta_angles[1]
        G=np.zeros((2,1))
        G[0,1]=(m1+m2)*g*l1*np.sin(theta_1)
        G[1,1]= m2*g*l2*np.sin(theta_2)
        self.G=G
        return G

    def dynamics(self, state, control):
        # state: angular velocity of rods 1 and 2 (w_1, w_2), angle of arms 1 and 2
        theta_1,theta_2,omega_1,omega_2= state[0], state[1],state[2],state[3]
        diff_theta=theta_1-theta_2
        #mass matrix
        M=self._make_mass_matrix(diff_theta)
        # coriolis matrix
        C=self._make_coriolis_matrix(diff_theta,np.array([omega_1,omega_2]))
        #gravity vector
        G=self._make_gravity_vector(np.array(theta_1,theta_2))

        b= control-C*np.array([omega_1,omega_2])-G
        thetaddot=np.linalg.solve(M,b)

        return np.array([omega_1,omega_2,thetaddot[0],thetaddot[1]])

    def get_model(self, x0=None, u0= None, eps=1e-6):
        return linearize_plant(self, x0, u0, eps=eps)

    def step(self,u):
        
        