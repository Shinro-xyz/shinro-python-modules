import numpy as np

class PIDCcontroller:
    def __init__(self,kp:np.ndarray, ki:np.ndarray,kd:np.ndarray, dt:float,
        output_limits:tuple[np.ndarray,np.ndarray]=None):
            self.kp=np.atleast_1d(kp)
            self.kd=np.atleast_1d(kd)
            self.ki=np.atleast_1d(ki)
            self.dt=dt

            self.min_limits=output_limits[0] if output_limits else None
            self.max_limits=output_limits[1] if output_limits else None
    def compute(self,current_state:np.ndarray,target_state:np.ndarray):
        error=target_state-current_state
        p_term= self.kp*error
        self._integral+=error*self.dt
        i_term=self.ki*self._integral
        der=(error-self._prev_error)/self.dt
        d_term=self.kd*der

        control_effort= p_term+i_term+d_term

        if self.min_limits is not None and self.max_limits is not None:
            
        self._prev_error=error.copy()
        
        
        