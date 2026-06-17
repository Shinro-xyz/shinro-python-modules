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
        
        
        