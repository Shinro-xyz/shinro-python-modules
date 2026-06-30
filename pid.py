import numpy as np
from components import Controller

class PIDController(Controller):
    """
    Proportional-Integral-Derivative (PID) controller for state tracking.
    """
    def __init__(self,kp:np.ndarray, ki:np.ndarray,kd:np.ndarray, dt:float,
        output_limits:tuple[np.ndarray,np.ndarray]=None):
        """
        Initialize the PID controller.

        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
            dt: Time step duration.
            output_limits: Optional tuple of (min_limits, max_limits) for control effort clamping.
        """
        self.kp=np.atleast_1d(kp)
        self.kd=np.atleast_1d(kd)
        self.ki=np.atleast_1d(ki)
        self.dt=dt
    
        self.min_limits=output_limits[0] if output_limits else None
        self.max_limits=output_limits[1] if output_limits else None
        self._integral=np.zeros_like(self.ki)
        self._prev_error=np.zeros_like(self.kd)
        self.has_run=False
    def compute(self,current_state:np.ndarray,target_state:np.ndarray):
        """
        Compute the control effort based on the error between target and current state.

        Args:
            current_state: The measured current state.
            target_state: The desired target state.

        Returns:
            The computed control effort.
        """
        error=target_state-current_state
        p_term= self.kp*error
        self._integral+=error*self.dt
        i_term=self.ki*self._integral
        
        if self.has_run is True:
            der=(error-self._prev_error)/self.dt
        else:
            der=np.zeros_like(error)
            self.has_run=True
        d_term=self.kd*der

        control_effort= p_term+i_term+d_term

        if self.min_limits is not None and self.max_limits is not None:
            clamped_effort=np.clip(control_effort, self.min_limits, self.max_limits)
            saturated_indices = control_effort != clamped_effort
            if np.any(saturated_indices):
                # Back-step the integral component for saturated channels
                self._integral[saturated_indices] -= error[saturated_indices] * self.dt
                control_effort = clamped_effort
            
        self._prev_error=error.copy()
        return control_effort
    
    def reset(self):
        """
        Reset the controller's internal state (integral and previous error).
        """
        self._integral=np.zeros_like(self.ki)
        self._prev_error=np.zeros_like(self.kd)
        self.has_run=False
        
        
        
        