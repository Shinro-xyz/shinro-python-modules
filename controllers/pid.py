import numpy as np
from typing import Optional, Tuple
from components import Controller


class PIDController(Controller):
    """Proportional-Integral-Derivative controller with anti-windup.

    Computes control effort as:
        u(t) = Kp e(t) + Ki ∫ e(τ) dτ + Kd de/dt

    Features:
    - Independent gains per channel (Kp, Ki, Kd as vectors)
    - Output clamping with integral anti-windup back-calculation
    - Derivative on error (standard form)

    Usage:
        pid = PIDController(
            kp=np.array([10.0, 10.0, 5.0]),
            ki=np.array([1.0, 1.0, 0.5]),
            kd=np.array([0.1, 0.1, 0.05]),
            dt=0.02,
            output_limits=(np.array([-1, -1, -1]), np.array([1, 1, 1])),
        )
        u = pid.compute(current_state, target_state)
    """

    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        dt: float,
        output_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """Initialize the PID controller.

        Args:
            kp: Proportional gain vector (n,).
            ki: Integral gain vector (n,).
            kd: Derivative gain vector (n,).
            dt: Time step in seconds.
            output_limits: Optional (min_limits, max_limits) for output clamping.
                Each is an array of shape (n,). When clamped, the integral term
                is back-calculated to prevent windup on saturated channels.
        """
        self.kp = np.atleast_1d(kp)
        self.kd = np.atleast_1d(kd)
        self.ki = np.atleast_1d(ki)
        self.dt = dt

        self.min_limits = output_limits[0] if output_limits else None
        self.max_limits = output_limits[1] if output_limits else None
        self._integral = np.zeros_like(self.ki)
        self._prev_error = np.zeros_like(self.kd)
        self.has_run = False

    def compute(self, current_state: np.ndarray, target_state: np.ndarray):
        """Compute the PID control effort.

        Args:
            current_state: Measured current state (n,).
            target_state: Desired target state (n,).

        Returns:
            Control effort vector (n,).
        """
        error = target_state - current_state
        p_term = self.kp * error
        self._integral += error * self.dt
        i_term = self.ki * self._integral

        if self.has_run is True:
            der = (error - self._prev_error) / self.dt
        else:
            der = np.zeros_like(error)
            self.has_run = True
        d_term = self.kd * der

        control_effort = p_term + i_term + d_term

        if self.min_limits is not None and self.max_limits is not None:
            clamped_effort = np.clip(control_effort, self.min_limits, self.max_limits)
            saturated_indices = control_effort != clamped_effort
            if np.any(saturated_indices):
                # Back-step the integral component for saturated channels
                self._integral[saturated_indices] -= error[saturated_indices] * self.dt
                control_effort = clamped_effort

        self._prev_error = error.copy()
        return control_effort

    def reset(self):
        """Reset the controller's internal state (integral and previous error)."""
        self._integral = np.zeros_like(self.ki)
        self._prev_error = np.zeros_like(self.kd)
        self.has_run = False