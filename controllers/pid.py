from typing import Optional, Tuple
from components import Controller
from factories.registry import register_controller
from utils.array_backend import ArrayBackend, NumpyBackend


@register_controller("PID")
class PIDController(Controller):
    def __init__(
        self,
        kp,
        ki,
        kd,
        dt: float,
        output_limits: Optional[Tuple] = None,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt

        self.min_limits = output_limits[0] if output_limits else None
        self.max_limits = output_limits[1] if output_limits else None
        self._integral = self.bk.zeros_like(self.ki)
        self._prev_error = self.bk.zeros_like(self.kd)
        self.has_run = False

    def compute(self, current_state, target_state):
        error = target_state - current_state
        p_term = self.kp * error
        self._integral = self._integral + error * self.dt
        i_term = self.ki * self._integral

        if self.has_run is True:
            der = (error - self._prev_error) / self.dt
        else:
            der = self.bk.zeros_like(error)
            self.has_run = True
        d_term = self.kd * der

        control_effort = p_term + i_term + d_term

        if self.min_limits is not None and self.max_limits is not None:
            clamped_effort = self.bk.clip(control_effort, self.min_limits, self.max_limits)
            saturated_indices = control_effort != clamped_effort
            if self.bk.any(saturated_indices):
                self._integral = self.bk.where(
                    saturated_indices,
                    self._integral - error * self.dt,
                    self._integral,
                )
                control_effort = clamped_effort

        self._prev_error = self.bk.copy(error)
        return control_effort

    def reset(self):
        self._integral = self.bk.zeros_like(self.ki)
        self._prev_error = self.bk.zeros_like(self.kd)
        self.has_run = False

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        bk = backend or NumpyBackend()
        n = len(config.get("kp", [1]))
        return cls(
            kp=bk.from_numpy(config.get("kp", [1.0] * n)),
            ki=bk.from_numpy(config.get("ki", [0.0] * n)),
            kd=bk.from_numpy(config.get("kd", [0.0] * n)),
            dt=config["dt"],
            backend=bk,
        )
