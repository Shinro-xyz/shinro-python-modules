
from shinro.components import Controller
from shinro.factories.registry import register_controller
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


@register_controller("PID")
class PIDController(Controller):
    """Proportional-Integral-Derivative controller with anti-windup.

    Computes control effort as:

    .. math::

        u(t) = K_p e(t) + K_i \\int e(\\tau) d\\tau + K_d \\frac{de}{dt}

    Features:
    - Independent gains per channel (Kp, Ki, Kd as vectors).
    - Output clamping with integral anti-windup back-calculation.
    - Derivative on error (standard form).

    When output is clamped, the integral term is back-calculated on saturated
    channels only to prevent integral windup.

    Args:
        kp: Proportional gain vector (n,).
        ki: Integral gain vector (n,).
        kd: Derivative gain vector (n,).
        dt: Time step in seconds.
        output_limits: Optional (min_limits, max_limits) for output clamping.
            Each is an array of shape (n,).
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        kp,
        ki,
        kd,
        dt: float,
        output_limits: tuple | None = None,
        backend: ArrayBackend | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt

        self.min_limits = output_limits[0] if output_limits else None
        self.max_limits = output_limits[1] if output_limits else None
        self.min_limits = output_limits[0] if output_limits else None
        self.max_limits = output_limits[1] if output_limits else None
        self._integral = self.bk.zeros_like(self.ki)
        self._prev_error = self.bk.zeros_like(self.kd)
        # First-tick gate as a 0/1 array (not a Python bool): under tracing a
        # bool branch would bake the first-tick path into the graph forever.
        # As a recurrent state port the graph gates the D-term at runtime.
        self._has_run = self.bk.zeros_like(self.ki)

    def compute(self, current_state, target_state):
        """Compute the PID control effort.

        Args:
            current_state: Measured current state (n,).
            target_state: Desired target state (n,).

        Returns:
            Control effort vector (n,).
        """
        error = target_state - current_state
        p_term = self.kp * error
        self._integral = self._integral + error * self.dt
        i_term = self.ki * self._integral

        # Where-gate: der = has_run ? (e - e_prev)/dt : 0. A Python branch
        # would bake the first-tick path into the traced graph forever; the
        # 0/1 flag is a recurrent state port the host feeds back (0 on tick
        # 0, 1 afterwards). where (not multiply): on tick 0 the candidate
        # divides by dt — with dt=0 that's inf, and inf*0 = NaN under a
        # multiply gate, while where simply discards the unselected branch.
        # The condition is an explicit `!= 0` so torch gets a boolean tensor
        # (torch.where rejects float conditions); under tracing it emits the
        # same ne node.
        der = self.bk.where(
            self._has_run != 0,
            (error - self._prev_error) / self.dt,
            self.bk.zeros_like(error),
        )
        d_term = self.kd * der

        control_effort = p_term + i_term + d_term

        if self.min_limits is not None and self.max_limits is not None:
            clamped_effort = self.bk.clip(control_effort, self.min_limits, self.max_limits)
            # Branch-free anti-windup: an elementwise mask replaces the old
            # `if bk.any(saturated)` early-out (a Python branch on a traced
            # value, which would silently apply the back-calculation every
            # tick). Where a channel is unsaturated the mask is 0 and the
            # integral passes through unchanged — identical results.
            saturated = control_effort != clamped_effort
            self._integral = self.bk.where(
                saturated,
                self._integral - error * self.dt,
                self._integral,
            )
            control_effort = clamped_effort

        self._prev_error = self.bk.copy(error)
        self._has_run = self.bk.zeros_like(self.ki) + 1.0
        return control_effort

    def reset(self):
        """Reset the controller's internal state (integral and previous error)."""
        self._integral = self.bk.zeros_like(self.ki)
        self._prev_error = self.bk.zeros_like(self.kd)
        self._has_run = self.bk.zeros_like(self.ki)

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a PID controller from a TOML config dict.

        Config fields:
            kp: List of proportional gains (n,).
            ki: List of integral gains (n,).
            kd: List of derivative gains (n,).
            dt: Time step.
            output_limits: Optional dict with ``min`` and ``max`` lists.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            PIDController instance.
        """
        bk = backend or NumpyBackend()
        n = len(config.get("kp", [1]))
        output_limits = config.get("output_limits")
        limits = None
        if output_limits:
            limits = (
                bk.array(output_limits["min"]),
                bk.array(output_limits["max"]),
            )
        return cls(
            kp=bk.from_numpy(config.get("kp", [1.0] * n)),
            ki=bk.from_numpy(config.get("ki", [0.0] * n)),
            kd=bk.from_numpy(config.get("kd", [0.0] * n)),
            dt=config["dt"],
            output_limits=limits,
            backend=bk,
        )
