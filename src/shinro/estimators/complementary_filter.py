"""Complementary filter for fusing a rate source with an absolute-angle source.

A complementary filter blends two measurements of the same quantity — one
accurate at high frequency (a rate, e.g. a gyro) and one accurate at low
frequency (an absolute angle, e.g. an accelerometer) — into one estimate. Its
transfer functions are complementary (low-pass + high-pass sum to one), which
is why it rejects both the rate's slow drift and the absolute sensor's noise.
"""

from dataclasses import dataclass
from typing import Any

from shinro.components import StateEstimator
from shinro.factories.registry import register_estimator
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


@dataclass(frozen=True)
class ComplementaryFilterConfig:
    """Strict TOML schema for :class:`ComplementaryFilter`.

    ``dt`` is optional: scenario builds inject it from the plant. Standalone
    use must supply it. ``channels`` is the number of fused axes (n_x); the
    measurement carries a rate and an absolute for each, so ``n_y = 2 * channels``.
    """

    alpha: float
    channels: int
    dt: float | None = None
    initial_state: list[float] | None = None
    name: str = "complementary"


@register_estimator("ComplementaryFilter")
class ComplementaryFilter(StateEstimator):
    """Complementary filter for rate + absolute-angle sensor fusion.

    Estimates an angle :math:`\\theta` from a rate :math:`\\omega` (good at high
    frequency) and an absolute measurement :math:`\\theta_a` (good at low
    frequency). The discrete form is

    .. math::

        \\hat{\\theta}_k = \\alpha \\,(\\hat{\\theta}_{k-1} + \\omega_{k-1} \\Delta t)
            + (1 - \\alpha)\\, \\theta_{a,k}

    where :math:`\\alpha \\in [0, 1]` is the blend weight on the integrated rate.
    It is the discrete equivalent of the continuous blend
    :math:`\\hat{\\theta} = \\frac{1}{1+\\tau s}\\theta_a + \\frac{\\tau s}{1+\\tau s}\\frac{\\omega}{s}`
    with :math:`\\alpha = \\tau / (\\tau + \\Delta t)`: the rate is high-passed
    (its drift is rejected), the absolute is low-passed (its noise is rejected).

    **Measurement layout.** The framework passes one measurement vector; it is
    the concatenation of the per-axis rate followed by the per-axis absolute,
    ``[\\omega_0 .. \\omega_{N-1}, \\theta_{a,0} .. \\theta_{a,N-1}]``
    (length ``2 * channels``). Everything is flat ``(N,)``, matching the EKF/UKF.

    **State.** The estimate :math:`\\hat{\\theta}` (``channels``,) is the only
    recurrent state; ``self.x_hat`` carries it across ticks (the composed graph
    threads it as ``state_x_hat``).

    Args:
        alpha: Blend weight on the integrated rate, in ``[0, 1]``. Higher means
            more trust in the rate (faster response, slower drift rejection);
            ``1`` is pure integration, ``0`` is pure measurement.
        channels: Number of independent angles fused (the state dimension n_x).
        dt: Sample time in seconds.
        x0: Initial angle estimate (channels,). Defaults to zeros.
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        alpha: float,
        channels: int,
        dt: float,
        x0: Any | None = None,
        backend: ArrayBackend | None = None,
    ):
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"ComplementaryFilter: alpha must be in [0, 1]; got {alpha}")
        if int(channels) < 1:
            raise ValueError(f"ComplementaryFilter: channels must be >= 1; got {channels}")
        self.bk = backend or NumpyBackend()
        self.alpha = float(alpha)
        self.dt = dt
        self.n_x = int(channels)
        self.n_y = 2 * self.n_x
        self.x_hat = self.bk.zeros(self.n_x) if x0 is None else self.bk.ravel(self.bk.copy(x0))

    def estimate(self, measurement, control_input):
        """Run one fuse step and return the posterior angle estimate.

        Splits the measurement into its rate and absolute halves, integrates the
        rate over ``dt``, and blends:

        .. math::

            \\hat{\\theta}_k = \\alpha (\\hat{\\theta}_{k-1} + \\omega \\Delta t)
                + (1 - \\alpha) \\theta_a

        Args:
            measurement: Concatenated ``[rates, absolutes]`` (2 * n_x,).
            control_input: Unused — a complementary filter has no control input,
                but the estimator contract requires the argument.

        Returns:
            Posterior angle estimate :math:`\\hat{\\theta}` (n_x,).
        """
        m = self.bk.ravel(measurement)
        n = self.n_x
        rates = self.bk.slice_(m, 0, n)
        absolutes = self.bk.slice_(m, n, 2 * n)
        prior = self.x_hat + self.dt * rates
        self.x_hat = self.alpha * prior + (1.0 - self.alpha) * absolutes
        return self.x_hat

    def reset(self, x0: Any | None = None):
        """Reset the filter to its initial state.

        Args:
            x0: Initial angle estimate (n_x,). Defaults to zeros.
        """
        self.x_hat = self.bk.zeros(self.n_x) if x0 is None else self.bk.ravel(self.bk.copy(x0))

    Config = ComplementaryFilterConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create a complementary filter from a TOML config dict or :class:`ComplementaryFilterConfig`.

        Config fields:
            alpha: Blend weight on the integrated rate, in ``[0, 1]``.
            channels: Number of fused axes (n_x); the measurement is ``2 * channels``.
            dt: Time step — required at runtime; injected from the plant in
                scenario builds.
            initial_state: Optional initial angle estimate (n_x,). Defaults to zeros.

        Args:
            config: TOML config dict or ComplementaryFilterConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            ComplementaryFilter instance.

        Raises:
            ValueError: If ``dt`` is None (standalone use requires it) or ``alpha``
                is outside ``[0, 1]``.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        if cfg.dt is None:
            raise ValueError(
                "ComplementaryFilter: dt is required — omit it only in scenario builds, "
                "where the plant's dt is injected"
            )
        x0 = None if cfg.initial_state is None else bk.array(cfg.initial_state)
        return cls(alpha=cfg.alpha, channels=cfg.channels, dt=cfg.dt, x0=x0, backend=bk)
