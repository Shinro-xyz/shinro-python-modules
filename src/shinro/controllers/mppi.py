"""Model Predictive Path Integral (MPPI) controller — sampling-based, information-theoretic optimal control.

At each step, MPPI samples :math:`N` control perturbation sequences from a
Gaussian distribution around the current nominal control sequence, rolls out
the dynamics for each perturbed sequence, and weights the perturbations by
their total cost using a softmax (exponential) weighting:

.. math::

    w_i = \\frac{\\exp\\left(-\\frac{S_i - \\min_j S_j}{\\lambda}\\right)}
               {\\sum_j \\exp\\left(-\\frac{S_j - \\min_j S_j}{\\lambda}\\right)}

where :math:`S_i` is the total cost of rollout :math:`i` and :math:`\\lambda`
is the temperature. The nominal control sequence is updated by the weighted
average of the perturbations:

.. math::

    u_k \\gets u_k + \\sum_i w_i \\, \\epsilon_{i,k}

The sampling, rollout, and softmax operations are inherently numpy-based.
Following the ``MPC_LTI`` precedent (where the OSQP solver is C-based and
always numpy), MPPI converts the input state to numpy, runs the entire
information-theoretic update in numpy, and converts the returned action back
to the backend's native type. The ``dynamics_fn`` and ``cost_fn`` callables
always receive and return numpy arrays.

Usage:
    controller = MPPIController(
        dynamics_fn=my_dynamics, cost_fn=my_cost,
        num_samples=100, temperature=1.0, dt=0.02, horizon=10,
        noise_sigma=[0.5, 0.5],
    )
    action = controller.compute(x0)
    controller.reset()
"""

from typing import Any

import numpy as np

from shinro.components import Controller
from shinro.factories.registry import register_controller
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


@register_controller("MPPI")
class MPPIController(Controller):
    """Model Predictive Path Integral controller.

    Each ``compute()`` call samples :math:`N` Gaussian control perturbations,
    rolls out the dynamics over a horizon of :math:`K` steps, weights the
    perturbations by the softmax of their total cost, and advances the
    receding-horizon nominal control sequence by one step. The returned action
    is the first element of the updated nominal sequence, clipped to the
    configured control bounds.

    Args:
        dynamics_fn: Callable ``dynamics_fn(x, u, dt) -> x_next`` stepping
            the state forward one time step. Receives numpy arrays of shape
            ``(N, D_x)`` and ``(N, D_u)`` and returns shape ``(N, D_x)``.
            May be ``None`` and injected later.
        cost_fn: Callable ``cost_fn(x, u) -> c`` returning the per-sample
            stage cost as an array of shape ``(N,)``. May be ``None`` and
            injected later.
        num_samples: Number of sampled perturbations N.
        temperature: Softmax temperature :math:`\\lambda` (> 0).
        dt: Time step passed to the dynamics.
        horizon: Prediction horizon K.
        noise_sigma: Standard deviation of the control perturbation per
            input channel ``(D_u,)``. If None, defaults to ``[0.5]``.
        u_min: Optional lower control bound ``(D_u,)`` or scalar.
        u_max: Optional upper control bound ``(D_u,)`` or scalar.
        seed: Optional RNG seed for reproducible sampling.
        backend: Array backend. Defaults to NumpyBackend. Used to bridge the
            input state and returned action; the internal sampling and
            rollout loop is numpy-based.
    """

    def __init__(
        self,
        dynamics_fn: Any | None = None,
        cost_fn: Any | None = None,
        num_samples: int = 100,
        temperature: float = 1.0,
        dt: float = 0.01,
        horizon: int = 10,
        noise_sigma=None,
        u_min=None,
        u_max=None,
        seed: int | None = None,
        backend: ArrayBackend | None = None,
    ):
        self.bk = backend or NumpyBackend()
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.N = num_samples
        self.K = horizon
        self.dt = float(dt)
        self.lam = float(temperature)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if noise_sigma is None:
            noise_sigma = [0.5]
        self.noise_sigma = np.atleast_1d(np.asarray(noise_sigma, dtype=np.float64))
        self.D_u = self.noise_sigma.shape[0]

        self.u_min = np.atleast_1d(np.asarray(u_min, dtype=np.float64)) if u_min is not None else None
        self.u_max = np.atleast_1d(np.asarray(u_max, dtype=np.float64)) if u_max is not None else None

        self.u = np.zeros((self.K, self.D_u))

        self._last_epsilon = None
        self._last_costs = None

    def compute(self, x0):
        """Compute the MPPI control action for a given initial state.

        Samples :math:`N` Gaussian perturbation sequences, rolls out the
        dynamics over the horizon, computes the softmax-weighted update, and
        returns the first action of the updated nominal sequence (clipped to
        bounds if configured).

        Args:
            x0: Initial state vector (D_x,). Accepts the backend's native
                array type (numpy or torch).

        Returns:
            First control action (D_u,) in the backend's native type.

        Raises:
            RuntimeError: If ``dynamics_fn`` or ``cost_fn`` has not been set.
        """
        if self.dynamics_fn is None or self.cost_fn is None:
            raise RuntimeError(
                "dynamics_fn and cost_fn must be set before calling compute(). "
                "If the controller was built with from_config, inject them via "
                "ctrl.dynamics_fn = ... and ctrl.cost_fn = ..."
            )
        dynamics_fn = self.dynamics_fn
        cost_fn = self.cost_fn

        x0_np = self.bk.to_numpy(x0)

        epsilon = self._rng.normal(loc=0.0, scale=self.noise_sigma, size=(self.N, self.K, self.D_u))
        self._last_epsilon = epsilon

        v = np.expand_dims(self.u, axis=0) + epsilon
        if self.u_min is not None or self.u_max is not None:
            v = np.clip(v, self.u_min, self.u_max)

        x_current = np.tile(x0_np, (self.N, 1))
        costs = np.zeros(self.N)

        for k in range(self.K):
            u_k = v[:, k, :]
            costs += cost_fn(x_current, u_k)
            inv_var_weighted_u = self.u[k] / (self.noise_sigma**2)
            control_penalty = self.lam * np.sum(inv_var_weighted_u * epsilon[:, k, :], axis=1)
            costs += control_penalty
            x_current = dynamics_fn(x_current, u_k, self.dt)

        costs += cost_fn(x_current, np.zeros((self.N, self.D_u)))
        self._last_costs = costs.copy()

        beta = np.min(costs)
        softmax_w = np.exp(-(costs - beta) / self.lam)
        softmax_w /= np.sum(softmax_w)

        weighted_eps = np.sum(softmax_w[:, np.newaxis, np.newaxis] * epsilon, axis=0)

        self.u += weighted_eps

        u_0 = self.u[0].copy()
        self.u[:-1] = self.u[1:]
        self.u[-1] = self.u[-2]

        if self.u_min is not None or self.u_max is not None:
            u_0 = np.clip(u_0, self.u_min, self.u_max)

        return self.bk.from_numpy(u_0)

    def reset(self):
        """Reset the controller to its initial state.

        Zeros the nominal control sequence and clears the last-sample
        bookkeeping attributes.
        """
        self.u = np.zeros((self.K, self.D_u))
        self._last_epsilon = None
        self._last_costs = None

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an MPPI controller from a TOML config dict.

        Config fields:
            num_samples: Number of sampled perturbations N.
            temperature: Softmax temperature.
            dt: Time step.
            horizon: Prediction horizon K.
            noise_sigma: Per-channel perturbation std dev (D_u,).
            u_min: Optional lower bound list (D_u,).
            u_max: Optional upper bound list (D_u,).
            seed: Optional RNG seed.

        The ``dynamics_fn`` and ``cost_fn`` callables cannot be serialized to
        TOML. They are created as ``None`` and must be injected after
        construction:

        .. code-block:: python

            ctrl = MPPIController.from_config(config)
            ctrl.dynamics_fn = my_dynamics
            ctrl.cost_fn = my_cost

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            MPPIController instance.
        """
        bk = backend or NumpyBackend()
        return cls(
            dynamics_fn=None,
            cost_fn=None,
            num_samples=config["num_samples"],
            temperature=config["temperature"],
            dt=config["dt"],
            horizon=config["horizon"],
            noise_sigma=config["noise_sigma"],
            u_min=config.get("u_min"),
            u_max=config.get("u_max"),
            seed=config.get("seed"),
            backend=bk,
        )
