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

The :math:`N` rollouts are parallel and run on the controller's
:class:`ArrayBackend` (``self.bk``) — when a :class:`TorchBackend` is set the
rollout loop executes as batched tensor operations. When built via
:func:`attach_plant`, the dynamics and cost come from a
:class:`BatchedDynamicsAdapter`, which vectorizes the plant's single-state
model over the sample batch.

Sampling is the one step of a tick with no dataflow representation, so it
stays on the host: perturbations are drawn with numpy (``ArrayBackend`` has no
RNG abstraction) and bridged to the backend via ``bk.from_numpy``. Everything
after sampling — the batched rollout, the softmax weighting, the
nominal-sequence update — runs through ``self.bk``, which is what makes the
controller lowerable: a traced ``compute`` receives the already-drawn
perturbations through a free graph input port (``epsilon``) instead of
sampling, and the compiled kernel computes the rest.

Usage:
    controller = MPPIController(
        dynamics_fn=my_dynamics, cost_fn=my_cost,
        num_samples=100, temperature=1.0, dt=0.02, horizon=10,
        noise_sigma=[0.5, 0.5],
    )
    action = controller.compute(x0)
    controller.reset()

    # Or wire a plant directly (dynamics/cost come from the plant's model):
    controller = MPPIController(...)
    controller.attach_plant(plant, Q=..., R=...)
    action = controller.compute(x0, x_ref=reference)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from shinro.components import Controller
from shinro.factories.registry import register_controller
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


def _as_float(value: Any, field: str) -> float:
    """Coerce a config scalar to float, naming the field when it fails.

    Mirrors :func:`shinro.controllers.smc._as_float`: config values arrive from
    TOML (or hand-written dicts), so a typo is a user error worth naming — a
    bare ``float("abc")`` reports only that a conversion failed, not which
    field or config was wrong.

    Args:
        value: The raw config value.
        field: The config field name, for the error message.

    Returns:
        The value as a ``float``.

    Raises:
        ValueError: If the value is not numeric.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MPPIConfig.{field} must be a number, got {value!r}") from exc


@dataclass(frozen=True)
class MPPIConfig:
    """Strict TOML schema for :class:`MPPIController`.

    ``dynamics_fn``/``cost_fn`` are not TOML-serializable: they are ``None``
    at construction and injected afterwards (``attach_plant`` or direct
    attribute assignment). ``dt`` is required at runtime (rollout stepping);
    scenario builds inject it from the plant, so it may be omitted there.
    """

    num_samples: int
    temperature: float
    horizon: int
    noise_sigma: list[float]
    dt: float | None = None
    u_min: list[float] | None = None
    u_max: list[float] | None = None
    seed: int | None = None
    state_cost: Any = None
    control_cost: Any = None
    name: str = "mppi"


@register_controller("MPPI")
class MPPIController(Controller):
    """Model Predictive Path Integral controller.

    Each ``compute()`` call samples :math:`N` Gaussian control perturbations,
    rolls out the dynamics over a horizon of :math:`K` steps, weights the
    perturbations by the softmax of their total cost, and advances the
    receding-horizon nominal control sequence by one step. The returned action
    is the first element of the updated nominal sequence, clipped to the
    configured control bounds.

    The dynamics and cost callables receive batched arrays of shape
    ``(N, D_x)`` / ``(N, D_u)`` and may be injected directly or produced by
    :func:`attach_plant` from a plant's model.

    Args:
        dynamics_fn: Callable ``dynamics_fn(x, u, dt) -> x_next`` stepping a
            batch of states forward one time step. Receives backend arrays of
            shape ``(N, D_x)`` and ``(N, D_u)`` and returns ``(N, D_x)``.
            May be ``None`` and injected later or via ``attach_plant``.
        cost_fn: Callable ``cost_fn(x, u) -> c`` returning the per-sample
            stage cost as an array of shape ``(N,)``. May be ``None`` and
            injected later or via ``attach_plant``.
        num_samples: Number of sampled perturbations N.
        temperature: Softmax temperature :math:`\\lambda` (> 0).
        dt: Time step passed to the dynamics.
        horizon: Prediction horizon K.
        noise_sigma: Standard deviation of the control perturbation per
            input channel ``(D_u,)``. If None, defaults to ``[0.5]``.
        u_min: Optional lower control bound ``(D_u,)`` or scalar.
        u_max: Optional upper control bound ``(D_u,)`` or scalar.
        seed: Optional RNG seed for reproducible sampling.
        backend: Array backend. Defaults to NumpyBackend. The rollout loop
            runs on this backend, so a TorchBackend executes batched tensor
            ops over the sample dimension.
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
        self.dt = _as_float(dt, "dt")
        self.lam = _as_float(temperature, "temperature")
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
        self._adapter = None
        self._Q = None
        self._R = None
        # Tracking reference for the attach_plant cost. Held in a 1-element
        # list rather than as an array-like attr: the tracer's attr-diff
        # promotes every reassigned ndarray/Tracer attr to a recurrent state
        # port, and this is a per-call input, not state.
        self._x_ref_holder = [None]

    def attach_plant(self, plant, Q: Any | None = None, R: Any | None = None):
        """Wire a plant into the controller via a batched dynamics adapter.

        Builds a :class:`BatchedDynamicsAdapter` from the plant and sets
        ``dynamics_fn`` / ``cost_fn`` from it. The cost uses the quadratic
        stage cost :math:`x^T Q x + u^T R u`, optionally tracking the reference
        passed to :func:`compute` — that reference is threaded through
        ``self._x_ref_holder``, a plain list, so the tracer does not mistake it
        for recurrent state. ``Q`` and ``R`` may be diagonal ``(D,)`` or full
        ``(D, D)`` matrices; they default to identity.

        Args:
            plant: A :class:`Plant` exposing ``get_model()`` (and optionally
                ``dynamics()`` for nonlinear plants).
            Q: State cost matrix (D_x, D_x) or diagonal (D_x,).
            R: Control cost matrix (D_u, D_u) or diagonal (D_u,).

        Raises:
            ValueError: If the plant's control dimension disagrees with the
                controller's ``noise_sigma`` length.
        """
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter

        adapter = BatchedDynamicsAdapter(plant)
        if adapter.control_dim != self.D_u:
            raise ValueError(
                f"Plant control dimension {adapter.control_dim} does not match "
                f"controller noise_sigma dimension {self.D_u}."
            )
        self._adapter = adapter
        self._Q = self._Q if Q is None else Q
        self._R = self._R if R is None else R
        self.D_x = adapter.state_dim
        self.D_u = adapter.control_dim

        self.dynamics_fn = adapter.dynamics_fn
        self.cost_fn = lambda x, u: adapter.cost_fn(x, u, self._Q, self._R, x_ref=self._x_ref_holder[0])

    def compute(self, current_state, target_state: Any | None = None, epsilon: Any | None = None):
        """Compute the MPPI control action for a given initial state.

        Draws :math:`N` Gaussian perturbation sequences (host-side — see the
        module docstring), rolls out the dynamics over the horizon, computes
        the softmax-weighted update, and returns the first action of the
        updated nominal sequence (clipped to bounds if configured).

        Everything after sampling runs through ``self.bk`` on batched tensors,
        so it evaluates identically on numpy, torch, and the tracing backend.
        Every reduction is written as a matmul identity — a sum over an axis is
        a contraction with a ones vector or with the softmax weights — except
        the softmax shift ``beta = min(costs)``, which is the ``min`` graph op.

        Args:
            current_state: Initial state vector (D_x,). Accepts the backend's
                native array type (numpy or torch).
            target_state: Optional reference state (D_x,) to track. When given,
                the cost penalizes deviation ``(x - target_state)``; otherwise
                the controller regulates to the origin.
            epsilon: Optional pre-drawn perturbations, shape ``(N, K*D_u)``,
                sample-major (row ``i``, column ``k*D_u + d``). This is the
                trace hook: when lowering, the perturbations become a free
                graph input port the host fills each tick, so the traced call
                never samples. ``None`` (the eager path) draws them from
                ``self._rng``.

        Returns:
            First control action (D_u,) in the backend's native type.

        Raises:
            RuntimeError: If ``dynamics_fn`` or ``cost_fn`` has not been set.
        """
        if self.dynamics_fn is None or self.cost_fn is None:
            raise RuntimeError(
                "dynamics_fn and cost_fn must be set before calling compute(). "
                "Inject them directly, call attach_plant(plant, ...), or if the "
                "controller was built with from_config, set ctrl.dynamics_fn = ... "
                "and ctrl.cost_fn = ..."
            )
        dynamics_fn = self.dynamics_fn
        cost_fn = self.cost_fn

        # Sampling is host-side by design: the one step of the tick with no
        # dataflow representation. The traced path receives epsilon as a graph
        # input port, so this branch is a trace-time constant and the traced
        # call never touches the RNG.
        if epsilon is None:
            eps_sample = self._rng.normal(loc=0.0, scale=self.noise_sigma, size=(self.N, self.K, self.D_u))
            self._last_epsilon = eps_sample
            eps_in = self.bk.from_numpy(eps_sample.reshape(self.N, self.K * self.D_u))
        else:
            # Host-supplied perturbations (the trace path, or a caller wanting
            # reproducible noise). Bridged like the sampled path so a numpy
            # array works on a torch backend; a no-op under tracing, where
            # epsilon is already the graph input Tracer.
            eps_in = self.bk.from_numpy(epsilon)

        N, K, Du = self.N, self.K, self.D_u
        lam = self.lam

        # Batch the initial state to (N, D_x) with a same-rank broadcast; the
        # tracer rejects numpy's rank-differing tile.
        d_x = current_state.shape[0]
        x_current = self.bk.zeros((N, d_x)) + self.bk.reshape(current_state, (1, d_x))

        # Tracking reference for the attach_plant cost closure (see __init__),
        # stored as a (1, D_x) row: the closure subtracts it from the batched
        # states, and (N, D_x) - (1, D_x) stays same-rank for the tracer (numpy
        # would broadcast a (D_x,) there; the tracer rejects the rank gap).
        self._x_ref_holder[0] = self.bk.reshape(target_state, (1, d_x)) if target_state is not None else None

        # eps_in is (N, K*D_u) sample-major, so its transpose is (K*D_u, N) and
        # step k's perturbations are the row block [k*D_u:(k+1)*D_u] transposed
        # back to (N, D_u).
        eps_t = eps_in.T
        ones_du = self.bk.array(np.ones(Du))
        ones_n_col = self.bk.array(np.ones((N, 1)))
        sigma2 = self.bk.reshape(self.bk.array(self.noise_sigma**2), (1, Du))
        costs = self.bk.zeros(N)
        # Bridge the nominal sequence to the backend once. It is numpy-hosted
        # (see the assignment at the end), and `numpy + torch` raises — only
        # `torch + numpy` works — so every use goes through this backend-native
        # copy. Under tracing this is a no-op passthrough and u_nominal is the
        # recurrent state input Tracer.
        u_nominal = self.bk.from_numpy(self.u)

        for k in range(K):
            eps_k = self.bk.slice_(eps_t, k * Du, (k + 1) * Du).T
            u_k = self.bk.slice_(u_nominal, k, k + 1)  # nominal row -> (1, D_u)
            v_k = eps_k + u_k
            if self.u_min is not None or self.u_max is not None:
                v_k = self.bk.clip(v_k, self.u_min, self.u_max)
            costs = costs + cost_fn(x_current, v_k)
            # lam * sum_d (u_k,d / sigma_d^2) eps_i,k,d — a row sum, i.e. a
            # contraction with a ones vector.
            control_penalty = lam * ((u_k / sigma2 * eps_k) @ ones_du)
            costs = costs + control_penalty
            x_current = dynamics_fn(x_current, v_k, self.dt)

        costs = costs + cost_fn(x_current, self.bk.zeros((N, Du)))

        # Diagnostics: the traced backend publishes `costs` as an auxiliary
        # graph output port (SMC's `healthy` precedent); eager backends keep the
        # concrete array. Assigning a Tracer here would be mis-detected as
        # recurrent state.
        self.bk.emit_named_output("costs", costs)
        costs_np = self.bk.to_numpy(costs)
        if isinstance(costs_np, np.ndarray):
            self._last_costs = costs_np.copy()

        # Softmax weights. beta is the shift that keeps exp from overflowing.
        # The normalizer is a contraction with a ones vector, not a 1-D @ 1-D
        # product: (1,N) @ (N,1) with the weights on *both* sides would sum the
        # squares. The (1,1) result ravels to (1,), which broadcasts against (N,).
        beta = self.bk.min(costs)
        w = self.bk.exp(-(costs - beta) / lam)
        w_sum = self.bk.ravel(self.bk.reshape(w, (1, N)) @ ones_n_col)
        w = w / w_sum

        # Weighted average of the perturbations: sum_i w_i eps_i is the matmul
        # (1,N) @ (N, K*D_u). to_numpy keeps the nominal sequence numpy-hosted
        # on the eager backends; under tracing it is a no-op passthrough, so
        # self.u stays a Tracer and is emitted as the recurrent state output.
        w_row = self.bk.reshape(w, (1, N))
        weighted_eps = self.bk.reshape(w_row @ eps_in, (K, Du))
        u_updated = u_nominal + weighted_eps

        # The returned action is the first element *before* the receding-horizon
        # shift (self.u[:-1] = self.u[1:]; self.u[-1] = self.u[-2], i.e. rows
        # 1..K-1 followed by row K-1 again). The tracer has no in-place slice
        # assignment, so the shift is K row slices + a stack.
        u_0 = self.bk.slice_(u_updated, 0, 1)
        if K > 1:
            rows = [self.bk.ravel(self.bk.slice_(u_updated, j, j + 1)) for j in range(1, K)]
            rows.append(self.bk.ravel(self.bk.slice_(u_updated, K - 1, K)))
            shifted = self.bk.stack(rows)
        else:
            shifted = u_updated
        self.u = self.bk.to_numpy(shifted)

        u_0 = self.bk.ravel(u_0)
        if self.u_min is not None or self.u_max is not None:
            u_0 = self.bk.clip(u_0, self.u_min, self.u_max)

        return u_0

    def reset(self):
        """Reset the controller to its initial state.

        Zeros the nominal control sequence and clears the last-sample
        bookkeeping attributes.
        """
        self.u = np.zeros((self.K, self.D_u))
        self._last_epsilon = None
        self._last_costs = None

    Config = MPPIConfig

    @classmethod
    def from_config(cls, config, backend: ArrayBackend | None = None):
        """Create an MPPI controller from a TOML config dict or :class:`MPPIConfig`.

        Config fields:
            num_samples: Number of sampled perturbations N.
            temperature: Softmax temperature.
            horizon: Prediction horizon K.
            noise_sigma: Per-channel perturbation std dev (D_u,).
            dt: Time step. Required at runtime; injected from the plant in
                scenario builds.
            u_min: Optional lower bound list (D_u,).
            u_max: Optional upper bound list (D_u,).
            seed: Optional RNG seed.
            state_cost: Optional diagonal Q weights (D_x,) or full Q matrix.
            control_cost: Optional diagonal R weights (D_u,) or full R matrix.

        The ``dynamics_fn`` and ``cost_fn`` callables cannot be serialized to
        TOML. They are created as ``None`` and must be injected after
        construction, either by setting the attributes directly or by calling
        ``attach_plant(plant)``:

        .. code-block:: python

            ctrl = MPPIController.from_config(config)
            ctrl.dynamics_fn = my_dynamics
            ctrl.cost_fn = my_cost
            # or:
            ctrl.attach_plant(plant)

        Args:
            config: TOML config dict or MPPIConfig.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            MPPIController instance.
        """
        bk = backend or NumpyBackend()
        cfg = cls.parse_config(config)
        if cfg.dt is None:
            raise ValueError(
                "MPPI: dt is required (rollout stepping) — omit it only in scenario "
                "builds, where the plant's dt is injected"
            )
        ctrl = cls(
            dynamics_fn=None,
            cost_fn=None,
            num_samples=cfg.num_samples,
            temperature=cfg.temperature,
            dt=cfg.dt,
            horizon=cfg.horizon,
            noise_sigma=cfg.noise_sigma,
            u_min=cfg.u_min,
            u_max=cfg.u_max,
            seed=cfg.seed,
            backend=bk,
        )
        ctrl._Q = bk.array(cfg.state_cost) if cfg.state_cost is not None else None
        ctrl._R = bk.array(cfg.control_cost) if cfg.control_cost is not None else None
        return ctrl
