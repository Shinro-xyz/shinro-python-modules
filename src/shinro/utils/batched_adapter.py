"""Batched dynamics and cost adapter — adapts a Plant's single-state interface to batched ``(N, ...)`` arrays.

Sampling-based controllers (MPPI, CEM, iLQR, particle filters) roll out
:math:`N` parallel trajectories over a prediction horizon. The :class:`Plant`
interface is single-state: ``dynamics(x, u)`` returns :math:`\\dot{x}` for one
state, and ``get_model()`` returns ``(A, B)`` for one system. This adapter
bridges the two by exposing batched ``dynamics_fn(x_batch, u_batch, dt)`` and
``cost_fn(x_batch, u_batch, Q, R, x_ref=None)`` callables that operate on a
leading batch dimension :math:`N`.

Two dynamics paths are supported, dispatched on what the plant exposes:

* **LTI path** — plants whose ``get_model()`` returns ``(A, B)``. The state
  update is a single batched matmul :math:`x_{k+1} = x_k A^T + u_k B^T`,
  which runs as one native kernel on both numpy and torch.

* **Nonlinear path** — plants that implement ``dynamics(x, u)``. The update is
  a semi-implicit Euler step :math:`x_{k+1} = x_k + dt\\, f(x_k, u_k)`.
  On torch the single-state dynamics is auto-vectorized over the batch with
  ``torch.vmap``; on numpy a Python loop over the batch is used.

The adapter uses the plant's own ``ArrayBackend`` throughout, so numpy and
torch both work with batched tensors and no hard-coded numpy in the hot loop.
For torch backends, the LTI path is a true batched matmul and the nonlinear
path runs through ``torch.vmap`` — both leverage torch's native batched ops.

Args:
    plant: A :class:`Plant` instance exposing ``get_model()`` and (for the
        nonlinear path) ``dynamics(state, control)``.
"""

from typing import Any

from shinro.components import Plant
from shinro.utils.array_backend import ArrayBackend, NumpyBackend


class BatchedDynamicsAdapter:
    """Adapt a Plant's single-state dynamics/cost to batched ``(N, ...)`` arrays.

    The adapter detects the dynamics path once at construction:

    * If ``plant.dynamics`` returns ``None`` (the ABC default — LTI plants
      need not override it), the LTI matmul path is used.
    * Otherwise the nonlinear path is used, integrating ``plant.dynamics``
      with semi-implicit Euler. On torch this is vectorized with
      ``torch.vmap``; on numpy it loops over the batch.

    All arrays live in the plant's backend, so torch inputs stay torch
    throughout and run as batched native ops.

    Usage:
        adapter = BatchedDynamicsAdapter(plant)
        x_next = adapter.dynamics_fn(x_batch, u_batch, dt)
        cost = adapter.cost_fn(x_batch, u_batch, Q, R)
    """

    def __init__(self, plant: Plant):
        self.plant = plant
        self.bk: ArrayBackend = getattr(plant, "bk", None) or NumpyBackend()
        self.dt = getattr(plant, "dt", 0.01)

        A, B = plant.get_model()
        self._A = A
        self._B = B
        self.D_x = self._A.shape[0]
        self.D_u = self._B.shape[1]

        self._has_dynamics = plant.dynamics(state=self.bk.zeros(self.D_x), control=self.bk.zeros(self.D_u)) is not None
        self._vmap = None
        torch = getattr(self.bk, "torch", None)
        if self._has_dynamics and torch is not None:
            # Vectorize the single-state nonlinear dynamics over the batch.
            self._vmap = torch.vmap(plant.dynamics, in_dims=(0, 0))

    @property
    def state_dim(self) -> int:
        """State dimension :math:`D_x`."""
        return self.D_x

    @property
    def control_dim(self) -> int:
        """Control dimension :math:`D_u`."""
        return self.D_u

    def dynamics_fn(self, x_batch, u_batch, dt: float):
        """Batched dynamics update.

        Args:
            x_batch: Batch of states (N, D_x).
            u_batch: Batch of controls (N, D_u).
            dt: Time step (s).

        Returns:
            Batch of next states (N, D_x).
        """
        if self._has_dynamics:
            return self._integrate(x_batch, u_batch, dt)
        return x_batch @ self._A.T + u_batch @ self._B.T

    def _integrate(self, x_batch, u_batch, dt: float):
        """Semi-implicit Euler integration of the plant's nonlinear dynamics.

        Args:
            x_batch: Batch of states (N, D_x).
            u_batch: Batch of controls (N, D_u).
            dt: Time step (s).

        Returns:
            Batch of next states (N, D_x).
        """
        if self._vmap is not None:
            return x_batch + dt * self._vmap(x_batch, u_batch)
        return x_batch + dt * self.bk.stack(
            [self.plant.dynamics(x_batch[i], u_batch[i]) for i in range(x_batch.shape[0])]
        )

    def cost_fn(self, x_batch, u_batch, Q, R, x_ref: Any | None = None):
        """Batched quadratic stage cost.

        Computes :math:`c(x, u) = (x - x_{ref})^T Q (x - x_{ref}) + u^T R u`
        for each sample, returning a per-sample cost vector of shape ``(N,)``.

        Args:
            x_batch: Batch of states (N, D_x).
            u_batch: Batch of controls (N, D_u).
            Q: State cost matrix (D_x, D_x) or diagonal (D_x,).
            R: Control cost matrix (D_u, D_u) or diagonal (D_u,).
            x_ref: Optional reference state (D_x,) to track. If None,
                regulates to the origin.

        Returns:
            Per-sample cost vector (N,).
        """
        if x_ref is not None:
            x_err = x_batch - x_ref
        else:
            x_err = x_batch
        x_cost = self._quad_form(x_err, Q)
        u_cost = self._quad_form(u_batch, R)
        return x_cost + u_cost

    def _quad_form(self, z, W) -> Any:
        """Batched quadratic form :math:`z^T W z` per sample.

        Accepts W as a diagonal ``(D,)`` vector (elementwise
        :math:`\\sum_i W_i z_i^2`) or a full ``(D, D)`` matrix (batched
        matmul). Returns a per-sample vector of shape ``(N,)``.

        Args:
            z: Batch of vectors (N, D).
            W: Diagonal (D,) or full (D, D) weight matrix.

        Returns:
            Per-sample quadratic value (N,).
        """
        if W is None:
            return self.bk.zeros(z.shape[0])
        if W.ndim == 1:
            return self.bk.sum(z * z * W, axis=1)
        return self.bk.sum(z * (z @ W.T), axis=1)
