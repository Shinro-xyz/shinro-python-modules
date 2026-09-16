"""Batched dynamics and cost adapter — batched ``(N, ...)`` callables for sampling controllers.

Sampling-based controllers (MPPI, CEM, iLQR, particle filters) roll out
:math:`N` parallel trajectories over a prediction horizon. This adapter
exposes the two callables they need — ``dynamics_fn(x_batch, u_batch, dt)``
and ``cost_fn(x_batch, u_batch, Q, R, x_ref=None)`` — operating on a
leading batch dimension :math:`N`, from a plant's ``dynamics`` and
``get_model()``.

Two dynamics paths are supported, dispatched on what the plant exposes:

* **Nonlinear path** — plants that implement ``dynamics`` (the ABC default is
  ``None``, which linear plants keep). The plant evaluates its own
  batch-capable derivative — one ``sin``/``mul`` node of shape ``(N, 1)`` per
  formula term, not ``N`` private copies — and this adapter integrates it with
  explicit Euler, ``x_{k+1} = x_k + dt f(x_k, u_k)``. Because the whole batch
  rides in the node shapes, the traced graph's node count is independent of
  the number of samples. The path is trace-safe: the plant routes every
  backend call through the ``bk`` it is handed, because ``trace_node`` swaps
  only the traced component's ``self.bk`` and leaves the plant's backend
  concrete.

* **LTI path** — plants whose ``dynamics`` returns ``None`` and whose
  ``get_model()`` returns ``(A, B)``. The state update is a single batched
  matmul ``x_{k+1} = x_k A^T + u_k B^T``, one native kernel on numpy and torch
  alike.

There is deliberately one nonlinear implementation, not two: a scalar formula
plus a batched rewrite would be two transcriptions of the same physics, free
to drift. ``Plant.dynamics`` accepts a single state ``(n_x,)`` **or** a batch
``(N, n_x)`` — a single state is a batch of one — so the eager per-sample
rollout, the finite-difference linearization, and the lowered graph all run
the same function.

The adapter evaluates with the backend the caller passes (the component's
backend when tracing), defaulting to the plant's.


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

    * ``plant.dynamics`` returning non-``None`` (the ABC default is ``None``)
      — the nonlinear path, integrating the plant's batch-capable derivative
      with explicit Euler. Trace-safe.
    * Otherwise the LTI matmul path.

    Arrays live in the plant's backend, or in the backend passed to
    :meth:`dynamics_fn` (tracing passes the component's ``TraceBackend``).

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

        # Path dispatch: a plant that overrides ``dynamics`` (the ABC default
        # is None) rolls out nonlinearly; otherwise the linearized (A, B)
        # matmul is exact and cheaper. The probe doubles as the capability
        # check the ABC used to leave to ``dynamics() is not None``.
        probe = plant.dynamics(state=self.bk.zeros(self.D_x), control=self.bk.zeros(self.D_u))
        self._nonlinear = probe is not None
        self._path = "nonlinear" if self._nonlinear else "lti"

    @property
    def state_dim(self) -> int:
        """State dimension :math:`D_x`."""
        return self.D_x

    @property
    def control_dim(self) -> int:
        """Control dimension :math:`D_u`."""
        return self.D_u

    @property
    def dynamics_path(self) -> str:
        """Which dynamics path this adapter dispatches to.

        ``"nonlinear"`` (the plant's own batch-capable derivative, integrated
        with explicit Euler) or ``"lti"`` (one batched matmul).
        """
        return self._path

    def dynamics_fn(self, x_batch, u_batch, dt: float, bk: Any | None = None):
        """Batched dynamics update.

        Dispatches to the path chosen at construction. ``bk`` is the backend
        the plant's nonlinear ``dynamics`` evaluates with, defaulting to the
        plant's own: tracing passes the component's ``TraceBackend`` here,
        because ``trace_node`` swaps only the traced component's ``self.bk`` —
        a plant calling ``self.bk.sin`` on a tracer would otherwise reach a
        concrete backend and fail. The LTI path ignores ``bk`` (operator
        arithmetic already routes through the tracer).

        Args:
            x_batch: Batch of states (N, D_x).
            u_batch: Batch of controls (N, D_u).
            dt: Time step (s).
            bk: Backend for the nonlinear path; defaults to the plant's backend.

        Returns:
            Batch of next states (N, D_x).
        """
        if self._nonlinear:
            f = self.plant.dynamics(x_batch, u_batch, bk=self.bk if bk is None else bk)
            return x_batch + dt * f
        return x_batch @ self._A.T + u_batch @ self._B.T

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

        The row sum is written as a contraction rather than
        ``bk.sum(..., axis=1)``: a sum over an axis *is* a matmul identity, and
        the operator form traces — ``@`` lifts the concrete ones operand into a
        const node — whereas ``bk.sum`` dispatches through whichever backend
        the *plant* holds and never sees a traced operand. Same reduction, no
        new op in the VM.

        Args:
            z: Batch of vectors (N, D).
            W: Diagonal (D,) or full (D, D) weight matrix.

        Returns:
            Per-sample quadratic value (N,).
        """
        if W is None:
            return self.bk.zeros(z.shape[0])
        if W.ndim == 1:
            # Diagonal weights: (z*z) contracted with W is exactly
            # Σ_i W_i z_i² — W plays the contraction vector's role, so no ones
            # vector is needed (and no rank-differing broadcast, which the
            # tracer's elementwise ops reject even though the VM supports it).
            return (z * z) @ W
        # Full W: Σ_j z_j (z Wᵀ)_j, a row-wise dot product — contracted with a
        # ones vector (there is no matmul identity for that one without it).
        ones = self.bk.array([1.0] * z.shape[-1])
        return (z * (z @ W.T)) @ ones
