"""Rank helpers for batch-capable plant dynamics.

``Plant.dynamics`` is called two ways:

* **Per-sample** — a single ``(n_x,)`` state. The eager nonlinear rollout, the
  finite-difference linearization (``linearize_plant``), and ``step()`` all use
  this form.
* **Whole-batch** — an ``(N, n_x)`` state, evaluated by the MPPI rollout. This
  is the form that is lowered, and its graph cost must not grow with ``N``.

These helpers make the rank handling explicit and identical in every plant, so
the physics body is written once, in the column idiom, with no branching:

    x, single = as_batch(bk, state)      # (1, n) or (N, n), plus the flag
    theta = column(bk, x, 2)             # (N, 1)
    ...
    return as_vector(bk, f, single)      # (n,) or (N, n)

The column idiom — ``bk.slice_(x.T, j, j + 1).T`` — transposes the ``(N, D)``
batch to ``(D, N)``, slices one coordinate of every sample, and transposes back
to an ``(N, 1)`` column. Scalar indexing (``state[j]``) cannot be used on the
batched path because the tracing backend's :class:`~shinro.codegen.tracing.Tracer`
has no ``__getitem__``.

Every helper takes the backend explicitly: plant code must call the backend it
was handed, not ``self.bk``, because ``trace_node`` swaps only the traced
component's backend — never the plant's.
"""

from typing import Any

from shinro.utils.array_backend import ArrayBackend


def as_batch(bk: ArrayBackend, x: Any) -> tuple[Any, bool]:
    """Promote a 1-D vector to a ``(1, n)`` row.

    Args:
        bk: Backend to reshape with.
        x: State vector ``(n,)`` or batch ``(N, n)``.

    Returns:
        ``(x_2d, was_1d)`` — the batch form and whether ``x`` was a single
        vector (so the derivative can be converted back).
    """
    if getattr(x, "shape", None) is None:
        x = bk.array(x)
    if len(x.shape) == 1:
        return bk.reshape(x, (1, x.shape[0])), True
    return x, False


def as_vector(bk: ArrayBackend, f: Any, was_1d: bool) -> Any:
    """Undo :func:`as_batch` for a derivative.

    Args:
        bk: Backend to reshape with.
        f: Derivative batch ``(N, n)``.
        was_1d: The flag returned by :func:`as_batch`.

    Returns:
        ``(n,)`` when the state was a single vector, else ``f`` unchanged.
    """
    return bk.ravel(f) if was_1d else f


def column(bk: ArrayBackend, x: Any, j: int) -> Any:
    """Column ``j`` of a 2-D ``(N, D)`` batch as an ``(N, 1)`` column.

    Args:
        bk: Backend to slice with.
        x: 2-D batch ``(N, D)``.
        j: Column index (compile-time constant).

    Returns:
        Column ``j``, shape ``(N, 1)``.
    """
    return bk.slice_(x.T, j, j + 1).T


def control_batch(bk: ArrayBackend, control: Any, n_u: int) -> Any:
    """Normalize a control to a 2-D ``(N, n_u)`` (or ``(1, n_u)``) row.

    Accepts a scalar (``n_u == 1``, or the first input with the rest zero for
    ``n_u > 1`` — the historical scalar-control convention), a ``(n_u,)``
    vector, or an ``(N, n_u)`` batch.

    Args:
        bk: Backend to build with.
        control: Scalar, ``(n_u,)``, or ``(N, n_u)``.
        n_u: Control dimension.

    Returns:
        A 2-D control row/batch.
    """
    if getattr(control, "shape", None) is None:
        control = bk.array([control] + [0.0] * (n_u - 1))
    if len(control.shape) == 0:
        control = bk.reshape(bk.array([control] + [0.0] * (n_u - 1)), (n_u,))
    if len(control.shape) == 1:
        return bk.reshape(control, (1, control.shape[0]))
    return control
