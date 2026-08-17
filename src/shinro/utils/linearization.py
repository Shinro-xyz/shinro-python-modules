from collections.abc import Callable
from typing import Any

import numpy as np

from shinro.utils.array_backend import ArrayBackend, NumpyBackend


def linearize(
    f: Callable[[Any, Any], Any],
    x0: Any,
    u0: Any,
    backend: ArrayBackend | None = None,
    eps: float = 1e-6,
) -> tuple[Any, Any]:
    """First-order Taylor expansion of f(x, u) around (x0, u0).

    Computes the Jacobians A = ∂f/∂x and B = ∂f/∂u at the operating point
    (x0, u0) using central finite differences. The user's dynamics function
    always receives and returns numpy arrays regardless of the backend.

    Args:
        f: Continuous-time dynamics ``f(x, u) -> dx/dt`` where x is (n_x,)
            and u is (n_u,), returns (n_x,).
        x0: Operating point state, shape (n_x,).
        u0: Operating point input, shape (n_u,).
        backend: Array backend. Defaults to NumpyBackend.
        eps: Step size for finite differences.

    Returns:
        Tuple (A, B) where A = ∂f/∂x has shape (n_x, n_x) and
        B = ∂f/∂u has shape (n_x, n_u), in the backend's native type.
    """
    bk = backend or NumpyBackend()
    x0_np = np.asarray(bk.to_numpy(x0) if hasattr(bk, 'to_numpy') else x0, dtype=np.float64)
    u0_np = np.asarray(bk.to_numpy(u0) if hasattr(bk, 'to_numpy') else u0, dtype=np.float64)

    def f_x(x):
        return np.asarray(f(x, u0_np), dtype=np.float64)

    def f_u(u):
        return np.asarray(f(x0_np, u), dtype=np.float64)

    n = x0_np.shape[0]
    m = f_x(x0_np).shape[0]
    A = np.zeros((m, n), dtype=np.float64)
    for i in range(n):
        h = np.zeros(n, dtype=np.float64)
        h[i] = eps
        A[:, i] = (f_x(x0_np + h) - f_x(x0_np - h)) / (2.0 * eps)

    r = u0_np.shape[0]
    B = np.zeros((m, r), dtype=np.float64)
    for i in range(r):
        h = np.zeros(r, dtype=np.float64)
        h[i] = eps
        B[:, i] = (f_u(u0_np + h) - f_u(u0_np - h)) / (2.0 * eps)

    return bk.from_numpy(A), bk.from_numpy(B)


def as_numpy_f(f, backend):
    """Wrap a backend-bound f(x, u) into a numpy-in/numpy-out callable.

    :func:`linearize` requires ``f`` to take and return numpy arrays
    regardless of the backend. This bridges a backend-native callable such
    as ``plant.dynamics(x_b, u_b) -> dx_b`` into that contract via
    ``backend.from_numpy`` / ``backend.to_numpy``.

    Args:
        f: Backend-bound callable ``f(x, u) -> y`` operating on backend arrays.
        backend: ArrayBackend whose ``from_numpy``/``to_numpy`` do the bridging.

    Returns:
        ``f_np(x_np, u_np) -> y_np`` operating on float64 numpy arrays.
    """
    def f_np(x, u):
        x_b = backend.from_numpy(x)
        u_b = backend.from_numpy(u)
        return np.asarray(backend.to_numpy(f(x_b, u_b)), dtype=np.float64)
    return f_np


def linearize_plant(plant, x0=None, u0=None, eps=1e-6):
    """Linearize a Plant's dynamics around an operating point.

    Resolves ``(x0, u0)`` defaults, bridges ``plant.dynamics`` through
    :func:`as_numpy_f`, and delegates to :func:`linearize`.

    State default: ``x0 = plant.bk.zeros(len(plant.get_state()))``.
    Control default: ``u0 = plant.bk.zeros(plant.input_dim)`` when the plant
    declares ``input_dim``; otherwise raises ``ValueError`` unless ``u0`` is
    passed explicitly, so multi-input plants cannot silently get a wrong-dim
    ``u0``.

    Args:
        plant: Plant with ``.dynamics``, ``.bk``, ``.get_state()``, and
            optionally ``.input_dim``.
        x0: Operating point state. Defaults to zeros(state_dim).
        u0: Operating point input. Defaults to zeros(plant.input_dim), or
            required when ``plant.input_dim`` is None.
        eps: Finite-difference step size.

    Returns:
        Tuple (A, B) in the plant's backend native type.
    """
    if x0 is None:
        x0 = plant.bk.zeros(len(plant.get_state()))
    if u0 is None:
        if plant.input_dim is None:
            raise ValueError(
                "Cannot infer control dimension: plant.input_dim is not "
                "set and u0 was not passed. Set self.input_dim on the "
                "plant, or pass u0 explicitly."
            )
        u0 = plant.bk.zeros(plant.input_dim)
    return linearize(as_numpy_f(plant.dynamics, plant.bk), x0, u0, plant.bk, eps=eps)
