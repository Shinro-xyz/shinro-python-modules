from typing import Callable, Optional, Tuple, Any
import numpy as np
from utils.array_backend import ArrayBackend, NumpyBackend


def linearize(
    f: Callable[[Any, Any], Any],
    x0: Any,
    u0: Any,
    backend: Optional[ArrayBackend] = None,
    eps: float = 1e-6,
) -> Tuple[Any, Any]:
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
