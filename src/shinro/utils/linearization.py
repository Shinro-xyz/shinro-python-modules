import tomllib
from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any

import numpy as np

from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.utils.config_resolver import resolve_config_path


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


def discretize_euler(A_c, B_c, dt, backend: ArrayBackend | None = None):
    """First-order Euler discretization of a continuous-time linear model.

    Computes the discrete-time matrices :math:`A_d = I + dt \\cdot A_c` and
    :math:`B_d = dt \\cdot B_c`, matching the semi-implicit Euler integration
    used by the analytical plants. At small ``dt`` the discretization error is
    negligible, and unlike ``scipy.linalg.expm`` it is backend-agnostic (the
    Kalman filter stays torch-capable).

    Args:
        A_c: Continuous-time state matrix (n_x, n_x).
        B_c: Continuous-time input matrix (n_x, n_u).
        dt: Time step in seconds.
        backend: Array backend. Defaults to NumpyBackend.

    Returns:
        Tuple (A_d, B_d) in the backend's native type.
    """
    bk = backend or NumpyBackend()
    n = A_c.shape[0]
    A_d = bk.eye(n) + dt * A_c
    B_d = dt * B_c
    return A_d, B_d


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


def derive_model(plant):
    """Return the plant's discrete-time ``(A_d, B_d)`` model.

    Delegates to ``plant.get_model()``, which every plant implements as the
    discrete-time model (analytical plants linearize + Euler-discretize at
    their ``dt``; velocity-commanded plants return ``A = I, B = dt·I``
    directly). This is the single source of the derived model shared by the
    simulation and compile paths.

    Args:
        plant: A plant exposing ``get_model()``.

    Returns:
        Tuple ``(A_d, B_d)`` in the plant's backend native type.
    """
    return plant.get_model()


def inject_model(cfg, plant) -> dict:
    """Fill ``A_dynamics``/``B_dynamics`` from the plant when a config omits them.

    ``cfg`` may be a config dict or a TOML path (resolved via
    :func:`shinro.utils.config_resolver.resolve_config_path`); a dict is
    returned either way. A config that already declares either matrix is left
    untouched — explicit model wins over the derived one.

    Args:
        cfg: Controller/estimator config dict or TOML path.
        plant: The plant to derive the model from.

    Returns:
        The config dict with ``A_dynamics``/``B_dynamics`` filled in when they
        were absent.
    """
    if isinstance(cfg, str):
        with open(resolve_config_path(cfg), "rb") as f:
            cfg = tomllib.load(f)
    if "A_dynamics" not in cfg and "B_dynamics" not in cfg:
        A_d, B_d = derive_model(plant)
        cfg = {**cfg, "A_dynamics": A_d, "B_dynamics": B_d}
    return cfg


def inject_plant_derived(cfg, plant, *, with_model: bool = False):
    """Fill ``dt`` / ``A_dynamics`` / ``B_dynamics`` on a Config dataclass from the plant.

    The plant is the single source of physics truth. Precedence: **explicit
    wins, derived fills, disagreement is loud**.

    - ``dt``: filled from ``plant.dt`` when the config omits it; a declared
      ``dt`` that disagrees with the plant's is a loud error (a mismatched
      PID/MPPI ``dt`` silently mis-scales integration otherwise).
    - ``A_dynamics``/``B_dynamics``: derived via :func:`derive_model` and
      injected (as TOML-serializable lists) when the config declares *neither*
      — only when ``with_model`` is set, so sim-backed velocity-commanded
      plants keep their untouched ``A = I, B = dt·I`` defaults.

    Args:
        cfg: A component Config dataclass instance (from
            :meth:`shinro.components.ConfigDriven.parse_config`).
        plant: The plant to derive from.
        with_model: Also derive the model when A/B are both absent.

    Returns:
        The (possibly replaced) Config dataclass.

    Raises:
        ValueError: On a ``dt`` disagreement, or a ``dt``-less config without
            ``B_dynamics`` in standalone use.
    """
    names = {f.name for f in fields(cfg)}
    if "dt" in names:
        if cfg.dt is None:
            cfg = replace(cfg, dt=float(plant.dt))
        elif abs(float(cfg.dt) - float(plant.dt)) > 1e-12:
            raise ValueError(
                f"{type(cfg).__name__}: config dt ({cfg.dt}) disagrees with plant dt "
                f"({plant.dt}) — the plant is the source of truth; omit dt to inherit it."
            )
    if with_model and "A_dynamics" in names and cfg.A_dynamics is None and cfg.B_dynamics is None:
        A_d, B_d = derive_model(plant)
        cfg = replace(cfg, A_dynamics=A_d.tolist(), B_dynamics=B_d.tolist())
    return cfg
