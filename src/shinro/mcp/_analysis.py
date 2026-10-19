"""MCP tools for LTI system analysis: rank tests, Gramians, truncation."""

import json

import numpy as np

from shinro.mcp._common import _make_analyzer, server
from shinro.utils.controllability_checker import LTISystemsAnalyzer

# ── System analysis tools ─────────────────────────────────────────────────────


@server.tool()
def analyze_controllability(
    A: list[list[float]],
    B: list[list[float]],
) -> str:
    """Check controllability of a state-space system via Kalman rank test.

    Returns the controllability matrix, its rank, condition number,
    and whether the system is controllable.

    Args:
        A: State matrix (n x n).
        B: Input matrix (n x m).
    """
    try:
        analyzer = _make_analyzer(A=A, B=B)
        C_mat = analyzer.controllabilty()
        rank = int(np.linalg.matrix_rank(C_mat))
        n = np.array(A, dtype=np.float64).shape[0]
        return json.dumps(
            {
                "controllable": rank == n,
                "rank": rank,
                "n": n,
                "condition": float(np.linalg.cond(C_mat)) if rank == n else None,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
def analyze_observability(
    A: list[list[float]],
    C: list[list[float]],
) -> str:
    """Check observability of a state-space system via Kalman rank test.

    Returns the observability matrix, its rank, condition number,
    and whether the system is observable.

    Args:
        A: State matrix (n x n).
        C: Output matrix (p x n).
    """
    try:
        analyzer = _make_analyzer(A=A, C=C)
        O_mat = analyzer.observability()
        rank = int(np.linalg.matrix_rank(O_mat))
        n = np.array(A, dtype=np.float64).shape[0]
        return json.dumps(
            {
                "observable": rank == n,
                "rank": rank,
                "n": n,
                "condition": float(np.linalg.cond(O_mat)) if rank == n else None,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
def gramian_continuous(
    A: list[list[float]],
    B: list[list[float]] | None = None,
    C: list[list[float]] | None = None,
) -> str:
    """Compute infinite-horizon continuous-time Gramians and Hankel singular values.

    Requires A to be Hurwitz (all eigenvalues with negative real part).
    For unstable systems, use gramian_finite() instead.

    Args:
        A: State matrix (n x n).
        B: Input matrix (n x m). Defaults to zeros.
        C: Output matrix (p x n). Defaults to zeros.
    """
    try:
        analyzer = _make_analyzer(A=A, B=B, C=C)
        result = {}
        try:
            wc = analyzer.controllability_gramian()
            result["controllability_gramian"] = wc.tolist()
            result["controllability_gramian_eigs"] = np.sort(np.real(np.linalg.eigvals(wc)))[::-1].tolist()
        except ValueError as e:
            result["controllability_gramian"] = None
            result["controllability_gramian_error"] = str(e)

        try:
            wo = analyzer.observability_gramian()
            result["observability_gramian"] = wo.tolist()
            result["observability_gramian_eigs"] = np.sort(np.real(np.linalg.eigvals(wo)))[::-1].tolist()
        except ValueError as e:
            result["observability_gramian"] = None
            result["observability_gramian_error"] = str(e)

        try:
            hsv = analyzer.hankel_singular_values()
            result["hankel_singular_values"] = hsv.flatten().tolist()
        except Exception:
            result["hankel_singular_values"] = None

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
def gramian_discrete(
    A: list[list[float]],
    B: list[list[float]] | None = None,
    C: list[list[float]] | None = None,
    dt: float = 0.01,
) -> str:
    """Compute infinite-horizon discrete-time Gramians and Hankel singular values.

    Requires the discrete-time system to be asymptotically stable
    (all eigenvalues inside the unit circle).

    Args:
        A: State matrix (n x n).
        B: Input matrix (n x m). Defaults to zeros.
        C: Output matrix (p x n). Defaults to zeros.
        dt: Sampling time in seconds (default: 0.01).
    """
    try:
        analyzer = _make_analyzer(A=A, B=B, C=C, dt=dt)
        result = {}
        try:
            wc = analyzer.discrete_controllability_gramian()
            result["controllability_gramian"] = wc.tolist()
            result["controllability_gramian_eigs"] = np.sort(np.real(np.linalg.eigvals(wc)))[::-1].tolist()
        except ValueError as e:
            result["controllability_gramian"] = None
            result["controllability_gramian_error"] = str(e)

        try:
            wo = analyzer.discrete_observability_gramian()
            result["observability_gramian"] = wo.tolist()
            result["observability_gramian_eigs"] = np.sort(np.real(np.linalg.eigvals(wo)))[::-1].tolist()
        except ValueError as e:
            result["observability_gramian"] = None
            result["observability_gramian_error"] = str(e)

        try:
            hsv = analyzer.discrete_hankel_singular_values()
            result["hankel_singular_values"] = hsv.flatten().tolist()
        except Exception:
            result["hankel_singular_values"] = None

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
def gramian_finite(
    A: list[list[float]],
    B: list[list[float]] | None = None,
    C: list[list[float]] | None = None,
    T: float = 5.0,
) -> str:
    """Compute finite-horizon Gramians via ODE integration.

    Works for any A (stable or unstable). Integrates the Lyapunov
    differential equation from 0 to T.

    Args:
        A: State matrix (n x n).
        B: Input matrix (n x m). Defaults to zeros.
        C: Output matrix (p x n). Defaults to zeros.
        T: Horizon length in seconds (default: 5.0).
    """
    try:
        analyzer = _make_analyzer(A=A, B=B, C=C)
        wc = analyzer.controllability_gramian_finite(T)
        wo = analyzer.observability_gramian_finite(T)
        return json.dumps(
            {
                "controllability_gramian": wc.tolist(),
                "observability_gramian": wo.tolist(),
                "horizon": T,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
def system_summary(
    A: list[list[float]],
    B: list[list[float]] | None = None,
    C: list[list[float]] | None = None,
    D: list[list[float]] | None = None,
    dt: float | None = None,
) -> str:
    """Produce a human-readable summary of system properties.

    Includes controllability/observability status, Gramian eigenvalues,
    Hankel singular values, and balanced-truncation error bound.

    Args:
        A: State matrix (n x n).
        B: Input matrix (n x m). Defaults to zeros.
        C: Output matrix (p x n). Defaults to zeros.
        D: Feedthrough matrix (p x m). Defaults to zeros.
        dt: Sampling time for discrete-time analysis. Defaults to None (continuous).
    """
    try:
        analyzer = _make_analyzer(A=A, B=B, C=C, D=D, dt=dt)
        return json.dumps({"summary": analyzer.summary()})
    except Exception as e:
        return json.dumps({"error": str(e)})


@server.tool()
def balanced_truncation(
    A: list[list[float]],
    B: list[list[float]],
    C: list[list[float]],
    r: int,
    D: list[list[float]] | None = None,
) -> str:
    """Perform balanced truncation to obtain an order-r reduced model.

    Requires A to be Hurwitz (all eigenvalues with negative real part).

    Args:
        A: State matrix (n x n).
        B: Input matrix (n x m).
        C: Output matrix (p x n).
        r: Desired reduced order (0 < r <= n).
        D: Feedthrough matrix (p x m). Defaults to zeros.
    """
    A_arr = np.array(A, dtype=np.float64)
    B_arr = np.array(B, dtype=np.float64)
    C_arr = np.array(C, dtype=np.float64)
    D_arr = np.array(D, dtype=np.float64) if D is not None else None

    try:
        analyzer = LTISystemsAnalyzer(A=A_arr, B=B_arr, C=C_arr, D=D_arr)
        Ar, Br, Cr, Dr = analyzer.balanced_truncate(r)
        sigma = analyzer.hankel_singular_values()
        error_bound = 2.0 * float(np.sum(sigma[r:]))
        return json.dumps(
            {
                "Ar": Ar.tolist(),
                "Br": Br.tolist(),
                "Cr": Cr.tolist(),
                "Dr": Dr.tolist(),
                "reduced_order": r,
                "error_bound": error_bound,
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})
