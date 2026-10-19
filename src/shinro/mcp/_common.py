"""Shared MCP state and conversion helpers.

The FastMCP instance, the instance store, and the small helpers every tool
module needs live here so the tool modules register against one server object
and read/write one store. Nothing in this module imports a tool module, so the
import graph stays acyclic: ``_common`` <- tool modules <- ``server``.
"""

from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

from shinro.utils.array_backend import NumpyBackend, TorchBackend
from shinro.utils.controllability_checker import LTISystemsAnalyzer

server = FastMCP("shinro")
_store: dict[str, Any] = {}
#: Traced/lowered graphs live in their own registry: a ``NodeGraph`` and a
#: controller can share a name without either tool mistaking one for the other.
_graph_store: dict[str, Any] = {}


def _resolve_backend(backend: str | None = None):
    if backend is None or backend == "numpy":
        return NumpyBackend()
    elif backend == "torch":
        return TorchBackend()
    raise ValueError(f"Unknown backend '{backend}'. Use 'numpy' or 'torch'.")


def _to_list(arr) -> list[float]:
    return arr.flatten().tolist()


def _from_list(data: list[float]):
    return np.array(data, dtype=np.float64)


def _set_mpc_default_constraints(ctrl: Any) -> None:
    if not hasattr(ctrl, "A_constraints"):
        import scipy.sparse as sp

        bk = ctrl.bk
        n_c = ctrl.m
        F = bk.eye(n_c)
        lo = bk.zeros(n_c) - 1e10
        hi = bk.zeros(n_c) + 1e10
        F_np = bk.to_numpy(F)
        ctrl.A_constraints = sp.csc_matrix(sp.block_diag([sp.coo_array(F_np)] * ctrl.N))
        ctrl.lcons = bk.tile(lo, ctrl.N)
        ctrl.ucons = bk.tile(hi, ctrl.N)


def _make_analyzer(
    A: list[list[float]],
    B: list[list[float]] | None = None,
    C: list[list[float]] | None = None,
    D: list[list[float]] | None = None,
    dt: float | None = None,
) -> LTISystemsAnalyzer:
    A_arr = np.array(A, dtype=np.float64)
    B_arr = np.array(B, dtype=np.float64) if B is not None else None
    C_arr = np.array(C, dtype=np.float64) if C is not None else None
    D_arr = np.array(D, dtype=np.float64) if D is not None else None
    return LTISystemsAnalyzer(A=A_arr, B=B_arr, C=C_arr, D=D_arr, dt=dt)
