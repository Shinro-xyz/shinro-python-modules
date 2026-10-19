"""Shared MCP state and conversion helpers for both servers.

There are **two** servers, because the tool families serve different phases of
work and a client should not pay for the ones it isn't using:

- :data:`server` (``shinro-mcp``) — designing and running components:
  controllers, estimators, trajectories, LTI analysis.
- :data:`codegen_server` (``shinro-mcp-codegen``) — the compile pipeline:
  tracing, lowering, and the verification gates.

Each tool module registers against one of them at import time. Nothing in this
module imports a tool module, so the import graph stays acyclic:
``_common`` <- tool modules <- ``server`` / ``codegen_server``.
"""

from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

from shinro.utils.array_backend import NumpyBackend, TorchBackend
from shinro.utils.controllability_checker import LTISystemsAnalyzer

server = FastMCP("shinro")
#: The compile-pipeline server. Tools registered here are only reachable from a
#: client connected to ``shinro-mcp-codegen``; importing the tool modules does
#: not leak them onto :data:`server`.
codegen_server = FastMCP("shinro-codegen")
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
