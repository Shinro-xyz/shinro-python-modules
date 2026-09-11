"""The unified ``.so``-vs-interpreter oracle kit.

ONE home for the comparison conventions every consumer shares: seeded random
inputs (square 2-D state ports made SPD), flat C-ABI packing, per-port max-abs
comparison, and the QP/non-QP tolerance tier. Consumed by the build-time gate
(:mod:`shinro.codegen.scenario_build`), the pytest oracle suite
(``tests/test_zig_lowering.py`` and friends), and shinro-bench's fidelity
payload builder. The on-target compare script stays numpy-free by design —
payload production is laptop-side, which is this module.

The pytest suite exercises this module directly, so a bug in the oracle
itself fails the suite instead of silently invalidating every comparison
downstream (build gate, CI, on-Pi verification).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from shinro.codegen import interpret

#: Tolerance for the .so-vs-interpreter oracle. Non-QP graphs (KF+LQR, KF+PID)
#: agree to float-exactness (the Zig Gauss-Jordan inv differs from LAPACK only
#: in last ulps); QP graphs settle within the solver's eps (same precedent as
#: the TestSolveQpOracle / TestMpcComposedOracle tolerances).
TOL_NON_QP = 1e-12
TOL_QP = 1e-3


def input_shape(graph, name: str) -> tuple[int, ...]:
    """The declared shape of a named input port."""
    for node in graph.nodes:
        if node.op == "input" and node.attrs["name"] == name:
            return node.shape
    raise KeyError(f"input port '{name}' not found in graph")


def output_shape(graph, name: str) -> tuple[int, ...]:
    """The declared shape of a named output port."""
    for node in graph.nodes:
        if node.op == "output" and node.attrs["name"] == name:
            return node.shape
    raise KeyError(f"output port '{name}' not found in graph")


def output_size(graph, name: str) -> int:
    """The flat size of a named output port."""
    return int(np.prod(output_shape(graph, name)))


def output_split(cg) -> tuple[int, int]:
    """Flat sizes of the (outputs, state) buffers for a composed graph."""
    n_out = sum(output_size(cg.graph, name) for name in cg.outputs)
    n_state = sum(output_size(cg.graph, name) for name in cg.state_outputs)
    return n_out, n_state


def state_slices(cg) -> dict[str, tuple[int, int]]:
    """Map each state-output port name to its (start, stop) in the flat state buffer."""
    slices: dict[str, tuple[int, int]] = {}
    off = 0
    for name in cg.state_outputs:
        size = output_size(cg.graph, name)
        slices[name] = (off, off + size)
        off += size
    return slices


def pack_arrays(cg, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Pack a port-name -> array dict into the flat C-ABI input buffer."""
    return np.concatenate([np.asarray(arrays[name], dtype=np.float64).ravel() for name in cg.inputs])


def random_inputs(cg, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Random inputs per port; square 2-D state ports are made SPD (covariance-like)."""
    inputs: dict[str, np.ndarray] = {}
    for name in cg.inputs:
        shape = input_shape(cg.graph, name)
        if len(shape) == 2 and shape[0] == shape[1]:
            a = rng.normal(0.0, 0.1, shape)
            inputs[name] = a @ a.T + 0.1 * np.eye(shape[0])
        else:
            inputs[name] = rng.normal(0.0, 0.1, shape)
    return inputs


def load_so(prefix: str | Path):
    """dlopen ``<prefix>/lib/libbase.so`` and wire up the shinro_step C ABI."""
    so_path = Path(prefix) / "lib" / "libbase.so"
    if not so_path.exists():
        raise FileNotFoundError(f"zig build produced no libbase.so at {so_path}")
    lib = ctypes.CDLL(str(so_path))
    lib.shinro_step.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3
    lib.shinro_step.restype = None
    return lib


def step_so(lib, inputs: np.ndarray, n_out: int, n_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Run one shinro_step: outputs and state into two separate flat buffers."""
    out = np.zeros(n_out, dtype=np.float64)
    state = np.zeros(n_state, dtype=np.float64)
    lib.shinro_step(
        inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out, state


def compare_ports(cg, out: np.ndarray, state: np.ndarray, traced: dict[str, np.ndarray]) -> float:
    """Max abs error across every output and state port, in port order."""
    max_err = 0.0
    off = 0
    for name in cg.outputs:
        expected = np.asarray(traced[name]).ravel()
        max_err = max(max_err, float(np.max(np.abs(out[off : off + expected.size] - expected))))
        off += expected.size
    slices = state_slices(cg)
    for name in cg.state_outputs:
        expected = np.asarray(traced[name]).ravel()
        start, stop = slices[name]
        max_err = max(max_err, float(np.max(np.abs(state[start:stop] - expected))))
    return max_err


def run_oracle(lib, cg, samples: int, seed: int) -> float:
    """Compare the .so's shinro_step against interpret() on random inputs.

    The full host-side oracle loop: seeded samples -> interpret() reference ->
    shinro_step -> per-port max-abs error. Returns the max error across every
    sample and port; the caller applies the tolerance (from :func:`tol_for`
    or a ``[compile].oracle_tol`` override).
    """
    rng = np.random.default_rng(seed)
    n_out, n_state = output_split(cg)
    max_err = 0.0
    for _ in range(samples):
        inputs = random_inputs(cg, rng)
        packed = pack_arrays(cg, inputs)
        out, state = step_so(lib, packed, n_out, n_state)
        traced = interpret(cg.graph, inputs)
        max_err = max(max_err, compare_ports(cg, out, state, traced))
    return max_err


def tol_for(manifest: dict) -> float:
    """The oracle gate for a graph: QP tier when the graph has a .solve_qp node."""
    return TOL_QP if manifest.get("has_solve_qp") else TOL_NON_QP
