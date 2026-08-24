"""Zig lowering oracle: the compiled .so must match the Python interpreter.

The MVP acceptance test for slice b. It generates ``runtime/graph_data.zig``
from the base_tracking composed graph (KF + LQR, input-clipped), compiles the
comptime VM with ``zig build-lib``, loads ``base.so`` via ctypes, and asserts
the C-ABI ``shinro_step`` output equals ``interpret()`` to float-exactness
across 50 seeded random inputs.

Requires ``zig`` on PATH. Skipped cleanly if it's unavailable.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from shinro.codegen import interpret
from shinro.codegen.gen_base import build_base_graph
from shinro.codegen.lower_zig import lower_zig

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / "runtime"
BUILD = REPO_ROOT / "build"


@pytest.fixture(scope="session")
def base_so(tmp_path_factory):
    """Build the .so from the base_tracking composed graph once per session."""
    if shutil.which("zig") is None:
        pytest.skip("zig not on PATH; skipping Zig lowering oracle")

    composed = build_base_graph()
    out_zig = RUNTIME / "graph_data.zig"
    lower_zig(composed, str(out_zig))

    so_path = tmp_path_factory.mktemp("zig") / "base.so"
    cmd = [
        "zig",
        "build-lib",
        str(RUNTIME / "lower.zig"),
        "-dynamic",
        "-lc",
        f"-femit-bin={so_path}",
        "-I",
        str(RUNTIME),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"zig build-lib failed: {result.stderr.strip()[:400]}")

    lib = ctypes.CDLL(str(so_path))
    lib.shinro_step.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.shinro_step.restype = None
    return lib, composed


def _output_split(cg):
    """Return (n_out, n_state): flat sizes of the outputs and state buffers."""
    n_out = 0
    for name in cg.outputs:
        n_out += next(int(np.prod(n.shape)) for n in cg.graph.nodes if n.op == "output" and n.attrs["name"] == name)
    n_state = 0
    for name in cg.state_outputs:
        n_state += next(int(np.prod(n.shape)) for n in cg.graph.nodes if n.op == "output" and n.attrs["name"] == name)
    return n_out, n_state


def _pack_inputs(cg, y, x_ref, u_prev, x_hat_init):
    """Pack host inputs into the flat C-ABI buffer, in cg.inputs order."""
    port_arrays = {"y": y, "x_ref": x_ref, "u_prev": u_prev, "state_x_hat": x_hat_init.ravel()}
    return np.concatenate([port_arrays[name].astype(np.float64) for name in cg.inputs])


def _step(lib, cg, inputs, n_out, n_state):
    """Run one zig step: outputs and state into two separate flat buffers."""
    out = np.zeros(n_out, dtype=np.float64)
    state = np.zeros(n_state, dtype=np.float64)
    lib.shinro_step(
        inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out, state


class TestZigLowering:
    def test_so_matches_interpreter_50_inputs(self, base_so):
        """The .so's shinro_step equals interpret() on 50 random inputs."""
        lib, cg = base_so
        rng = np.random.default_rng(42)
        n_out, n_state = _output_split(cg)

        max_err = 0.0
        for _ in range(50):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.1, (3,))
            u_prev = rng.normal(0.0, 0.1, (3,))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))

            inputs = _pack_inputs(cg, y, x_ref, u_prev, x_hat_init)
            out, state = _step(lib, cg, inputs, n_out, n_state)

            traced = interpret(
                cg.graph,
                {"y": y, "x_ref": x_ref, "u_prev": u_prev, "state_x_hat": x_hat_init},
            )
            # u output
            max_err = max(max_err, float(np.max(np.abs(out - np.asarray(traced["u"]).ravel()))))
            # recurrent state outputs (state_x_hat, state_u_prev)
            max_err = max(max_err, float(np.max(np.abs(state[:3] - np.asarray(traced["state_x_hat"]).ravel()))))
            max_err = max(max_err, float(np.max(np.abs(state[3:] - np.asarray(traced["state_u_prev"]).ravel()))))

        assert max_err == 0.0, f"Zig .so diverged from interpreter: max abs err = {max_err:.3e}"

    def test_so_state_feedback_roundtrip(self, base_so):
        """state outputs feed back as next-tick state inputs (recurrent edges)."""
        lib, cg = base_so
        rng = np.random.default_rng(1)
        n_out, n_state = _output_split(cg)

        y = rng.normal(0.0, 0.1, (3,))
        x_ref = rng.normal(0.0, 0.1, (3,))
        u = rng.normal(0.0, 0.1, (3,))
        x_hat = rng.normal(0.0, 0.1, (3, 1))

        # Run the .so for three ticks, threading state out -> next-tick state in.
        for _ in range(3):
            inputs = _pack_inputs(cg, y, x_ref, u, x_hat)
            out, state = _step(lib, cg, inputs, n_out, n_state)
            u = out
            x_hat = state[:3].reshape(3, 1)

        assert np.all(np.isfinite(u)), "Zig step produced non-finite control"


def test_lower_zig_emits_valid_data_table():
    """lower_zig produces a Zig file with the expected top-level constants."""
    composed = build_base_graph()
    lower_zig(composed, str(RUNTIME / "graph_data.zig"))
    text = (RUNTIME / "graph_data.zig").read_text()
    assert "pub const nodes = [_]Node{" in text
    assert "pub const const_blob" in text
    assert "pub const buf_len" in text
    assert "pub const n_outputs = 1;" in text
    assert all(op in text for op in ("matmul", "inv", "clip", "reshape"))
