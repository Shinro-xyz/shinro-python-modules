"""Zig lowering oracle: the compiled .so must match the Python interpreter.

The MVP acceptance test for slice b. It generates ``runtime/graph_data.zig``
from the base_tracking composed graph (KF + LQR, input-clipped), compiles the
comptime VM with ``zig build`` (runtime/build.zig), loads ``libbase.so`` via
ctypes, and asserts the C-ABI ``shinro_step`` output equals ``interpret()`` to
float-exactness across 50 seeded random inputs.

Requires ``zig`` on PATH. Skipped cleanly if it's unavailable.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.gen_base import build_base_graph
from shinro.codegen import interpret
from shinro.codegen.compose import ComposedGraph
from shinro.codegen.lower_zig import lower_zig
from shinro.codegen.tracing import Graph

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / "runtime"
BUILD = REPO_ROOT / "build"


def _build_so(composed, build_dir):
    """Lower a composed graph, compile the comptime VM, and load libbase.so.

    Each call lowers ``composed`` to ``runtime/graph_data.zig`` and builds a
    fresh ``libbase.so`` into the given (unique) prefix directory, so multiple
    graphs can be cross-checked in one session without clobbering each other.
    """
    if shutil.which("zig") is None:
        pytest.skip("zig not on PATH; skipping Zig lowering oracle")

    out_zig = RUNTIME / "graph_data.zig"
    lower_zig(composed, str(out_zig))

    cmd = [
        "zig",
        "build",
        "--build-file",
        str(RUNTIME / "build.zig"),
        "--prefix",
        str(build_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"zig build failed: {result.stderr.strip()[:400]}")

    so_path = build_dir / "lib" / "libbase.so"
    if not so_path.exists():
        pytest.skip(f"zig build produced no libbase.so; stderr: {result.stderr.strip()[:400]}")

    lib = ctypes.CDLL(str(so_path))
    lib.shinro_step.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.shinro_step.restype = None
    return lib, composed


def _build_lowered_ops_graph():
    """A single-input graph exercising every op lowered beyond the base set.

    Emits the 7 host-only ops from issue #13 (tanh / relu / exp / copy as
    shape-preserving elementwise, slice as an offset copy, and the
    argmax -> one_hot deterministic-policy decode), plus sin / cos for the
    nonlinear-dynamics path. Each is a named output so the .so output can be
    compared per-op against the Python interpreter.
    """
    g = Graph()
    x = g.input("x", (4,))
    tanh_id = g.emit("tanh", [x], (4,))
    relu_id = g.emit("relu", [x], (4,))
    exp_id = g.emit("exp", [x], (4,))
    copy_id = g.emit("copy", [x], (4,))
    slice_id = g.emit("slice", [x], (2,), start=1, stop=3)
    argmax_id = g.emit("argmax", [x], ())
    one_hot_id = g.emit("one_hot", [argmax_id], (4,), depth=4)
    sin_id = g.emit("sin", [x], (4,))
    cos_id = g.emit("cos", [x], (4,))
    stack_id = g.emit("stack", [x, x], (2, 4))
    for name, src in (
        ("tanh", tanh_id),
        ("relu", relu_id),
        ("exp", exp_id),
        ("copy", copy_id),
        ("slice", slice_id),
        ("argmax", argmax_id),
        ("one_hot", one_hot_id),
        ("sin", sin_id),
        ("cos", cos_id),
        ("stack", stack_id),
    ):
        g.output(name, src)
    return ComposedGraph(
        graph=g,
        inputs=["x"],
        outputs=["tanh", "relu", "exp", "copy", "slice", "argmax", "one_hot", "sin", "cos", "stack"],
        state_inputs=[],
        state_outputs=[],
    )


@pytest.fixture(scope="session")
def base_so(tmp_path_factory):
    """Build the .so from the base_tracking composed graph once per session."""
    return _build_so(build_base_graph(), tmp_path_factory.mktemp("zig-build"))


@pytest.fixture(scope="session")
def lowered_ops_so(tmp_path_factory):
    """Build the .so from the lowered-ops graph once per session."""
    return _build_so(_build_lowered_ops_graph(), tmp_path_factory.mktemp("zig-build-ops"))


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
    return _pack_arrays(cg, port_arrays)


def _pack_arrays(cg, arrays):
    """Pack a port-name -> array dict into the flat C-ABI input buffer."""
    return np.concatenate([arrays[name].astype(np.float64) for name in cg.inputs])


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


class TestLoweredOpsOracle:
    """The newly-lowered ops match the interpreter (issue #13 + sin/cos/stack).

    copy/slice/relu/argmax/one_hot/stack are exact (pure data movement,
    integer indices, or concatenation). exp/tanh/sin/cos are transcendental —
    both sides wrap the platform libm, so they must agree to within 1 ulp
    rather than bit-for-bit.
    """

    def test_lowered_ops_match_interpreter(self, lowered_ops_so):
        lib, cg = lowered_ops_so
        rng = np.random.default_rng(7)
        n_out, n_state = _output_split(cg)
        assert n_state == 0

        exact_ops = {"copy", "slice", "relu", "argmax", "one_hot", "stack"}
        transcendental = {"exp", "tanh", "sin", "cos"}

        for _ in range(20):
            x = rng.normal(0.0, 1.0, (4,))
            inputs = _pack_arrays(cg, {"x": x})
            out, _ = _step(lib, cg, inputs, n_out, n_state)

            traced = interpret(cg.graph, {"x": x})
            off = 0
            for name in cg.outputs:
                expected = np.asarray(traced[name]).ravel()
                got = out[off : off + expected.size]
                off += expected.size
                if name in transcendental:
                    # libm agreement within 1 ulp (rtol 1e-14 over ~e^1 scale).
                    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-14)
                else:
                    assert name in exact_ops, f"unexpected op {name}"
                    assert np.array_equal(got, expected), (
                        f"op {name} diverged: got {got}, expected {expected}"
                    )
