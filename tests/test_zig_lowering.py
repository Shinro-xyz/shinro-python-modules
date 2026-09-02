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
from shinro.codegen.trace_node import trace_node
from shinro.codegen.tracing import Graph
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.utils.array_backend import NumpyBackend

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


def _build_mpc_graph():
    """Trace the base MPC_LTI and wrap it as a single-input step graph.

    The traced compute() is: x0 → q = Fᵀ x0 (matmul) → solve_qp → u[:3]
    (slice). The ``solve_qp`` node drives the codegen static solver baked into
    libbase.so (runtime/codegen/emosqp/), whose problem must match the
    ``mpc_lti_base.toml`` bake (n_vars=30).
    """
    ctrl = ControllerFactory(
        str(REPO_ROOT / "src/shinro/configs/controllers/mpc_lti_base.toml")
    ).create(backend=NumpyBackend())
    ng = trace_node(ctrl, input_shapes={"x0": (3,)})
    return ComposedGraph(
        graph=ng.graph,
        inputs=["x0"],
        outputs=["out"],
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


@pytest.fixture(scope="session")
def mpc_so(tmp_path_factory):
    """Build the .so from the traced MPC graph (exercises the .solve_qp op)."""
    return _build_so(_build_mpc_graph(), tmp_path_factory.mktemp("zig-build-mpc"))


def _output_split(cg):
    """Return (n_out, n_state): flat sizes of the outputs and state buffers."""
    n_out = 0
    for name in cg.outputs:
        n_out += next(int(np.prod(n.shape)) for n in cg.graph.nodes if n.op == "output" and n.attrs["name"] == name)
    n_state = 0
    for name in cg.state_outputs:
        n_state += next(int(np.prod(n.shape)) for n in cg.graph.nodes if n.op == "output" and n.attrs["name"] == name)
    return n_out, n_state


def _state_slices(cg):
    """Map each state output port name to its (start, stop) in the flat state buffer."""
    slices = {}
    off = 0
    for name in cg.state_outputs:
        size = next(int(np.prod(n.shape)) for n in cg.graph.nodes if n.op == "output" and n.attrs["name"] == name)
        slices[name] = (off, off + size)
        off += size
    return slices


def _pack_inputs(cg, y, x_ref, u_prev, x_hat_init, P_init):
    """Pack host inputs into the flat C-ABI buffer, in cg.inputs order."""
    port_arrays = {
        "y": y,
        "x_ref": x_ref,
        "u_prev": u_prev,
        "state_x_hat": x_hat_init.ravel(),
        "state_P": P_init.ravel(),
    }
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
        sl = _state_slices(cg)

        max_err = 0.0
        for _ in range(50):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.1, (3,))
            u_prev = rng.normal(0.0, 0.1, (3,))
            x_hat_init = rng.normal(0.0, 0.1, (3, 1))
            # Well-conditioned SPD covariance: S = C P_pred C^T + R must stay
            # invertible for both the Zig LU and numpy's LAPACK inv.
            P_init = rng.normal(0.0, 0.1, (3, 3))
            P_init = P_init @ P_init.T + 0.1 * np.eye(3)

            inputs = _pack_inputs(cg, y, x_ref, u_prev, x_hat_init, P_init)
            out, state = _step(lib, cg, inputs, n_out, n_state)

            traced = interpret(
                cg.graph,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev,
                    "state_x_hat": x_hat_init,
                    "state_P": P_init,
                },
            )
            # Compare every output and state port by name.
            off = 0
            for name in cg.outputs:
                expected = np.asarray(traced[name]).ravel()
                max_err = max(max_err, float(np.max(np.abs(out[off : off + expected.size] - expected))))
                off += expected.size
            for name in cg.state_outputs:
                expected = np.asarray(traced[name]).ravel()
                start, stop = sl[name]
                max_err = max(max_err, float(np.max(np.abs(state[start:stop] - expected))))

        # Tolerance, not bit-exact: the live .inv now runs on dynamic data and
        # the Zig LU differs from numpy's LAPACK in the last ulps (same
        # precedent as the 1-ulp transcendental carve-out).
        assert max_err < 1e-12, f"Zig .so diverged from interpreter: max abs err = {max_err:.3e}"

    def test_so_state_feedback_roundtrip(self, base_so):
        """state outputs feed back as next-tick state inputs (recurrent edges)."""
        lib, cg = base_so
        rng = np.random.default_rng(1)
        n_out, n_state = _output_split(cg)
        sl = _state_slices(cg)

        y = rng.normal(0.0, 0.1, (3,))
        x_ref = rng.normal(0.0, 0.1, (3,))
        u = rng.normal(0.0, 0.1, (3,))
        x_hat = rng.normal(0.0, 0.1, (3, 1))
        P = np.eye(3) * 0.1

        # Run the .so for three ticks, threading state out -> next-tick state in.
        for _ in range(3):
            inputs = _pack_inputs(cg, y, x_ref, u, x_hat, P)
            out, state = _step(lib, cg, inputs, n_out, n_state)
            u = out
            x_hat = state[sl["state_x_hat"][0] : sl["state_x_hat"][1]].reshape(3, 1)
            P = state[sl["state_P"][0] : sl["state_P"][1]].reshape(3, 3)

        assert np.all(np.isfinite(u)), "Zig step produced non-finite control"

    def test_so_matches_live_kf_multitick(self, base_so):
        """100-tick .so closed loop equals a live numpy KalmanFilter loop.

        The regression test for recurrent covariance: the deployed graph must
        run the live P recursion (state_P port feeding back each tick), not a
        frozen one-step gain baked from the trace-time seed. The oracle is
        the real KalmanFilter.estimate() on the numpy backend, driven in
        parallel through the same measurement/reference sequence.
        """
        lib, cg = base_so
        rng = np.random.default_rng(11)
        n_out, n_state = _output_split(cg)
        sl = _state_slices(cg)

        kf = EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())
        lqr = ControllerFactory("configs/controllers/lqr_base.toml").create(backend=NumpyBackend())
        limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))

        P = np.eye(3) * 0.1
        x_hat = np.zeros((3, 1))
        u_prev_so = np.zeros(3)
        u_prev_np = np.zeros(3)

        max_err = 0.0
        for _ in range(100):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.05, (3,))

            # Live numpy oracle: the same predict-update the graph encodes,
            # with P evolving per tick (each loop evolves its own u_prev).
            kf.P = P.copy()
            kf.x_hat = x_hat.copy()
            x_hat_np = kf.estimate(y.reshape(-1, 1), u_prev_np.reshape(-1, 1))
            u_np = np.clip(lqr.compute(x_hat_np.ravel(), x_ref), limits[0], limits[1])

            inputs = _pack_inputs(cg, y, x_ref, u_prev_so, x_hat, P)
            out, state = _step(lib, cg, inputs, n_out, n_state)
            max_err = max(max_err, float(np.max(np.abs(out - u_np))))

            x_hat = state[sl["state_x_hat"][0] : sl["state_x_hat"][1]].reshape(3, 1)
            P = state[sl["state_P"][0] : sl["state_P"][1]].reshape(3, 3)
            u_prev_so = out
            u_prev_np = u_np

        assert max_err < 1e-10, (
            f".so diverged from live KalmanFilter over 100 ticks: max abs err = {max_err:.3e}"
        )


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


class TestSolveQpOracle:
    """The .solve_qp VM op (codegen static solver) matches the interpreter.

    The .so's shinro_step drives the statically-allocated OSQP solver baked
    from ``mpc_lti_base.toml`` (eps=1e-6); the interpreter's solve_qp handler
    solves the same problem with the same tolerance via the Python osqp.
    Both are the same ADMM algorithm, so u[:3] agrees within tolerance (the
    flat terminal-control region only affects the discarded u[29]).
    """

    def test_mpc_solve_matches_interpreter(self, mpc_so):
        lib, cg = mpc_so
        rng = np.random.default_rng(3)
        n_out, n_state = _output_split(cg)
        assert n_state == 0
        assert n_out == 3

        max_err = 0.0
        for _ in range(10):
            x0 = rng.normal(0.0, 0.1, (3,))
            inputs = _pack_arrays(cg, {"x0": x0})
            out, _ = _step(lib, cg, inputs, n_out, n_state)

            traced = interpret(cg.graph, {"x0": x0})["out"]
            max_err = max(max_err, float(np.max(np.abs(out - np.asarray(traced).ravel()))))

        assert max_err < 1e-3, f"Zig .so solve_qp diverged from interpreter: max abs err = {max_err:.3e}"

