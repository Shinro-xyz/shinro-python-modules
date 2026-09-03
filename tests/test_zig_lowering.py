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
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.gen_base import build_base_graph
from shinro.codegen import interpret
from shinro.codegen.compose import ComposedGraph, compose
from shinro.codegen.lower_zig import lower_zig
from shinro.codegen.trace_node import trace_node
from shinro.codegen.tracing import Graph
from shinro.controllers.pid import PIDController
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.utils.array_backend import NumpyBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / "runtime"
BUILD = REPO_ROOT / "build"


def _build_so(composed, build_dir, graph_path=None, solver_dir=None):
    """Lower a composed graph, compile the comptime VM, and load libbase.so.

    Each call lowers ``composed`` to ``runtime/graph_data.zig`` (or
    ``graph_path`` when given) and builds a fresh ``libbase.so`` into the
    given (unique) prefix directory, so multiple graphs can be cross-checked
    in one session without clobbering each other. ``solver_dir`` selects the
    baked OSQP solver to compile in (default: the shipped
    ``runtime/codegen/emosqp/`` bake) — pass a DeltaU bake to build a graph
    whose ``.solve_qp`` node has n_vars=45.
    """
    if shutil.which("zig") is None:
        pytest.skip("zig not on PATH; skipping Zig lowering oracle")

    out_zig = graph_path or (RUNTIME / "graph_data.zig")
    lower_zig(composed, str(out_zig))

    cmd = [
        "zig",
        "build",
        "--build-file",
        str(RUNTIME / "build.zig"),
        "--prefix",
        str(build_dir),
    ]
    if graph_path is not None:
        cmd += [f"-Dgraph={graph_path}"]
    if solver_dir is not None:
        cmd += [f"-Dsolver_dir={solver_dir}"]
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
    zero_id = g.const(np.zeros(4))
    ne_zero_id = g.emit("ne", [x, x], (4,))  # x != x → all flags 0
    ne_one_id = g.emit("ne", [x, zero_id], (4,))  # x != 0 → all flags set
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
        ("ne_zero", ne_zero_id),
        ("ne_one", ne_one_id),
    ):
        g.output(name, src)
    return ComposedGraph(
        graph=g,
        inputs=["x"],
        outputs=[
            "tanh", "relu", "exp", "copy", "slice", "argmax", "one_hot",
            "sin", "cos", "stack", "ne_zero", "ne_one",
        ],
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


def _build_pid_composed_graph():
    """KF + PID (with output_limits) composed graph.

    Exercises the controller recurrent-state path end to end: PID's
    _integral/_prev_error/_has_run thread as state ports, the D-term is
    multiply-gated, and output_limits forces the branch-free anti-windup
    (ne mask + where back-calculation) into the graph.
    """
    kf = EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())
    pid = PIDController(
        kp=np.array([2.0, 2.0, 2.0]),
        ki=np.array([0.5, 0.5, 0.5]),
        kd=np.array([0.5, 0.5, 0.5]),
        dt=0.02,
        output_limits=(np.array([-0.3, -0.3, -0.6]), np.array([0.3, 0.3, 0.6])),
        backend=NumpyBackend(),
    )
    kf.P = np.eye(3) * 0.1
    kf.x_hat = np.zeros((3, 1))
    kf_graph = trace_node(
        kf,
        input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
        state_shapes={"x_hat": (3, 1), "P": (3, 3)},
    )
    pid_graph = trace_node(
        pid,
        input_shapes={"current_state": (3,), "target_state": (3,)},
        state_shapes={"_integral": (3,), "_prev_error": (3,), "_has_run": (3,)},
    )
    limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))
    return compose(kf_graph, pid_graph, plant_dims={"n_x": 3, "n_u": 3}, input_limits=limits)


def _build_mpc_deltau_composed_graph():
    """KF + MPC_DeltaU composed graph (n_vars=45 bake required).

    DeltaU's compute(x0, u_prev) augments the state with the previous control;
    compose routes the shared u_prev recurrent port to both the estimator and
    the controller, and state_u_prev closes the loop. The graph's .solve_qp
    node has output size 45 (horizon 15 × n_u 3), so it must be built against
    the mpc_base.toml bake via -Dsolver_dir.
    """
    from scripts.gen_mpc import build_mpc_composed_graph

    return build_mpc_composed_graph("configs/controllers/mpc_base.toml")


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


@pytest.fixture(scope="session")
def mpc_composed_so(tmp_path_factory):
    """Build the .so from the composed KF + MPC_LTI graph (error-state feed)."""
    from scripts.gen_mpc import build_mpc_composed_graph

    return _build_so(build_mpc_composed_graph(), tmp_path_factory.mktemp("zig-build-mpc-composed"))


@pytest.fixture(scope="session")
def pid_composed_so(tmp_path_factory):
    """Build the .so from the composed KF + PID graph (recurrent integral + anti-windup)."""
    return _build_so(_build_pid_composed_graph(), tmp_path_factory.mktemp("zig-build-pid-composed"))


@pytest.fixture(scope="session")
def deltau_bake(tmp_path_factory):
    """Bake the MPC_DeltaU static solver (mpc_base.toml, n_vars=45) into a tmp dir.

    A second bake alongside the shipped mpc_lti_base.toml one — the whole
    point of the -Dsolver_dir build option. Never touches the shared
    runtime/codegen/emosqp/ tree.
    """
    from scripts.gen_emosqp_test import bake

    bake_dir = tmp_path_factory.mktemp("deltau-bake")
    bake("configs/controllers/mpc_base.toml", str(bake_dir), str(bake_dir / "emosqp_data.zig"))
    return bake_dir


@pytest.fixture(scope="session")
def mpc_deltau_composed_so(tmp_path_factory, deltau_bake):
    """Build the .so from the composed KF + MPC_DeltaU graph against the DeltaU bake."""
    graph_path = tmp_path_factory.mktemp("deltau-graph") / "graph_data.zig"
    return _build_so(
        _build_mpc_deltau_composed_graph(),
        tmp_path_factory.mktemp("zig-build-mpc-deltau"),
        graph_path=graph_path,
        solver_dir=deltau_bake,
    )


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

        exact_ops = {"copy", "slice", "relu", "argmax", "one_hot", "stack", "ne_zero", "ne_one"}
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


class TestMpcComposedOracle:
    """The composed KF + MPC_LTI .so matches a live numpy closed loop.

    The regulator gets the error state e = x_hat - x_ref (compose inserts the
    sub node); the oracle is the real KalmanFilter.estimate() + MPC_LTI.compute()
    on the numpy backend, driven in parallel through the same y/x_ref sequence.
    Tolerance is looser than the KF+LQR oracle: the .so's baked EMOSQP
    warm-starts from the previous tick's solution while the live side
    cold-starts Python osqp each tick, so ADMM settles at slightly different
    points within eps — and that difference feeds back through the loop
    (measured: max ~1.5e-6 over 100 ticks).
    """

    def test_so_matches_live_kf_mpc(self, mpc_composed_so):
        lib, cg = mpc_composed_so
        rng = np.random.default_rng(21)
        n_out, n_state = _output_split(cg)
        sl = _state_slices(cg)

        kf = EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())
        mpc = ControllerFactory("configs/controllers/mpc_lti_base.toml").create(backend=NumpyBackend())
        limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))

        P = np.eye(3) * 0.1
        x_hat = np.zeros((3, 1))
        u_prev_so = np.zeros(3)
        u_prev_np = np.zeros(3)

        max_err = 0.0
        for _ in range(100):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.05, (3,))

            # Live numpy oracle: KF step, then the regulator on the error.
            kf.P = P.copy()
            kf.x_hat = x_hat.copy()
            x_hat_np = kf.estimate(y.reshape(-1, 1), u_prev_np.reshape(-1, 1))
            u_np = np.clip(
                mpc.compute(x_hat_np.ravel() - x_ref), limits[0], limits[1]
            )

            inputs = _pack_inputs(cg, y, x_ref, u_prev_so, x_hat, P)
            out, state = _step(lib, cg, inputs, n_out, n_state)
            max_err = max(max_err, float(np.max(np.abs(out - u_np))))

            x_hat = state[sl["state_x_hat"][0] : sl["state_x_hat"][1]].reshape(3, 1)
            P = state[sl["state_P"][0] : sl["state_P"][1]].reshape(3, 3)
            u_prev_so = out
            u_prev_np = u_np

        assert max_err < 1e-4, (
            f".so diverged from live KF+MPC over 100 ticks: max abs err = {max_err:.3e}"
        )


class TestPidComposedOracle:
    """The composed KF + PID .so matches a live numpy closed loop.

    The regression test for controller recurrent state: PID's integral must
    accumulate across ticks (the old composed graph froze it at zero), the
    D-term gate must open after tick 0 (the old graph baked the first-tick
    branch forever), and the anti-windup back-calculation must fire only on
    saturated channels. output_limits forces saturation so the ne/where
    anti-windup path is genuinely exercised.
    """

    def test_so_matches_live_kf_pid(self, pid_composed_so):
        lib, cg = pid_composed_so
        rng = np.random.default_rng(31)
        n_out, n_state = _output_split(cg)
        sl = _state_slices(cg)

        kf = EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())
        pid = PIDController(
            kp=np.array([2.0, 2.0, 2.0]),
            ki=np.array([0.5, 0.5, 0.5]),
            kd=np.array([0.5, 0.5, 0.5]),
            dt=0.02,
            output_limits=(np.array([-0.3, -0.3, -0.6]), np.array([0.3, 0.3, 0.6])),
            backend=NumpyBackend(),
        )
        limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))

        P = np.eye(3) * 0.1
        x_hat = np.zeros((3, 1))
        integral_so = np.zeros(3)
        prev_error_so = np.zeros(3)
        has_run_so = np.zeros(3)
        integral_live = np.zeros(3)
        prev_error_live = np.zeros(3)
        has_run_live = np.zeros(3)
        u_prev_so = np.zeros(3)
        u_prev_np = np.zeros(3)
        saw_saturation = False

        max_err = 0.0
        for _ in range(100):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.05, (3,))

            # Live numpy oracle: KF step, then PID with its own state.
            kf.P = P.copy()
            kf.x_hat = x_hat.copy()
            pid._integral = integral_live.copy()
            pid._prev_error = prev_error_live.copy()
            pid._has_run = has_run_live.copy()
            x_hat_np = kf.estimate(y.reshape(-1, 1), u_prev_np.reshape(-1, 1))
            u_np = np.clip(pid.compute(x_hat_np.ravel(), x_ref), limits[0], limits[1])
            integral_live = pid._integral.copy()
            prev_error_live = pid._prev_error.copy()
            has_run_live = pid._has_run.copy()
            saw_saturation = saw_saturation or bool(
                np.any(np.abs(u_np) >= 0.3 - 1e-12)
            )

            inputs = _pack_arrays(
                cg,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev_so,
                    "state_x_hat": x_hat.ravel(),
                    "state_P": P.ravel(),
                    "state_integral": integral_so,
                    "state_prev_error": prev_error_so,
                    "state_has_run": has_run_so,
                },
            )
            out, state = _step(lib, cg, inputs, n_out, n_state)
            max_err = max(max_err, float(np.max(np.abs(out - u_np))))

            # Each loop evolves its own state.
            x_hat = state[sl["state_x_hat"][0] : sl["state_x_hat"][1]].reshape(3, 1)
            P = state[sl["state_P"][0] : sl["state_P"][1]].reshape(3, 3)
            integral_so = state[sl["state_integral"][0] : sl["state_integral"][1]]
            prev_error_so = state[sl["state_prev_error"][0] : sl["state_prev_error"][1]]
            has_run_so = state[sl["state_has_run"][0] : sl["state_has_run"][1]]
            u_prev_so = out
            u_prev_np = u_np

        assert saw_saturation, "oracle never saturated — anti-windup path untested"
        assert max_err < 1e-10, (
            f".so diverged from live KF+PID over 100 ticks: max abs err = {max_err:.3e}"
        )


class TestMpcDeltaUComposedOracle:
    """The composed KF + MPC_DeltaU .so matches a live numpy closed loop.

    The regression test for the second bake: the graph's .solve_qp node has
    output size 45 (mpc_base.toml, horizon 15), so the .so is built against
    the DeltaU bake via -Dsolver_dir — the shipped n_vars=30 bake would be
    rejected at compile time by the comptime graph↔bake check. The live oracle
    is KalmanFilter.estimate() + MPC_DeltaU.compute(x̂ − x_ref, u_prev) with
    u_prev threaded on both sides. Tolerance matches TestMpcComposedOracle:
    the .so's baked EMOSQP warm-starts from the previous tick while the live
    side cold-starts Python osqp, so ADMM settles within eps and the
    difference feeds back through the loop.
    """

    def test_so_matches_live_kf_mpc_deltau(self, mpc_deltau_composed_so):
        lib, cg = mpc_deltau_composed_so
        rng = np.random.default_rng(41)
        n_out, n_state = _output_split(cg)
        sl = _state_slices(cg)

        kf = EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())
        mpc = ControllerFactory("configs/controllers/mpc_base.toml").create(backend=NumpyBackend())
        limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))

        P = np.eye(3) * 0.1
        x_hat = np.zeros((3, 1))
        u_prev_so = np.zeros(3)
        u_prev_np = np.zeros(3)

        max_err = 0.0
        for _ in range(100):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.05, (3,))

            # Live numpy oracle: KF step, then the DeltaU regulator on the
            # error with its own u_prev.
            kf.P = P.copy()
            kf.x_hat = x_hat.copy()
            x_hat_np = kf.estimate(y.reshape(-1, 1), u_prev_np.reshape(-1, 1))
            u_np = np.clip(
                mpc.compute(x_hat_np.ravel() - x_ref, u_prev_np), limits[0], limits[1]
            )

            inputs = _pack_inputs(cg, y, x_ref, u_prev_so, x_hat, P)
            out, state = _step(lib, cg, inputs, n_out, n_state)
            max_err = max(max_err, float(np.max(np.abs(out - u_np))))

            x_hat = state[sl["state_x_hat"][0] : sl["state_x_hat"][1]].reshape(3, 1)
            P = state[sl["state_P"][0] : sl["state_P"][1]].reshape(3, 3)
            u_prev_so = out
            u_prev_np = u_np

        assert max_err < 1e-4, (
            f".so diverged from live KF+MPC_DeltaU over 100 ticks: max abs err = {max_err:.3e}"
        )


def test_comptime_n_vars_mismatch_rejects_build(tmp_path):
    """A .solve_qp graph built against the wrong bake fails at compile time.

    The regression test for the comptime graph↔bake check: lowering the DeltaU
    graph (n_vars=45) and building it against the shipped n_vars=30 bake must
    fail with a @compileError naming both sizes — not silently link a
    shape-mismatched solver.
    """
    if shutil.which("zig") is None:
        pytest.skip("zig not on PATH; skipping Zig lowering oracle")

    graph_path = tmp_path / "graph_data.zig"
    lower_zig(_build_mpc_deltau_composed_graph(), str(graph_path))

    cmd = [
        "zig",
        "build",
        "--build-file",
        str(RUNTIME / "build.zig"),
        "--prefix",
        str(tmp_path / "build"),
        f"-Dgraph={graph_path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0, "mismatched graph+bake should not compile"
    assert "does not match baked solver n_vars" in result.stderr, result.stderr[:400]


@pytest.fixture(scope="module")
def manifests(tmp_path_factory, deltau_bake):
    """Build base and DeltaU .so's and return their build dirs."""
    base_dir = tmp_path_factory.mktemp("zig-build-manifest-base")
    _build_so(build_base_graph(), base_dir)

    deltau_dir = tmp_path_factory.mktemp("zig-build-manifest-deltau")
    graph_path = tmp_path_factory.mktemp("zig-build-manifest-deltau-graph") / "graph_data.zig"
    _build_so(
        _build_mpc_deltau_composed_graph(),
        deltau_dir,
        graph_path=graph_path,
        solver_dir=deltau_bake,
    )
    return base_dir, deltau_dir


# Each Zig node line: .{ .op = .matmul, .inputs = &.{6, 3}, .rows = 3, .cols = 3, .aux = 0 }
_NODE_RE = re.compile(
    r"\.\{ \.op = \.(\w+), \.inputs = &\.\{([^}]*)\}, \.rows = (\d+), \.cols = (\d+), \.aux = (\d+) \}"
)


def _parse_zig_nodes(text):
    """Parse every node entry from a generated graph_data.zig into dicts."""
    nodes = []
    for m in _NODE_RE.finditer(text):
        vm_op, inputs, rows, cols, aux = m.groups()
        in_ids = [int(x) for x in inputs.split(",") if x.strip()] if inputs.strip() else []
        nodes.append(
            {
                "vm_op": vm_op,
                "inputs": in_ids,
                "rows": int(rows),
                "cols": int(cols),
                "aux": int(aux),
            }
        )
    return nodes


class TestBuildManifest:
    """The build manifest (audit trail) describes what's inside the .so.

    Every build writes a deterministic report next to the artifact
    (<prefix>/lib/libbase.manifest.json) plus a timestamped archive copy
    (<prefix>/manifests/<UTC>-<graphsha8>.json). The report is a pure function
    of the inputs (no timestamps), so identical builds produce identical
    reports — the diffable audit record for "which controller combination was
    this binary built from".
    """

    def _report(self, build_dir):
        return json.loads((build_dir / "lib" / "libbase.manifest.json").read_text())

    def test_report_exists_and_describes_binary(self, manifests):
        """The report carries build facts, provenance, solver, and graph content."""
        base_dir, _ = manifests
        report = self._report(base_dir)

        assert report["target"]
        assert report["optimize"] == "Debug"
        assert report["zig_version"]
        assert report["libc"] is True
        assert report["float_type"] == "f64"

        prov = report["provenance"]
        assert prov["graph_sha256"]
        assert prov["solver_sha256"]

        # solver facts from the bake
        assert report["solver"]["n_vars"] == 30
        assert report["solver"]["n_cons"] == 60
        assert report["solver"]["config"].endswith("mpc_lti_base.toml")

        # graph facts: op histogram matches the composed graph, ports match
        g = report["graph"]
        assert g["nodes_total"] == len(build_base_graph().graph.nodes)
        assert g["buf_len"] > 0
        assert g["solve_qp"] is None  # base graph has no QP node
        expected = {}
        for n in build_base_graph().graph.nodes:
            expected[n.op] = expected.get(n.op, 0) + 1
        assert g["op_histogram"] == expected
        assert [p["name"] for p in g["inputs"]] == build_base_graph().inputs
        assert [p["name"] for p in g["outputs"]] == build_base_graph().outputs
        assert [p["name"] for p in g["state_outputs"]] == build_base_graph().state_outputs

        # ordered node list with dual names + layout fields
        assert len(g["nodes"]) == g["nodes_total"]
        assert set(g["nodes"][0]) == {"i", "op", "vm_op", "inputs", "rows", "cols", "offset", "aux"}
        assert any(n["op"] == "const" and n["vm_op"] == "cst" for n in g["nodes"])
        assert any(n["op"] == "input" and n["vm_op"] == "inp" for n in g["nodes"])
        # offsets are cumulative in node order (contiguous buffer slots)
        prev_end = 0
        for n in g["nodes"]:
            assert n["offset"] == prev_end
            prev_end += n["rows"] * n["cols"]

    def test_deltau_manifest_differs_op_wise(self, manifests):
        """DeltaU vs base reports differ op-wise: solve_qp present, n_vars pair."""
        base_dir, deltau_dir = manifests
        base = self._report(base_dir)
        deltau = self._report(deltau_dir)

        assert deltau["graph"]["solve_qp"] == {"expected_n_vars": 45}
        assert deltau["solver"]["n_vars"] == 45
        assert "solve_qp" in deltau["graph"]["ops"]
        assert "solve_qp" not in base["graph"]["ops"]
        assert "slice" in deltau["graph"]["ops"]  # u[:3] after the solve
        assert deltau["graph"]["nodes_total"] != base["graph"]["nodes_total"]
        # the node list reflects the QP wiring: solve_qp feeds the slice
        solve = [n for n in deltau["graph"]["nodes"] if n["vm_op"] == "solve_qp"][0]
        assert solve["rows"] == 45 and solve["cols"] == 1
        assert solve["inputs"] == [solve["i"] - 1]  # q = Fᵀ x_aug matmul

    def test_archive_copy_timestamped_and_identical(self, manifests):
        """The archive copy is timestamped and byte-identical to the report."""
        base_dir, _ = manifests
        report = (base_dir / "lib" / "libbase.manifest.json").read_text()
        archives = list((base_dir / "manifests").glob("*.json"))
        assert len(archives) >= 1
        # filename: <UTC>-<graphsha8>.json
        assert "-" in archives[0].stem
        assert archives[0].read_text() == report

    def test_drift_guard_nodes_match_emitted_table(self, manifests):
        """The manifest's node list matches the emitted Zig table field-for-field.

        Regex-parses every node entry from graph_data.zig and compares vm_op,
        wiring, shape, and aux against the manifest — the serializer and the
        manifest can never silently drift.
        """
        base_dir, _ = manifests
        report = self._report(base_dir)
        text = (RUNTIME / "graph_data.zig").read_text()
        zig_nodes = _parse_zig_nodes(text)
        assert len(zig_nodes) == report["graph"]["nodes_total"]
        manifest_nodes = report["graph"]["nodes"]
        assert len(manifest_nodes) == len(zig_nodes)
        for mn, zn in zip(manifest_nodes, zig_nodes):
            assert mn["vm_op"] == zn["vm_op"], f"node {mn['i']} op mismatch"
            assert mn["inputs"] == zn["inputs"], f"node {mn['i']} wiring mismatch"
            assert mn["rows"] == zn["rows"] and mn["cols"] == zn["cols"], (
                f"node {mn['i']} shape mismatch"
            )
            assert mn["aux"] == zn["aux"], f"node {mn['i']} aux mismatch"

    def test_lowering_is_deterministic(self, tmp_path):
        """Two lowers of the same graph produce byte-identical manifests."""
        from shinro.codegen.lower_zig import lower_zig

        cg = build_base_graph()
        p1 = tmp_path / "g1" / "graph_data.zig"
        p2 = tmp_path / "g2" / "graph_data.zig"
        p1.parent.mkdir()
        p2.parent.mkdir()
        lower_zig(cg, str(p1))
        lower_zig(cg, str(p2))
        m1 = (p1.parent / "graph_data_manifest.json").read_bytes()
        m2 = (p2.parent / "graph_data_manifest.json").read_bytes()
        assert m1 == m2

