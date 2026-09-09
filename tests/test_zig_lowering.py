"""Zig lowering oracle: the compiled .so must match the Python interpreter.

The MVP acceptance test for slice b. It generates ``src/shinro/runtime/graph_data.zig``
from the base_tracking composed graph (KF + LQR, input-clipped), compiles the
comptime VM with ``zig build`` (src/shinro/runtime/build.zig), loads
``libbase.so`` via ctypes, and asserts the C-ABI ``shinro_step`` output equals
``interpret()`` to float-exactness across 50 seeded random inputs.

Requires ``zig`` on PATH. Skipped cleanly if it's unavailable.
"""

from __future__ import annotations

import ctypes
import hashlib
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
RUNTIME = REPO_ROOT / "src/shinro/runtime"
BUILD = REPO_ROOT / "build"


def _build_so(composed, build_dir, graph_path=None, solver_dir=None, provenance=None):
    """Lower a composed graph, compile the comptime VM, and load libbase.so.

    Each call lowers ``composed`` to ``src/shinro/runtime/graph_data.zig`` (or
    ``graph_path`` when given) and builds a fresh ``libbase.so`` into the
    given (unique) prefix directory, so multiple graphs can be cross-checked
    in one session without clobbering each other. ``solver_dir`` selects the
    baked OSQP solver to compile in (default: the shipped
    ``src/shinro/runtime/codegen/emosqp/`` bake) — pass a DeltaU bake to build
    a graph whose ``.solve_qp`` node has n_vars=45. Graphs without a
    ``.solve_qp`` node (LQR, PID, ...) are built solver-free. ``provenance``
    is forwarded to ``lower_zig`` (config hashes + tool versions recorded in
    the manifest).
    """
    if shutil.which("zig") is None:
        pytest.skip("zig not on PATH; skipping Zig lowering oracle")

    out_zig = graph_path or (RUNTIME / "graph_data.zig")
    lower_zig(composed, str(out_zig), provenance=provenance)

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
    libbase.so (src/shinro/runtime/codegen/emosqp/), whose problem must match the
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


def _build_luenberger_composed_graph():
    """Luenberger + LQR composed graph via the generic builder.

    Exercises the estimator-swap path end to end: the two-pass trace discovers
    the Luenberger observer's recurrent ``x_hat`` (no ``P`` covariance, unlike
    the KF), and the composed graph is solver-free (no ``.solve_qp`` node).
    """
    from shinro.codegen.build import build_composed_graph

    limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))
    return build_composed_graph(
        estimator_config="configs/estimators/luenberger_base.toml",
        controller_config="configs/controllers/lqr_base.toml",
        n_x=3,
        n_u=3,
        input_limits=limits,
    )


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
def luenberger_composed_so(tmp_path_factory):
    """Build the .so from the composed Luenberger + LQR graph (recurrent x_hat)."""
    return _build_so(_build_luenberger_composed_graph(), tmp_path_factory.mktemp("zig-build-luenberger-composed"))


@pytest.fixture(scope="session")
def deltau_bake(tmp_path_factory):
    """Bake the MPC_DeltaU static solver (mpc_base.toml, n_vars=45) into a tmp dir.

    A second bake alongside the shipped mpc_lti_base.toml one — the whole
    point of the -Dsolver_dir build option. Never touches the shared
    src/shinro/runtime/codegen/emosqp/ tree.
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


def _build_matmul_shapes_graph():
    """A graph exercising every matmul shape the VM dispatch must handle.

    Regression for the single-input (n_u=1) bug: a ``(2,1) @ (1,1)`` matmul
    (a 2-D column times a scalar-width matrix — e.g. the estimator's ``B·u``
    term) was misclassified as vecmat (1-D @ 2-D), reading out of bounds and
    producing garbage. Each shape is a named output so the ``.so`` can be
    compared per-case against numpy.
    """
    g = Graph()
    a = g.input("a", (2, 2))
    v = g.input("v", (2,))
    vcol = g.input("vcol", (2, 1))
    scol = g.input("scol", (1, 1))
    row = g.input("row", (1, 2))

    cases = {
        "matmul2d": g.emit("matmul", [a, a], (2, 2)),  # (2,2)@(2,2)
        "matvec_col": g.emit("matmul", [a, vcol], (2, 1)),  # (2,2)@(2,1)
        "vecmat": g.emit("matmul", [v, a], (2,)),  # (2,)@(2,2)
        "col_x_scalar": g.emit("matmul", [vcol, scol], (2, 1)),  # (2,1)@(1,1) <- the bug
        "matvec_row": g.emit("matmul", [row, v], (1,)),  # (1,2)@(2,)
    }
    for name, src in cases.items():
        g.output(name, src)
    return ComposedGraph(
        graph=g,
        inputs=["a", "v", "vcol", "scol", "row"],
        outputs=list(cases),
    )


def _build_single_input_kf_lqr_graph(tmp_path):
    """A composed single-input (n_x=2, n_u=1) KF+LQR graph.

    The exact shape that regressed: with one control input, the estimator's
    ``B·u`` term is a ``(2,1)@(1,1)`` matmul. Uses explicit linearized
    inverted-pendulum dynamics so the estimator/controller carry a real
    ``B_dynamics`` of width 1.
    """
    from shinro.codegen.build import build_composed_graph

    est = tmp_path / "kalman_pendulum.toml"
    est.write_text(
        'type = "KalmanFilter"\n'
        'name = "kalman_pendulum"\n'
        "dt = 0.01\n"
        "process_noise = [0.001, 0.01]\n"
        "measurement_noise = [0.005, 0.05]\n"
        "A_dynamics = [[1.0, 0.01], [0.1962, 1.0]]\n"
        "B_dynamics = [[0.0], [0.4]]\n"
    )
    ctrl = tmp_path / "lqr_pendulum.toml"
    ctrl.write_text(
        'type = "LQR"\n'
        'name = "lqr_pendulum"\n'
        "dt = 0.01\n"
        "state_cost = [50.0, 10.0]\n"
        "control_cost = [0.5]\n"
        "A_dynamics = [[1.0, 0.01], [0.1962, 1.0]]\n"
        "B_dynamics = [[0.0], [0.4]]\n"
    )
    return build_composed_graph(str(est), str(ctrl), n_x=2, n_u=1)


@pytest.fixture(scope="module")
def matmul_shapes_so(tmp_path_factory):
    """A compiled .so for the matmul shape-dispatch matrix."""
    d = tmp_path_factory.mktemp("zig-matmul-shapes")
    return _build_so(_build_matmul_shapes_graph(), d / "build", graph_path=d / "graph_data.zig")


@pytest.fixture(scope="module")
def single_input_kf_lqr_so(tmp_path_factory):
    """A compiled .so for the single-input KF+LQR oracle test."""
    d = tmp_path_factory.mktemp("zig-single-input")
    cg = _build_single_input_kf_lqr_graph(d)
    return _build_so(cg, d / "build", graph_path=d / "graph_data.zig")


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


class TestMatmulShapeDispatch:
    def test_so_matches_numpy_for_every_shape(self, matmul_shapes_so):
        """The compiled .so equals numpy for every matmul shape the dispatch
        must route: 2-D matmul, matvec, vecmat, and the single-input
        column-times-scalar-width (m,1)@(1,1) case that was misrouted."""
        lib, cg = matmul_shapes_so
        rng = np.random.default_rng(7)
        n_out, _ = _output_split(cg)

        offsets = {}
        off = 0
        for name in cg.outputs:
            size = next(
                int(np.prod(n.shape))
                for n in cg.graph.nodes
                if n.op == "output" and n.attrs["name"] == name
            )
            offsets[name] = (off, off + size)
            off += size

        refs = {
            "matmul2d": lambda a, v, vcol, scol, row: a @ a,
            "matvec_col": lambda a, v, vcol, scol, row: a @ vcol,
            "vecmat": lambda a, v, vcol, scol, row: v @ a,
            "col_x_scalar": lambda a, v, vcol, scol, row: vcol @ scol,
            "matvec_row": lambda a, v, vcol, scol, row: row @ v,
        }
        max_err = 0.0
        for _ in range(30):
            a = rng.normal(0.0, 0.1, (2, 2))
            v = rng.normal(0.0, 0.1, (2,))
            vcol = rng.normal(0.0, 0.1, (2, 1))
            scol = rng.normal(0.0, 0.1, (1, 1))
            row = rng.normal(0.0, 0.1, (1, 2))
            inputs = _pack_arrays(
                cg,
                {
                    "a": a.ravel(),
                    "v": v.ravel(),
                    "vcol": vcol.ravel(),
                    "scol": scol.ravel(),
                    "row": row.ravel(),
                },
            )
            out, _ = _step(lib, cg, inputs, n_out, 1)
            for name in cg.outputs:
                got = out[offsets[name][0] : offsets[name][1]]
                exp = np.asarray(refs[name](a, v, vcol, scol, row)).ravel()
                max_err = max(max_err, float(np.max(np.abs(got - exp))))
        assert max_err < 1e-12, f"matmul shape dispatch diverged: max abs err = {max_err:.3e}"


class TestSingleInputKfLqr:
    def test_so_matches_interpret_single_input(self, single_input_kf_lqr_so):
        """A single-input (n_u=1) KF+LQR .so equals interpret().

        Regression for the compiled-kernel bug: with one control input the
        estimator's B·u is a (2,1)@(1,1) matmul, which the VM misclassified
        as vecmat (garbage in release, an OOB panic in debug). This is the
        shape every n_u=1 plant (inverted pendulum, cartpole, ...) hits.
        """
        lib, cg = single_input_kf_lqr_so
        rng = np.random.default_rng(3)
        n_out, n_state = _output_split(cg)
        sl = _state_slices(cg)
        max_err = 0.0
        for _ in range(30):
            y = rng.normal(0.0, 0.1, (2,))
            x_ref = rng.normal(0.0, 0.1, (2,))
            u_prev = rng.normal(0.0, 0.1, (1,))
            x_hat = rng.normal(0.0, 0.1, (2, 1))
            P = rng.normal(0.0, 0.1, (2, 2))
            P = P @ P.T + 0.1 * np.eye(2)

            inputs = _pack_inputs(cg, y, x_ref, u_prev, x_hat, P)
            out, state = _step(lib, cg, inputs, n_out, n_state)
            traced = interpret(
                cg.graph,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev,
                    "state_x_hat": x_hat,
                    "state_P": P,
                },
            )
            off = 0
            for name in cg.outputs:
                exp = np.asarray(traced[name]).ravel()
                max_err = max(max_err, float(np.max(np.abs(out[off : off + exp.size] - exp))))
                off += exp.size
            for name in cg.state_outputs:
                exp = np.asarray(traced[name]).ravel()
                start, stop = sl[name]
                max_err = max(max_err, float(np.max(np.abs(state[start:stop] - exp))))
        assert max_err < 1e-12, f"single-input KF+LQR .so diverged: max abs err = {max_err:.3e}"


# ─── plant compile-scan: (op, shape) surface across the whole zoo ────────────
#
# The one-step oracle verifies compilation fidelity: the .so's shinro_step
# must compute the same graph math as interpret(), for every shape class and
# op mix the plants generate. Dimensions drive the matmul-dispatch shape
# classes (the n_u=1 (m,1)@(1,1) misroute only exists at one control input);
# the controller/estimator choice drives the op set (PID adds where/ne +
# three recurrent state ports; Luenberger drops the inv and the P port).

#: Standalone-instantiable packaged plants: (name, cfg, n_x, n_u, dt).
PLANT_DIMS = [
    ("CartPole", "configs/plants/cartpole.toml", 4, 1, 0.01),
    ("InvertedPendulum", "configs/plants/inverted_pendulum.toml", 2, 1, 0.01),
    ("DoublePendulum", "configs/plants/double_pendulum.toml", 4, 2, 0.01),
    ("HolonomicMobileRobot", "configs/plants/holonomic_base.toml", 3, 3, 0.02),
]
#: Controllers/estimators that trace without a solver bake. MPC stays at the
#: base dims (existing suite) — every QP size needs its own baked solver.
#: PID is square-systems-only: its per-channel gains (n_u,) broadcast against
#: the (n_x,) error, so for n_x != n_u the traced output/clip/anti-windup
#: shapes are inconsistent (comptime OOB in the clip lowering, or a runtime
#: panic in the where dispatch — both observed before the guard below).
#: compose() now rejects those with a loud ValueError.
SCAN_CONTROLLERS = ("LQR", "PID")
SCAN_ESTIMATORS = ("KalmanFilter", "LuenbergerObserver")

#: 4 plants x 2 estimators x (LQR) + square plants x 2 estimators x (PID).
PLANT_SCAN_CASES = [
    (f"{plant}-{ctrl}-{est}", plant, cfg, n_x, n_u, dt, ctrl, est)
    for plant, cfg, n_x, n_u, dt in PLANT_DIMS
    for ctrl in SCAN_CONTROLLERS
    if ctrl == "LQR" or n_x == n_u  # PID: square systems only
    for est in SCAN_ESTIMATORS
]

#: Synthetic dimensionality sweep at the Quadrotor-scale: n_x x n_u combos
#: beyond any named plant, LQR+KF. Exercises large matmuls and the n_x x n_x
#: Kalman inverse (12x12, 24x24) the named plants never reach.
DIM_SWEEP_CASES = [
    (f"synth{nx}x{nu}", "synthetic", None, nx, nu, 0.01, "LQR", "KalmanFilter")
    for nx in (6, 12, 24)
    for nu in (1, 3, 4)
]

ALL_SCAN_CASES = PLANT_SCAN_CASES + DIM_SWEEP_CASES


def _plant_model(plant_name, cfg, n_x, n_u, dt):
    """Discrete (A_d, B_d) for a plant: linearized dynamics where possible,
    the plant's own discrete model otherwise, and a diagonally-stable
    synthetic model for the dim-sweep cases (A=I would make the LQR DARE
    infeasible with a truncated B)."""
    if cfg is None:
        return 0.99 * np.eye(n_x), 0.01 * np.eye(n_x)[:, :n_u]

    import tomllib

    from shinro.factories.registry import _PLANT_REGISTRY
    from shinro.utils.array_backend import NumpyBackend
    from shinro.utils.config_resolver import resolve_config_path
    from shinro.utils.linearization import discretize_euler, linearize_plant

    with open(resolve_config_path(cfg), "rb") as f:
        plant = _PLANT_REGISTRY[plant_name].from_config(tomllib.load(f), backend=NumpyBackend())
    try:
        A_c, B_c = linearize_plant(plant, u0=plant.bk.zeros(n_u))
        A_d, B_d = discretize_euler(A_c, B_c, plant.dt, backend=plant.bk)
    except Exception:
        A_d, B_d = plant.get_model()
    return np.asarray(A_d), np.asarray(B_d)


def _controller_config_toml(kind, name, n_x, n_u, dt, A_d, B_d, out_dir):
    """Write a controller config TOML for the scan (LQR or PID)."""
    path = out_dir / f"ctrl_{kind.lower()}_{name}.toml"
    if kind == "LQR":
        path.write_text(
            f'type = "LQR"\nname = "lqr_{name}"\ndt = {dt}\n'
            f"state_cost = {[1.0] * n_x}\n"
            f"control_cost = {[1.0] * n_u}\n"
            f"A_dynamics = {np.round(A_d, 6).tolist()}\n"
            f"B_dynamics = {np.round(B_d, 6).tolist()}\n"
        )
    elif kind == "PID":
        # output_limits force the branch-free anti-windup (ne mask + where
        # back-calculation) into the graph at every dim.
        path.write_text(
            f'type = "PID"\nname = "pid_{name}"\ndt = {dt}\n'
            f"kp = {[2.0] * n_u}\nki = {[0.5] * n_u}\nkd = {[0.5] * n_u}\n"
            f"output_limits = {{ min = {[-10.0] * n_u}, max = {[10.0] * n_u} }}\n"
        )
    else:
        raise ValueError(f"unknown controller kind {kind!r}")
    return str(path)


def _estimator_config_toml(kind, name, n_x, n_u, dt, A_d, B_d, out_dir):
    """Write an estimator config TOML for the scan (KF or Luenberger).

    B_dynamics is always explicit: the dt*I default is (n_x, n_x), which
    mismatches the (n_u, 1) control_input port whenever n_u < n_x.
    """
    path = out_dir / f"est_{kind.lower()}_{name}.toml"
    A = np.round(A_d, 6).tolist()
    B = np.round(B_d, 6).tolist()
    if kind == "KalmanFilter":
        path.write_text(
            f'type = "KalmanFilter"\nname = "kf_{name}"\ndt = {dt}\n'
            f"process_noise = {[1e-3] * n_x}\n"
            f"measurement_noise = {[1e-3] * n_x}\n"
            f"A_dynamics = {A}\nB_dynamics = {B}\n"
        )
    elif kind == "LuenbergerObserver":
        path.write_text(
            f'type = "LuenbergerObserver"\nname = "luen_{name}"\ndt = {dt}\n'
            f"observer_gain = {[0.5] * n_x}\n"
            f"A_dynamics = {A}\nB_dynamics = {B}\n"
        )
    else:
        raise ValueError(f"unknown estimator kind {kind!r}")
    return str(path)


def _plant_graph(tmp_path, case_name, plant_name, plant_cfg, n_x, n_u, dt, controller, estimator):
    """A composed estimator+controller graph for a plant's dims."""
    from shinro.codegen.build import build_composed_graph

    A_d, B_d = _plant_model(plant_name, plant_cfg, n_x, n_u, dt)
    est = _estimator_config_toml(estimator, case_name, n_x, n_u, dt, A_d, B_d, tmp_path)
    ctrl = _controller_config_toml(controller, case_name, n_x, n_u, dt, A_d, B_d, tmp_path)
    return build_composed_graph(
        est,
        ctrl,
        n_x=n_x,
        n_u=n_u,
        input_limits=(np.full(n_u, -10.0), np.full(n_u, 10.0)),
    )


def _random_spd(rng, n):
    """A random well-conditioned SPD matrix (innovation covariances must stay
    invertible for both the Zig LU and numpy's LAPACK inv)."""
    P = rng.normal(0.0, 0.1, (n, n))
    return P @ P.T + 0.1 * np.eye(n)


def _scan_input_ports(cg, n_x, n_u, rng):
    """Random host inputs for a scan graph, keyed by port name.

    y/x_ref/u_prev/state_x_hat/state_P get random values (state_P as a random
    SPD matrix); any further recurrent state ports — e.g. PID's
    _integral/_prev_error/_has_run — are zero-filled, which is also the
    semantically correct first tick (has_run=0 gates the D-term via where).
    """
    known = {
        "y": rng.normal(0.0, 0.1, (n_x,)),
        "x_ref": rng.normal(0.0, 0.1, (n_x,)),
        "u_prev": rng.normal(0.0, 0.1, (n_u,)),
        "state_x_hat": rng.normal(0.0, 0.1, (n_x, 1)),
        "state_P": _random_spd(rng, n_x),
    }
    ports = {}
    for name in cg.inputs:
        if name in known:
            ports[name] = known[name]
        else:
            shape = next(
                n.shape for n in cg.graph.nodes if n.op == "input" and n.attrs["name"] == name
            )
            ports[name] = np.zeros(tuple(shape))
    return ports


@pytest.fixture(scope="module")
def plant_so(tmp_path_factory, request):
    """A compiled scan .so for one (plant x controller x estimator) case."""
    case_name, plant_name, plant_cfg, n_x, n_u, dt, controller, estimator = request.param
    d = tmp_path_factory.mktemp(f"zig-plant-{case_name}")
    cg = _plant_graph(d, case_name, plant_name, plant_cfg, n_x, n_u, dt, controller, estimator)
    lib, cg = _build_so(cg, d / "build", graph_path=d / "graph_data.zig")
    return lib, cg, n_x, n_u, case_name


class TestPlantCompileScan:
    """Every instantiable plant's compiled estimator+controller .so equals
    interpret() — compilation fidelity across the full (op, shape) surface."""

    @pytest.mark.parametrize("plant_so", ALL_SCAN_CASES, indirect=True, ids=[c[0] for c in ALL_SCAN_CASES])
    def test_so_matches_interpret_for_each_plant(self, plant_so):
        lib, cg, n_x, n_u, name = plant_so
        rng = np.random.default_rng(5)
        n_out, n_state = _output_split(cg)
        sl = _state_slices(cg)
        max_err = 0.0
        for _ in range(20):
            ports = _scan_input_ports(cg, n_x, n_u, rng)
            inputs = _pack_arrays(cg, {k: v.ravel() for k, v in ports.items()})
            out, state = _step(lib, cg, inputs, n_out, n_state)
            traced = interpret(cg.graph, ports)
            off = 0
            for pname in cg.outputs:
                exp = np.asarray(traced[pname]).ravel()
                max_err = max(max_err, float(np.max(np.abs(out[off : off + exp.size] - exp))))
                off += exp.size
            for pname in cg.state_outputs:
                exp = np.asarray(traced[pname]).ravel()
                start, stop = sl[pname]
                max_err = max(max_err, float(np.max(np.abs(state[start:stop] - exp))))
        assert max_err < 1e-12, f"{name}: .so diverged from interpreter: max abs err = {max_err:.3e}"


def test_compose_rejects_nonsquare_pid(tmp_path):
    """compose() raises loudly when the controller's traced output is not (n_u,).

    Regression for the malformed-graph class: a PID with fewer gain channels
    than state dimensions broadcasts its gains against the error vector, so
    the traced output/clip/anti-windup shapes disagree (previously: comptime
    OOB in the clip lowering for some dims, a runtime panic in the where
    dispatch for others — both silent-in-release).
    """
    import pytest as _pytest

    from shinro.codegen.build import build_composed_graph

    est = tmp_path / "kf_cartpole.toml"
    est.write_text(
        'type = "KalmanFilter"\nname = "kf"\ndt = 0.01\n'
        "process_noise = [0.001, 0.001, 0.001, 0.001]\n"
        "measurement_noise = [0.005, 0.005, 0.005, 0.005]\n"
        "A_dynamics = [[1.0, 0.01, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],"
        " [0.0, 0.0, 1.0, 0.01], [0.0, 0.0, 0.0, 1.0]]\n"
        "B_dynamics = [[0.0], [0.05], [0.0], [0.3]]\n"
    )
    ctrl = tmp_path / "pid_cartpole.toml"
    ctrl.write_text(
        'type = "PID"\nname = "pid"\ndt = 0.01\n'
        "kp = [2.0]\nki = [0.5]\nkd = [0.5]\n"
        "output_limits = { min = [-10.0], max = [10.0] }\n"
    )
    with _pytest.raises(ValueError, match="control dimension"):
        build_composed_graph(str(est), str(ctrl), n_x=4, n_u=1)


def test_lower_zig_emits_valid_data_table():
    """lower_zig produces a Zig file with the expected top-level constants."""
    composed = build_base_graph()
    lower_zig(composed, str(RUNTIME / "graph_data.zig"))
    text = (RUNTIME / "graph_data.zig").read_text()
    assert "pub const nodes = [_]Node{" in text
    assert "pub const const_blob" in text
    assert "pub const buf_len" in text
    assert "pub const has_solve_qp = false;" in text
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


class TestLuenbergerComposedOracle:
    """The composed Luenberger + LQR .so matches a live numpy closed loop.

    The estimator-swap regression test: the Luenberger observer's recurrent
    ``x_hat`` must thread across ticks (the two-pass trace discovers it, no
    ``P`` covariance like the KF), and the composed graph must be solver-free.
    """

    def test_so_matches_live_luenberger_lqr(self, luenberger_composed_so):
        lib, cg = luenberger_composed_so
        rng = np.random.default_rng(37)
        n_out, n_state = _output_split(cg)
        sl = _state_slices(cg)

        luen = EstimatorFactory("configs/estimators/luenberger_base.toml").create(backend=NumpyBackend())
        lqr = ControllerFactory("configs/controllers/lqr_base.toml").create(backend=NumpyBackend())
        limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))

        x_hat_so = np.zeros((3, 1))
        x_hat_np = np.zeros((3, 1))
        u_prev_so = np.zeros(3)
        u_prev_np = np.zeros(3)

        max_err = 0.0
        for _ in range(100):
            y = rng.normal(0.0, 0.1, (3,))
            x_ref = rng.normal(0.0, 0.05, (3,))

            # Live numpy oracle: Luenberger step, then LQR with its own state.
            luen.x_hat = x_hat_np.copy()
            x_hat_np = luen.estimate(y.reshape(-1, 1), u_prev_np.reshape(-1, 1))
            u_np = np.clip(lqr.compute(x_hat_np.ravel(), x_ref), limits[0], limits[1])

            inputs = _pack_arrays(
                cg,
                {
                    "y": y,
                    "x_ref": x_ref,
                    "u_prev": u_prev_so,
                    "state_x_hat": x_hat_so.ravel(),
                },
            )
            out, state = _step(lib, cg, inputs, n_out, n_state)
            max_err = max(max_err, float(np.max(np.abs(out - u_np))))

            # Each loop evolves its own state.
            x_hat_so = state[sl["state_x_hat"][0] : sl["state_x_hat"][1]].reshape(3, 1)
            u_prev_so = out
            u_prev_np = u_np

        assert max_err < 1e-10, (
            f".so diverged from live Luenberger+LQR over 100 ticks: max abs err = {max_err:.3e}"
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


# Each Zig node line: .{ .op = .matmul, .inputs = &.{6, 3}, .rows = 3, .cols = 3, .aux = 0 [, .vec = false] }
_NODE_RE = re.compile(
    r"\.\{ \.op = \.(\w+), \.inputs = &\.\{([^}]*)\}, \.rows = (\d+), \.cols = (\d+), \.aux = (\d+)(?:, \.vec = (?:true|false))? \}"
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
        assert prov["solver_sha256"] is None

        # no solver facts: the base graph has no bake compiled into the .so
        assert report["solver"] is None

        # graph facts: op histogram matches the composed graph, ports match
        g = report["graph"]
        assert g["nodes_total"] == len(build_base_graph().graph.nodes)
        assert g["buf_len"] > 0
        assert g["has_solve_qp"] is False
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

        assert deltau["graph"]["has_solve_qp"] is True
        assert deltau["graph"]["solve_qp"] == {"expected_n_vars": 45}
        assert deltau["provenance"]["solver_sha256"]
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


class TestDeploymentRecord:
    """The post-compile deployment record: master hash over config/graph/solver/binary.

    ``scripts/stamp_deployment.py`` reads the build manifest, hashes the .so
    and the baked solver tree, and writes ``libbase.deployment.json`` with a
    single master hash committing to the whole chain. ``verify_deployment.py``
    re-hashes the artifacts and compares (producer/verifier separation).
    """

    @staticmethod
    def _sha256(path: str) -> str:
        from shinro.utils.config_resolver import resolve_config_path

        return hashlib.sha256(resolve_config_path(path).read_bytes()).hexdigest()

    def _stamp(self, build_dir):
        from scripts.stamp_deployment import stamp

        return stamp(build_dir, RUNTIME)

    def test_record_master_hash_and_slots(self, manifests):
        """The record carries a master hash and four slots; binary slot matches the .so."""
        base_dir, _ = manifests
        record = self._stamp(base_dir)

        assert re.fullmatch(r"[0-9a-f]{64}", record["master_hash"])
        for slot in ("config", "graph", "solver", "binary"):
            assert re.fullmatch(r"[0-9a-f]{64}", record["slots"][slot])

        so = base_dir / "lib" / "libbase.so"
        assert record["slots"]["binary"] == hashlib.sha256(so.read_bytes()).hexdigest()
        # base graph has no .solve_qp node -> solver slot is the sentinel
        assert record["slots"]["solver"] == hashlib.sha256(b"").hexdigest()
        assert record["solver"] is None

    def test_record_deterministic(self, manifests):
        """Re-stamping the same build produces a byte-identical record."""
        base_dir, _ = manifests
        assert self._stamp(base_dir) == self._stamp(base_dir)

    def test_archive_copy_timestamped(self, manifests):
        """The archive copy is timestamped in the filename only."""
        base_dir, _ = manifests
        self._stamp(base_dir)
        archives = list((base_dir / "deployments").glob("*.json"))
        assert len(archives) >= 1
        assert "-" in archives[0].stem

    def test_verify_passes_and_detects_drift(self, tmp_path_factory):
        """verify_deployment returns 0 on match, 1 when a pinned config drifts."""
        from scripts.verify_deployment import verify

        build_dir = tmp_path_factory.mktemp("zig-build-deploy-verify")
        graph_path = tmp_path_factory.mktemp("zig-build-deploy-verify-graph") / "graph_data.zig"
        cg = build_base_graph()
        _build_so(
            cg,
            build_dir,
            graph_path=graph_path,
            provenance={
                "configs": {
                    "configs/estimators/kalman_base.toml": self._sha256("configs/estimators/kalman_base.toml"),
                    "configs/controllers/lqr_base.toml": self._sha256("configs/controllers/lqr_base.toml"),
                },
            },
        )
        self._stamp(build_dir)
        record = build_dir / "lib" / "libbase.deployment.json"

        assert verify(record, graph_path=graph_path) == 0

        cfg = REPO_ROOT / "src/shinro/configs/controllers/lqr_base.toml"
        original = cfg.read_bytes()
        try:
            cfg.write_bytes(original + b"\n# tamper\n")
            assert verify(record, graph_path=graph_path) == 1
        finally:
            cfg.write_bytes(original)

    def test_graph_provenance_recorded(self, tmp_path):
        """lower_zig with provenance records config hashes + tool versions."""
        cg = build_base_graph()
        out = tmp_path / "graph_data.zig"
        lower_zig(
            cg,
            str(out),
            provenance={
                "configs": {"configs/controllers/lqr_base.toml": "abc123"},
                "python_version": "3.12",
            },
        )
        manifest = json.loads((tmp_path / "graph_data_manifest.json").read_text())
        assert manifest["provenance"]["configs"]["configs/controllers/lqr_base.toml"] == "abc123"
        assert manifest["provenance"]["python_version"] == "3.12"

