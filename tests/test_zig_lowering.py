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
import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.gen_base import build_base_graph
from shinro.codegen import interpret
from shinro.codegen.compose import ComposedGraph, compose
from shinro.codegen.lower_zig import lower_zig
from shinro.codegen.oracle import input_shape, output_split, pack_arrays, state_slices, step_so
from shinro.codegen.trace_node import trace_node
from shinro.codegen.tracing import Graph
from shinro.controllers.mppi import MPPIController
from shinro.controllers.pid import PIDController
from shinro.controllers.smc import SlidingModeController
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
            "tanh",
            "relu",
            "exp",
            "copy",
            "slice",
            "argmax",
            "one_hot",
            "sin",
            "cos",
            "stack",
            "ne_zero",
            "ne_one",
        ],
        state_inputs=[],
        state_outputs=[],
    )


def _build_mpc_graph():
    """Trace the base MPC_LTI and wrap it as a single-input step graph.

    The traced compute() is: e = current_state - target_state → q = Fᵀ e
    (matmul) → solve_qp → u[:3] (slice). The ``solve_qp`` node drives the
    codegen static solver baked into libbase.so
    (src/shinro/runtime/codegen/emosqp/), whose problem must match the
    ``mpc_lti_base.toml`` bake (n_vars=30).
    """
    ctrl = ControllerFactory(str(REPO_ROOT / "src/shinro/configs/controllers/mpc_lti_base.toml")).create(backend=NumpyBackend())
    ng = trace_node(
        ctrl,
        input_shapes={"current_state": (3,), "target_state": (3,)},
    )
    return ComposedGraph(
        graph=ng.graph,
        inputs=["current_state", "target_state"],
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


def _smc_controller(
    smoother: str = "sat",
    phi: float = 0.1,
    alpha: float = 0.0,
    controllability_eps: float = 1e-12,
) -> SlidingModeController:
    """The live (numpy) SMC the lowered graph is compared against.

    Same gains as the shipped ``configs/controllers/smc.toml`` (c=[1, 2],
    k1=1, k2=0, sat, phi=0.1) so the fixture graph and the live component
    cannot drift apart silently.
    """
    return SlidingModeController(
        c=[1.0, 2.0],
        k1=1.0,
        k2=0.0,
        phi=phi,
        smoother=smoother,
        alpha=alpha,
        controllability_eps=controllability_eps,
        backend=NumpyBackend(),
    )


def _build_smc_graph(
    smoother: str = "sat",
    phi: float = 0.1,
    alpha: float = 0.0,
    controllability_eps: float = 1e-12,
    n_u: int = 1,
) -> ComposedGraph:
    """Standalone SMC graph: ``(x, f_x, g_x)`` in, ``(out, healthy)`` out.

    SMC is the first lowered controller whose runtime inputs are live plant
    evaluations (``f_x = f(x)``, ``g_x = g(x)``), so there is no estimator to
    compose with — and ``compose()`` deliberately has no role for ``f_x``/
    ``g_x`` (it refuses them rather than mis-wiring; see ``compose.py``). The
    graph is therefore traced standalone and lowered directly, with the
    dynamics terms as free C-ABI ports the host fills each tick. The plant
    stays on the host, exactly as it does for every other lowered controller.

    Two outputs: ``out`` is the control (fail-safe-guarded — zero when the
    controllability flag trips) and ``healthy`` is the flag SMC publishes via
    :meth:`ArrayBackend.emit_named_output`. No recurrent state: SMC is
    memoryless, so ``state_outputs`` is empty.

    Args:
        smoother: ``sat`` / ``tanh`` / ``sigmoid`` boundary layer.
        phi: Boundary layer thickness (0 selects the pure ``sign`` switch).
        alpha: Fractional power on the switching term (exercises ``pow``).
        controllability_eps: Near-zero ``‖c^T g‖`` threshold, baked into the
            graph as a const — a plant-scaled deployment design parameter.
        n_u: Number of control inputs. ``n_u == 1`` traces the division
            branch; ``n_u > 1`` traces the minimum-norm pseudo-inverse branch
            (``transpose`` + ``matmul``), with ``g_x`` shaped ``(2, n_u)``.

    Returns:
        The traced, lowered-ready :class:`ComposedGraph`.
    """
    smc = _smc_controller(smoother=smoother, phi=phi, alpha=alpha, controllability_eps=controllability_eps)
    ng = trace_node(smc, input_shapes={"x": (2,), "f_x": (2,), "g_x": (2, n_u)})
    return ComposedGraph(
        graph=ng.graph,
        inputs=["x", "f_x", "g_x"],
        outputs=["out", "healthy"],
        state_inputs=[],
        state_outputs=[],
    )


def _smc_rand_inputs(rng: np.random.Generator, min_cg: float = 0.2, n_u: int = 1) -> dict[str, np.ndarray]:
    """Random SMC inputs with ``‖c^T g‖`` kept above the guard.

    ``c^T g = 0`` is measure-zero for continuous random data, so the samples
    are rejection-filtered to stay clear of the fail-safe branch — this keeps
    the numpy reference on its raising path, where it agrees with the graph.
    For ``n_u == 1`` the norm check reduces to the old ``|c^T g|`` one.
    """
    x = rng.normal(0.0, 0.5, (2,))
    f_x = rng.normal(0.0, 0.5, (2,))
    c = np.array([1.0, 2.0])
    while True:
        g_x = rng.normal(0.0, 1.0, (2, n_u))
        if np.linalg.norm(c @ g_x) >= min_cg:
            return {"x": x, "f_x": f_x, "g_x": g_x}


# MPPI graph dims stay deliberately small: the comptime VM unrolls the whole
# K-step rollout, so node count and the stack buffer grow with N*K*D_u.
MPPI_N, MPPI_K, MPPI_DX, MPPI_DU = 6, 3, 3, 3


def _mppi_controller(N: int = MPPI_N, K: int = MPPI_K, dt: float = 0.02) -> MPPIController:
    """The live (numpy) MPPI the lowered graph is compared against.

    Wired from an LTI plant through ``attach_plant`` — the production path
    (``ScenarioFactory`` does the same) — so the traced graph and the live
    reference cannot drift apart silently. The plant fixes D_x = D_u = 3.
    """
    from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot

    bk = NumpyBackend()
    plant = HolonomicMobileRobot(
        num_wheels=3, radius_robots=0.1, gamma=0.0, radius_wheels=0.03, dt=dt, backend=bk
    )
    ctrl = MPPIController(
        num_samples=N,
        temperature=1.0,
        dt=dt,
        horizon=K,
        noise_sigma=[0.5, 0.5, 0.5],
        u_min=[-0.5, -0.5, -0.5],
        u_max=[0.5, 0.5, 0.5],
        seed=1,
        backend=bk,
    )
    ctrl.attach_plant(plant, Q=np.array([1.0, 1.0, 1.0]), R=np.array([0.1, 0.1, 0.1]))
    return ctrl


def _build_mppi_graph(N: int = MPPI_N, K: int = MPPI_K, dt: float = 0.02) -> ComposedGraph:
    """Standalone MPPI graph: the perturbations arrive as an input port.

    MPPI's Gaussian sampling stays on the host (see ``mppi.py``), so unlike
    every other lowered controller there is no RNG in the graph: ``epsilon``
    is a free C-ABI port of shape ``(N, K*D_u)`` (sample-major) the host fills
    each tick. That is what makes parity checkable — the same draw goes to the
    graph and to the live controller.

    Recurrent state is the nominal control sequence ``u`` (``(K, D_u)``): the
    controller rebinds it each tick, so ``trace_node`` detects it and the graph
    emits ``state_u``. Two non-state outputs: ``out`` (the action) and
    ``costs``, the per-sample rollout costs published through
    :meth:`ArrayBackend.emit_named_output` — a host-visible diagnostic, the way
    SMC publishes ``healthy``.

    Args:
        N: Number of sampled perturbations.
        K: Prediction horizon.
        dt: Rollout time step (baked into the plant's model).

    Returns:
        The traced, lowered-ready :class:`ComposedGraph`.
    """
    ctrl = _mppi_controller(N=N, K=K, dt=dt)
    ng = trace_node(
        ctrl,
        input_shapes={
            "current_state": (MPPI_DX,),
            "target_state": (MPPI_DX,),
            "epsilon": (N, K * MPPI_DU),
        },
        state_shapes={"u": (K, MPPI_DU)},
    )
    return ComposedGraph(
        graph=ng.graph,
        inputs=["current_state", "target_state", "epsilon", "state_u"],
        outputs=["out", "costs"],
        state_inputs=["state_u"],
        state_outputs=["state_u"],
    )


# The nonlinear rollout graph: MPPI on an InvertedPendulum (D_x = 2, D_u = 1).
MPPI_NL_DX, MPPI_NL_DU = 2, 1


def _mppi_pendulum_controller(N: int = MPPI_N, K: int = MPPI_K, dt: float = 0.02) -> MPPIController:
    """Live MPPI on a nonlinear plant — the reference the traced graph is checked against.

    Wired through ``attach_plant`` (the production path, as ``ScenarioFactory``
    does), so the rollout integrates the plant's own batch-capable
    ``dynamics``. The plant fixes D_x = 2 (theta, theta_dot) and D_u = 1 (tau).
    """
    from shinro.plants.inverted_pendulum import InvertedPendulum

    bk = NumpyBackend()
    plant = InvertedPendulum(mass=0.1, length=0.5, damping=0.0, gravity=9.81, dt=dt, backend=bk)
    ctrl = MPPIController(
        num_samples=N,
        temperature=1.0,
        dt=dt,
        horizon=K,
        noise_sigma=[0.5],
        u_min=[-0.5],
        u_max=[0.5],
        seed=1,
        backend=bk,
    )
    ctrl.attach_plant(plant, Q=np.array([1.0, 1.0]), R=np.array([0.1]))
    return ctrl


def _build_mppi_pendulum_graph(N: int = MPPI_N, K: int = MPPI_K, dt: float = 0.02) -> ComposedGraph:
    """Standalone nonlinear MPPI graph — same ports, a plant-dynamics rollout.

    The only difference from :func:`_build_mppi_graph` is the plant: the
    rollout evaluates ``InvertedPendulum.dynamics`` over the whole sample batch
    (``sin``/``mul`` nodes of shape ``(N, 1)``, not ``N`` copies of a scalar
    body). The C-ABI contract is unchanged — ``epsilon`` in, ``out`` + ``costs``
    out, ``state_u`` recurrent — which is the point: the lowering path does not
    care whether the dynamics are a matmul or a formula.
    """
    ctrl = _mppi_pendulum_controller(N=N, K=K, dt=dt)
    ng = trace_node(
        ctrl,
        input_shapes={
            "current_state": (MPPI_NL_DX,),
            "target_state": (MPPI_NL_DX,),
            "epsilon": (N, K * MPPI_NL_DU),
        },
        state_shapes={"u": (K, MPPI_NL_DU)},
    )
    return ComposedGraph(
        graph=ng.graph,
        inputs=["current_state", "target_state", "epsilon", "state_u"],
        outputs=["out", "costs"],
        state_inputs=["state_u"],
        state_outputs=["state_u"],
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


@pytest.fixture(scope="session")
def smc_so(tmp_path_factory):
    """Build the .so for the shipped smc.toml config (sat, phi=0.1, alpha=0).

    Lowers to a tmp graph_path so the shared src/shinro/runtime/graph_data.zig
    is not clobbered by this fixture (same discipline as the DeltaU case).
    """
    d = tmp_path_factory.mktemp("zig-build-smc")
    return _build_so(_build_smc_graph(), d, graph_path=d / "graph_data.zig")


@pytest.fixture(scope="session")
def smc_multi_so(tmp_path_factory):
    """Build the .so for the n_u=2 SMC graph (minimum-norm pseudo-inverse branch).

    Two control inputs exercise the transpose/matmul closed form that the
    scalar branch never emits. Tmp graph_path, same discipline as smc_so.
    """
    d = tmp_path_factory.mktemp("zig-build-smc-multi")
    return _build_so(_build_smc_graph(n_u=2), d, graph_path=d / "graph_data.zig")


@pytest.fixture(scope="session")
def mppi_so(tmp_path_factory):
    """Build the .so from the standalone MPPI graph (the sampling-port contract).

    MPPI's perturbations arrive as a free C-ABI port, so this kernel does no
    sampling: the host draws ``epsilon``, and three-way parity (.so vs
    interpreter vs live numpy) is checkable exactly on that same draw. Lowers
    to a tmp graph_path so the shared src/shinro/runtime/graph_data.zig is not
    clobbered (same discipline as smc_so).
    """
    d = tmp_path_factory.mktemp("zig-build-mppi")
    return _build_so(_build_mppi_graph(), d, graph_path=d / "graph_data.zig")


@pytest.fixture(scope="session")
def mppi_pendulum_so(tmp_path_factory):
    """Build the .so from the nonlinear MPPI graph (plant-``dynamics`` rollout).

    The same C-ABI contract as ``mppi_so`` — a free ``epsilon`` port, so parity
    is checkable on one shared draw — but the rollout evaluates the plant's
    batch-capable ``dynamics`` (``sin``/``mul`` nodes over the sample batch)
    instead of a batched matmul. Lowers to a tmp graph_path so the shipped
    runtime graph is never clobbered.
    """
    d = tmp_path_factory.mktemp("zig-build-mppi-pendulum")
    return _build_so(_build_mppi_pendulum_graph(), d, graph_path=d / "graph_data.zig")


# SMC config variants, each a graph-structure specialization: phi=0 swaps the
# clip boundary layer for the `sign` op, sigmoid adds the `abs` + `div` path,
# and alpha=0.5 exercises `pow` with a fractional exponent.
SMC_VARIANTS = [
    pytest.param({"phi": 0.0}, id="sign-phi0"),
    pytest.param({"smoother": "sigmoid"}, id="sigmoid"),
    pytest.param({"alpha": 0.5}, id="sat-alpha-pow"),
]


@pytest.fixture(scope="session", params=SMC_VARIANTS)
def smc_variant_so(request, tmp_path_factory):
    """Build a .so per SMC config variant (each is its own lowered graph)."""
    cg = _build_smc_graph(**request.param)
    slug = "-".join(f"{k}-{v}" for k, v in sorted(request.param.items()))
    d = tmp_path_factory.mktemp(f"zig-build-smc-{slug}")
    lib, composed = _build_so(cg, d, graph_path=d / "graph_data.zig")
    return lib, composed, request.param


def _pack_inputs(cg, y, x_ref, u_prev, x_hat_init, P_init):
    """Pack host inputs into the flat C-ABI buffer, in cg.inputs order."""
    port_arrays = {
        "y": y,
        "x_ref": x_ref,
        "u_prev": u_prev,
        "state_x_hat": x_hat_init.ravel(),
        "state_P": P_init.ravel(),
    }
    return pack_arrays(cg, port_arrays)


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
        n_out, n_state = output_split(cg)
        sl = state_slices(cg)

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
            out, state = step_so(lib, inputs, n_out, n_state)

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
        n_out, n_state = output_split(cg)
        sl = state_slices(cg)

        y = rng.normal(0.0, 0.1, (3,))
        x_ref = rng.normal(0.0, 0.1, (3,))
        u = rng.normal(0.0, 0.1, (3,))
        x_hat = rng.normal(0.0, 0.1, (3, 1))
        P = np.eye(3) * 0.1

        # Run the .so for three ticks, threading state out -> next-tick state in.
        for _ in range(3):
            inputs = _pack_inputs(cg, y, x_ref, u, x_hat, P)
            out, state = step_so(lib, inputs, n_out, n_state)
            u = out
            x_hat = state[sl["state_x_hat"][0] : sl["state_x_hat"][1]].reshape(3, 1)
            P = state[sl["state_P"][0] : sl["state_P"][1]].reshape(3, 3)

        assert np.all(np.isfinite(u)), "Zig step produced non-finite control"


class TestMatmulShapeDispatch:
    def test_so_matches_numpy_for_every_shape(self, matmul_shapes_so):
        """The compiled .so equals numpy for every matmul shape the dispatch
        must route: 2-D matmul, matvec, vecmat, and the single-input
        column-times-scalar-width (m,1)@(1,1) case that was misrouted."""
        lib, cg = matmul_shapes_so
        rng = np.random.default_rng(7)
        n_out, _ = output_split(cg)

        offsets = {}
        off = 0
        for name in cg.outputs:
            size = next(int(np.prod(n.shape)) for n in cg.graph.nodes if n.op == "output" and n.attrs["name"] == name)
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
            inputs = pack_arrays(
                cg,
                {
                    "a": a.ravel(),
                    "v": v.ravel(),
                    "vcol": vcol.ravel(),
                    "scol": scol.ravel(),
                    "row": row.ravel(),
                },
            )
            out, _ = step_so(lib, inputs, n_out, 1)
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
        n_out, n_state = output_split(cg)
        sl = state_slices(cg)
        max_err = 0.0
        for _ in range(30):
            y = rng.normal(0.0, 0.1, (2,))
            x_ref = rng.normal(0.0, 0.1, (2,))
            u_prev = rng.normal(0.0, 0.1, (1,))
            x_hat = rng.normal(0.0, 0.1, (2, 1))
            P = rng.normal(0.0, 0.1, (2, 2))
            P = P @ P.T + 0.1 * np.eye(2)

            inputs = _pack_inputs(cg, y, x_ref, u_prev, x_hat, P)
            out, state = step_so(lib, inputs, n_out, n_state)
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
DIM_SWEEP_CASES = [(f"synth{nx}x{nu}", "synthetic", None, nx, nu, 0.01, "LQR", "KalmanFilter") for nx in (6, 12, 24) for nu in (1, 3, 4)]

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
            shape = next(n.shape for n in cg.graph.nodes if n.op == "input" and n.attrs["name"] == name)
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


# ─── glue-op shape semantics: the VM must reproduce numpy for every shape ───


def _glue_probe_case(name):
    """(builder, in_specs, feed) for one glue-op shape probe."""
    if name == "transpose-nonsquare":

        def build(g):
            return {"t": g.emit("transpose", [g.input("x", (2, 3))], (3, 2))}

        return build, [("x", (2, 3))], {"x": np.arange(6, dtype=float).reshape(2, 3)}
    if name == "clip-scalar-bounds":

        def build(g):
            x = g.input("x", (4,))
            return {"c": g.emit("clip", [x], (4,), lo=np.float64(-0.5), hi=np.float64(0.5))}

        return build, [("x", (4,))], {"x": np.array([-2.0, -0.1, 0.1, 2.0])}
    if name == "where-scalar-branch":

        def build(g):
            x = g.input("x", (3,))
            one = g.emit("const", [], (), value=np.float64(1.0))
            zero = g.emit("const", [], (), value=np.float64(0.0))
            cond = g.emit("ne", [x, zero], (3,))
            return {"w": g.emit("where", [cond, one, x], (3,))}

        return build, [("x", (3,))], {"x": np.array([0.0, 1.0, 2.0])}
    if name == "where-row-broadcast":

        def build(g):
            x = g.input("x", (3, 2))
            bias = g.emit("const", [], (1, 2), value=np.array([[1.0, 2.0]]))
            zero = g.emit("const", [], (1, 2), value=np.zeros((1, 2)))
            cond = g.emit("ne", [x, zero], (3, 2))
            return {"w": g.emit("where", [cond, bias, x], (3, 2))}

        return build, [("x", (3, 2))], {"x": np.arange(6, dtype=float).reshape(3, 2)}
    if name == "ew2-row-broadcast":

        def build(g):
            x = g.input("x", (3, 2))
            bias = g.emit("const", [], (1, 2), value=np.array([[10.0, 20.0]]))
            return {"s": g.emit("add", [x, bias], (3, 2))}

        return build, [("x", (3, 2))], {"x": np.ones((3, 2))}
    if name == "ew2-col-broadcast":

        def build(g):
            x = g.input("x", (3, 2))
            scale = g.emit("const", [], (3, 1), value=np.array([[2.0], [3.0], [4.0]]))
            return {"s": g.emit("mul", [x, scale], (3, 2))}

        return build, [("x", (3, 2))], {"x": np.full((3, 2), 1.5)}
    if name == "slice-2d-rows":

        def build(g):
            x = g.input("x", (4, 2))
            return {"s": g.emit("slice", [x], (2, 2), start=1, stop=3)}

        return build, [("x", (4, 2))], {"x": np.arange(8, dtype=float).reshape(4, 2)}
    if name == "slice-1d-control":

        def build(g):
            x = g.input("x", (6,))
            return {"s": g.emit("slice", [x], (3,), start=2, stop=5)}

        return build, [("x", (6,))], {"x": np.arange(6, dtype=float)}
    if name == "transcendentals":

        def build(g):
            x = g.input("x", (8,))
            return {
                "tanh": g.emit("tanh", [x], (8,)),
                "exp": g.emit("exp", [x], (8,)),
                "sin": g.emit("sin", [x], (8,)),
                "cos": g.emit("cos", [x], (8,)),
            }

        rng = np.random.default_rng(7)
        return build, [("x", (8,))], {"x": rng.normal(0, 2, 8)}
    raise ValueError(name)


GLUE_CASES = [
    # the five found-bug cells ...
    "transpose-nonsquare",  # VM had square-only stride symmetry -> silent garbage
    "clip-scalar-bounds",  # scalar bounds -> flat-blob comptime OOB
    "where-scalar-branch",  # scalar branch -> runtime OOB panic
    "ew2-row-broadcast",  # (1,2)+(3,2) -> runtime OOB panic
    "slice-2d-rows",  # flat-offset indexing on a row slice -> silent garbage
    # ... and broadcast/shape cells adjacent to them
    "where-row-broadcast",
    "ew2-col-broadcast",
    "slice-1d-control",
    "transcendentals",
]


class TestGlueOpShapeSemantics:
    """The lowered VM must reproduce numpy's op semantics for every shape
    class, not just the same-shape ones the control configs generate.

    Regression class for the 2026-09-09 audit: transpose scrambled non-square
    inputs, clip/where/ew2 only handled same-shape or size-1 operands (numpy
    broadcasts scalars and (1, m) rows), and slice treated a row offset as a
    flat element offset on 2-D sources. All five failed silently (wrong
    values) or panicked in debug / corrupted in release.
    """

    @pytest.mark.parametrize("name", GLUE_CASES)
    def test_so_matches_numpy(self, tmp_path, name):
        build, in_specs, feed = _glue_probe_case(name)
        g = Graph()
        outs = build(g)
        for oname, src in outs.items():
            g.output(oname, src)
        cg = ComposedGraph(graph=g, inputs=[n for n, _ in in_specs], outputs=list(outs))
        d = tmp_path / name
        d.mkdir()
        lib, cg2 = _build_so(cg, d / "build", graph_path=d / "graph_data.zig")
        n_out, n_state = output_split(cg2)
        inp = np.concatenate([np.asarray(feed[k]).ravel() for k, _ in in_specs])
        out, _ = step_so(lib, inp, n_out, n_state)
        traced = interpret(cg2.graph, dict(feed))
        off = 0
        max_err = 0.0
        for oname in cg2.outputs:
            exp = np.asarray(traced[oname]).ravel()
            max_err = max(max_err, float(np.max(np.abs(out[off : off + exp.size] - exp))))
            off += exp.size
        assert max_err < 1e-12, f"{name}: .so diverged from numpy: max abs err = {max_err:.3e}"


class TestPlantCompileScan:
    """Every instantiable plant's compiled estimator+controller .so equals
    interpret() — compilation fidelity across the full (op, shape) surface."""

    @pytest.mark.parametrize("plant_so", ALL_SCAN_CASES, indirect=True, ids=[c[0] for c in ALL_SCAN_CASES])
    def test_so_matches_interpret_for_each_plant(self, plant_so):
        lib, cg, n_x, n_u, name = plant_so
        rng = np.random.default_rng(5)
        n_out, n_state = output_split(cg)
        sl = state_slices(cg)
        max_err = 0.0
        for _ in range(20):
            ports = _scan_input_ports(cg, n_x, n_u, rng)
            inputs = pack_arrays(cg, {k: v.ravel() for k, v in ports.items()})
            out, state = step_so(lib, inputs, n_out, n_state)
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
        'type = "PID"\nname = "pid"\ndt = 0.01\nkp = [2.0]\nki = [0.5]\nkd = [0.5]\noutput_limits = { min = [-10.0], max = [10.0] }\n'
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
        n_out, n_state = output_split(cg)
        assert n_state == 0

        exact_ops = {"copy", "slice", "relu", "argmax", "one_hot", "stack", "ne_zero", "ne_one"}
        transcendental = {"exp", "tanh", "sin", "cos"}

        for _ in range(20):
            x = rng.normal(0.0, 1.0, (4,))
            inputs = pack_arrays(cg, {"x": x})
            out, _ = step_so(lib, inputs, n_out, n_state)

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
                    assert np.array_equal(got, expected), f"op {name} diverged: got {got}, expected {expected}"


class TestSmcOracle:
    """The lowered SMC control law matches the interpreter and live numpy.

    SMC is the first lowered controller whose runtime inputs are live plant
    evaluations, so the graph is standalone (no estimator) with ``f_x``/
    ``g_x`` as free C-ABI ports. This suite covers the ops added for it
    (``abs`` / ``sign`` / ``pow`` / ``lt``) plus the auxiliary ``healthy``
    port and the where-guarded fail-safe: a graph has no exceptions, so the
    near-zero ``c^T g`` guard compiles to a zero command + a flag instead of
    a raise.
    """

    def test_so_matches_interpreter_and_numpy(self, smc_so):
        """.so, graph interpreter, and live numpy agree on 25 seeded samples."""
        lib, cg = smc_so
        n_out, n_state = output_split(cg)
        assert n_state == 0
        assert n_out == 2  # out + healthy

        rng = np.random.default_rng(11)
        smc = _smc_controller()
        max_err = 0.0
        for _ in range(25):
            arrays = _smc_rand_inputs(rng)
            out, _ = step_so(lib, pack_arrays(cg, arrays), n_out, n_state)
            traced = interpret(cg.graph, arrays)
            want = np.asarray(smc.compute(arrays["x"], arrays["f_x"], arrays["g_x"])).ravel()

            # healthy = 1 on this path (|c^T g| ≥ 0.2 ≫ 1e-12)
            assert out[1] == 1.0
            assert traced["healthy"][0] == 1.0
            np.testing.assert_allclose(out[0], traced["out"][0], rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(out[0], want[0], rtol=1e-12, atol=1e-12)
            max_err = max(max_err, abs(out[0] - want[0]))
        assert max_err < 1e-12, f"SMC .so drifted from live numpy: {max_err:.3e}"

    @pytest.mark.parametrize(
        "g_x",
        [
            np.array([[1.0], [-0.5]]),  # c^T g == 0 exactly
            np.array([[0.0], [0.0]]),  # degenerate actuator channel
        ],
        ids=["exact-zero", "zero-g"],
    )
    def test_lost_controllability_is_failsafe_and_flagged(self, smc_so, g_x):
        """A graph cannot raise: |c^T g| below eps → u == 0 and healthy == 0.

        The compiled analogue of numpy's RuntimeError. Zero-command is the
        kernel's floor, not a safety guarantee — the flag is what lets the
        host run its own fault policy in the same tick.
        """
        lib, cg = smc_so
        n_out, n_state = output_split(cg)
        arrays = {"x": np.array([1.0, 0.5]), "f_x": np.array([0.3, -0.2]), "g_x": g_x}

        out, _ = step_so(lib, pack_arrays(cg, arrays), n_out, n_state)
        assert out[0] == 0.0, "fail-safe must emit exactly zero"
        assert out[1] == 0.0, "controllability flag must be low"

        # The interpreter (the graph's own reference) agrees, and the live
        # component still raises — the two paths differ only here, by design.
        with np.errstate(divide="ignore", invalid="ignore"):
            traced = interpret(cg.graph, arrays)
            assert traced["out"][0] == 0.0
            assert traced["healthy"][0] == 0.0
        with pytest.raises(RuntimeError, match="near-zero"):
            _smc_controller().compute(arrays["x"], arrays["f_x"], arrays["g_x"])

    def test_so_matches_interpreter_and_numpy_multi_input(self, smc_multi_so):
        """n_u=2: .so, interpreter, and live numpy agree on the min-norm branch."""
        lib, cg = smc_multi_so
        n_out, n_state = output_split(cg)
        assert n_state == 0
        assert n_out == 3  # out (2 inputs) + healthy

        rng = np.random.default_rng(17)
        smc = _smc_controller()
        c = np.array([1.0, 2.0])
        max_err = 0.0
        for _ in range(25):
            arrays = _smc_rand_inputs(rng, n_u=2)
            out, _ = step_so(lib, pack_arrays(cg, arrays), n_out, n_state)
            traced = interpret(cg.graph, arrays)
            want = np.asarray(smc.compute(arrays["x"], arrays["f_x"], arrays["g_x"])).ravel()

            # healthy = 1 on this path (||c^T g|| >= 0.2 >> 1e-12)
            assert out[2] == 1.0
            assert traced["healthy"][0] == 1.0
            np.testing.assert_allclose(out[:2], traced["out"], rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(out[:2], want, rtol=1e-12, atol=1e-12)

            # The command delivers the reaching law along the surface: the
            # pseudo-inverse is not just close to numpy, it solves the system.
            s = float(c @ arrays["x"])
            num = -smc.k1 * abs(s) ** smc.alpha * np.clip(s / smc.phi, -1.0, 1.0) - float(c @ arrays["f_x"])
            assert np.isclose((c @ arrays["g_x"]) @ want, num, atol=1e-10)
            max_err = max(max_err, float(np.max(np.abs(out[:2] - want))))
        assert max_err < 1e-12, f"SMC n_u=2 .so drifted from live numpy: {max_err:.3e}"

    def test_multi_input_lost_controllability_is_failsafe_and_flagged(self, smc_multi_so):
        """n_u>1: ||c^T g|| below eps → u == 0 (both inputs) and healthy == 0."""
        lib, cg = smc_multi_so
        n_out, n_state = output_split(cg)
        # c = [1, 2]; each column is orthogonal to c, so c^T g == [0, 0].
        g_x = np.array([[2.0, -2.0], [-1.0, 1.0]])
        assert np.allclose(np.array([1.0, 2.0]) @ g_x, 0.0)
        arrays = {"x": np.array([1.0, 0.5]), "f_x": np.array([0.3, -0.2]), "g_x": g_x}

        out, _ = step_so(lib, pack_arrays(cg, arrays), n_out, n_state)
        np.testing.assert_array_equal(out[:2], np.zeros(2))
        assert out[2] == 0.0, "controllability flag must be low"

        with np.errstate(divide="ignore", invalid="ignore"):
            traced = interpret(cg.graph, arrays)
            np.testing.assert_array_equal(traced["out"], np.zeros(2))
            assert traced["healthy"][0] == 0.0
        with pytest.raises(RuntimeError, match="near-zero"):
            _smc_controller().compute(arrays["x"], arrays["f_x"], arrays["g_x"])

    def test_multi_input_graph_uses_the_pseudo_inverse_ops(self, smc_multi_so):
        """The n_u>1 branch is matmul + transpose, not a baked solve constant."""
        _, cg = smc_multi_so
        ops = {node.op for node in cg.graph.nodes}
        assert {"transpose", "matmul", "abs", "lt", "where"} <= ops, f"missing ops: {sorted(ops)}"
        assert set(cg.outputs) == {"out", "healthy"}
        assert cg.state_outputs == []  # SMC is memoryless

    def test_graph_uses_the_new_ops_and_aux_port(self, smc_so):
        """Drift guard: the guard/flag structure is actually in the graph."""
        _, cg = smc_so
        ops = {node.op for node in cg.graph.nodes}
        assert {"abs", "pow", "lt", "where"} <= ops, f"missing guard ops: {sorted(ops)}"
        assert set(cg.outputs) == {"out", "healthy"}
        assert cg.state_outputs == []  # SMC is memoryless

    def test_graph_contains_sign_for_pure_switching(self, smc_variant_so):
        """phi == 0 selects the sign path; the variant graph must contain it."""
        _, cg, cfg = smc_variant_so
        if cfg.get("phi") != 0.0:
            pytest.skip("sign only appears when phi == 0")
        ops = {node.op for node in cg.graph.nodes}
        assert "sign" in ops

    def test_variant_graphs_match_interpreter(self, smc_variant_so):
        """Each config variant is its own graph and matches the interpreter.

        phi=0 swaps clip for sign; sigmoid (s/(|s|+phi)) adds abs; alpha=0.5
        exercises pow with a fractional exponent — all against the graph's
        own interpreter reference, with the live numpy component as the
        second opinion.
        """
        lib, cg, cfg = smc_variant_so
        n_out, n_state = output_split(cg)
        rng = np.random.default_rng(29)
        smc = _smc_controller(**cfg)

        for _ in range(15):
            arrays = _smc_rand_inputs(rng)
            out, _ = step_so(lib, pack_arrays(cg, arrays), n_out, n_state)
            traced = interpret(cg.graph, arrays)
            want = np.asarray(smc.compute(arrays["x"], arrays["f_x"], arrays["g_x"])).ravel()
            assert out[1] == 1.0
            np.testing.assert_allclose(out[0], traced["out"][0], rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(out[0], want[0], rtol=1e-12, atol=1e-12)

    def test_controllability_eps_is_baked_from_config(self, tmp_path):
        """eps is a deployment knob: a plant-scaled value trips the flag earlier.

        With eps = 0.5, a perfectly usable-but-small |c^T g| = 0.2 is treated
        as lost controllability — the arithmetic-only 1e-12 default would let
        it through and amplify 1/0.2 instead. The graph is built with its own
        graph_path so the shared src/shinro/runtime/graph_data.zig is untouched.
        """
        cg = _build_smc_graph(controllability_eps=0.5)
        lib, _ = _build_so(cg, tmp_path, graph_path=tmp_path / "graph_data.zig")
        n_out, n_state = output_split(cg)
        arrays = {"x": np.array([1.0, 0.5]), "f_x": np.array([0.3, -0.2]), "g_x": np.array([[0.2], [0.0]])}
        assert abs(0.2) > 1e-12  # the default guard would not fire here

        out, _ = step_so(lib, pack_arrays(cg, arrays), n_out, n_state)
        assert out[0] == 0.0
        assert out[1] == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            traced = interpret(cg.graph, arrays)
        assert traced["healthy"][0] == 0.0


class TestMppiOracle:
    """The interpreted MPPI graph matches live numpy — including recurrence.

    MPPI's Gaussian sampling stays on the host, so the perturbations arrive
    through a free ``epsilon`` port and the traced call never touches the RNG.
    That is what makes parity checkable at all: feed the same draw to the graph
    and to the live controller and they must agree, tick after tick.

    These cases are interpreter-only (no .so): they pin that the traced graph
    computes the same control law the live component does, which is the oracle
    the Zig VM is checked against next.
    """

    def _feeds(self, x0, x_ref, epsilon, state_u=None):
        """The four C-ABI input ports; ``state_u`` defaults to a zero plan."""
        return {
            "current_state": x0,
            "target_state": x_ref,
            "epsilon": epsilon,
            "state_u": np.zeros((MPPI_K, MPPI_DU)) if state_u is None else state_u,
        }

    def test_interpreter_matches_numpy(self):
        """interpret() == live numpy across 10 seeded perturbation draws."""
        cg = _build_mppi_graph()
        ref = _mppi_controller()
        rng = np.random.default_rng(23)
        x_ref = np.array([1.0, 0.0, 0.0])
        max_u_err = 0.0
        max_state_err = 0.0
        for _ in range(10):
            # Each iteration is a fresh tick: the graph is fed a zero plan, so
            # the live reference must start from one too.
            ref.reset()
            x0 = rng.normal(0.0, 0.5, MPPI_DX)
            eps = rng.normal(0.0, 0.5, (MPPI_N, MPPI_K * MPPI_DU))
            out = interpret(cg.graph, self._feeds(x0, x_ref, eps))
            want_u = np.asarray(ref.compute(x0, x_ref, eps)).ravel()
            want_state = np.asarray(ref.u).reshape(MPPI_K, MPPI_DU)

            np.testing.assert_allclose(out["out"], want_u, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(out["state_u"], want_state, rtol=1e-11, atol=1e-11)
            max_u_err = max(max_u_err, np.max(np.abs(out["out"] - want_u)))
            max_state_err = max(max_state_err, np.max(np.abs(out["state_u"] - want_state)))
        assert max_u_err < 1e-11, f"MPPI graph drifted from live numpy: {max_u_err:.3e}"
        assert max_state_err < 1e-11

    def test_costs_port_is_published(self):
        """The per-sample rollout costs are a graph output port (diagnostic)."""
        cg = _build_mppi_graph()
        rng = np.random.default_rng(31)
        out = interpret(
            cg.graph,
            self._feeds(
                rng.normal(0.0, 0.5, MPPI_DX),
                np.array([1.0, 0.0, 0.0]),
                rng.normal(0.0, 0.5, (MPPI_N, MPPI_K * MPPI_DU)),
            ),
        )
        assert out["costs"].shape == (MPPI_N,)
        assert np.all(np.isfinite(out["costs"]))

    def test_recurrence_matches_sequential_ticks(self):
        """Feeding state_u back reproduces a second live tick (recurrent edge)."""
        cg = _build_mppi_graph()
        ref = _mppi_controller()
        rng = np.random.default_rng(29)
        x0 = rng.normal(0.0, 0.5, MPPI_DX)
        x_ref = np.array([1.0, 0.0, 0.0])
        eps1 = rng.normal(0.0, 0.5, (MPPI_N, MPPI_K * MPPI_DU))
        eps2 = rng.normal(0.0, 0.5, (MPPI_N, MPPI_K * MPPI_DU))

        tick1 = interpret(cg.graph, self._feeds(x0, x_ref, eps1))
        want1 = np.asarray(ref.compute(x0, x_ref, eps1)).ravel()
        np.testing.assert_allclose(tick1["out"], want1, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            tick1["state_u"], np.asarray(ref.u).reshape(MPPI_K, MPPI_DU), rtol=1e-11, atol=1e-11
        )

        # The graph's own state output feeds the next tick — no numpy state.
        tick2 = interpret(cg.graph, self._feeds(x0, x_ref, eps2, state_u=tick1["state_u"]))
        want2 = np.asarray(ref.compute(x0, x_ref, eps2)).ravel()
        np.testing.assert_allclose(tick2["out"], want2, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            tick2["state_u"], np.asarray(ref.u).reshape(MPPI_K, MPPI_DU), rtol=1e-11, atol=1e-11
        )

    def test_graph_structure_and_ports(self):
        """Drift guard: the ops MPPI relies on and the C-ABI port layout."""
        cg = _build_mppi_graph()
        ops = {node.op for node in cg.graph.nodes}
        for op in ("min", "matmul", "clip", "slice", "stack", "transpose", "exp", "reshape"):
            assert op in ops, f"MPPI graph lost the {op!r} op"

        assert cg.outputs == ["out", "costs"]
        assert cg.state_outputs == ["state_u"]
        port_shapes = {n.attrs["name"]: n.shape for n in cg.graph.nodes if n.op == "input"}
        # The sampling contract: (N, K*D_u), sample-major — what the host packs.
        assert port_shapes["epsilon"] == (MPPI_N, MPPI_K * MPPI_DU)
        assert port_shapes["state_u"] == (MPPI_K, MPPI_DU)
        assert port_shapes["current_state"] == (MPPI_DX,)
        assert port_shapes["target_state"] == (MPPI_DX,)

    def test_so_matches_interpreter_and_numpy(self, mppi_so):
        """.so, interpreter, and live numpy agree on the same seeded draws."""
        lib, cg = mppi_so
        n_out, n_state = output_split(cg)
        assert n_state == MPPI_K * MPPI_DU  # the nominal plan recurs
        assert n_out == MPPI_DU + MPPI_N  # out (D_u) + costs (N)

        ref = _mppi_controller()
        rng = np.random.default_rng(41)
        x_ref = np.array([1.0, 0.0, 0.0])
        max_u_err = 0.0
        for _ in range(10):
            ref.reset()
            x0 = rng.normal(0.0, 0.5, MPPI_DX)
            eps = rng.normal(0.0, 0.5, (MPPI_N, MPPI_K * MPPI_DU))
            feeds = self._feeds(x0, x_ref, eps)
            out, state = step_so(lib, pack_arrays(cg, feeds), n_out, n_state)
            traced = interpret(cg.graph, feeds)
            want_u = np.asarray(ref.compute(x0, x_ref, eps)).ravel()

            # kernel vs its own interpreter (the tight tier), then vs numpy
            np.testing.assert_allclose(out[:MPPI_DU], traced["out"], rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(state, np.asarray(traced["state_u"]).ravel(), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(out[MPPI_DU:], np.asarray(traced["costs"]).ravel(), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(out[:MPPI_DU], want_u, rtol=1e-11, atol=1e-11)
            max_u_err = max(max_u_err, np.max(np.abs(out[:MPPI_DU] - want_u)))
        assert max_u_err < 1e-11, f"MPPI .so drifted from live numpy: {max_u_err:.3e}"

    def test_cabi_recurrence_matches_numpy(self, mppi_so):
        """The kernel's own state buffer reproduces a second live tick.

        The host feeds ``state_out`` straight back as the next tick's
        ``state_u`` — no numpy state in the loop. Two ticks must match two
        sequential live ``compute()`` calls.
        """
        lib, cg = mppi_so
        n_out, n_state = output_split(cg)
        start, stop = state_slices(cg)["state_u"]
        ref = _mppi_controller()
        rng = np.random.default_rng(43)
        x0 = rng.normal(0.0, 0.5, MPPI_DX)
        x_ref = np.array([1.0, 0.0, 0.0])
        eps1 = rng.normal(0.0, 0.5, (MPPI_N, MPPI_K * MPPI_DU))
        eps2 = rng.normal(0.0, 0.5, (MPPI_N, MPPI_K * MPPI_DU))

        out1, state1 = step_so(lib, pack_arrays(cg, self._feeds(x0, x_ref, eps1)), n_out, n_state)
        want1 = np.asarray(ref.compute(x0, x_ref, eps1)).ravel()
        np.testing.assert_allclose(out1[:MPPI_DU], want1, rtol=1e-11, atol=1e-11)

        # Feed the kernel's state back through the C-ABI input buffer.
        plan = state1[start:stop].reshape(MPPI_K, MPPI_DU)
        out2, state2 = step_so(lib, pack_arrays(cg, self._feeds(x0, x_ref, eps2, state_u=plan)), n_out, n_state)
        want2 = np.asarray(ref.compute(x0, x_ref, eps2)).ravel()
        np.testing.assert_allclose(out2[:MPPI_DU], want2, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(state2[start:stop], np.asarray(ref.u).ravel(), rtol=1e-11, atol=1e-11)


class TestMppiNonlinearOracle:
    """The nonlinear MPPI graph: same C-ABI ports, a plant-``dynamics`` rollout.

    MPPI on a nonlinear plant rolls out with the plant's own batch-capable
    ``dynamics``, so the graph contains one ``sin``/``mul`` node of shape
    ``(N, 1)`` per formula term instead of ``N`` copies of a scalar body — the
    property that keeps the node count independent of the sample count. These
    cases are interpreter-only; the ``.so`` three-way parity follows.
    """

    def _feeds(self, x0, x_ref, epsilon, state_u=None):
        """The four C-ABI input ports; ``state_u`` defaults to a zero plan."""
        return {
            "current_state": x0,
            "target_state": x_ref,
            "epsilon": epsilon,
            "state_u": np.zeros((MPPI_K, MPPI_NL_DU)) if state_u is None else state_u,
        }

    def _draw(self, rng):
        """A tick's state, reference, and perturbation draw."""
        x0 = rng.normal(0.0, 0.3, MPPI_NL_DX)
        x_ref = np.array([0.5, 0.0])
        eps = rng.normal(0.0, 0.5, (MPPI_N, MPPI_K * MPPI_NL_DU))
        return x0, x_ref, eps

    def test_interpreter_matches_numpy(self):
        """interpret() == live numpy across 10 seeded draws (nonlinear rollout)."""
        cg = _build_mppi_pendulum_graph()
        ref = _mppi_pendulum_controller()
        rng = np.random.default_rng(23)
        max_u_err = 0.0
        max_state_err = 0.0
        for _ in range(10):
            ref.reset()
            x0, x_ref, eps = self._draw(rng)
            out = interpret(cg.graph, self._feeds(x0, x_ref, eps))
            want_u = np.asarray(ref.compute(x0, x_ref, eps)).ravel()
            want_state = np.asarray(ref.u).reshape(MPPI_K, MPPI_NL_DU)
            np.testing.assert_allclose(out["out"], want_u, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(out["state_u"], want_state, rtol=1e-11, atol=1e-11)
            max_u_err = max(max_u_err, float(np.max(np.abs(out["out"] - want_u))))
            max_state_err = max(max_state_err, float(np.max(np.abs(out["state_u"] - want_state))))
        assert max_u_err < 1e-11, f"nonlinear MPPI graph drifted from live numpy: {max_u_err:.3e}"
        assert max_state_err < 1e-11

    def test_costs_port_is_published(self):
        """The per-sample rollout costs remain a graph output port."""
        cg = _build_mppi_pendulum_graph()
        rng = np.random.default_rng(31)
        out = interpret(cg.graph, self._feeds(*self._draw(rng)))
        assert out["costs"].shape == (MPPI_N,)
        assert np.all(np.isfinite(out["costs"]))

    def test_recurrence_matches_sequential_ticks(self):
        """Feeding state_u back reproduces a second live tick."""
        cg = _build_mppi_pendulum_graph()
        ref = _mppi_pendulum_controller()
        rng = np.random.default_rng(29)
        x0, x_ref, eps1 = self._draw(rng)
        _, _, eps2 = self._draw(rng)

        tick1 = interpret(cg.graph, self._feeds(x0, x_ref, eps1))
        want1 = np.asarray(ref.compute(x0, x_ref, eps1)).ravel()
        np.testing.assert_allclose(tick1["out"], want1, rtol=1e-11, atol=1e-11)

        tick2 = interpret(cg.graph, self._feeds(x0, x_ref, eps2, state_u=tick1["state_u"]))
        want2 = np.asarray(ref.compute(x0, x_ref, eps2)).ravel()
        np.testing.assert_allclose(tick2["out"], want2, rtol=1e-11, atol=1e-11)

    def test_graph_uses_plant_dynamics_and_only_u_recurs(self):
        """Drift guard: the nonlinear op, the port layout, and the single state."""
        cg = _build_mppi_pendulum_graph()
        ops = {node.op for node in cg.graph.nodes}
        assert "sin" in ops, "the nonlinear rollout lost the plant's sin term"
        for op in ("min", "matmul", "clip", "slice", "stack", "exp", "reshape"):
            assert op in ops, f"nonlinear MPPI graph lost the {op!r} op"

        assert cg.outputs == ["out", "costs"]
        # ``state_outputs`` is exactly what trace-time state detection found:
        # the nominal plan, and nothing else (e.g. the tracking reference must
        # not be promoted to a recurrent port).
        assert cg.state_outputs == ["state_u"]
        port_shapes = {n.attrs["name"]: n.shape for n in cg.graph.nodes if n.op == "input"}
        assert port_shapes["epsilon"] == (MPPI_N, MPPI_K * MPPI_NL_DU)
        assert port_shapes["state_u"] == (MPPI_K, MPPI_NL_DU)
        assert port_shapes["current_state"] == (MPPI_NL_DX,)
        assert port_shapes["target_state"] == (MPPI_NL_DX,)

    def test_node_count_is_independent_of_sample_count(self):
        """The headline property: nodes track K*D_u, not the number of samples.

        A per-sample (looped) rollout would grow as ``N*K`` — the reason the
        batched ``dynamics`` contract exists. Tracing the same policy with 4x
        the samples must produce the identical graph.
        """
        small = _build_mppi_pendulum_graph(N=6)
        large = _build_mppi_pendulum_graph(N=24)

        def eps_shape(cg):
            return next(n.shape for n in cg.graph.nodes if n.op == "input" and n.attrs["name"] == "epsilon")

        # The 4x batch really is in the graph — in the *shape* of the port, not
        # in the number of nodes.
        assert eps_shape(small) == (6, MPPI_K * MPPI_NL_DU)
        assert eps_shape(large) == (24, MPPI_K * MPPI_NL_DU)
        assert len(large.graph.nodes) == len(small.graph.nodes)

    def test_so_matches_interpreter_and_numpy(self, mppi_pendulum_so):
        """.so, interpreter, and live numpy agree on the same seeded draws."""
        lib, cg = mppi_pendulum_so
        n_out, n_state = output_split(cg)
        assert n_state == MPPI_K * MPPI_NL_DU  # the nominal plan recurs
        assert n_out == MPPI_NL_DU + MPPI_N  # out (D_u) + costs (N)

        ref = _mppi_pendulum_controller()
        rng = np.random.default_rng(41)
        max_u_err = 0.0
        for _ in range(10):
            ref.reset()
            x0, x_ref, eps = self._draw(rng)
            feeds = self._feeds(x0, x_ref, eps)
            out, state = step_so(lib, pack_arrays(cg, feeds), n_out, n_state)
            traced = interpret(cg.graph, feeds)
            want_u = np.asarray(ref.compute(x0, x_ref, eps)).ravel()

            # kernel vs its own interpreter (the tight tier), then vs numpy
            np.testing.assert_allclose(out[:MPPI_NL_DU], traced["out"], rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(state, np.asarray(traced["state_u"]).ravel(), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(out[MPPI_NL_DU:], np.asarray(traced["costs"]).ravel(), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(out[:MPPI_NL_DU], want_u, rtol=1e-11, atol=1e-11)
            max_u_err = max(max_u_err, float(np.max(np.abs(out[:MPPI_NL_DU] - want_u))))
        assert max_u_err < 1e-11, f"nonlinear MPPI .so drifted from live numpy: {max_u_err:.3e}"

    def test_cabi_recurrence_matches_numpy(self, mppi_pendulum_so):
        """The kernel's own state buffer reproduces a second live tick.

        The host feeds ``state_out`` straight back as the next tick's
        ``state_u`` — no numpy state in the loop.
        """
        lib, cg = mppi_pendulum_so
        n_out, n_state = output_split(cg)
        start, stop = state_slices(cg)["state_u"]
        ref = _mppi_pendulum_controller()
        rng = np.random.default_rng(43)
        x0, x_ref, eps1 = self._draw(rng)
        _, _, eps2 = self._draw(rng)

        out1, state1 = step_so(lib, pack_arrays(cg, self._feeds(x0, x_ref, eps1)), n_out, n_state)
        want1 = np.asarray(ref.compute(x0, x_ref, eps1)).ravel()
        np.testing.assert_allclose(out1[:MPPI_NL_DU], want1, rtol=1e-11, atol=1e-11)

        plan = state1[start:stop].reshape(MPPI_K, MPPI_NL_DU)
        out2, state2 = step_so(lib, pack_arrays(cg, self._feeds(x0, x_ref, eps2, state_u=plan)), n_out, n_state)
        want2 = np.asarray(ref.compute(x0, x_ref, eps2)).ravel()
        np.testing.assert_allclose(out2[:MPPI_NL_DU], want2, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(state2[start:stop], np.asarray(ref.u).ravel(), rtol=1e-11, atol=1e-11)


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
        n_out, n_state = output_split(cg)
        assert n_state == 0
        assert n_out == 3

        max_err = 0.0
        for _ in range(10):
            x0 = rng.normal(0.0, 0.1, (3,))
            zero = np.zeros(3)
            inputs = pack_arrays(cg, {"current_state": x0, "target_state": zero})
            out, _ = step_so(lib, inputs, n_out, n_state)

            traced = interpret(cg.graph, {"current_state": x0, "target_state": zero})["out"]
            max_err = max(max_err, float(np.max(np.abs(out - np.asarray(traced).ravel()))))

        assert max_err < 1e-3, f"Zig .so solve_qp diverged from interpreter: max abs err = {max_err:.3e}"


# ─── closed-loop oracles: .so vs live numpy components over 100 ticks ───────


def _make_kf():
    return EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())


def _make_luenberger():
    return EstimatorFactory("configs/estimators/luenberger_base.toml").create(backend=NumpyBackend())


def _make_lqr():
    return ControllerFactory("configs/controllers/lqr_base.toml").create(backend=NumpyBackend())


def _make_mpc_lti():
    return ControllerFactory("configs/controllers/mpc_lti_base.toml").create(backend=NumpyBackend())


def _make_mpc_deltau():
    return ControllerFactory("configs/controllers/mpc_base.toml").create(backend=NumpyBackend())


def _make_pid():
    return PIDController(
        kp=np.array([2.0, 2.0, 2.0]),
        ki=np.array([0.5, 0.5, 0.5]),
        kd=np.array([0.5, 0.5, 0.5]),
        dt=0.02,
        output_limits=(np.array([-0.3, -0.3, -0.6]), np.array([0.3, 0.3, 0.6])),
        backend=NumpyBackend(),
    )


_CLOSED_LOOP_LIMITS = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))
KF_P_INIT = np.eye(3) * 0.1

KF_PORTS = (("state_x_hat", "x_hat", (3, 1)), ("state_P", "P", (3, 3)))
LUENBERGER_PORTS = (("state_x_hat", "x_hat", (3, 1)),)
PID_PORTS = (
    ("state_integral", "_integral", (3,)),
    ("state_prev_error", "_prev_error", (3,)),
    ("state_has_run", "_has_run", (3,)),
)


@dataclasses.dataclass(frozen=True)
class ClosedLoopCase:
    """One (estimator, controller) closed-loop oracle config.

    est_state_ports: (graph state port, live estimator attr, shape) triples.
    est_seed_mode: "cross" re-seeds the live estimator from the .so's threaded
        state each tick (KF cases — isolates single-tick predict-update
        fidelity from long-horizon drift toward the gate); "self" lets both
        sides evolve their own states independently (Luenberger — the
        stronger property).
    ctrl_state_ports: controller recurrent state (PID's integral/prev_error/
        has_run), threaded on both sides.
    mode: how the controller consumes the estimate — "tracking" (x_hat,
        x_ref), "error" (x_hat - x_ref, MPC_LTI), "error_deltau"
        (MPC_DeltaU, which also takes u_prev).
    sat_threshold: when set, the loop must actually saturate (asserts the
        anti-windup ne/where path is genuinely exercised).
    """

    name: str
    fixture: str
    estimator: Callable[[], Any]
    controller: Callable[[], Any]
    mode: str
    est_state_ports: tuple[tuple[str, str, tuple[int, ...]], ...]
    est_seed_mode: str
    ctrl_state_ports: tuple[tuple[str, str, tuple[int, ...]], ...]
    est_init: dict[str, np.ndarray]
    tol: float
    seed: int
    sat_threshold: float | None = None


CASES = [
    ClosedLoopCase(
        name="kf_lqr",
        fixture="base_so",
        estimator=_make_kf,
        controller=_make_lqr,
        mode="tracking",
        est_state_ports=KF_PORTS,
        est_seed_mode="cross",
        ctrl_state_ports=(),
        est_init={"state_P": KF_P_INIT},
        tol=1e-10,
        seed=11,
    ),
    ClosedLoopCase(
        name="kf_mpc_lti",
        fixture="mpc_composed_so",
        estimator=_make_kf,
        controller=_make_mpc_lti,
        mode="error",
        est_state_ports=KF_PORTS,
        est_seed_mode="cross",
        ctrl_state_ports=(),
        est_init={"state_P": KF_P_INIT},
        tol=1e-4,
        seed=21,
    ),
    ClosedLoopCase(
        name="kf_pid",
        fixture="pid_composed_so",
        estimator=_make_kf,
        controller=_make_pid,
        mode="tracking",
        est_state_ports=KF_PORTS,
        est_seed_mode="cross",
        ctrl_state_ports=PID_PORTS,
        est_init={"state_P": KF_P_INIT},
        tol=1e-10,
        seed=31,
        sat_threshold=0.3,
    ),
    ClosedLoopCase(
        name="luenberger_lqr",
        fixture="luenberger_composed_so",
        estimator=_make_luenberger,
        controller=_make_lqr,
        mode="tracking",
        est_state_ports=LUENBERGER_PORTS,
        est_seed_mode="self",
        ctrl_state_ports=(),
        est_init={},
        tol=1e-10,
        seed=37,
    ),
    ClosedLoopCase(
        name="kf_mpc_deltau",
        fixture="mpc_deltau_composed_so",
        estimator=_make_kf,
        controller=_make_mpc_deltau,
        mode="error_deltau",
        est_state_ports=KF_PORTS,
        est_seed_mode="cross",
        ctrl_state_ports=(),
        est_init={"state_P": KF_P_INIT},
        tol=1e-4,
        seed=41,
    ),
]


def _run_closed_loop(lib, cg, case, ticks=100):
    """Thread the .so and the live components through `ticks` shared y/x_ref ticks.

    The .so side threads its recurrent state ports across ticks
    (state_slices); the live side drives the real estimator/controller on the
    numpy backend. Each side evolves its own u_prev; the max abs error
    between the two controls is the oracle metric.
    """
    rng = np.random.default_rng(case.seed)
    n_out, n_state = output_split(cg)
    sl = state_slices(cg)
    n_x = int(np.prod(input_shape(cg.graph, "y")))
    n_u = int(np.prod(input_shape(cg.graph, "u_prev")))
    est = case.estimator()
    ctrl = case.controller()

    so_est = {port: case.est_init.get(port, np.zeros(shape)).astype(np.float64).copy() for port, _, shape in case.est_state_ports}
    live_est = {attr: case.est_init.get(port, np.zeros(shape)).astype(np.float64).copy() for port, attr, shape in case.est_state_ports}
    so_ctrl = {port: np.zeros(shape) for port, _, shape in case.ctrl_state_ports}
    live_ctrl = {attr: np.zeros(shape) for _, attr, shape in case.ctrl_state_ports}
    u_prev_so = np.zeros(n_u)
    u_prev_np = np.zeros(n_u)

    max_err = 0.0
    saw_saturation = False
    for _ in range(ticks):
        y = rng.normal(0.0, 0.1, (n_x,))
        x_ref = rng.normal(0.0, 0.05, (n_x,))

        if case.est_seed_mode == "cross":
            for port, attr, shape in case.est_state_ports:
                setattr(est, attr, so_est[port].reshape(shape).copy())
        else:
            for port, attr, shape in case.est_state_ports:
                setattr(est, attr, live_est[attr].reshape(shape).copy())
        x_hat_np = est.estimate(y.reshape(-1, 1), u_prev_np.reshape(-1, 1))
        if case.est_seed_mode == "self":
            for port, attr, shape in case.est_state_ports:
                live_est[attr] = np.asarray(getattr(est, attr)).reshape(shape).copy()

        for port, attr, shape in case.ctrl_state_ports:
            setattr(ctrl, attr, live_ctrl[attr].copy())
        if case.mode == "tracking":
            u_raw = ctrl.compute(x_hat_np.ravel(), x_ref)
        elif case.mode == "error":
            u_raw = ctrl.compute(x_hat_np.ravel() - x_ref)
        else:
            u_raw = ctrl.compute(x_hat_np.ravel() - x_ref, u_prev=u_prev_np)
        u_np = np.clip(u_raw, _CLOSED_LOOP_LIMITS[0], _CLOSED_LOOP_LIMITS[1])
        for port, attr, shape in case.ctrl_state_ports:
            live_ctrl[attr] = np.asarray(getattr(ctrl, attr)).reshape(shape).copy()
        if case.sat_threshold is not None:
            saw_saturation = saw_saturation or bool(np.any(np.abs(u_np) >= case.sat_threshold - 1e-12))

        ports = {"y": y, "x_ref": x_ref, "u_prev": u_prev_so}
        for port, _, shape in case.est_state_ports:
            ports[port] = so_est[port].reshape(shape)
        for port, _, shape in case.ctrl_state_ports:
            ports[port] = so_ctrl[port].reshape(shape)
        out, state = step_so(lib, pack_arrays(cg, ports), n_out, n_state)
        max_err = max(max_err, float(np.max(np.abs(out - u_np))))
        for port, _, shape in case.est_state_ports:
            a, b = sl[port]
            so_est[port] = state[a:b].reshape(shape).copy()
        for port, _, shape in case.ctrl_state_ports:
            a, b = sl[port]
            so_ctrl[port] = state[a:b].reshape(shape).copy()
        u_prev_so = out
        u_prev_np = u_np

    return max_err, saw_saturation


class TestClosedLoopOracles:
    """Composed .so vs live numpy closed loops: one driver, five declarative cases.

    The KF cases are the recurrent-covariance regression (the live P
    recursion must run each tick, not a frozen trace-time gain); the
    Luenberger case is the estimator-swap regression (x_hat threads with no
    P, solver-free graph); the PID case is the controller-recurrent-state
    regression (integral accumulates, D-gate opens after tick 0, anti-windup
    fires only on saturated channels); the MPC cases compare the baked
    EMOSQP solver (warm-started from the previous tick) against cold-started
    Python osqp — ADMM settles at slightly different points within eps, and
    that difference feeds back through the loop (measured max ~1.5e-6 over
    100 ticks → the 1e-4 gate). The DeltaU case additionally builds against
    the n_vars=45 bake via -Dsolver_dir (the comptime graph↔bake check
    rejects the shipped n_vars=30 bake at compile time).
    """

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_so_matches_live_components(self, request, case):
        lib, cg = request.getfixturevalue(case.fixture)
        max_err, saw_saturation = _run_closed_loop(lib, cg, case)
        if case.sat_threshold is not None:
            assert saw_saturation, f"{case.name}: oracle never saturated — anti-windup path untested"
        assert max_err < case.tol, f"{case.name}: .so diverged from live components over 100 ticks: max abs err = {max_err:.3e}"


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
            assert mn["rows"] == zn["rows"] and mn["cols"] == zn["cols"], f"node {mn['i']} shape mismatch"
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


# ─── ONNX policy oracle (imported graph -> .so) ─────────────────────────────

#: The committed toy policy (scripts/gen_toy_onnx.py): 3 -> 4 -> 2 tanh MLP with
#: the closed form action = [tanh(x0) + 0.5, tanh(x1) - 0.5].
TOY_ONNX = REPO_ROOT / "tests" / "fixtures" / "models" / "toy_mlp.onnx"


def _onnx_policy_graph(action_cfg: dict):
    """Import the toy ONNX policy with the given (baked) action-space config."""
    pytest.importorskip("onnx")
    from shinro.codegen.onnx_import import import_onnx_policy

    return import_onnx_policy(str(TOY_ONNX), obs_cfg={"state_keys": [0, 1, 2]}, action_cfg=action_cfg)


@pytest.fixture(scope="session")
def onnx_continuous_so(tmp_path_factory):
    """The continuous (no epsilon port) baked policy kernel."""
    d = tmp_path_factory.mktemp("zig-build-onnx-continuous")
    return _build_so(_onnx_policy_graph({"action_space": "continuous"}), d, graph_path=d / "graph_data.zig")


@pytest.fixture(scope="session")
def onnx_discrete_so(tmp_path_factory):
    """The deterministic discrete kernel (argmax + one_hot, no epsilon port)."""
    d = tmp_path_factory.mktemp("zig-build-onnx-discrete")
    return _build_so(_onnx_policy_graph({"action_space": "discrete"}), d, graph_path=d / "graph_data.zig")


@pytest.fixture(scope="session")
def onnx_discrete_eps_so(tmp_path_factory):
    """The sampling discrete kernel: it consumes host Gumbel noise."""
    d = tmp_path_factory.mktemp("zig-build-onnx-discrete-eps")
    cfg = {"action_space": "discrete", "deterministic": False}
    return _build_so(_onnx_policy_graph(cfg), d, graph_path=d / "graph_data.zig")


@pytest.fixture(scope="session")
def onnx_stochastic_eps_so(tmp_path_factory):
    """The sampling stochastic kernel; the toy's 2 outputs read as [mean; log_std]."""
    d = tmp_path_factory.mktemp("zig-build-onnx-stochastic-eps")
    cfg = {"action_space": "stochastic", "deterministic": False}
    return _build_so(_onnx_policy_graph(cfg), d, graph_path=d / "graph_data.zig")


class TestOnnxPolicyOracle:
    """An imported ONNX policy lowers to a .so that matches the interpreter.

    The graph comes from the committed toy fixture, so this is the only oracle
    whose subject is a *learned* policy rather than a hand-written control law:
    it proves the importer's output (baked encoder, transposed Gemms, composed
    activations, and the action post-processing) compiles bit-for-bit. Each
    action space gets its own kernel because the space — and whether an
    ``epsilon`` port exists — is baked at import time. Graphs lower to tmp
    paths, never the shared ``src/shinro/runtime/graph_data.zig``.
    """

    def test_continuous_matches_interpreter_and_closed_form(self, onnx_continuous_so):
        lib, cg = onnx_continuous_so
        assert cg.inputs == ["state"]  # deterministic: no noise port
        assert cg.state_outputs == []
        n_out, n_state = output_split(cg)
        assert (n_out, n_state) == (2, 0)

        rng = np.random.default_rng(7)
        for _ in range(50):
            state = rng.normal(0.0, 1.0, 3)
            out, _ = step_so(lib, pack_arrays(cg, {"state": state}), n_out, n_state)
            traced = interpret(cg.graph, {"state": state})["u"]
            closed = np.array([np.tanh(state[0]) + 0.5, np.tanh(state[1]) - 0.5])
            np.testing.assert_allclose(out, traced, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(out, closed, rtol=1e-6, atol=1e-7)

    def test_discrete_deterministic_one_hot(self, onnx_discrete_so):
        lib, cg = onnx_discrete_so
        assert cg.inputs == ["state"]
        n_out, n_state = output_split(cg)

        rng = np.random.default_rng(8)
        for _ in range(25):
            state = rng.normal(0.0, 1.0, 3)
            out, _ = step_so(lib, pack_arrays(cg, {"state": state}), n_out, n_state)
            traced = interpret(cg.graph, {"state": state})["u"]
            logits = np.array([np.tanh(state[0]) + 0.5, np.tanh(state[1]) - 0.5])
            want = np.zeros(2)
            want[int(np.argmax(logits))] = 1.0
            np.testing.assert_allclose(out, traced, rtol=0, atol=0)
            np.testing.assert_allclose(out, want, rtol=0, atol=0)

    def test_discrete_sampling_consumes_gumbel_noise(self, onnx_discrete_eps_so):
        lib, cg = onnx_discrete_eps_so
        assert cg.inputs == ["state", "epsilon"]
        assert input_shape(cg.graph, "epsilon") == (2,)
        n_out, n_state = output_split(cg)

        rng = np.random.default_rng(9)
        for _ in range(25):
            state = rng.normal(0.0, 1.0, 3)
            gumbel = -np.log(-np.log(rng.uniform(size=2)))  # the host's Gumbel noise
            arrays = {"state": state, "epsilon": gumbel}
            out, _ = step_so(lib, pack_arrays(cg, arrays), n_out, n_state)
            traced = interpret(cg.graph, arrays)["u"]
            logits = np.array([np.tanh(state[0]) + 0.5, np.tanh(state[1]) - 0.5])
            want = np.zeros(2)
            want[int(np.argmax(logits + gumbel))] = 1.0
            np.testing.assert_allclose(out, traced, rtol=0, atol=0)
            np.testing.assert_allclose(out, want, rtol=0, atol=0)

    def test_stochastic_sampling_matches_interpreter_and_formula(self, onnx_stochastic_eps_so):
        lib, cg = onnx_stochastic_eps_so
        assert cg.inputs == ["state", "epsilon"]
        assert input_shape(cg.graph, "epsilon") == (1,)  # the toy's 2 outputs -> n_u = 1
        n_out, n_state = output_split(cg)

        rng = np.random.default_rng(10)
        for _ in range(25):
            state = rng.normal(0.0, 1.0, 3)
            eps = rng.normal(size=1)
            arrays = {"state": state, "epsilon": eps}
            out, _ = step_so(lib, pack_arrays(cg, arrays), n_out, n_state)
            traced = interpret(cg.graph, arrays)["u"]
            mean = np.tanh(state[0]) + 0.5
            log_std = np.clip(np.tanh(state[1]) - 0.5, -10.0, 2.0)
            want = mean + np.exp(log_std) * eps
            np.testing.assert_allclose(out, traced, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(out, want, rtol=1e-12, atol=1e-12)
