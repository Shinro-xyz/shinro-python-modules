"""SMC demo: a nonlinear plant, defined as f(x)/g(x), deployed exactly as it lowers.

Sliding Mode Control is the one controller whose runtime inputs are *live plant
evaluations*. There is no estimator and no plant inside the graph: the host
evaluates ``f(x)`` and ``g(x)`` each tick, and the lowered kernel is pure
arithmetic on ``(x, f_x, g_x)``. This demo walks that contract end to end on a
nonlinear control-affine plant:

    x1_dot = x2
    x2_dot = -a*sin(x1) - b*x2 + c*u          (pendulum-like, SISO)

1. **Define the plant as two functions** — ``f(x)`` (drift, ``(n_x,)``) and
   ``g(x)`` (input matrix, ``(n_x, n_u)``). The controller never sees them.
2. **Close the loop eagerly** — the controller drives ``s = c^T x`` to zero.
3. **Model mismatch** — tell the controller the wrong ``a``; the switching
   gain ``k1`` is what absorbs the error (that is the point of SMC).
4. **Config-driven** — the shipped ``samples/controllers/smc.toml``.
5. **The controllability guard, both ways** — eager numpy raises
   ``RuntimeError``; the lowered graph cannot raise, so it emits ``u = 0`` and
   a ``healthy = 0`` flag the host acts on in the same tick.
6. **Trace it** — the same ``compute`` call becomes a graph; the graph
   interpreter is checked against live numpy (it is the ``.so``'s oracle).
7. **Deploy it** (``--build``) — lower to a temp graph, compile the Zig VM,
   ``dlopen`` the ``.so``, and run the *same* nonlinear closed loop against the
   compiled kernel, checking every tick against live numpy. This is "SMC
   deployed as is": host owns ``f``/``g``, kernel owns the arithmetic.

``--build`` needs ``zig`` on PATH (the first build takes ~30 s; later builds
reuse the cache and finish in ~1 s); without it every other section still runs.
No MuJoCo, no torch required.

Scope note: this is single-surface SMC (one row ``c``). An ``n_u > 1`` plant
needs no code change — ``g_x`` shaped ``(n_x, n_u)`` selects the minimum-norm
branch automatically — and a bank of instances covers multi-axis control. See
``TestSmcOracle`` in ``tests/test_zig_lowering.py`` for the ``n_u = 2`` oracle.

Usage:
  python -m demos.demo_smc            # eager + traced (default install)
  python -m demos.demo_smc --build    # + compile the .so and run it
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from shinro.codegen import interpret
from shinro.codegen.compose import ComposedGraph
from shinro.codegen.lower_zig import lower_zig
from shinro.codegen.oracle import load_so, output_split, pack_arrays, step_so
from shinro.codegen.trace_node import trace_node
from shinro.controllers.smc import SlidingModeController

BUILD_SO = "--build" in sys.argv

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / "src" / "shinro" / "runtime"

DT = 0.001
STEPS = 4000


# ─── 1. the plant, as two functions the host owns ──────────────────────────

A_GRAV = 1.0  # gravity gain in the true plant
B_DAMP = 0.5  # damping
C_CTRL = 1.5  # control authority


def f(x) -> np.ndarray:
    """The TRUE drift f(x): ``(n_x,) -> (n_x,)``. Evaluated host-side."""
    return np.array([x[1], -A_GRAV * np.sin(x[0]) - B_DAMP * x[1]])


def g(x) -> np.ndarray:
    """The TRUE input matrix g(x): ``(n_x,) -> (n_x, n_u)``.

    Constant here, but it may depend on ``x`` — the controller treats it as
    data either way. A 1-D ``(n_x,)`` column is also accepted.
    """
    return np.array([[0.0], [C_CTRL]])


def f_model(x, a_hat: float) -> np.ndarray:
    """The drift the CONTROLLER is TOLD about — ``a_hat`` is our lie."""
    return np.array([x[1], -a_hat * np.sin(x[0]) - B_DAMP * x[1]])


def build_smc(
    c: tuple[float, ...] = (1.0, 2.0),
    k1: float = 3.0,
    k2: float = 1.0,
    phi: float = 0.1,
    smoother: str = "tanh",
    alpha: float = 0.0,
) -> SlidingModeController:
    """The demo's SMC: ``c=[1, 2]`` (Hurwitz), tanh boundary layer.

    ``c`` defines the surface ``s = c^T x``; ``c=[1, 2]`` means the polynomial
    ``1 + 2p`` (root ``p = -0.5``), so on the surface the error decays as
    ``x1_dot = -0.5*x1``. Same gain shape as the shipped ``smc.toml``, with a
    stronger ``k1`` so the nonlinear demo converges fast.
    """
    return SlidingModeController(
        c=list(c), k1=k1, k2=k2, phi=phi, smoother=smoother, alpha=alpha
    )


def integrate(x: np.ndarray, u, dt: float) -> np.ndarray:
    """One Euler step of the TRUE plant (the demo's ground truth)."""
    return x + dt * (f(x) + g(x) @ np.asarray(u).ravel())


# ─── 2. eager closed loop: exact model ─────────────────────────────────────


def demo_eager_closed_loop() -> None:
    print("=== 2. Eager closed loop on the nonlinear plant (exact model) ===")
    ctrl = build_smc()
    c = np.array([1.0, 2.0])
    x = np.array([1.0, 0.0])  # 1 rad from equilibrium
    print(f"  {'step':>5} {'x1':>9} {'x2':>9} {'s=c^Tx':>9} {'u':>9}")
    for step in range(STEPS):
        u = ctrl.compute(x, f(x), g(x))  # <-- the whole contract, one line
        x = integrate(x, u, DT)
        if step % 400 == 0:
            print(f"  {step:>5} {x[0]:>9.5f} {x[1]:>9.5f} {(c @ x).item():>9.5f} {u[0].item():>9.5f}")
    s = (c @ x).item()
    print(f"  final x = {np.round(x, 5)}, |s| = {abs(s):.2e} (surface reached)\n")


# ─── 3. model mismatch: this is what k1 pays for ───────────────────────────


def demo_model_mismatch() -> None:
    print("=== 3. Model mismatch: the switching gain k1 absorbs the error ===")
    a_hat = 2.0  # controller believes gravity gain is 2.0; the plant uses 1.0
    # The mismatch enters s_dot as c^T(f_true - f_model), bounded here by
    # |2*(A_GRAV - a_hat)*sin(x1)| = 2.0, so k1 must exceed ~2.0.
    for k1 in (1.0, 3.0):
        ctrl = build_smc(k1=k1)
        x = np.array([1.0, 0.0])
        for _ in range(STEPS):
            u = ctrl.compute(x, f_model(x, a_hat), g(x))
            x = integrate(x, u, DT)
        s = (np.array([1.0, 2.0]) @ x).item()
        verdict = "converges" if abs(s) < 0.1 else "stalled on the mismatch"
        print(f"  k1={k1:>4.1f} (mismatch bound ~2.0): |s| = {abs(s):.4f}  -> {verdict}")
    print(f"  x = {np.round(x, 5)} at k1=3.0\n")


# ─── 4. the shipped config, through the factory ────────────────────────────


def demo_config_driven() -> None:
    print("=== 4. Config-driven: the shipped samples/controllers/smc.toml ===")
    from shinro.factories import ControllerFactory
    from shinro.utils.config_resolver import resolve_config_path

    ctrl = ControllerFactory(str(resolve_config_path("samples/controllers/smc.toml"))).create()
    u = ctrl.compute(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([[0.0], [1.0]]))
    print(f"  c = {ctrl.c.tolist()}, k1 = {ctrl.k1}, smoother = {ctrl._smoother_name}")
    print(f"  compute(x=[1, 0]) -> u = {np.asarray(u).ravel()[0].item():+.5f}")
    print("  (deployment re-lowers one kernel per config; a config change is a re-lower)\n")


# ─── 5. the controllability guard: eager raises, the graph flags ───────────


def build_smc_graph(n_u: int = 1) -> ComposedGraph:
    """Trace standalone SMC: ``(x, f_x, g_x)`` in, ``(out, healthy)`` out.

    There is no estimator and no plant to compose with — ``compose()``
    deliberately has no role for ``f_x``/``g_x`` (see ``compose.py``), so the
    graph is traced standalone and lowered directly, with the dynamics terms as
    free C-ABI ports. SMC is memoryless: ``state_outputs`` is empty.
    """
    smc = build_smc()
    ng = trace_node(smc, input_shapes={"x": (2,), "f_x": (2,), "g_x": (2, n_u)})
    return ComposedGraph(
        graph=ng.graph,
        inputs=["x", "f_x", "g_x"],
        outputs=["out", "healthy"],
        state_inputs=[],
        state_outputs=[],
    )


def demo_guard(graph: ComposedGraph) -> None:
    print("=== 5. Loss of controllability: eager raises, the graph flags ===")
    # c^T g = 1*1 + 2*(-0.5) = 0 exactly
    arrays = {"x": np.array([1.0, 0.5]), "f_x": np.array([0.3, -0.2]), "g_x": np.array([[1.0], [-0.5]])}
    print(f"  c^T g = {(np.array([1.0, 2.0]) @ arrays['g_x'].ravel()).item():.1f} (below controllability_eps)")

    try:
        build_smc().compute(arrays["x"], arrays["f_x"], arrays["g_x"])
        raise AssertionError("expected the eager backend to raise")
    except RuntimeError as exc:
        print(f"  eager numpy : RuntimeError({exc})")

    with np.errstate(divide="ignore", invalid="ignore"):
        traced = interpret(graph.graph, arrays)
    print(f"  traced graph: u = {np.asarray(traced['out']).ravel()[0].item():.1f}, "
          f"healthy = {np.asarray(traced['healthy']).ravel()[0].item():.1f}")
    print("  A compiled kernel cannot raise (a Zig panic across the C ABI aborts the host),")
    print("  so the guard becomes data: zero command + a flag for the host's fault policy.")
    print("  Zero-command is the kernel's floor, NOT a safety guarantee for an unstable plant.\n")


# ─── 6. trace it: the interpreter is the .so's oracle ──────────────────────


def demo_trace_and_check(graph: ComposedGraph) -> float:
    print("=== 6. Trace it: interpreter vs live numpy (the .so's oracle) ===")
    ctrl = build_smc()
    rng = np.random.default_rng(11)
    max_err = 0.0
    for _ in range(25):
        x = rng.normal(0.0, 0.5, (2,))
        arrays = {"x": x, "f_x": f(x), "g_x": g(x)}  # host-evaluated, as in deployment
        traced = interpret(graph.graph, arrays)
        want = np.asarray(ctrl.compute(arrays["x"], arrays["f_x"], arrays["g_x"])).ravel()
        max_err = max(max_err, np.abs(np.asarray(traced["out"]).ravel() - want).max().item())
        assert np.asarray(traced["healthy"]).ravel()[0].item() == 1.0

    n_out, n_state = output_split(graph)
    print(f"  graph: {len(graph.graph.nodes)} nodes, inputs {graph.inputs}, outputs {graph.outputs}")
    print(f"  no recurrent state (memoryless): state_outputs = {graph.state_outputs}")
    print(f"  max |interpreter - live numpy| over 25 samples = {max_err:.2e}")
    print(f"  C-ABI buffers: {n_out} output f64 ({n_out * 8} B), {n_state} state f64\n")
    return max_err


# ─── 7. deploy it: same graph, compiled ────────────────────────────────────


def demo_deploy(graph: ComposedGraph) -> None:
    print("=== 7. Deploy it: lower, compile the Zig VM, dlopen, run it ===")
    if shutil.which("zig") is None:
        print("  zig not on PATH — skipping. The trace above is the same graph the .so runs.")
        print("  Install zig and re-run `python -m demos.demo_smc --build` to see it compiled.\n")
        return

    # Lower to a TEMP graph path: never clobber the shipped
    # src/shinro/runtime/graph_data.zig (the KF+LQR base graph).
    tmp = Path(tempfile.mkdtemp(prefix="shinro-smc-demo-"))
    graph_path = tmp / "graph_data.zig"
    lower_zig(graph, str(graph_path))
    print(f"  lowered to {graph_path} (temp; shipped graph untouched)")

    build = subprocess.run(
        [
            "zig",
            "build",
            "--build-file",
            str(RUNTIME / "build.zig"),
            "--prefix",
            str(tmp),
            f"-Dgraph={graph_path}",
            "-Doptimize=ReleaseFast",  # the production mode; Debug is ~11 MB unstripped
        ],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        print(f"  zig build failed:\n{build.stderr.strip()[:400]}")
        return

    lib = load_so(tmp)
    so_bytes = (tmp / "lib" / "libbase.so").stat().st_size
    n_out, n_state = output_split(graph)
    print(f"  compiled libbase.so (ReleaseFast, stripped): {so_bytes / 1024:.0f} KiB, "
          f"{n_out} outputs, {n_state} state")

    # The deployment loop: host evaluates f/g, kernel does the arithmetic.
    ctrl = build_smc()
    x = np.array([1.0, 0.0])
    max_err = 0.0
    for step in range(STEPS):
        packed = pack_arrays(graph, {"x": x, "f_x": f(x), "g_x": g(x)})
        out, _state = step_so(lib, packed, n_out, n_state)
        u_kernel, healthy = out[0], out[1]
        assert healthy == 1.0, "unexpected controllability fault in the demo trajectory"

        u_ref = np.asarray(ctrl.compute(x, f(x), g(x))).ravel()[0].item()
        max_err = max(max_err, abs(u_kernel - u_ref).item())
        x = integrate(x, u_kernel, DT)
        if step % 2000 == 0:
            print(f"    step {step:>5}: x = {np.round(x, 5)}, u = {u_kernel:+.5f}, healthy = {healthy:.0f}")

    print(f"  final x = {np.round(x, 5)}")
    print(f"  max |compiled .so - live numpy| per tick = {max_err:.2e} (bit-parity tier is 1e-12)")
    print(f"  artifacts: {tmp}\n")


def main() -> None:
    print(f"SMC demo (plant: x2_dot = -{A_GRAV}*sin(x1) - {B_DAMP}*x2 + {C_CTRL}*u, dt={DT})\n")
    demo_eager_closed_loop()
    demo_model_mismatch()
    demo_config_driven()
    graph = build_smc_graph()
    demo_guard(graph)
    demo_trace_and_check(graph)
    if BUILD_SO:
        demo_deploy(graph)
    else:
        print("=== 7. Deploy it ===")
        print("  re-run with --build to lower this graph, compile the Zig VM, and run it\n")
    print("Done.")


if __name__ == "__main__":
    main()
