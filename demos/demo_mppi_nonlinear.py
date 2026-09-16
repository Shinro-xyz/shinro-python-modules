"""MPPI on a nonlinear plant: the rollout *is* the plant's dynamics, and it lowers.

MPPI's rollout is the one place a plant model enters the graph. For a linear
plant that is a batched matmul; for a nonlinear plant it is the plant's own
``dynamics`` — one batch-capable method that serves the eager per-sample path,
the finite-difference linearization, and the traced graph. This demo walks that
from development to a compiled kernel, on an ``InvertedPendulum``:

1. **One method, two ranks** — ``dynamics(state)`` takes a single ``(2,)`` state
   *or* a batch ``(N, 2)`` and returns the derivative with the same rank. It is
   the only implementation of the physics; there is no second, batched copy to
   drift from it.
2. **Close the loop eagerly** — MPPI (via ``attach_plant``) swings the pendulum
   up from a 23-degree tilt and holds it.
3. **Why the nonlinear rollout** — the *predicted* horizon is compared with the
   plant's real motion, for the plant's nonlinear model and for the linearized
   ``(A, B)``. Far from upright the linearization is wrong; the nonlinear model
   is not.
4. **Trace it** — the same ``compute`` call becomes a graph: one ``sin`` node of
   shape ``(N, 1)`` per horizon step, so the node count is independent of the
   sample count. The graph interpreter is checked against live numpy (it is the
   ``.so``'s oracle).
5. **The C-ABI contract** — ``epsilon`` (the perturbations) is a free input
   port: sampling stays on the host, which is what makes parity checkable.
6. **Deploy it** (``--build``) — lower to a temp graph, compile the Zig VM with
   ReleaseFast, ``dlopen`` the ``.so``, and run the *same* closed loop against
   the compiled kernel on the *same* noise, tick by tick.

``--build`` needs ``zig`` on PATH; without it every other section still runs.
No MuJoCo, no torch required.

Usage:
  python -m demos.demo_mppi_nonlinear            # eager + traced (default install)
  python -m demos.demo_mppi_nonlinear --build    # + compile the .so and run it
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
from shinro.controllers.mppi import MPPIController
from shinro.plants.inverted_pendulum import InvertedPendulum
from shinro.utils.array_backend import NumpyBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / "src" / "shinro" / "runtime"
BUILD_SO = "--build" in sys.argv

DT = 0.01
N_SAMPLES = 64
HORIZON = 10
STEPS = 600
TARGET = np.array([0.0, 0.0])  # upright
X0 = np.array([0.4, 0.0])  # ~23 degrees off upright
SIGMA = 0.5
U_LIMIT = 1.0


# ─── the plant + the controller, wired the production way ──────────────────


def build_plant() -> InvertedPendulum:
    return InvertedPendulum(mass=0.1, length=0.5, damping=0.0, gravity=9.81, dt=DT, backend=NumpyBackend())


def build_controller(plant: InvertedPendulum, N: int = N_SAMPLES, K: int = HORIZON) -> MPPIController:
    """The live (numpy) MPPI — the reference the traced graph and .so are checked against.

    ``attach_plant`` is the production wiring (``ScenarioFactory`` does the same):
    it builds the batched adapter and points MPPI's rollout at the plant's own
    ``dynamics``.
    """
    ctrl = MPPIController(
        num_samples=N,
        temperature=0.5,
        dt=DT,
        horizon=K,
        noise_sigma=[SIGMA],
        u_min=[-U_LIMIT],
        u_max=[U_LIMIT],
        seed=42,
        backend=NumpyBackend(),
    )
    ctrl.attach_plant(plant, Q=np.array([20.0, 1.0]), R=np.array([0.1]))
    return ctrl


def build_graph(N: int = N_SAMPLES, K: int = HORIZON) -> ComposedGraph:
    """Trace the standalone MPPI graph: perturbations in, action + costs out.

    MPPI's Gaussian sampling stays on the host, so ``epsilon`` becomes a free
    C-ABI input port of shape ``(N, K*D_u)`` and the traced call never touches
    the RNG. The nominal plan ``u`` is the recurrent state.
    """
    plant = build_plant()
    ctrl = build_controller(plant, N=N, K=K)
    ng = trace_node(
        ctrl,
        input_shapes={"current_state": (2,), "target_state": (2,), "epsilon": (N, K)},
        state_shapes={"u": (K, 1)},
    )
    return ComposedGraph(
        graph=ng.graph,
        inputs=["current_state", "target_state", "epsilon", "state_u"],
        outputs=["out", "costs"],
        state_inputs=["state_u"],
        state_outputs=["state_u"],
    )


# ─── 1. one method, two ranks ──────────────────────────────────────────────


def demo_dynamics() -> None:
    print("=== 1. The plant's dynamics: one batch-capable method ===")
    plant = build_plant()
    single = plant.dynamics(np.array([0.3, 0.0]), np.array([0.0]))
    batch = plant.dynamics(np.array([[0.3, 0.0], [0.6, 0.1], [-0.2, 0.4]]), np.array([[0.0], [0.1], [-0.3]]))
    print("  theta_ddot = (g/l)*sin(theta) + tau/(m*l^2) - (b/(m*l^2))*theta_dot")
    print(f"  dynamics((2,), (1,))       -> {np.round(single, 6)}   shape {single.shape}")
    print(f"  dynamics((3,2), (3,1))     -> shape {batch.shape} (row i is sample i)")
    print(f"  batch row 0 == single call : {np.allclose(batch[0], single)}")
    print("  Same function serves step(), the finite-difference linearization, and the tracer.\n")


# ─── 2. close the loop eagerly ─────────────────────────────────────────────


def run_eager(steps: int = STEPS, seed: int = 7):
    """Drive the pendulum upright with live numpy MPPI; return (states, us, epsilons)."""
    plant = build_plant()
    ctrl = build_controller(plant)
    ctrl.reset()
    plant.state = NumpyBackend().array(X0)
    rng = np.random.default_rng(seed)
    states, us, epsilons = [X0.copy()], [], []
    for _ in range(steps):
        eps = rng.normal(0.0, SIGMA, (N_SAMPLES, HORIZON))
        u = np.asarray(ctrl.compute(plant.get_state(), TARGET, eps)).ravel()
        epsilons.append(eps)
        us.append(u)
        plant.step(u)
        states.append(np.asarray(plant.get_state()).ravel().copy())
    return np.array(states), np.array(us), epsilons


def demo_closed_loop() -> None:
    print("=== 2. Eager closed loop: MPPI swings the pendulum upright ===")
    states, _, _ = run_eager()
    for i in (0, 50, 150, 300, STEPS):
        print(f"    t = {i * DT:>5.2f} s   theta = {states[i][0]:+.5f} rad   theta_dot = {states[i][1]:+.5f}")
    print(f"  |theta| at the end = {abs(states[-1][0]):.2e} rad (target 0)\n")


# ─── 3. why the nonlinear rollout ──────────────────────────────────────────


def _rollout(fn, x0, us, dt=DT):
    """Explicit Euler over a control sequence — the update MPPI's rollout uses."""
    x = np.array(x0, dtype=float)
    for u in us:
        x = x + dt * fn(x, np.array([u]))
    return x


def demo_model_error() -> None:
    print("=== 3. Why the nonlinear rollout: the linearization is wrong at a tilt ===")
    plant = build_plant()
    A_d, B_d = (np.asarray(m) for m in plant.get_model())  # Euler-discretized about upright
    A_c, B_c = (A_d - np.eye(2)) / DT, B_d / DT  # back to continuous time
    theta, tau = 0.8, 0.0  # ~46 degrees — far from the linearization point
    x0 = np.array([theta, 0.0])

    f_true = np.asarray(plant.dynamics(x0, np.array([tau]))).ravel()
    f_lin = A_c @ x0 + (B_c @ np.array([tau])).ravel()
    print(f"  at theta = {theta:.2f} rad, tau = 0:")
    print(f"    true theta_ddot       = {f_true[1]:+.4f}   ((g/l)*sin(theta))")
    print(f"    linearized theta_ddot = {f_lin[1]:+.4f}   ((g/l)*theta)")
    print(f"    -> the linearization is off by {abs(f_lin[1] - f_true[1]) / abs(f_true[1]) * 100:.1f}% at this angle")

    steps = 50  # 0.5 s — long enough for the error to compound
    us = [tau] * steps
    truth = _rollout(lambda x, u: np.asarray(plant.dynamics(x, u)).ravel(), x0, us)
    linear = _rollout(lambda x, u: A_c @ x + (B_c @ u).ravel(), x0, us)
    print(f"  over {steps} steps ({steps * DT:.2f} s) of the same explicit-Euler rollout:")
    print(f"    nonlinear model  theta = {truth[0]:+.4f}")
    print(f"    linearized model theta = {linear[0]:+.4f}   err {abs(linear[0] - truth[0]):.2e}")
    print("  The plant's own dynamics is exact by construction; the (A, B) model is not.")
    print("  (The plant's step() integrates semi-implicitly; the rollout uses explicit Euler —")
    print("   a small, consistent integration difference, not a model error.)\n")


# ─── 4. trace it ───────────────────────────────────────────────────────────


def demo_trace() -> ComposedGraph:
    print("=== 4. Trace it: the rollout becomes graph nodes ===")
    cg = build_graph()
    nodes = cg.graph.nodes
    sin_ids = [i for i, n in enumerate(nodes) if n.op == "sin"]
    print(f"  {len(nodes)} nodes | inputs={cg.inputs}")
    print(f"  outputs={cg.outputs} | state_outputs={cg.state_outputs}")
    print(f"  sin nodes at {sin_ids} — one per horizon step, each covering all {N_SAMPLES} samples:")
    for i in range(sin_ids[0] - 1, sin_ids[0] + 6):
        n = nodes[i]
        print(f"    [{i:3d}] {n.op:<10} shape={str(n.shape):<8} inputs={list(n.inputs)}")

    # The oracle the .so is checked against: interpreter vs live numpy.
    ref = build_controller(build_plant())
    rng = np.random.default_rng(11)
    max_err = 0.0
    for _ in range(10):
        ref.reset()
        x0 = rng.normal(0.0, 0.3, 2)
        eps = rng.normal(0.0, SIGMA, (N_SAMPLES, HORIZON))
        feeds = {"current_state": x0, "target_state": TARGET, "epsilon": eps, "state_u": np.zeros((HORIZON, 1))}
        out = interpret(cg.graph, feeds)
        u_ref = np.asarray(ref.compute(x0, TARGET, eps)).ravel()
        max_err = max(max_err, np.max(np.abs(out["out"] - u_ref)).item())
    print(f"  interpret() vs live numpy over 10 seeded draws: max |du| = {max_err:.2e} (tier 1e-11)")

    # The structural claim: the batch lives in the shapes, not in the node count.
    small, large = build_graph(N=16, K=HORIZON), build_graph(N=64, K=HORIZON)
    print(f"  node count: N=16 -> {len(small.graph.nodes)}, N=64 -> {len(large.graph.nodes)} (identical)")
    print("  only the tensor shapes (and the stack buffer) grow with N.\n")
    return cg


# ─── 5. the C-ABI contract ─────────────────────────────────────────────────


def demo_abi(cg: ComposedGraph) -> None:
    print("=== 5. What the host packs (C-ABI) ===")
    shapes = (cg.inputs, cg.outputs, cg.state_outputs)
    port_nodes = {n.attrs["name"]: n.shape for n in cg.graph.nodes if n.op in ("input", "output")}
    for label, names in zip(("in ", "out", "st "), shapes):
        for name in names:
            if name in port_nodes:
                print(f"  [{label}] {name:<14} {port_nodes[name]}")
    print(f"  epsilon = (N, K*D_u) = ({N_SAMPLES}, {HORIZON * 1}) — drawn by the host every tick,")
    print("  so the kernel contains no RNG and the same noise can be replayed for parity.\n")


# ─── 6. deploy it: the same graph, compiled ────────────────────────────────


def demo_deploy(cg: ComposedGraph) -> None:
    print("=== 6. Deploy it: lower, compile the Zig VM, dlopen, run the same loop ===")
    if shutil.which("zig") is None:
        print("  zig not on PATH — skipping. The trace above is the same graph the .so runs.")
        print("  Install zig and re-run `python -m demos.demo_mppi_nonlinear --build`.\n")
        return

    # Lower to a TEMP graph path: never clobber the shipped
    # src/shinro/runtime/graph_data.zig (the KF+LQR base graph).
    tmp = Path(tempfile.mkdtemp(prefix="shinro-mppi-nonlinear-demo-"))
    graph_path = tmp / "graph_data.zig"
    lower_zig(cg, str(graph_path))
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
            "-Doptimize=ReleaseFast",  # production mode; Debug is unstripped and huge
        ],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        print(f"  zig build failed:\n{build.stderr.strip()[:400]}")
        return

    lib = load_so(tmp)
    so_bytes = (tmp / "lib" / "libbase.so").stat().st_size
    n_out, n_state = output_split(cg)
    print(f"  compiled libbase.so (ReleaseFast): {so_bytes / 1024:.0f} KiB, "
          f"{n_out} output elements, {n_state} state elements")

    # The deployment loop: the host draws epsilon, the kernel does the rollout.
    # The SAME epsilon sequence drives the eager reference, tick for tick.
    eager_states, eager_us, epsilons = run_eager()
    plant = build_plant()
    plant.state = NumpyBackend().array(X0)
    plan = np.zeros((HORIZON, 1))
    max_du = 0.0
    max_dx = 0.0
    for step, eps in enumerate(epsilons):
        feeds = {
            "current_state": np.asarray(plant.get_state()).ravel(),
            "target_state": TARGET,
            "epsilon": eps,
            "state_u": plan,
        }
        out, state = step_so(lib, pack_arrays(cg, feeds), n_out, n_state)
        u_kernel = out[:1]
        plan = state[: HORIZON * 1].reshape(HORIZON, 1)
        max_du = max(max_du, np.max(np.abs(u_kernel - eager_us[step])).item())
        plant.step(u_kernel)
        x_kernel = np.asarray(plant.get_state()).ravel()
        max_dx = max(max_dx, np.max(np.abs(x_kernel - eager_states[step + 1])).item())

    print("  closed loop, kernel vs eager numpy on identical noise:")
    print(f"    max |u_kernel - u_eager| per tick = {max_du:.2e} (bit-parity tier is 1e-12)")
    print(f"    max |x_kernel - x_eager| per tick = {max_dx:.2e}")
    print(f"    final state = {np.round(np.asarray(plant.get_state()).ravel(), 6)}")
    print(f"  artifacts: {tmp}\n")


def main() -> None:
    print(f"MPPI on a nonlinear plant (InvertedPendulum, N={N_SAMPLES}, K={HORIZON}, dt={DT})\n")
    demo_dynamics()
    demo_closed_loop()
    demo_model_error()
    cg = demo_trace()
    demo_abi(cg)
    if BUILD_SO:
        demo_deploy(cg)
    else:
        print("=== 6. Deploy it ===")
        print("  re-run with --build to lower this graph, compile the Zig VM, and run the .so\n")
    print("Done.")


if __name__ == "__main__":
    main()
