#!/usr/bin/env python3
"""Demo: trace the base_tracking closed-loop step and verify it matches numpy.

This script exercises the XLA-style tracing layer in ``shinro.codegen`` on
the ``base_tracking`` scenario (KalmanFilter + LQR on the holonomic base).

It does four things, printing the results at each stage:

1. **Trace a single component** (KalmanFilter) and show the captured graph
   node-by-node, then interpret it and verify the output matches the live
   ``NumpyBackend`` computation. The KF's two array attrs (``x_hat`` and the
   covariance ``P``) are declared as recurrent state, so they become graph
   input/output ports rather than being frozen at trace-time values.

2. **Compose** the estimator and controller into one closed-loop step graph
   (with the scenario's input_limits clip), show the combined graph's
   structure, and verify the composed step matches the live numpy loop over
   several ticks — threading the graph's own ``state_*`` outputs back in, so
   the KF covariance recursion runs live instead of being frozen.

3. **Swap the estimator** (KalmanFilter → LuenbergerObserver) and re-compose,
   showing the controller's captured graph is reused without re-tracing —
   the modularity proof.

4. **A different plant** (CartPole, 4-state / 1-input) built from scratch,
   showing the same tracer works with no tracer-side changes.

Run from the repo root:

    python -m demos.demo_codegen
"""

from __future__ import annotations

import os
import sys

# Make the package importable without an installed `shinro` (e.g. when run
# from a different Python env like a conda env that doesn't have the package
# installed). Adds the repo's src/ to sys.path so `import shinro` resolves
# to the source checkout.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
from scipy.linalg import solve_discrete_are  # noqa: E402

from shinro.codegen import interpret, trace_node  # noqa: E402
from shinro.codegen.compose import compose  # noqa: E402
from shinro.controllers.lqr import LQR  # noqa: E402
from shinro.estimators.kalman_filter import KalmanFilter  # noqa: E402
from shinro.factories.controller_factory import ControllerFactory  # noqa: E402
from shinro.factories.estimator_factory import EstimatorFactory  # noqa: E402
from shinro.plants.cartpole import CartPole  # noqa: E402
from shinro.utils.array_backend import NumpyBackend  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def show_graph(graph, max_nodes: int = 25) -> None:
    """Print a graph's nodes in a compact table."""
    print(f"  {len(graph.nodes)} nodes:")
    print(f"  {'#':>3}  {'op':<10} {'inputs':<14} {'shape':<12}")
    print(f"  {'-' * 3}  {'-' * 10} {'-' * 14} {'-' * 12}")
    for i, node in enumerate(graph.nodes[:max_nodes]):
        ins = str(node.inputs) if node.inputs else "-"
        shape = str(node.shape)
        print(f"  {i:>3}  {node.op:<10} {ins:<14} {shape:<12}")
    if len(graph.nodes) > max_nodes:
        print(f"  ... ({len(graph.nodes) - max_nodes} more)")


def _p0(n: int) -> np.ndarray:
    """The KF's initial covariance — the same 0.1*I the host seeds at tick 0."""
    return np.eye(n) * 0.1


def main() -> None:
    rng = np.random.default_rng(42)

    # ─── Load components from the base_tracking configs ─────────────────
    banner("Loading components from base_tracking configs")
    kf = EstimatorFactory("samples/estimators/kalman_base.toml").create(backend=NumpyBackend())
    luen = EstimatorFactory("samples/estimators/luenberger_base.toml").create(backend=NumpyBackend())
    lqr = ControllerFactory("samples/controllers/lqr_base.toml").create(backend=NumpyBackend())
    print(f"  KalmanFilter: A{kf.A.shape} B{kf.B.shape} C{kf.C.shape}")
    print(f"  Luenberger:   A{luen.A.shape} B{luen.B.shape} L{luen.L.shape}")
    print(f"  LQR:          K{lqr.K.shape}  (gain from DARE, baked at trace time)")

    # ─── Stage 1: trace KalmanFilter alone ──────────────────────────────
    banner("Stage 1: Trace KalmanFilter.estimate() alone")
    # The estimator owns two recurrent arrays: the state estimate x_hat and
    # the covariance P. Both are declared via state_shapes so the tracer
    # pre-injects them as placeholders — they become graph inputs/outputs,
    # and the covariance recursion stays LIVE instead of being baked into a
    # trace-time constant (which compose would reject as a frozen recursion).
    kf.P = _p0(3)
    kf.x_hat = np.zeros(3)
    kf_graph = trace_node(
        kf,
        input_shapes={"measurement": (3,), "control_input": (3,)},
        state_shapes={"x_hat": (3,), "P": (3, 3)},
    )
    print("  Captured graph:")
    show_graph(kf_graph.graph)
    print(f"  Detected state attrs: {kf_graph.state_attrs}")
    ops_used = sorted({n.op for n in kf_graph.graph.nodes})
    print(f"  Ops used: {ops_used}")

    # Verify: interpret the graph on random inputs, compare to live numpy.
    y = rng.normal(0, 0.1, (3,))
    u = rng.normal(0, 0.1, (3,))
    x_hat_init = rng.normal(0, 0.1, (3,))
    kf.P = _p0(3)
    kf.x_hat = x_hat_init.copy()
    expected = kf.estimate(y, u)
    traced = interpret(
        kf_graph.graph,
        {"measurement": y, "control_input": u,
         "state_x_hat": x_hat_init, "state_P": _p0(3)},
    )
    got = traced.get("out")
    if got is None:
        got = traced.get("state_x_hat")
    assert got is not None, f"no output in traced graph: {list(traced)}"
    err = float(np.max(np.abs(got - expected)))
    print("\n  Interpret vs NumpyBackend:")
    print(f"    numpy x_hat:   {expected.ravel()}")
    print(f"    traced x_hat:  {got.ravel()}")
    print(f"    max abs err:   {err:.2e}  {'PASS' if err < 1e-12 else 'FAIL'}")

    # ─── Stage 2: compose KF + LQR into one closed-loop step ────────────
    banner("Stage 2: Compose KalmanFilter + LQR into one shinro_step")
    ctrl_graph = trace_node(
        lqr,
        input_shapes={"current_state": (3,), "target_state": (3,)},
    )
    print(f"  LQR graph: {len(ctrl_graph.graph.nodes)} nodes, "
          f"stateless={ctrl_graph.state_attrs == []}")

    # The base_tracking scenario's input_limits.
    limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))
    composed = compose(
        kf_graph, ctrl_graph,
        plant_dims={"n_x": 3, "n_u": 3},
        input_limits=limits,
    )
    print("\n  Composed graph (one closed-loop step):")
    show_graph(composed.graph)
    print(f"  Inputs:  {composed.inputs}")
    print(f"  Outputs: {composed.outputs}")
    print(f"  State (recurrent): {composed.state_inputs}")

    # Verify the composed step matches the live numpy loop. Because P is a
    # recurrent input port (state_P), we thread the graph's own state outputs
    # back in each tick: the KF covariance must track numpy's live Riccati
    # recursion rather than stay frozen at its trace-time value.
    print("\n  Verifying composed step vs live numpy loop (10 trials x 4 ticks):")
    max_err = 0.0
    for trial in range(10):
        y_seq = rng.normal(0, 0.1, (4, 3))
        x_ref = rng.normal(0, 0.1, (3,))
        x_hat_init = rng.normal(0, 0.1, (3,))

        # Ground truth: live numpy loop, same start + inputs, full recurrence.
        kf.P = _p0(3)
        kf.x_hat = x_hat_init.copy()
        u_prev = np.zeros(3)

        # Traced: seed the recurrent ports, then thread them tick by tick.
        state = {"state_x_hat": x_hat_init.copy(), "state_P": _p0(3)}
        for t in range(4):
            x_hat_np = kf.estimate(y_seq[t], u_prev)
            u_np = np.clip(lqr.compute(x_hat_np, x_ref), limits[0], limits[1])

            traced = interpret(
                composed.graph,
                {"y": y_seq[t], "x_ref": x_ref, "u_prev": u_prev,
                 "state_x_hat": state["state_x_hat"], "state_P": state["state_P"]},
            )
            max_err = max(max_err, float(np.max(np.abs(traced["u"] - u_np))))
            state["state_x_hat"] = traced["state_x_hat"]
            state["state_P"] = traced["state_P"]
            u_prev = u_np
    print(f"    max abs err over 10 trials: {max_err:.2e}  "
          f"{'PASS' if max_err < 1e-12 else 'FAIL'}")

    # ─── Stage 3: swap estimator, reuse the LQR graph ──────────────────
    banner("Stage 3: Swap estimator (KF → Luenberger), reuse LQR graph")
    luen.x_hat = np.zeros(3)
    luen_graph = trace_node(
        luen,
        input_shapes={"measurement": (3,), "control_input": (3,)},
        state_shapes={"x_hat": (3,)},
    )
    print(f"  Luenberger graph: {len(luen_graph.graph.nodes)} nodes")
    print(f"  LQR graph: REUSED ({len(ctrl_graph.graph.nodes)} nodes, not re-traced)")

    composed_swap = compose(
        luen_graph, ctrl_graph,  # ← ctrl_graph reused from Stage 2
        plant_dims={"n_x": 3, "n_u": 3},
        input_limits=limits,
    )
    print(f"  Swapped composed graph: {len(composed_swap.graph.nodes)} nodes")

    # Verify the swapped composition matches the live numpy loop.
    print("\n  Verifying Luenberger+LQR vs live numpy loop (10 trials x 4 ticks):")
    max_err = 0.0
    for trial in range(10):
        y_seq = rng.normal(0, 0.1, (4, 3))
        x_ref = rng.normal(0, 0.1, (3,))
        x_hat_init = rng.normal(0, 0.1, (3,))

        luen.x_hat = x_hat_init.copy()
        u_prev = np.zeros(3)

        state = {"state_x_hat": x_hat_init.copy()}
        for t in range(4):
            x_hat_np = luen.estimate(y_seq[t], u_prev)
            u_np = np.clip(lqr.compute(x_hat_np, x_ref), limits[0], limits[1])

            traced = interpret(
                composed_swap.graph,
                {"y": y_seq[t], "x_ref": x_ref, "u_prev": u_prev,
                 "state_x_hat": state["state_x_hat"]},
            )
            max_err = max(max_err, float(np.max(np.abs(traced["u"] - u_np))))
            state["state_x_hat"] = traced["state_x_hat"]
            u_prev = u_np
    print(f"    max abs err over 10 trials: {max_err:.2e}  "
          f"{'PASS' if max_err < 1e-12 else 'FAIL'}")

    # ─── Stage 4: the cartpole system (4-state, 1-input) ────────────────
    banner("Stage 4: CartPole system (4-state, 1-input) — define, trace, compose")

    # There are no cartpole-tuned controller/estimator configs in the repo,
    # so we build the LQR gain and Kalman filter directly from the plant's
    # linearized model — exactly what LQR.from_config / KalmanFilter.from_config
    # would do if a config existed. This shows the tracer works for a
    # different plant (4D vs 3D) with no tracer-side changes.
    print("  Building CartPole and linearizing around the upright equilibrium...")
    cp = CartPole(dt=0.01, backend=NumpyBackend())
    A_ct, B_ct = cp.get_model()  # continuous-time (4,4), (4,1)
    dt_cp = cp.dt
    # Discretize: x_{k+1} = A_d x_k + B_d u_k  (forward Euler — matches the
    # plant's semi-implicit step's linearization).
    A_d = np.eye(4) + A_ct * dt_cp
    B_d = B_ct * dt_cp
    print(f"    A_d = {A_d.shape}  B_d = {B_d.shape}  dt = {dt_cp}")

    # LQR gain via DARE (same as LQR.gain_calculation()).
    Q_cp = np.diag([10.0, 1.0, 100.0, 1.0])  # penalize x and θ heavily
    R_cp = np.diag([0.1])
    P_dare = solve_discrete_are(A_d, B_d, Q_cp, R_cp)
    K_cp = np.linalg.inv(R_cp + B_d.T @ P_dare @ B_d) @ (B_d.T @ P_dare @ A_d)
    print(f"    K_cp = {K_cp.ravel()}  (1x4 gain, baked at trace time)")

    # Kalman filter for the cartpole (4-state, 4-measurement: full state).
    Q_kf = np.diag([0.01, 0.01, 0.01, 0.01])  # process noise
    R_kf = np.diag([0.1, 0.1, 0.1, 0.1])      # measurement noise
    C_kf = np.eye(4)
    kf_cp = KalmanFilter(
        A=A_d, B=B_d, Q=Q_kf, R=R_kf, C=C_kf, backend=NumpyBackend(),
    )
    # LQR controller with the cartpole gains.
    lqr_cp = LQR(
        state_cost_matrix=Q_cp, control_cost_matrix=R_cp,
        dynamics_state_matrix=A_d, dynamics_control_matrix=B_d,
        backend=NumpyBackend(),
    )
    # Override the gain with our cartpole-tuned K (from_config would solve
    # the same DARE; we set it explicitly here for clarity).
    lqr_cp.K = K_cp
    print(f"    KF: A{A_d.shape} B{B_d.shape} C{C_kf.shape}")
    print(f"    LQR: K{K_cp.shape} (cartpole-tuned, 1-input)")

    # Trace both components (4-state, 1-input — different shapes from Stage 1-3).
    print("\n  Tracing KF (cartpole) and LQR (cartpole)...")
    kf_cp.P = _p0(4)
    kf_cp.x_hat = np.zeros(4)
    kf_cp_graph = trace_node(
        kf_cp,
        input_shapes={"measurement": (4,), "control_input": (1,)},
        state_shapes={"x_hat": (4,), "P": (4, 4)},
    )
    lqr_cp_graph = trace_node(
        lqr_cp,
        input_shapes={"current_state": (4,), "target_state": (4,)},
    )
    print(f"    KF graph:  {len(kf_cp_graph.graph.nodes)} nodes, "
          f"state={kf_cp_graph.state_attrs}")
    print(f"    LQR graph: {len(lqr_cp_graph.graph.nodes)} nodes, "
          f"stateless={lqr_cp_graph.state_attrs == []}")

    # Compose (n_x=4, n_u=1; clip the force to ±10 N).
    cp_limits = (np.array([-10.0]), np.array([10.0]))
    composed_cp = compose(
        kf_cp_graph, lqr_cp_graph,
        plant_dims={"n_x": 4, "n_u": 1},
        input_limits=cp_limits,
    )
    print(f"\n  Composed cartpole step graph: {len(composed_cp.graph.nodes)} nodes")
    print(f"    Inputs:  {composed_cp.inputs}")
    print(f"    Outputs: {composed_cp.outputs}")

    # Verify the composed cartpole step matches the live numpy loop.
    print("\n  Verifying cartpole composed step vs live numpy loop (10 trials x 4 ticks):")
    max_err = 0.0
    for trial in range(10):
        y_seq = rng.normal(0, 0.1, (4, 4))       # 4 measurements per tick
        x_ref = rng.normal(0, 0.1, (4,))         # reference (e.g. upright at x=0)
        x_hat_init = rng.normal(0, 0.1, (4,))    # initial estimate

        kf_cp.P = _p0(4)
        kf_cp.x_hat = x_hat_init.copy()
        u_prev = np.zeros(1)

        state = {"state_x_hat": x_hat_init.copy(), "state_P": _p0(4)}
        for t in range(4):
            x_hat_np = kf_cp.estimate(y_seq[t], u_prev)
            u_np = np.clip(lqr_cp.compute(x_hat_np, x_ref), cp_limits[0], cp_limits[1])

            traced = interpret(
                composed_cp.graph,
                {"y": y_seq[t], "x_ref": x_ref, "u_prev": u_prev,
                 "state_x_hat": state["state_x_hat"], "state_P": state["state_P"]},
            )
            max_err = max(max_err, float(np.max(np.abs(traced["u"] - u_np))))
            state["state_x_hat"] = traced["state_x_hat"]
            state["state_P"] = traced["state_P"]
            u_prev = u_np
    print(f"    max abs err over 10 trials: {max_err:.2e}  "
          f"{'PASS' if max_err < 1e-12 else 'FAIL'}")

    # ─── Summary ────────────────────────────────────────────────────────
    banner("Summary")
    print("  The tracer captures the control-law math as a primitive graph.")
    print("  The interpreter replays it to float-exactness vs NumpyBackend.")
    print("  Recurrent state (KF x_hat/P) becomes explicit state_* ports — no frozen recursion.")
    print("  The composition pass stitches per-node graphs with auto-reshape.")
    print("  Swapping a component re-traces only that node; the rest is reused.")
    print("  Works for any plant (3-DOF base, 4-DOF cartpole) — no tracer-side changes.")
    print("  Next slice (b): lower the captured graph to Zig → base.so.")


if __name__ == "__main__":
    main()
