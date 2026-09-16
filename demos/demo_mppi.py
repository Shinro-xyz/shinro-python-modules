# FILE: demos/demo_mppi.py
"""MPPI demo: how to wire ``dynamics_fn`` / ``cost_fn``, and the lowering contract.

Four ways exist to give MPPI its model, and which one you pick decides whether
the controller can be compiled to a Zig kernel:

1. **Constructor injection** — ``MPPIController(dynamics_fn=..., cost_fn=...)``
   for a hand-written model.
2. **Attribute injection** — set ``ctrl.dynamics_fn`` / ``ctrl.cost_fn`` after
   construction (the ``from_config`` path).
3. **Plant-driven** — ``ctrl.attach_plant(plant, Q=..., R=...)`` builds both
   from the plant's model; this is what ``ScenarioFactory`` does.
4. **Config-driven** — ``MPPIController.from_config(cfg)`` (TOML cannot hold a
   callable, so the callables are injected afterwards by one of the above).

The callables are **batched** and **traced through** (they are inlined into the
lowered graph), so for a lowerable controller they must be trace-safe: operator
and ``bk``-op only, no raw numpy, no ``x[i]`` indexing, no ``bk.sum``, and no
rank-differing broadcast. ``attach_plant``'s LTI path already satisfies this.

Everything here runs on the default install — no MuJoCo, no torch required.

Usage:
  python -m demos.demo_mppi
"""

from __future__ import annotations

import numpy as np

from shinro.codegen import interpret
from shinro.codegen.trace_node import trace_node
from shinro.controllers.mppi import MPPIController
from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
from shinro.utils.array_backend import NumpyBackend

DT = 0.02
N_SAMPLES = 200
HORIZON = 15

# A first-order integrator, x' = x + dt*u, as diagonal weights. These are the
# model constants a user would normally derive from their own system.
D_X, D_U = 1, 1
Q_DIAG = np.array([1.0])  # (D_x,) state weights
R_DIAG = np.array([0.1])  # (D_u,) control weights


# ─── the two callables, written trace-safely ───────────────────────────────


def integrator_dynamics(x_batch, u_batch, dt):
    """Batched dynamics: ``(N, D_x), (N, D_u)`` -> ``(N, D_x)``.

    Operator-only — no numpy, no ``x[i]`` indexing, no ``bk.sum`` — so this is
    also what tracing records when the controller is lowered.
    """
    return x_batch + dt * u_batch


def quadratic_cost(x_batch, u_batch):
    """Batched stage cost: ``(N, D_x), (N, D_u)`` -> ``(N,)`` per-sample scalar.

    Each quadratic form is written as a contraction (``(N, D) @ (D,) -> (N,)``)
    rather than ``bk.sum(x*x*W, axis=1)``: a sum over an axis is a matmul
    identity, and this form traces (``TraceBackend`` has no ``sum`` op).
    """
    return (x_batch * x_batch) @ Q_DIAG + (u_batch * u_batch) @ R_DIAG


def build_custom_controller(bk):
    """Path 1 — constructor injection: a hand-written model."""
    return MPPIController(
        dynamics_fn=integrator_dynamics,
        cost_fn=quadratic_cost,
        num_samples=N_SAMPLES,
        temperature=1.0,
        dt=DT,
        horizon=HORIZON,
        noise_sigma=[1.0],
        seed=0,
        backend=bk,
    )


# ─── 1. constructor injection ──────────────────────────────────────────────


def demo_custom_model(bk):
    print("=== 1. Constructor injection: hand-written dynamics + cost ===")
    ctrl = build_custom_controller(bk)

    x = np.array([1.0])  # (D_x,)
    for step in range(200):
        u = bk.to_numpy(ctrl.compute(x))
        x = x + DT * u  # the plant the controller was told about
        if step % 50 == 0:
            print(f"  step {step:3d}: x = {x[0]:+.4f}  u = {u[0]:+.4f}")
    print(f"  final |x| = {abs(x[0]):.4f} (regulated toward the origin)\n")


# ─── 2. plant-driven (attach_plant) ────────────────────────────────────────


def demo_plant_model(bk):
    print("=== 2. Plant-driven: attach_plant(plant, Q=, R=) ===")
    plant = HolonomicMobileRobot(
        num_wheels=3, radius_robots=0.1, gamma=0.0, radius_wheels=0.03, dt=DT, backend=bk
    )
    ctrl = MPPIController(
        num_samples=200,
        temperature=1.0,
        dt=DT,
        horizon=10,
        noise_sigma=[1.0, 1.0, 1.0],
        seed=1,
        backend=bk,
    )
    # The plant's LTI model supplies dynamics (x @ A.T + u @ B.T); Q/R are the
    # quadratic cost weights. attach_plant OVERWRITES an earlier cost_fn.
    ctrl.attach_plant(plant, Q=np.array([10.0, 10.0, 10.0]), R=np.array([0.1, 0.1, 0.1]))

    x = np.zeros(3)
    x_ref = np.array([1.0, 0.0, 0.0])
    for _ in range(300):
        u = bk.to_numpy(ctrl.compute(bk.array(x), bk.array(x_ref)))  # x_ref tracked
        x = x + DT * u  # A = I, B = dt*I for this plant
    print(f"  tracked x_ref = {x_ref}, reached x = {np.round(x, 4)}")
    print(f"  |x - x_ref| = {np.linalg.norm(x - x_ref):.4f}\n")


# ─── 3. from_config ────────────────────────────────────────────────────────


def demo_config_model(bk):
    print("=== 3. Config-driven: from_config() + injection ===")
    config = {
        "num_samples": 100,
        "temperature": 1.0,
        "dt": DT,
        "horizon": 10,
        "noise_sigma": [1.0],
        "u_min": [-5.0],
        "u_max": [5.0],
        "seed": 7,
        "state_cost": [1.0],  # -> ctrl._Q
        "control_cost": [0.1],  # -> ctrl._R
    }
    ctrl = MPPIController.from_config(config, backend=bk)
    # A callable cannot live in TOML: inject it (path 1 or 2), or call
    # attach_plant(plant) to build it from the plant instead.
    ctrl.dynamics_fn = integrator_dynamics
    ctrl.cost_fn = quadratic_cost
    u = bk.to_numpy(ctrl.compute(bk.array([1.0])))
    print(f"  from_config + injection: u = {u[0]:+.4f} (bounded by u_min/u_max)")
    print("  (attach_plant(plant) is the alternative — it sets both callables)\n")


# ─── 4. host-supplied perturbations (the lowering contract) ────────────────


def demo_host_supplied_noise(bk):
    print("=== 4. Host-supplied epsilon: sampling stays on the host ===")
    sampled = build_custom_controller(bk)
    x0 = np.array([1.0])
    u_sampled = bk.to_numpy(sampled.compute(x0))
    eps = sampled._last_epsilon  # the draw the controller actually used
    if eps is None:  # the eager path always records its draw
        raise RuntimeError("expected compute() to record the sampled perturbations")

    # A lowered kernel does no sampling: the host draws the perturbations and
    # feeds them in, so compute(..., epsilon=...) must reproduce the run. The
    # layout is (N, K*D_u), sample-major — what the C-ABI input port expects.
    fed = build_custom_controller(bk)
    u_fed = bk.to_numpy(fed.compute(x0, None, np.ascontiguousarray(eps.reshape(N_SAMPLES, HORIZON * D_U))))
    print(f"  sampled u = {u_sampled[0]:+.10f}")
    print(f"  fed-eps u = {u_fed[0]:+.10f}  (identical: same noise in -> same u out)\n")


# ─── 5. trace it: the same callables become graph nodes ────────────────────


def demo_trace_and_lower(bk):
    print("=== 5. Tracing: the callables become lowered graph nodes ===")
    ctrl = build_custom_controller(bk)
    ng = trace_node(
        ctrl,
        input_shapes={"current_state": (D_X,), "target_state": (D_X,), "epsilon": (N_SAMPLES, HORIZON * D_U)},
        state_shapes={"u": (HORIZON, D_U)},  # recurrent: the nominal sequence
    )
    print(f"  traced graph: {len(ng.graph.nodes)} nodes, state attrs = {ng.state_attrs}")
    print(f"  graph outputs: {sorted(ng.output_nodes)}")

    x0 = np.array([1.0])
    eps = np.random.default_rng(0).normal(0.0, 1.0, (N_SAMPLES, HORIZON * D_U))
    feeds = {
        "current_state": x0,
        "target_state": np.zeros(D_X),
        "epsilon": eps,
        "state_u": np.zeros((HORIZON, D_U)),
    }
    out = interpret(ng.graph, feeds)  # the interpreter is the .so's oracle

    ref = build_custom_controller(bk)
    u_ref = bk.to_numpy(ref.compute(x0, None, eps))
    err = np.max(np.abs(out["out"] - u_ref))
    print(f"  interpreter vs live numpy: u max err = {err:.2e} (the Zig VM is checked the same way)")
    print(f"  published diagnostics: costs{np.asarray(out['costs']).shape}\n")


# ─── the rules that decide whether a model can lower ───────────────────────


def print_trace_safety_rules():
    print("=== Trace-safety rules for a lowerable dynamics_fn/cost_fn ===")
    for i, rule in enumerate(
        (
            "use operators and self.bk ops — no raw numpy (np.sum, np.tile, ...)",
            "no x[i] indexing — index with bk.slice_ (row blocks) and .T",
            "no bk.sum — write sums as contractions ((z * z) @ W)",
            "no rank-differing broadcast — (N, D) * (D,) is rejected; use (1, D)",
            "stateful attrs must be rebound (self.u = ...), never mutated in place",
            "plant dynamics must be batch-capable — a per-sample x[i] loop cannot be traced",
        ),
        start=1,
    ):
        print(f"  {i}. {rule}")
    print("\n  attach_plant's dynamics already follows all of these: LTI plants use one")
    print("  batched matmul, nonlinear plants evaluate their own dynamics over the batch.")


def main():
    bk = NumpyBackend()
    print(f"MPPI demo (backend: {type(bk).__name__}, dt={DT}, N={N_SAMPLES}, K={HORIZON})\n")
    demo_custom_model(bk)
    demo_plant_model(bk)
    demo_config_model(bk)
    demo_host_supplied_noise(bk)
    demo_trace_and_lower(bk)
    print_trace_safety_rules()
    print("\nDone.")


if __name__ == "__main__":
    main()
