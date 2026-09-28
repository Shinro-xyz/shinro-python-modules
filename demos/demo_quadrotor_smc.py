"""Quadrotor multi-surface sliding-mode demo.

A single sliding surface cannot stabilize a 12-state, 4-input quadrotor: the
surface ``s = cᵀx`` is a *scalar*, so driving it to zero leaves 11 dimensions
free and the min-norm law is underdetermined (one equation, four rotors). The
fix is to **stack one surface per input-matched degree of freedom** — altitude
plus roll/pitch/yaw — so ``C`` is a ``(4, 12)`` matrix:

    s            = C x
    s_dot_want   = -k1 |s|^alpha sat(s/phi) - k2 s
    (C g) (u - u_trim) = s_dot_want - C f(x, u_trim)
    u            = u_trim + solve(C g, s_dot_want - C f)

With ``m = n_u = 4`` the matrix product ``C g`` is a square **decoupling
matrix** (near hover it is exactly the rotor mixer: collective thrust and the
roll/pitch/yaw torques mapped back onto the four rotor speeds), so the control
is fully determined — no min-norm projection.

The host owns ``f(x)`` and ``g(x)``: that is SMC's runtime contract (the
controller is pure arithmetic on ``(x, f, g)``). Here the plant supplies the
nonlinear drift ``f`` and the finite-difference control effectiveness ``g`` via
``linearize_plant``, evaluated at the hover trim so the quadratic rotor-speed
input does not vanish the Jacobian.

Usage:
  python -m demos.demo_quadrotor_smc
"""

from __future__ import annotations

import numpy as np

from shinro.plants.quadrotor import Quadrotor
from shinro.utils.array_backend import NumpyBackend
from shinro.utils.linearization import linearize_plant

DT = 0.01
STEPS = 500

# One surface per controlled DOF: [altitude, roll, pitch, yaw].
#   s_i = rate_i + lambda_i * (pos_i - pos_i_ref)   (ref = 0, the trim)
LAM = (3.0, 6.0, 6.0, 4.0)
K1, K2, PHI, ALPHA = 1.5, 1.0, 0.05, 1.0

POS_IDX = (2, 3, 4, 5)  # z, roll, pitch, yaw
RATE_IDX = (8, 9, 10, 11)  # z_dot, p, q, r

X0 = np.array([0.0, 0.0, 0.3, 0.15, -0.12, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# ─── the plant + the multi-surface law ─────────────────────────────────────


def build_plant() -> Quadrotor:
    return Quadrotor(backend=NumpyBackend())


def trim_speed(plant: Quadrotor) -> float:
    """Rotor speed that balances gravity on all four rotors: sqrt(m g / 4k)."""
    return (plant.m * plant.g / (4.0 * plant.k)) ** 0.5


def surface_matrix(lam=LAM) -> np.ndarray:
    """``C`` (4, 12): column ``pos`` gets ``lambda``, column ``rate`` gets 1."""
    C = np.zeros((4, 12))
    for row, (pos, rate, lam_i) in enumerate(zip(POS_IDX, RATE_IDX, lam)):
        C[row, pos] = lam_i
        C[row, rate] = 1.0
    return C


def drift(plant: Quadrotor, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``f(x)`` — the plant's nonlinear continuous dynamics, host-evaluated."""
    return np.asarray(plant.dynamics(x, u), dtype=float)


def control_effectiveness(plant: Quadrotor, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``g(x)`` — ``df/du`` (12, 4), host-evaluated at the trim control."""
    _, g = linearize_plant(plant, x, u)
    return np.asarray(g, dtype=float)


def reaching_law(s: np.ndarray) -> np.ndarray:
    """``s_dot_want`` — per-surface reaching law with a saturation layer."""
    return -K1 * np.abs(s) ** ALPHA * np.clip(s / PHI, -1.0, 1.0) - K2 * s


def mssmc_step(plant: Quadrotor, x: np.ndarray, u_trim: np.ndarray, C: np.ndarray):
    """One multi-surface SMC tick: return ``(u, s, C g)``.

    ``g`` is the control effectiveness at the hover trim, so ``u - u_trim`` is
    what the linearization is valid for and ``solve`` returns the increment.
    """
    f = drift(plant, x, u_trim)
    g = control_effectiveness(plant, x, u_trim)
    s = C @ x
    Cg = C @ g
    du = np.linalg.solve(Cg, reaching_law(s) - C @ f)
    return u_trim + du, s, Cg


def integrate(plant: Quadrotor, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Advance the plant one tick and return the new state as numpy."""
    plant.state = x
    return np.asarray(plant.step(u), dtype=float)


# ─── 1. plant + trim ───────────────────────────────────────────────────────


def demo_plant(plant: Quadrotor, u_trim: np.ndarray) -> None:
    print("=== 1. Quadrotor plant and hover trim ===")
    print(f"  m={plant.m} kg, I={plant.I}, k={plant.k}, b={plant.b}, dt={plant.dt}")
    print("  state [x,y,z,roll,pitch,yaw,vx,vy,vz,p,q,r] (12), control [w1..w4] (4)")
    print(f"  trim rotor speed = sqrt(m g / 4k) = {u_trim[0]:.4f} rad/s (each rotor)\n")


# ─── 2. the surfaces and the decoupling matrix ─────────────────────────────


def demo_surfaces(plant: Quadrotor, u_trim: np.ndarray, C: np.ndarray) -> None:
    print("=== 2. Stacked surfaces and the decoupling matrix ===")
    names = ["altitude", "roll", "pitch", "yaw"]
    for name, lam, pos, rate in zip(names, LAM, POS_IDX, RATE_IDX):
        print(f"  s_{name:<8} = x[{rate:>2}] + {lam} * x[{pos}]")
    print(f"  C is {C.shape}  ->  s = C x is a {C.shape[0]}-vector (one per rotor DOF)")

    Cg = C @ control_effectiveness(plant, np.zeros(12), u_trim)
    print("\n  C g at hover (rows: surfaces, cols: rotors) =")
    for name, row in zip(names, Cg):
        print(f"    {name:<8} " + "  ".join(f"{v:+.4f}" for v in row))
    print(f"  det(C g) = {np.linalg.det(Cg):+.6e}, cond = {np.linalg.cond(Cg):.1f}")
    print("  The rows are the rotor mixer (thrust, tau_x, tau_y, tau_z) -> invertible,")
    print("  so u = u_trim + (C g)^-1 (s_dot_want - C f) is fully determined.\n")


# ─── 3. close the loop ─────────────────────────────────────────────────────


def demo_closed_loop(plant: Quadrotor, u_trim: np.ndarray, C: np.ndarray) -> np.ndarray:
    print("=== 3. Closed loop: level the quadrotor and hold altitude ===")
    x = X0.copy()
    print(f"  {'step':>5} {'z':>9} {'roll':>9} {'pitch':>9} {'yaw':>9} {'|s|':>9} {'thrust':>9}")
    for step in range(STEPS):
        u, s, _ = mssmc_step(plant, x, u_trim, C)
        x = integrate(plant, x, u)
        if step % 50 == 0 or step == STEPS - 1:
            thrust = plant.k * np.sum(u**2)
            print(f"  {step:>5} {x[2]:>9.5f} {x[3]:>9.5f} {x[4]:>9.5f} {x[5]:>9.5f} "
                  f"{np.linalg.norm(s):>9.2e} {thrust:>9.4f}")
    print(f"  final state = {np.round(x, 5)}")
    print(f"  (weight = {plant.m * plant.g:.4f} N; thrust converged to it at level attitude)")
    print("  z + roll/pitch/yaw converged. x, y drift — they are unactuated here; a real")
    print("  quadrotor cascades an outer position loop onto the roll/pitch setpoints.\n")
    return x


# ─── 4. why one surface is not enough ──────────────────────────────────────


def demo_single_surface_fails(plant: Quadrotor, u_trim: np.ndarray) -> None:
    print("=== 4. Contrast: one surface is underdetermined and cannot stabilize it ===")
    from shinro.controllers.smc import SlidingModeController

    c = np.poly(-np.arange(1.0, 12.0))[::-1].copy()  # a valid 12-state Hurwitz surface
    smc = SlidingModeController(c=c, k1=1.0, k2=0.5, phi=PHI, backend=NumpyBackend())
    x = X0.copy()
    u = u_trim.copy()
    for step in range(200):
        f = drift(plant, x, u_trim)
        g = control_effectiveness(plant, x, u_trim)
        try:
            u = np.asarray(smc.compute(x, f, g), dtype=float).ravel()
        except RuntimeError as exc:
            print(f"  step {step}: single-surface SMC faulted -> {exc}")
            print(f"  state = {np.round(x, 4)}")
            print("  |s| is scalar: one equation, four rotors; the law's min-norm member")
            print("  leaves the other 11 state dimensions uncontrolled.\n")
            return
        u = np.clip(u, 0.0, 2.0 * u_trim[0])  # keep the plant finite so the failure is legible
        x = integrate(plant, x, u)
    print(f"  survived 200 steps (no guard trip) but drifted to {np.round(x[[2, 3, 4, 5]], 4)}")
    print("  (z, roll, pitch, yaw) — one surface zeroed, the other dimensions left free.\n")


# ─── 5. loss of effectiveness: the fail-safe, not a crash ──────────────────


def demo_guard(plant: Quadrotor, u_trim: np.ndarray, C: np.ndarray) -> None:
    print("=== 5. Loss of effectiveness: C g loses rank, so guard before solving ===")
    x_pitch_up = np.zeros(12)
    x_pitch_up[4] = np.pi / 2.0  # gimbal lock
    cases = [
        ("hover", u_trim, np.zeros(12)),
        ("rotors stopped", np.zeros(4), np.zeros(12)),
        ("pitch = 90 deg", u_trim, x_pitch_up),
    ]
    for label, u_op, x_op in cases:
        g = control_effectiveness(plant, x_op, u_op)
        det = np.linalg.det(C @ g)
        healthy = 1.0 if abs(det) > 1e-9 else 0.0
        print(f"  {label:<16} det(C g) = {det:+.3e}  ->  healthy = {healthy:.0f}")
    print("  A compiled kernel cannot raise (a Zig panic across the C ABI aborts the host),")
    print("  so the guard is data: command zero and let the host's fault policy act on")
    print("  `healthy` in the same tick. Zero command is a floor, not a stability promise.\n")


def main() -> None:
    plant = build_plant()
    u_trim = np.full(4, trim_speed(plant))
    C = surface_matrix()
    print(f"Quadrotor multi-surface SMC demo (dt={DT})\n")
    demo_plant(plant, u_trim)
    demo_surfaces(plant, u_trim, C)
    demo_closed_loop(plant, u_trim, C)
    demo_single_surface_fails(plant, u_trim)
    demo_guard(plant, u_trim, C)
    print("Done.")


if __name__ == "__main__":
    main()
