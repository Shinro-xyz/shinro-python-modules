"""Quadrotor sliding-mode demo — a bank of the framework's SlidingModeController.

The framework SMC is **single-surface** by design: ``c`` is one row, so
``s = c^T x`` is a scalar and the law controls one error combination. The
supported multi-axis pattern is therefore a *bank* of instances — one per
controlled degree of freedom — plus a mixer. This demo does exactly that with
the built-in controller, no bespoke control law:

    altitude:  s_z = z_dot  + lambda_z * z        -> collective thrust  F
    roll:      s_phi = p    + lambda_phi * phi    -> body torque        tau_x
    pitch:     s_theta = q  + lambda_theta * theta-> body torque        tau_y
    yaw:       s_psi = r    + lambda_psi * psi    -> body torque        tau_z

Each instance is handed its own two-state channel ``(pos, rate)`` plus the
reduced drift ``f_i`` and control effectiveness ``g_i`` that embed the plant's
gravity / gyroscopic / kinematic terms. The built-in law then produces a scalar
virtual control per channel:

    u_i = (c_i^T g_i)^-1 ( -c_i^T f_i - k1 |s_i|^alpha smooth(s_i) - k2 s_i )

The four virtual controls ``[F, tau_x, tau_y, tau_z]`` map back to rotor speeds
through the plant's (constant, config-derived) mixer:

    [F, tau_x, tau_y, tau_z]^T = M [w1^2, w2^2, w3^2, w4^2]^T ,  w = sqrt(M^-1 v)

so the only per-tick linear algebra is four scalar divisions inside the
controllers — no matrix inverse, no new ops. The bank is coupled only through
those feedforward terms, which is the documented decentralized multi-axis use.

The host owns ``f`` and ``g`` (SMC's runtime contract). x/y are on no surface —
they need a cascaded outer position loop — so this demo stabilizes the four
input-matched DOF.

Usage:
  python -m demos.demo_quadrotor_smc
"""

from __future__ import annotations

import numpy as np

from shinro.controllers.smc import SlidingModeController
from shinro.plants.quadrotor import Quadrotor
from shinro.utils.array_backend import NumpyBackend

DT = 0.01
STEPS = 500

# One surface per controlled DOF: s_i = rate_i + lambda_i * (pos_i - ref), ref = 0.
LAM = (3.0, 6.0, 6.0, 4.0)  # z, roll, pitch, yaw
K1, K2, PHI, ALPHA = 1.5, 1.0, 0.05, 1.0

X0 = np.array([0.0, 0.0, 0.3, 0.15, -0.12, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# ─── the plant, the bank of built-in SMCs, and the mixer ───────────────────


def build_plant() -> Quadrotor:
    return Quadrotor(backend=NumpyBackend())


def trim_speed(plant: Quadrotor) -> float:
    """Rotor speed that balances gravity on all four rotors: sqrt(m g / 4k)."""
    return (plant.m * plant.g / (4.0 * plant.k)) ** 0.5


def build_bank(backend=None) -> list[SlidingModeController]:
    """Four built-in SMC instances, one per controlled DOF."""
    bk = backend or NumpyBackend()
    return [
        SlidingModeController(c=[lam, 1.0], k1=K1, k2=K2, phi=PHI, smoother="sat", alpha=ALPHA, backend=bk)
        for lam in LAM
    ]


def mixer_matrix(plant: Quadrotor) -> np.ndarray:
    """``M`` (4x4): ``[F, tau_x, tau_y, tau_z] = M [w1^2, w2^2, w3^2, w4^2]``.

    Built from the plant's rotor table (``plant.rotors``) and coefficients, so
    it follows a configured ``[[rotors]]`` layout for free. Constant, hence
    inverted once.
    """
    x = np.array([r[0] for r in plant.rotors])
    y = np.array([r[1] for r in plant.rotors])
    spin = np.array([r[2] for r in plant.rotors])
    k, b = plant.k, plant.b
    return np.vstack([
        plant.k * np.ones(4),  # collective thrust  F     = k * sum(a_i)
        k * y,                 # roll torque        tau_x = k * sum(y_i a_i)
        -k * x,                # pitch torque       tau_y = -k * sum(x_i a_i)
        -b * spin,             # yaw torque         tau_z = -b * sum(sigma_i a_i)
    ])


def channel_inputs(x: np.ndarray, plant: Quadrotor):
    """Per-channel ``(x_i, f_i, g_i)`` for the bank, with the coupling embedded.

    Each channel is the two-state ``(pos_i, rate_i)`` with rate_i in
    ``{z_dot, p, q, r}``. ``f_i`` is the channel's drift (gravity, gyroscopic
    and Euler-kinematics terms) and ``g_i`` maps the scalar virtual control to
    the channel's acceleration.
    """
    m, g = plant.m, plant.g
    i_xx, i_yy, i_zz = plant.I
    roll, pitch = x[3], x[4]
    p, q, r = x[9], x[10], x[11]
    s_r, c_r = np.sin(roll), np.cos(roll)
    c_p = np.cos(pitch)

    z_dot = x[8]
    phi_dot = p + q * s_r * np.tan(pitch) + r * c_r * np.tan(pitch)
    theta_dot = q * c_r - r * s_r
    psi_dot = (q * s_r + r * c_r) / c_p

    channels = [
        (np.array([x[2], z_dot]),      np.array([z_dot, -g]),                         np.array([0.0, c_p * c_r / m])),
        (np.array([roll, p]),          np.array([phi_dot, q * r * (i_yy - i_zz) / i_xx]), np.array([0.0, 1.0 / i_xx])),
        (np.array([pitch, q]),         np.array([theta_dot, r * p * (i_zz - i_xx) / i_yy]), np.array([0.0, 1.0 / i_yy])),
        (np.array([x[5], r]),          np.array([psi_dot, p * q * (i_xx - i_yy) / i_zz]), np.array([0.0, 1.0 / i_zz])),
    ]
    return channels


def virtual_controls(bank, x: np.ndarray, plant: Quadrotor) -> list[float]:
    """Run each built-in SMC on its channel: return ``[F, tau_x, tau_y, tau_z]``."""
    out = []
    for smc, (x_i, f_i, g_i) in zip(bank, channel_inputs(x, plant)):
        out.append(np.asarray(smc.compute(x_i, f_i, g_i)).ravel()[0].item())
    return out


def mix(mixer_inv: np.ndarray, virtual: list[float]) -> np.ndarray:
    """Invert the mixer to rotor speeds: ``w = sqrt(M^-1 v)`` (clipped at 0)."""
    a = mixer_inv @ np.array(virtual)
    return np.sqrt(np.clip(a, 0.0, None))


def integrate(plant: Quadrotor, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    plant.state = x
    return np.asarray(plant.step(u), dtype=float)


# ─── 1. plant + trim ───────────────────────────────────────────────────────


def demo_plant(plant: Quadrotor, u_trim: float) -> None:
    print("=== 1. Quadrotor plant and hover trim ===")
    print(f"  m={plant.m} kg, I={plant.I}, k={plant.k}, b={plant.b}, dt={plant.dt}")
    print("  state [x,y,z,roll,pitch,yaw,vx,vy,vz,p,q,r] (12), control [w1..w4] (4)")
    print(f"  trim rotor speed = sqrt(m g / 4k) = {u_trim:.4f} rad/s (each rotor)\n")


# ─── 2. the bank and the mixer ─────────────────────────────────────────────


def demo_bank(plant: Quadrotor, bank, mixer: np.ndarray, mixer_inv: np.ndarray) -> None:
    print("=== 2. A bank of built-in SlidingModeControllers + the rotor mixer ===")
    names = ["altitude", "roll", "pitch", "yaw"]
    rates = ["z_dot", "p", "q", "r"]
    for name, smc, lam, rate in zip(names, bank, LAM, rates):
        print(f"  {name:<8} s = {rate} + {lam} * pos   (c = {smc.c.tolist()})")

    print("\n  mixer  [F, tau_x, tau_y, tau_z] = M @ [w1^2, w2^2, w3^2, w4^2]:")
    for row in mixer:
        print("    " + "  ".join(f"{v:+8.4f}" for v in row))
    print(f"  det(M) = {np.linalg.det(mixer):+.6e}, cond(M) = {np.linalg.cond(mixer):.1f}")
    print("  M is constant (rotor geometry), so M^-1 is computed once; each built-in")
    print("  SMC does a scalar division per tick — no matrix inverse at runtime.\n")


# ─── 3. close the loop ─────────────────────────────────────────────────────


def demo_closed_loop(plant: Quadrotor, bank, mixer_inv: np.ndarray) -> np.ndarray:
    print("=== 3. Closed loop: level the quadrotor and hold altitude ===")
    x = X0.copy()
    print(f"  {'step':>5} {'z':>9} {'roll':>9} {'pitch':>9} {'yaw':>9} {'thrust':>9}")
    for step in range(STEPS):
        u = mix(mixer_inv, virtual_controls(bank, x, plant))
        x = integrate(plant, x, u)
        if step % 50 == 0 or step == STEPS - 1:
            thrust = plant.k * np.sum(u**2)
            print(f"  {step:>5} {x[2]:>9.5f} {x[3]:>9.5f} {x[4]:>9.5f} {x[5]:>9.5f} {thrust:>9.4f}")
    print(f"  final state = {np.round(x, 5)}")
    print(f"  (weight = {plant.m * plant.g:.4f} N; thrust converged to it at level attitude)")
    print("  x, y drift — unactuated here; a real stack cascades a position loop onto")
    print("  the roll/pitch setpoints.\n")
    return x


# ─── 4. why one surface is not enough ──────────────────────────────────────


def demo_single_surface_fails(plant: Quadrotor, u_trim: float) -> None:
    print("=== 4. Contrast: one built-in SMC on the whole 12-state plant cannot stabilize it ===")
    c = np.poly(-np.arange(1.0, 12.0))[::-1].copy()  # a valid 12-state Hurwitz surface
    smc = SlidingModeController(c=c, k1=1.0, k2=0.5, phi=PHI, backend=NumpyBackend())
    x = X0.copy()
    u = np.full(4, u_trim)
    for step in range(200):
        f = np.asarray(plant.dynamics(x, u), dtype=float)
        _, g = _linearize(plant, x, u)
        try:
            u = np.asarray(smc.compute(x, f, g), dtype=float).ravel()
        except RuntimeError as exc:
            print(f"  step {step}: single-surface SMC faulted -> {exc}")
            print(f"  state = {np.round(x, 4)}")
            print("  |s| is scalar: one equation, four rotors; the law's min-norm member")
            print("  leaves the other 11 state dimensions uncontrolled.\n")
            return
        u = np.clip(u, 0.0, 2.0 * u_trim)  # keep the plant finite so the failure is legible
        x = integrate(plant, x, u)
    print(f"  survived 200 steps but drifted to (z, roll, pitch, yaw) = {np.round(x[[2, 3, 4, 5]], 4)}")
    print("  one surface zeroed, the other dimensions left free.\n")


# ─── 5. the built-in controllability guard ─────────────────────────────────


def demo_guard(plant: Quadrotor, bank) -> None:
    print("=== 5. The built-in guard: c^T g -> 0 at gimbal lock ===")
    smc_z = bank[0]
    gimbal = np.zeros(12)
    gimbal[4] = np.pi / 2.0  # pitch = 90 deg -> cos(pitch) = 0
    _, f_z, g_z = channel_inputs(gimbal, plant)[0]
    cg = (smc_z.c @ g_z).item()
    print(f"  altitude channel at pitch = 90 deg: c^T g = cos(pitch)/m = {cg:+.3e}")
    try:
        smc_z.compute(np.array([0.0, 0.0]), f_z, g_z)
    except RuntimeError as exc:
        print(f"  eager numpy -> RuntimeError({exc})")
    print(f"  (controllability_eps = {smc_z.controllability_eps:g})")
    print("  A lowered kernel cannot raise (a Zig panic across the C ABI aborts the host),")
    print("  so the same guard compiles to a zero command + a `healthy = 0` flag.\n")


def _linearize(plant: Quadrotor, x: np.ndarray, u: np.ndarray):
    """Continuous (A, B) at (x, u) via the framework helper."""
    from shinro.utils.linearization import linearize_plant

    return linearize_plant(plant, x, u)


def main() -> None:
    plant = build_plant()
    u_trim = trim_speed(plant)
    bank = build_bank()
    mixer = mixer_matrix(plant)
    mixer_inv = np.linalg.inv(mixer)
    print(f"Quadrotor sliding-mode demo — bank of built-in SMCs (dt={DT})\n")
    demo_plant(plant, u_trim)
    demo_bank(plant, bank, mixer, mixer_inv)
    demo_closed_loop(plant, bank, mixer_inv)
    demo_single_surface_fails(plant, u_trim)
    demo_guard(plant, bank)
    print("Done.")


if __name__ == "__main__":
    main()
