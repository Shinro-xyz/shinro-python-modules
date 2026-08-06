# Compiling the Robot Control Loop to Zig

**Status:** Proposal / design sketch
**Audience:** Zig engineer evaluating feasibility
**Repo:** `shinro-python-modules` — a Python control suite for robotics

---

## 1. The problem

`shinro` is a Python library of controllers (LQR, MPC, MPPI, PID), plants
(holonomic base, 6-DOF arm, inverted pendulum), state estimators (Kalman,
Luenberger), and trajectory generators. Components are wired into a robot
simulation by `RobotSim`, which reads a TOML config, resolves each component
through a registry, and instantiates Python objects.

The control loop is **eager and interpreted**:

```
   robot_config.toml
        │
        ▼
   RobotSim ──► Python objects (ABC instances)
        │
        ▼
   sim.step()  ← every step: Python method dispatch,
                 ArrayBackend virtual calls, dict lookups,
                 GC pressure, GIL
```

This is fine for prototyping but wrong for deployment:

- **No hard real-time guarantee** — Python dispatch, GC, and the GIL make
  worst-case step time unbounded.
- **Per-step overhead** — a 3×3 LQR feedback is ~9 multiply-adds, but each
  `compute()` call pays Python method resolution, attribute lookups, and
  numpy call overhead.
- **No ahead-of-time validation** — a bad config fails 200 steps into a
  simulation, not at load time.

## 2. The idea

Compile the control loop to **Zig**, ahead of time, from the same TOML
config that `RobotSim` already reads. The result is a single native shared
library that runs the entire closed loop (estimator → controller → plant)
with no Python in the per-step path.

This is the robot equivalent of TensorFlow's static graph / XLA compilation,
but with a crucial difference: **we keep the eager Python path as a
first-class fallback** (the "inverse torch" model — compile by default,
eager opt-in for debugging).

```
   robot_config.toml
        │
        ├──► RobotSim(eager=True)   ← interpreted Python, for debugging
        │
        └──► compile_graph()        ← NEW, default path
                 │
                 ▼
             robot_loop.zig
                 │
                 ▼
             zig build -O ReleaseFast
                 │
                 ▼
             robot_loop.so  ──►  lib.control_loop_step(...)
```

## 3. Why Zig

- **C-ABI compatible** — trivial FFI from Python (`ctypes`/`cffi`), and
  direct calls into existing C libraries (MuJoCo, OSQP, BLAS, LAPACK).
- **`comptime`** — the graph's dimensions and gains are known at build time
  (they come from the TOML). Zig specializes the loop: unrolls small
  matrices, bakes gains as constants, eliminates dead branches. This is the
  XLA-fusion equivalent, done by the language instead of a custom compiler.
- **No runtime, no GC** — safe to call from a hard-real-time control thread.
- **`inline for`** — comptime loop unrolling for small dims; regular `for`
  for large dims (e.g. MPPI's 100×10 rollout).
- **`@cImport`** — read C headers at compile time; BLAS/LAPACK/MuJoCo/OSQP
  become type-checked Zig calls with zero wrapper layer.

## 4. Architecture

### 4.1 The graph is already declared — it's the TOML

The repo already has the three ingredients of a static graph; they're just
interpreted at runtime instead of compiled:

| TF v1 concept | shinro equivalent | File |
|---|---|---|
| op nodes | ABCs: `Controller`, `Plant`, `StateEstimator`, `TrajectoryGenerator`, `PhysicsEngine` | `src/shinro/components.py` |
| op registration | `@register_controller/plant/...` + registry dicts | `src/shinro/factories/registry.py` |
| serialized graph | TOML config (`[[plants]]`, `[[controllers]]`, ...) | `src/shinro/configs/robot_config.toml` |
| executor | `RobotSim` walks TOML, resolves registry, calls `from_config` | `src/shinro/simulation/robotsim.py` |
| device placement | `ArrayBackend` (NumpyBackend / TorchBackend) | `src/shinro/utils/array_backend.py` |

The only missing piece is a **compile pass** that turns the TOML into Zig
instead of Python objects.

### 4.2 The `zir()` contract

Each component that wants to be "freezable" implements a `zir()` method
(Zig IR) returning a `ZigIR` dataclass: the Zig source for its compute step
plus an I/O signature the codegen pass uses to wire nodes together.

```python
@dataclass
class ZigIR:
    name: str          # e.g. "base_controller"
    op_kind: str       # "controller" | "plant" | "estimator" | "trajectory"
    inputs: list[str]  # upstream node names (graph edges)
    outputs: list[str] # names this writes
    n_in: int          # input vector dim (comptime-known)
    n_out: int         # output vector dim
    src: str           # Zig source body
    deps: list[str]    # extra Zig imports / fns (e.g. plant dynamics for MPPI)
```

A `ZIGCompilable` mixin marks a component as freezable. Nodes that don't
implement it fall back to a **Python callback** in the compiled loop — the
same graph-break mechanism as `torch.compile`.

### 4.3 The codegen pass

`compile_graph()` reads the same TOML and registry as `RobotSim`, but emits
Zig instead of instantiating objects:

```
   for each [[plants]] / [[estimators]] / [[controllers]] / [[trajectories]]:
     cls = registry[entry["type"]]
     inst = cls.from_config(entry)          # binds params, solves DARE, etc.
     ir = inst.zir(entry["name"], entry.get("inputs", []))
     nodes.append(ir)

   topo-sort by edges → concatenate kernels → emit control_loop_step()
   → write robot_loop.zig → zig build → robot_loop.so
```

The Python codegen is deliberately dumb: it reads the TOML, bakes parameters
as `comptime` constants, and emits **generic** Zig. Zig's compiler does the
specialization (unroll small, BLAS large, eliminate dead branches).

### 4.4 The runtime

`RobotSim` gains an `eager` flag (default `False`). In compiled mode it
dlopens `robot_loop.so` and calls `control_loop_step()` per tick. Python-side
plant objects are kept as **read-only mirrors** so existing inspection code
(`plant.get_state()`, `sim.get_state()`) keeps working unchanged.

```python
sim = RobotSim("robot_config.toml")            # compiled by default
sim.step()                                      # → FFI into Zig

sim = RobotSim("robot_config.toml", eager=True)  # Python, for debugging
sim.step()                                      # → real stack traces
```

## 5. What the emitted Zig looks like

### 5.1 LQR — fully frozen

The gain `K` is solved offline (DARE) by `from_config()` and baked in as a
`comptime` constant. The runtime loop is one unrolled matvec:

```zig
const K_lqr = [3][3]f64{
    [_]f64{ -9.516259, 0.0,       0.0       },
    [_]f64{ 0.0,       -9.516259, 0.0       },
    [_]f64{ 0.0,       0.0,       -7.071068 },
};

fn lqr_step(x: [3]f64, x_ref: [3]f64) [3]f64 {
    var u: [3]f64 = .{0} ** 3;
    inline for (0..3) |i| {
        var acc: f64 = 0;
        inline for (0..3) |j| {
            acc += K_lqr[i][j] * (x_ref[j] - x[j]);
        }
        u[i] = acc;
    }
    return u;
}
```

### 5.2 Kalman filter — matrices frozen, covariance live

`A, B, C, Q, R` are baked as constants; the error covariance `P` and gain
`K_gain` are live state. The innovation-covariance inverse uses LAPACK
(`LAPACKE_dgesv`) rather than an explicit inverse:

```zig
const c = @cImport({
    @cInclude("cblas.h");
    @cInclude("lapacke.h");
});

fn kalman_step(state: *RobotState, y: [3]f64, u: [3]f64) [3]f64 {
    const x_pred = add3(matvec3(A_kf, state.x_hat), matvec3(B_kf, u));
    const P_pred = add3x3(matmul3(matmul3(A_kf, state.P), transpose3(A_kf)), Q_kf);
    const S = add3x3(matmul3(matmul3(C_kf, P_pred), transpose3(C_kf)), R_kf);
    // solve S @ K = P_pred @ C^T via LAPACK (no explicit inverse)
    var rhs = matmul3(P_pred, transpose3(C_kf));
    var S_copy = S;
    var K_gain: [3][3]f64 = .{0} ** 3;
    inline for (0..3) |col| {
        var b_col: [3]f64 = .{0} ** 3;
        inline for (0..3) |row| { b_col[row] = rhs[row][col]; }
        var pivots: [3]i32 = .{0} ** 3;
        _ = c.LAPACKE_dgesv(c.LAPACK_ROW_MAJOR, 3, 1,
            &S_copy[0][0], 3, &pivots[0], &b_col[0], 1);
        inline for (0..3) |row| { K_gain[row][col] = b_col[row]; }
    }
    state.x_hat = add3(x_pred, matvec3(K_gain, sub3(y, matvec3(C_kf, x_pred))));
    state.P = matmul3(sub3x3(I3, matmul3(K_gain, C_kf)), P_pred);
    return state.x_hat;
}
```

### 5.3 MPPI — structure frozen, samples live

The rollout loop unrolls at comptime (`N=100, K=10, D_u=3`); the RNG and
softmax weighting run live:

```zig
const N_MPPI: usize = 100;
const K_MPPI: usize = 10;
const SIGMA2 = [3]f64{ 0.25, 0.25, 0.25 };  // noise_sigma^2, baked
const LAM: f64 = 1.0;                        // temperature, baked

fn mppi_rollout(x0: [3]f64, u_nom: *const [K_MPPI][3]f64,
                eps: *const [N_MPPI][K_MPPI][3]f64) [N_MPPI]f64 {
    var costs: [N_MPPI]f64 = .{0} ** N_MPPI;
    inline for (0..N_MPPI) |i| {
        var x = x0;
        inline for (0..K_MPPI) |k| {
            const u = [3]f64{ u_nom[k][0]+eps[i][k][0],
                              u_nom[k][1]+eps[i][k][1],
                              u_nom[k][2]+eps[i][k][2] };
            costs[i] += quad_cost_3(x, u);          // Q, R baked
            costs[i] += LAM * (u_nom[k][0]/SIGMA2[0]*eps[i][k][0]
                            +  u_nom[k][1]/SIGMA2[1]*eps[i][k][1]
                            +  u_nom[k][2]/SIGMA2[2]*eps[i][k][2]);
            x = add3(matvec3(A, x), matvec3(B, u)); // plant dynamics baked
        }
        costs[i] += quad_cost_3(x, .{0, 0, 0});
    }
    return costs;
}
```

### 5.4 The frozen entry point

```zig
pub export fn control_loop_step(state: *RobotState, engine: *EngineHandle,
                               x_ref: [3]f64) void {
    const y = read_measurement(engine);          // MuJoCo qpos, extern fn
    const x_est = kalman_step(state, y, state.u_prev);
    var u = lqr_step(x_est, x_ref);
    u = clip3(u, U_MIN, U_MAX);
    _ = base_step(state, u);                     // plant dynamics
    write_control(engine, u);                    // MuJoCo ctrl, extern fn
    state.u_prev = u;
}
```

## 6. BLAS/LAPACK via `@cImport`

Every method in `ArrayBackend` maps to a BLAS/LAPACK call, so the emitted
Zig can use the same libraries the Python path already relies on:

| `ArrayBackend` method | BLAS/LAPACK |
|---|---|
| `bk.inv(S)` | `LAPACKE_dgesv` (solve, don't invert) |
| `bk.pinv(J)` | `LAPACKE_dgels` (least squares) |
| `bk.solve(A, b)` | `LAPACKE_dgesv` |
| `bk.cholesky(P)` | `LAPACKE_dpotrf` |
| `A @ B` | `cblas_dgemm` |
| `A @ x` | `cblas_dgemv` |
| `bk.eigvals(A)` | `LAPACKE_dsyev` |

The codegen pass picks the kernel by dimension at comptime:

```zig
fn solve_linear(comptime N: usize, A: *[N][N]f64, b: *[N]f64) [N]f64 {
    if (N <= 4) {
        return solve_unrolled(N, A, b);   // hand-rolled, unrolled
    } else {
        return solve_lapack(N, A, b);     // LAPACKE_dgesv
    }
}
```

This is what makes freezing the 6-DOF arm (FK/Jacobian/IK) and MPC's lifted
QP matrices realistic — no hand-rolled 6×6 or 60×60 linear algebra.

## 7. Migration tiers

| Tier | Components | Effort | Approach |
|---|---|---|---|
| 1 | LQR, KalmanFilter, LuenbergerObserver, HolonomicMobileRobot, InvertedPendulum, CartPole, cubic/quintic trajectories | ~30 lines `zir()` each | Hand-rolled unrolled math (≤4 dims) |
| 1.5 | ArmRobot (6-DOF FK/IK/Jacobian) | ~40 lines `zir()` | BLAS/LAPACK calls |
| 2 | MPC_LTI (OSQP QP), MPPI (rollout) | ~50 lines `zir()` | Bake H/F matrices; `extern fn` to OSQP; live RNG for MPPI |
| 3 | MuJoCoEngine, LeRobotAdapter (diffusion policy) | none | `extern fn` to MuJoCo C API; Python callback for LeRobot |

Non-`ZIGCompilable` nodes become Python callbacks in the compiled loop
(graph breaks), so the system works incrementally — freeze the hot linear
path first, leave the rest as callbacks.

## 8. What we need from the Zig engineer

1. **Feasibility review** of the `zir()` contract and the emitted Zig
   patterns (comptime specialization, `inline for` vs `for`, BLAS/LAPACK
   linking via `@cImport`).
2. **Build integration** — a `build.zig` that links BLAS/LAPACK and MuJoCo
   and produces a C-ABI shared library with `control_loop_step` /
   `control_loop_reset` entry points.
3. **FFI bridge** — the Python side (`ctypes`/`cffi`) that dlopens the `.so`
   and marshals the `RobotState` struct + engine handle.
4. **Comptime specialization guidance** — where `inline for` helps vs.
   hurts (binary size, compile time, instruction cache) for the MPPI
   rollout case.
5. **A reference `zir()` implementation** for one Tier-1 component (LQR or
   KalmanFilter) to validate the pattern end-to-end.

## 9. Open questions

- **Real-time guarantee** — what's the worst-case step time of the compiled
  loop on target hardware? (No GC, no dispatch, but BLAS calls have
  variable latency.)
- **MuJoCo coupling** — should the physics engine stay Python-side (called
  via FFI from the Zig loop) or be fully inlined into the compiled loop?
- **MPPI unrolling** — is `inline for` over `N=100, K=10` worth it, or
  should the sample/horizon loops stay as regular `for`?
- **State marshalling** — the `RobotState` struct crosses the FFI boundary
  every step. Is a shared-memory buffer better than per-call marshalling?
- **Build-time budget** — how long does `zig build` take for a full graph
  with comptime specialization? Is incremental rebuild feasible when only
  one component's config changes?

---

*This document accompanies the design discussion in the repo. The Python
side (codegen pass, `zir()` contract, `RobotSim` eager flag) is sketched in
`src/shinro/codegen/` (proposed). The Zig side is entirely open for the
Zig engineer to own.*
