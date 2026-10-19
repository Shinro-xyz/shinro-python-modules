# Shinro MCP Servers

Three Model Context Protocol (MCP) servers expose the shinro control library, split by
phase of work. Both run over `stdio` — compatible with any MCP client (Claude Desktop,
Cline, custom hosts).

| Server | Console script | Tools | For |
|--------|----------------|-------|-----|
| **component** | `shinro-mcp` | 33 | Designing and running controllers, estimators, trajectories, LTI analysis |
| **codegen** | `shinro-mcp-codegen` | 16 | Tracing, lowering to Zig, verification gates, building the kernel |
| **notebook** | `shinro-mcp-notebook` | 3 | Recording experimental runs to an append-only JSONL file |

They are separate servers because an MCP client puts **every** tool schema of a
connected server into the model's context on each turn. A session that is only tuning
a controller should not carry 14 tracing/lowering/verification schemas (~48% of the
combined tool payload), and a compile session should not carry the component tools.
Tool-selection accuracy degrades as the visible tool set grows (see arXiv
`2505.10570`, `2505.03275`, `2606.15508`), so the split is a per-query-relevance
measure, not just a token one.

### Scope of the codegen server

The codegen server **ends at a verified, stamped `lib<name>.so`** plus its
deployment record. It deliberately does **not**:

- deploy the kernel to a target or board,
- drive it closed-loop against a plant,
- measure it on target (timing, jitter, memory, protocol latency).

Those belong to the consuming project — for shinro's own harness, `shinro-bench`.
Keeping the boundary here is what lets the server stay free of any robot, transport,
or hardware knowledge: it needs only your configs, the ONNX weights if any, and
`zig` on `PATH`. If you want closed-loop behaviour, drive the built `.so` yourself
(see the `u_prev` note under the external-repository gotchas) or use the simulation
runner in your own code.

## Quick Start

Both ship as console scripts (installed with the package; entry points in
`pyproject.toml`):

```bash
shinro-mcp            # component server
shinro-mcp-codegen    # codegen server
shinro-mcp-notebook   # experiment notebook

# or, if the console script is not on PATH:
python -m shinro.mcp.server
python -m shinro.mcp.codegen_server
```

Each listens on `stdio` and registers only its own tools. Connect via an MCP client
that supports the stdio transport — either server alone, or both when a session spans
design and compile.

---

## Use from your own repository

Everything below was verified from a clean virtualenv against a package install (not a
source checkout) and an unrelated working directory.

### 1. Install

**There is no PyPI release yet** (`pip install shinro` fails). Install from a wheel or
the git remote:

```bash
pip install /path/to/shinro-0.3.1.dev20-py3-none-any.whl
# or, with repo access:
pip install "shinro[onnx-rl] @ git+ssh://git@github.com/Shinro-xyz/shinro-python-modules.git"
```

Extras: `[onnx-rl]` for ONNX policies (adds `onnx`; compile-time only), `[torch]`,
`[mujoco]`, `[media]`. `zig` must be on `PATH` for `build_kernel` — tracing and
lowering do not need it.

The wheel ships the Zig runtime (`runtime/build.zig`, `lower.zig`, the baked `emosqp`
solver) and all four console scripts, so a build works from an installed package.

### 2. Write your own configs

**shinro ships no config files.** The `samples/` directory in the repository is
documentation; it is not installed. Your project owns its TOMLs — copy the sample you
need if you have the repo, otherwise write them from `docs/components.md`:

```text
your-repo/
  configs/kf.toml            # estimator (type = "KalmanFilter", ...)
  configs/lqr.toml           # controller (type = "LQR", ...)
  scenario.toml              # [controller] [estimator] [compile]
```

```toml
# scenario.toml
[controller]
type = "LQR"
config = "configs/lqr.toml"

[estimator]
type = "KalmanFilter"
config = "configs/kf.toml"

[compile]
optimize = "debug"            # "debug" | "release" (ReleaseFast)
artifact_name = "libdemo"     # kernel installs as lib/libdemo.so
n_x = 3
n_u = 3
```

Paths inside a TOML resolve **relative to the server process's working directory**
(there is no package fallback, by design). So set your client's `cwd` to your repo root.

### 3. Point your MCP client at the server

```json
{
  "mcpServers": {
    "shinro": { "command": "shinro-mcp", "cwd": "/path/to/your-repo" },
    "shinro-codegen": { "command": "shinro-mcp-codegen", "cwd": "/path/to/your-repo" }
  }
}
```

Add only the codegen server if the session is compile-only (that is the split's
purpose); add both when a session spans design and compile. Clients that gate
*project-scoped* server definitions need your approval for each new repository.

### 4. The workflow, and what a healthy run looks like

| Tool | Expected |
|------|----------|
| `generate_scenario` | `nodes_traced: 47`, `nodes_lowered: 39`, `has_solve_qp: false` |
| `build_kernel` | `exit_code: 0`, `artifact_name: "libdemo"`, `artifact_bytes > 0`, `oracle.status: "passed"`, a `master_hash` |
| `verify_deployment` | `ok: true`, `failures: []` |

The oracle tolerance is 1e-12 for non-QP graphs (1e-3 with a `.solve_qp` node); a
KF+LQR graph typically lands around 5e-16. Artifacts appear in
`build/graph/lib/libdemo.{so,manifest.json,deployment.json}`.

### 5. Gotchas specific to a foreign repository

- **CWD is the contract.** `config`, `model_path`, `scenario_path` and `out_dir` are
  all resolved against the server's working directory. A wrong `cwd` is a loud
  `FileNotFoundError`, never a silent fallback.
- **Graph names are process-local.** A graph registered by `generate_scenario` lives
  in that server process; a new session must re-generate it (deterministic) or pass
  `scenario_path` to the verification tools.
- **Build artifacts land where you point them.** `build_kernel` defaults
  `prefix = graph_dir`; pass `prefix` to separate the lowered graph from the
  deployable tree, and pass `artifact_name` if it is not `libbase` (auto-detected only
  when exactly one `.so` is present).
- **The lowering tools refuse the packaged runtime dir.** They will not write into
  `shinro/runtime/`, because that is the shipped graph.
- **Third-party components** need their module imported before the server starts —
  `shinro-compile --import your_module` does this for the CLI; for the MCP server,
  import them from a `.pth`/sitecustomize or a thin wrapper script that imports your
  module and then calls `shinro.mcp.codegen_server.main()`. Registering a custom plant
  needs a frozen `Config` dataclass plus `physics_engine`, `get_state`, `get_model`,
  `dynamics` and `step`.
- **Driving the compiled `.so` yourself: `state_inputs` includes `u_prev`.** In a
  composed graph `u_prev` is declared with the recurrent ports but has **no** `state_*`
  slice in the state buffer. Feed it the previous control; treating it as a slice
  shortens the packed input buffer by `n_u` and the kernel silently reads shifted
  garbage. `cg.state_outputs`/`state_slices` cover only the real `state_*` buffers.
- **PID requires `n_x == n_u`** (element-wise gains). A non-square plant needs LQR/MPC
  or a reduced model — see `docs/components.md`.

---

## Module layout

Two entry points over one set of domain modules under `src/shinro/mcp/`:

| Module | Contents |
|--------|----------|
| `_common.py` | The two `FastMCP` instances (`server`, `codegen_server`), the instance stores, and the conversion/analyzer helpers |
| `_controllers.py` | Controller tools (create, compute, reset, MPC constraints, PID limits, listing) |
| `_estimators.py` | Estimator tools (create, estimate, reset, listing) |
| `_trajectories.py` | Trajectory tools (create, generate, position_at, listing) |
| `_plants.py` | Plant tools (create, attach, list, model) — the model seam for nonlinear components |
| `_simulation.py` | Design-time closed-loop simulation (`simulate_scenario`) |
| `_lifecycle.py` | Instance lifecycle (list every stored object, delete one) |
| `_notebook.py` | Run-record tools (append one record, list summaries, fetch a record) |
| `notebook_server.py` | Notebook entry point: imports `_notebook`, provides `main()` |
| `_analysis.py` | LTI analysis tools (rank tests, Gramians, summary, balanced truncation) |
| `_codegen.py` | Tracing and lowering tools (trace, compose, interpret, lower, gen scenario, measure) |
| `_verification.py` | Verification tools (gate A, zig build, oracle B, stamp, verify) |
| `server.py` | Component entry point: imports its tool modules, re-exports every tool, provides `main()` |
| `codegen_server.py` | Codegen entry point: same for the compile-pipeline tools |

Each tool module decorates its functions with `@server.tool()` against one of the two
instances in `_common.py`, so a tool registers on exactly one server by importing the
matching entry module. Adding a tool means adding it to the relevant domain module —
no registry to update. Import from `shinro.mcp.server` or `shinro.mcp.codegen_server`
(not the private modules) to keep that contract.

---

## Client Configuration

The project ships with `.mcp.json` at the root — the standard file for distributing MCP config with a project. Most clients (Claude Code, OpenCode, Cline) auto-discover it.

### Claude Code / OpenCode / Cline

No setup needed — `.mcp.json` is picked up automatically. Claude Code prompts for approval on first use.

`.mcp.json` ships **only the component server** (`shinro`), because the default working
set is controller/estimator design. Add the codegen server to the same file when a
session will trace, lower, or verify:

```json
{
  "mcpServers": {
    "shinro": { "command": "shinro-mcp" },
    "shinro-codegen": { "command": "shinro-mcp-codegen" }
  }
}
```

Connecting both is fine — they are separate processes with independent state, and each
advertises only its own tools. Enable the second one when you need it rather than by
default: that is the whole point of the split.

### Claude Desktop

Add to `claude_desktop_config.json` (add a second entry from the block above for the
codegen server):

```json
{
  "mcpServers": {
    "shinro": {
      "command": "shinro-mcp"
    }
  }
}
```

### OpenCode (explicit config)

Add to `opencode.json`:

```json
{
  "mcp": {
    "shinro": {
      "type": "local",
      "command": ["shinro-mcp"]
    }
  }
}
```

### Any stdio MCP client

```json
{
  "mcpServers": {
    "shinro": {
      "command": "shinro-mcp"
    }
  }
}
```

If the `shinro-mcp` script isn't on `PATH` (e.g. an active virtualenv), point the
command at the module form instead:

```json
{
  "mcpServers": {
    "shinro": {
      "command": "python",
      "args": ["-m", "shinro.mcp.server"]
    }
  }
}
```

---

## Tool Reference

### Controllers

| Tool | Description |
|------|-------------|
| `create_controller` | Create a PID, LQR, MPC_LTI, or MPC_DeltaU controller |
| `controller_compute` | Compute a control action from a stored controller |
| `controller_reset` | Reset a controller's internal state |
| `set_mpc_constraints` | Set input constraints on any controller exposing `constraints()` — MPC_LTI, MPC_DeltaU, or your own QP controller |
| `get_mpc_constraints` | Read current constraints from such a controller |
| `set_pid_output_limits` | Set output clamping limits on a PID controller (every PID variant shares this interface) |
| `list_controller_types` | List all registered controller types |
| `list_controllers` | List all created controller instances |
| `list_instances` | List **every** stored controller, estimator, trajectory and plant, with kind, type, and attachments |
| `delete_instance` | Delete a stored instance; refuses to delete a plant that components are attached to |

**Controller types:** every registered controller — `PID`, `LQR`, `MPC_LTI`,
`MPC_DeltaU`, `MPPI`, `SMC`, `onnx_rl`, plus any you register with
`@register_controller`. `create_controller` / `controller_compute` dispatch through the
registry and by input role, so a third-party controller needs no tool changes.

The MPC constraint tools are a **capability test**, not a name allowlist: a controller
gains them by implementing `constraints(constraint_matrix, upper_bounds, lower_bounds)`
plus the `N` / `m` / `bk` attributes the default-bound helper needs. `set_pid_output_limits`
stays PID-specific, since output clamping is part of the PID contract every variant
shares.

**Example — PID:**

```json
{
  "name": "create_controller",
  "arguments": {
    "name": "my_pid",
    "type": "PID",
    "params": {
      "dt": 0.02,
      "kp": [2.0, 2.0],
      "ki": [0.1, 0.1],
      "kd": [0.5, 0.5]
    }
  }
}
```

**Example — MPC with constraints:**

```json
{
  "name": "create_controller",
  "arguments": {
    "name": "my_mpc",
    "type": "MPC_LTI",
    "params": {
      "dt": 0.02,
      "horizon": 10,
      "state_cost": [1.0, 1.0],
      "control_cost": [0.1, 0.1]
    }
  }
}
```

Then set constraints:

```json
{
  "name": "set_mpc_constraints",
  "arguments": {
    "name": "my_mpc",
    "upper": [0.5, 0.5],
    "lower": [-0.5, -0.5]
  }
}
```

**Example — Compute:**

```json
{
  "name": "controller_compute",
  "arguments": {
    "name": "my_pid",
    "state": [1.0, 0.5],
    "reference": [0.0, 0.0]
  }
}
```

Returns: `{"action": [-2.0, -1.0]}`

---

### Estimators

| Tool | Description |
|------|-------------|
| `create_estimator` | Create a KalmanFilter or LuenbergerObserver |
| `estimator_estimate` | Run one predict-update cycle |
| `estimator_reset` | Reset estimator internal state |
| `list_estimator_types` | List all registered estimator types |
| `list_estimators` | List all created estimator instances |

**Estimator types:** `KalmanFilter`, `LuenbergerObserver`

**Example — Kalman filter:**

```json
{
  "name": "create_estimator",
  "arguments": {
    "name": "my_kf",
    "type": "KalmanFilter",
    "params": {
      "dt": 0.02,
      "process_noise": [0.01, 0.01],
      "measurement_noise": [0.1, 0.1]
    }
  }
}
```

**Example — Estimate:**

```json
{
  "name": "estimator_estimate",
  "arguments": {
    "name": "my_kf",
    "measurement": [0.0, 0.0],
    "control_input": [0.0, 0.0]
  }
}
```

Returns: `{"state_estimate": [0.0, 0.0]}`

---

### Trajectories

| Tool | Description |
|------|-------------|
| `create_trajectory` | Create a cubic, quintic, waypoint, or phase-list trajectory |
| `trajectory_generate` | Compute polynomial coefficients for cubic/quintic generators |
| `trajectory_position_at` | Evaluate position, velocity, acceleration at time t |
| `list_trajectory_types` | List all registered trajectory types |
| `list_trajectories` | List all created trajectory instances |

**Trajectory types:** `cubic_segments`, `quintic_segments`, `waypoints`, `phase_list`

**Example — Cubic trajectory:**

```json
{
  "name": "create_trajectory",
  "arguments": {
    "name": "my_traj",
    "type": "cubic_segments",
    "params": {"dt": 0.02}
  }
}
```

```json
{
  "name": "trajectory_generate",
  "arguments": {
    "name": "my_traj",
    "start_position": [0.0, 0.0, 0.0],
    "end_position": [0.0, 0.24, 0.15],
    "duration": 3.0,
    "start_vel": [0.0, 0.0, 0.0],
    "end_vel": [0.0, 0.0, 0.0]
  }
}
```

```json
{
  "name": "trajectory_position_at",
  "arguments": {
    "name": "my_traj",
    "t": 1.5
  }
}
```

Returns: `{"position": [...], "velocity": [...], "acceleration": [...]}`

---

### Plants

A plant supplies what non-linear components cannot carry in their own config: SMC's
`f_x`/`g_x`, MPPI's dynamics and cost, and the EKF/UKF process model. Linear
components (LQR, PID, MPC, KF, Luenberger) need none of this.

| Tool | Description |
|------|-------------|
| `create_plant` | Create a plant from a TOML config or inline params |
| `attach_plant` | Attach a plant to a controller or estimator |
| `plant_model` | Discrete-time `(A, B)` + dims; pass `state` for the true nonlinear `f(x,u)` and `(f_x, g_x)` |
| `list_plant_types` | Registered plant types |
| `list_plants` | Created plants, their dims, and what they are attached to |

`controller_compute` maps its inputs to the controller's `compute()` parameters **by
role** (state / reference / `u_prev` / model / free host input) using the same
dispatcher as the simulation runner and gate A. So once a plant is attached:

- **SMC** receives `f_x`/`g_x` derived from the plant — no extra arguments from you.
- **MPPI** rolls out the plant's dynamics; `epsilon` is left `None` so it samples.
- **EKF/UKF** get their process model; `estimate()` is unchanged.

A non-linear controller with **no** plant attached reports a clear error naming the
plant rather than a `TypeError` about `g_x`.

**Example — SMC on an inverted pendulum:**

```json
{"name": "create_plant", "arguments": {"name": "pend", "config_path": "configs/pendulum.toml"}}
{"name": "create_controller", "arguments": {"name": "smc", "type": "SMC", "params": {"dt": 0.02, "c": [1.0, 2.0], "k1": 1.0, "phi": 0.1}}}
{"name": "attach_plant", "arguments": {"component_name": "smc", "plant_name": "pend"}}
{"name": "controller_compute", "arguments": {"name": "smc", "state": [0.1, 0.0], "reference": [0.0, 0.0]}}
```

`attach_plant` on a self-contained component (e.g. LQR) is refused, because its model
is already in its config.

**Four model hooks, and what each is for.** There is no `get_dynamics`; a plant
exposes these instead:

| Hook | Meaning | Consumed by |
|------|---------|-------------|
| `get_model()` | discrete `(A, B)` — **linearized** at the operating point, Euler-discretized at `dt` | LQR, MPC, `plant_model` |
| `dynamics(x, u)` | continuous-time vector field `f(x, u)` — the exact nonlinear model | EKF/UKF process model, MPPI rollouts |
| `model_terms(x, n_u)` | control-affine pair `(f_x, g_x)` at `x`; `f = f_x + g_x·u` | SMC |
| `control_matrix(x, u)` | `g(x, u) = ∂f/∂u` | `model_terms` |

`plant_model(name)` returns **one stable shape** with both views and the exact
evaluation point it used — no mode-switching and no hidden default:

```json
{
  "n_x": 2, "n_u": 1, "state": [0.0, 0.0],
  "linearization": { "A": [[1.0, 0.01], [0.1962, 1.0]], "B": [[0.0], [0.4]] },
  "evaluation": {
    "x": [0.1, 0.0], "u": [0.5], "u_source": "caller",
    "dynamics": [0.0, 21.9587],
    "model_terms": { "f_x": [0.0, 1.9587], "g_x": [[0.0], [40.0]] },
    "affine_exact": true,
    "affine_residual": 2e-11
  }
}
```

- `linearization` is the plant's default operating point (`get_model()`).
- `evaluation` is at `x` (default: the plant's current state) and `u` (default:
  `operating_control()`, reported as `u_source`: `caller` / `operating_control` /
  `zero`). A rotor plant's default is therefore hover, not zero — `control_matrix(x, 0)`
  is identically zero there, which would report a `g_x` that looks broken.
- **`affine_exact` answers the question the prose used to answer:** it probes whether
  `g` depends on `u` at this `x`. `true` means `f = f_x + g_x·u` holds for all `u`
  (force/torque plants); `false` means it is the tangent at the operating point
  (rotor thrust ~ ω²), exact only there — `affine_residual` is `‖f − (f_x + g_x·u)‖∞`.

---

### Simulation

Runs a scenario closed-loop in numpy and reports how well it tracked — what tuning
needs. Change a gain or weight, re-run, compare `tracking.final` / `tracking.max`.

| Tool | Description |
|------|-------------|
| `simulate_scenario` | Run a scenario `steps` times; returns final state, tracking error, action magnitude, and a downsampled trajectory |

It wraps the library's own `run_scenario`, so a number here means the same thing as a
number anywhere else in the project. It is **numpy** simulation: it neither builds nor
loads a kernel — the compiled path stays on the codegen server, and the oracle is what
proves the two agree.

The scenario must declare a `[plant]` (or `[sim]`) — compile-only scenarios, which
declare their dims in `[compile]`, cannot be simulated.

```json
{
  "name": "simulate_scenario",
  "arguments": {"scenario_path": "scenarios/pendulum.toml", "steps": 800, "seed": 0}
}
```

```json
{"steps": 800, "dt": 0.01,
 "final": {"plant_state": [0.003, -0.0056], "estimated": [...], "reference": [0.0, 0.0]},
 "tracking": {"final": 0.00557, "max": 0.20052, "mean": 0.05589},
 "max_abs_action": 0.1086, "all_finite": true,
 "check": {"ok": null},
 "trajectory": {"t": [0.0, 2.66, 5.32, 7.99], "plant_state": [...], ...}}
```

`tracking` is the L∞ distance from the reference; `check` applies the scenario's own
`[scenario.tolerance]` block when it declares one (otherwise `ok` is `null` — the tool
does not invent a threshold). Pass `samples: 0` for summary metrics only.

---

### System Analysis

| Tool | Description |
|------|-------------|
| `analyze_controllability` | Kalman rank test for controllability |
| `analyze_observability` | Kalman rank test for observability |
| `gramian_continuous` | Infinite-horizon continuous-time Gramians + Hankel SVs |
| `gramian_discrete` | Infinite-horizon discrete-time Gramians + Hankel SVs |
| `gramian_finite` | Finite-horizon Gramians via ODE integration |
| `system_summary` | Human-readable system properties report |
| `balanced_truncation` | Order-r reduced model via balanced truncation |

**Example — Controllability:**

```json
{
  "name": "analyze_controllability",
  "arguments": {
    "A": [[0, 1], [0, 0]],
    "B": [[0], [1]]
  }
}
```

Returns: `{"controllable": true, "rank": 2, "n": 2, "condition": 4.0}`

**Example — Balanced truncation:**

```json
{
  "name": "balanced_truncation",
  "arguments": {
    "A": [[-1, 0], [0, -2]],
    "B": [[1], [1]],
    "C": [[1, 1]],
    "r": 1
  }
}
```

Returns: `{"Ar": [[-1.0]], "Br": [[1.414]], "Cr": [[1.414]], "Dr": [[0]], "reduced_order": 1, "error_bound": 0.0}`

---

### Codegen — trace and lower

The zig-free half of the compile pipeline (`src/shinro/codegen/` +
`shinro.scenarios.gen`). Graphs are traced and stored by name in their own
registry, so a graph and a controller may share a name.

| Tool | Description |
|------|-------------|
| `list_codegen_ops` | Registered graph ops + scenario graph recipes |
| `check_component` | Construct a config and report its trace contract (no trace) |
| `trace_component` | Trace one component into a graph; reports discovered `state_attrs` + shapes |
| `build_composed_graph` | Trace an estimator + controller into one closed-loop step graph |
| `graph_summary` | Ports, op histogram, and lowerability of a stored graph |
| `interpret_graph` | Run a stored graph on numpy inputs (the Python oracle) |
| `measure_graph` | Static size metrics: host buffers, VM stack buffer, baked blobs |
| `lower_graph` | Serialize a stored graph to `graph_data.zig` + manifest (no compiler) |
| `generate_scenario` | The gen stage: trace + compose + lower an isolated scenario graph |
| `list_graphs` | List the graphs held in this session (name, trace size, ports) |
| `delete_graph` | Drop a stored graph — already-built artifacts are untouched |

**Two-stage workflow:** `generate_scenario` (or `build_composed_graph` +
`lower_graph`) produces the graph; `build_kernel` compiles it.

**Node counts.** `nodes_traced` is the captured graph before lowering folds
constants/canonicalizes — this is what `interpret_graph` (and gate A) run on.
`nodes_lowered` is the table the Zig VM actually compiles, and can be smaller.

**Stored graphs are process-local — and now evictable.** A graph registered by
`generate_scenario` lives in that server process's registry, and holds a full copy of
the traced graph. A new client session must re-generate it (cheap, deterministic) or
pass `scenario_path` to the verification tools; a long session should `list_graphs`
and `delete_graph` what it no longer needs.

**Example — trace a Kalman filter:**

```json
{
  "name": "trace_component",
  "arguments": {
    "name": "kf",
    "config_path": "samples/estimators/kalman_base.toml",
    "input_shapes": {"measurement": [3], "control_input": [3]}
  }
}
```

Returns the ports, the op histogram, and `state_attrs` / `state_shapes` — pass the
latter back as `state_shapes` to re-trace with `state_*` ports instead of
freezing the recurrence at trace-time values.

**Safety:** the lowering tools refuse any `out_dir` inside the packaged
`shinro/runtime/` directory, because `runtime/graph_data.zig` is the shipped
graph that `make zig-gen` regenerates.

---

### Verification — gates and the zig stage

`src/shinro/verification/` plus the build stage. Gate A and oracle B are
complementary: gate A proves the *graph* reproduces the components, oracle B
proves the *Zig VM* reproduces the graph. Only both make a stamp's "verified"
claim true.

| Tool | Description |
|------|-------------|
| `run_gate_a` | `interpret(graph)` vs the live estimator + controller (no zig) |
| `build_kernel` | Integrity + gate A + `zig build` + oracle B + stamp + verify |
| `run_oracle` | `.so` vs `interpret()` on random inputs (oracle B) |
| `stamp_deployment` | Write the deployment record (master hash over the four slots) |
| `verify_deployment` | Re-hash artifacts against the record (drift gate) |

`build_kernel` requires `zig` on PATH and returns an `exit_code` / `status`
(`ok`, `usage_error`, `oracle_mismatch`, `build_or_verify_failed`,
`zig_missing`). Omitting `scenario_path` builds and stamps *unverified* — the
record then carries `oracle.status = "not_run"`. The artifact stem is resolved
from `lib/*.so` after the build, so a scenario whose `[compile].artifact_name`
is not `libbase` still reports its oracle and master hash.

**Known gap — policy-only graphs have no gate A.** With `[estimator]` absent
there is no live component pair to drive, so oracle B is the only equivalence
check. Oracle B compares the `.so` against `interpret()` of *the same graph*, so
it cannot catch a mis-wired state port or a wrong observation encoder — those
would have to be caught against the eager `onnx_rl` adapter. For policies, also
read the port layout from `graph_summary` and confirm the recurrence yourself.

The pipeline narrates on stdout, which on a stdio server is the JSON-RPC
channel; these tools capture it and return it as a `log` field.

**Example — build and verify the base kernel:**

```json
{
  "name": "build_kernel",
  "arguments": {
    "graph_dir": "build/mcp/base",
    "scenario_path": "tests/integration/scenarios/base_tracking.toml"
  }
}
```

Returns `exit_code`, `status`, `artifact`, `artifact_bytes`, the recorded
`oracle`, the `master_hash`, and the captured `log`.

---

## Notebook

The **notebook** server (``shinro-mcp-notebook``) records experimental runs so trials
outlive the session that produced them. Records go to an **append-only JSONL file you
name** — one JSON object per line, nothing rewritten, so a crashed session cannot
corrupt earlier entries.

| Tool | Description |
|------|-------------|
| `record_run` | Append one record: `path`, `name`, `components`, `metrics`, `notes`, `tags` |
| `list_runs` | Compact summaries (id, name, timestamp, tags, metric names), oldest first |
| `get_run` | One full record, by `id` first, then by `name` (most recent) |

The server fills **provenance** for you — `created_at`, `shinro_version`, `python`,
`platform`, `cwd`, and (inside a git repo) `git_sha` / `git_dirty`. That is the part an
agent could forget or get wrong, and it is what makes a record traceable to the code
that produced it.

**Record a run:**

```json
{
  "name": "record_run",
  "arguments": {
    "path": "runs/experiments.jsonl",
    "name": "pid-motor-ref2",
    "components": {
      "plant": {"type": "DCMotor", "config": "configs/plant.toml"},
      "controller": {"type": "PID", "config": "configs/pid.toml"}
    },
    "metrics": {"final_error": 0.0097, "steps": 600, "artifact_bytes": 11706317},
    "tags": ["motor", "baseline"]
  }
}
```

```json
{"recorded": "20261019T154518-a1b2", "name": "pid-motor-ref2", "path": "runs/experiments.jsonl"}
```

Notes: the path is resolved against the server's working directory (parent directories
are created); a missing notebook reads as an empty list rather than an error; and a
malformed line is skipped and counted in `skipped_lines` so one bad append does not make
the rest unreadable.

---

## Test Coverage Summary

The test suites (`tests/test_mcp_server.py`, `tests/test_mcp_plants.py`,
`tests/test_mcp_lifecycle.py`, `tests/test_mcp_notebook.py`,
`tests/test_mcp_simulation.py`, `tests/test_mcp_codegen.py`,
`tests/test_mcp_verification.py`) cover:

| Category | Tests | What's tested |
|----------|-------|---------------|
| **Controller creation** | 8 | Inline params, config files, duplicate names, unknown types, missing params, nonexistent config, missing type field, incomplete params |
| **Controller compute** | 10 | PID/LQR/MPC action dimensions, MPC_DeltaU with/without u_prev, unknown names, wrong dimensions, empty state, default reference, integral accumulation, broadcasting |
| **Controller reset** | 3 | PID clears integral, unknown name, LQR no-op |
| **Controller listing** | 3 | Types include all, empty store, name-to-type mapping |
| **Malicious edge cases** | 16 | NaN/Inf in state, very large values/gains, negative/zero gains, empty/long/unicode/special names, 100 controllers, string/None in state, negative/zero horizon MPC, negative dt, zero dt, negative state/control cost LQR, large horizon MPC timeout, large state dimension, MPC constraints |
| **Estimator creation** | 5 | Inline Kalman/Luenberger, config file, unknown type, no params, nonexistent config |
| **Estimator estimate** | 5 | Kalman/Luenberger returns estimate, convergence with noiseless measurements, unknown name, wrong dimension |
| **Estimator reset** | 3 | Kalman clears state, Luenberger clears state, unknown name |
| **Estimator listing** | 3 | Types include all, empty store, name-to-type mapping |
| **Trajectory creation** | 6 | Config file, cubic/quintic inline, unknown type, no params, nonexistent config |
| **Trajectory generate** | 4 | Cubic/quintic success, unknown name, config-based error |
| **Trajectory position_at** | 6 | Returns pos/vel/acc, start/end match, unknown name, before generate, time clamping |
| **Trajectory listing** | 3 | Types include all, empty store, name-to-type mapping |
| **System analysis** | 20 | Controllability (5), observability (4), continuous Gramians (5), discrete Gramians (3), finite Gramians (3), system summary (2), balanced truncation (4) |
| **MPC constraints** | 8 | Set/get on MPC_LTI/DeltaU, custom matrix, non-MPC error, unknown name, bounds respected in compute |
| **PID output limits** | 5 | Set success, clamping, anti-windup, non-PID error, unknown name, config-based limits |

**Total: 246 tests** across the seven suites (247 collected; `make test` deselects
the large-horizon MPC timeout test)

Codegen (`tests/test_mcp_codegen.py`, 31) and verification
(`tests/test_mcp_verification.py`, 21) add:

| Category | Tests | What's tested |
|----------|-------|---------------|
| **Introspection** | 3 | Op/recipe inventory, contract probe, missing config |
| **Tracing** | 6 | KF state discovery + shapes, op capture, re-trace with state ports, wrong-rank error, string/int shape specs, unknown type |
| **Compose + interpret** | 9 | Port layout, half-specified limits, zero-error control, flat-value reshaping, missing/wrong-size ports, unknown names, summary |
| **Lowering** | 6 | Writes graph + manifest, manifest matches disk, provenance, refuses runtime dir + subdirs, leaves shipped graph byte-identical, unknown name |
| **Gen stage** | 4 | Isolated graph, interpretable output, missing scenario, runtime-dir guard |
| **Measure** | 2 | Buffer/blob metrics, unknown name |
| **Store isolation** | 1 | A graph and a controller may share a name |
| **Gate A** | 4 | Passes on base, reports graph size, rejects policy-only, missing scenario |
| **Zig build** | 6 | Exit 0, passed oracle, deployment record, captured log, zig-missing, no-graph usage error |
| **Oracle B** | 4 | Passes on built kernel, stored-graph name, missing binary, missing graph source |
| **Stamp + verify** | 7 | Roundtrip hash, supplied oracle block, missing binary, verify ok, binary drift, graph drift, missing record |

### Edge cases covered

- **Numerical:** NaN, Inf, float max, negative values, zero values
- **Dimensional:** Wrong state/measurement dimensions, empty vectors, broadcasting
- **Naming:** Empty string, 10000-char names, unicode, special characters
- **Resource:** 100 concurrent controllers, 1000-dimensional state, large MPC horizon rejection
- **Stateful:** Integral accumulation, anti-windup, reset clears state, convergence over time
- **Error handling:** Unknown types, missing params, nonexistent files, type mismatches, solver failures
