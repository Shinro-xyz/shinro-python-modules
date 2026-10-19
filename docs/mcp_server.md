# Shinro MCP Servers

Two Model Context Protocol (MCP) servers expose the shinro control library, split by
phase of work. Both run over `stdio` — compatible with any MCP client (Claude Desktop,
Cline, custom hosts).

| Server | Console script | Tools | For |
|--------|----------------|-------|-----|
| **component** | `shinro-mcp` | 25 | Designing and running controllers, estimators, trajectories, LTI analysis |
| **codegen** | `shinro-mcp-codegen` | 14 | Tracing, lowering to Zig, verification gates, building the kernel |

They are separate servers because an MCP client puts **every** tool schema of a
connected server into the model's context on each turn. A session that is only tuning
a controller should not carry 14 tracing/lowering/verification schemas (~48% of the
combined tool payload), and a compile session should not carry the component tools.
Tool-selection accuracy degrades as the visible tool set grows (see arXiv
`2505.10570`, `2505.03275`, `2606.15508`), so the split is a per-query-relevance
measure, not just a token one.

## Quick Start

Both ship as console scripts (installed with the package; entry points in
`pyproject.toml`):

```bash
shinro-mcp            # component server
shinro-mcp-codegen    # codegen server

# or, if the console script is not on PATH:
python -m shinro.mcp.server
python -m shinro.mcp.codegen_server
```

Each listens on `stdio` and registers only its own tools. Connect via an MCP client
that supports the stdio transport — either server alone, or both when a session spans
design and compile.

---

## Module layout

Two entry points over one set of domain modules under `src/shinro/mcp/`:

| Module | Contents |
|--------|----------|
| `_common.py` | The two `FastMCP` instances (`server`, `codegen_server`), the instance stores, and the conversion/analyzer helpers |
| `_controllers.py` | Controller tools (create, compute, reset, MPC constraints, PID limits, listing) |
| `_estimators.py` | Estimator tools (create, estimate, reset, listing) |
| `_trajectories.py` | Trajectory tools (create, generate, position_at, listing) |
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
| `set_mpc_constraints` | Set input constraints on an MPC controller |
| `get_mpc_constraints` | Read current MPC constraints |
| `set_pid_output_limits` | Set output clamping limits on a PID controller |
| `list_controller_types` | List all registered controller types |
| `list_controllers` | List all created controller instances |

**Controller types:** `PID`, `LQR`, `MPC_LTI`, `MPC_DeltaU`

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

**Two-stage workflow:** `generate_scenario` (or `build_composed_graph` +
`lower_graph`) produces the graph; `build_kernel` compiles it.

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
record then carries `oracle.status = "not_run"`.

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

## Test Coverage Summary

The test suites (`tests/test_mcp_server.py`, `tests/test_mcp_codegen.py`,
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

**Total: 178 tests** across the three suites (179 collected; `make test` deselects
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
