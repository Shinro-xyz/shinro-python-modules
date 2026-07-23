# Shinro MCP Server

A Model Context Protocol (MCP) server exposing the shinro control library as callable tools. Runs over `stdio` transport — compatible with any MCP client (Claude Desktop, Cline, custom hosts).

## Quick Start

```bash
python shinro_mcp_server.py
```

The server listens on `stdio` and registers all tools below. Connect via an MCP client that supports the stdio transport.

---

## Client Configuration

The project ships with `.mcp.json` at the root — the standard file for distributing MCP config with a project. Most clients (Claude Code, OpenCode, Cline) auto-discover it.

### Claude Code / OpenCode / Cline

No setup needed — `.mcp.json` is picked up automatically. Claude Code prompts for approval on first use.

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "shinro": {
      "command": "python",
      "args": ["/absolute/path/to/shinro-python-modules/shinro_mcp_server.py"]
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
      "command": ["python", "shinro_mcp_server.py"]
    }
  }
}
```

### Any stdio MCP client

```json
{
  "mcpServers": {
    "shinro": {
      "command": "python",
      "args": ["/path/to/shinro_mcp_server.py"]
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

## Test Coverage Summary

The test suite (`tests/test_mcp_server.py`, 1213 lines) covers:

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

**Total: 108 tests**

### Edge cases covered

- **Numerical:** NaN, Inf, float max, negative values, zero values
- **Dimensional:** Wrong state/measurement dimensions, empty vectors, broadcasting
- **Naming:** Empty string, 10000-char names, unicode, special characters
- **Resource:** 100 concurrent controllers, 1000-dimensional state, large MPC horizon rejection
- **Stateful:** Integral accumulation, anti-windup, reset clears state, convergence over time
- **Error handling:** Unknown types, missing params, nonexistent files, type mismatches, solver failures
