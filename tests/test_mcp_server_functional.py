"""Functional tests for the MCP server — spawns the actual process and talks JSON-RPC over stdio.

Tests verify protocol-level correctness: server starts, responds to tools/list,
handles tool calls with correct JSON-RPC structure, and handles errors gracefully.
"""

import json
import subprocess
import sys
import pytest
from pathlib import Path

SERVER_PATH = Path(__file__).resolve().parent.parent / "shinro_mcp_server.py"


@pytest.fixture
def server():
    """Spawn the MCP server as a subprocess, perform initialize handshake, yield a client helper."""
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _next_id = [0]

    def _request(method: str, params: dict | None = None) -> dict:
        _next_id[0] += 1
        req = json.dumps({
            "jsonrpc": "2.0",
            "id": _next_id[0],
            "method": method,
            "params": params or {},
        })
        proc.stdin.write(req + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        return json.loads(line.strip())

    def _notify(method: str, params: dict | None = None) -> None:
        """Send a notification (no response expected)."""
        req = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        })
        proc.stdin.write(req + "\n")
        proc.stdin.flush()

    # Perform MCP initialize handshake
    init_resp = _request("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0"},
    })
    assert "result" in init_resp, f"Initialize failed: {init_resp}"

    # Notify initialized (no response expected)
    _notify("notifications/initialized")

    yield _request

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class TestServerProtocol:
    """Verify the server starts and speaks correct JSON-RPC."""

    def test_tools_list_returns_all_15_tools(self, server):
        """tools/list returns 15 tools with expected names."""
        resp = server("tools/list")
        assert resp["jsonrpc"] == "2.0"
        assert "id" in resp
        assert "result" in resp
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert len(tools) == 15
        assert "create_controller" in tool_names
        assert "controller_compute" in tool_names
        assert "controller_reset" in tool_names
        assert "list_controller_types" in tool_names
        assert "list_controllers" in tool_names
        assert "create_estimator" in tool_names
        assert "estimator_estimate" in tool_names
        assert "estimator_reset" in tool_names
        assert "list_estimator_types" in tool_names
        assert "list_estimators" in tool_names
        assert "create_trajectory" in tool_names
        assert "trajectory_generate" in tool_names
        assert "trajectory_position_at" in tool_names
        assert "list_trajectory_types" in tool_names
        assert "list_trajectories" in tool_names

    def test_tools_list_tool_has_input_schema(self, server):
        """Each tool in tools/list has an inputSchema with properties."""
        resp = server("tools/list")
        for tool in resp["result"]["tools"]:
            assert "inputSchema" in tool
            assert "properties" in tool["inputSchema"]

    def test_unknown_method_returns_error(self, server):
        """Calling a nonexistent method returns a JSON-RPC error."""
        resp = server("nonexistent_method")
        assert "error" in resp

    def test_missing_required_args_returns_error(self, server):
        """Calling a tool without required args returns an error."""
        resp = server("tools/call", {
            "name": "controller_compute",
            "arguments": {},
        })
        assert "error" in resp or "isError" in resp.get("result", {})

    def test_response_id_matches_request_id(self, server):
        """The response id matches the request id."""
        resp = server("tools/list")
        assert resp["id"] == 2  # id 1=initialize, notification has no id, 2=this call


class TestControllerFunctional:
    """End-to-end controller flow through the MCP server."""

    def test_create_pid_and_compute(self, server):
        """Create a PID controller inline, compute an action, verify result."""
        resp = server("tools/call", {
            "name": "create_controller",
            "arguments": {
                "name": "pid1",
                "type": "PID",
                "params": {"dt": 0.02, "kp": [2.0], "ki": [0.1], "kd": [0.5]},
            },
        })
        assert "Created PID controller 'pid1'" in resp["result"]["content"][0]["text"]

        resp = server("tools/call", {
            "name": "controller_compute",
            "arguments": {"name": "pid1", "state": [1.0], "reference": [0.0]},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "action" in data
        assert len(data["action"]) == 1

    def test_create_lqr_from_config_and_compute(self, server):
        """Create an LQR from config file, compute, verify action."""
        resp = server("tools/call", {
            "name": "create_controller",
            "arguments": {"name": "lqr1", "config_path": "configs/controllers/lqr_base.toml"},
        })
        assert "Created" in resp["result"]["content"][0]["text"]

        resp = server("tools/call", {
            "name": "controller_compute",
            "arguments": {"name": "lqr1", "state": [0.1, 0.2, 0.05], "reference": [0.0, 0.0, 0.0]},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "action" in data
        assert len(data["action"]) == 3

    def test_create_mpc_and_compute(self, server):
        """Create MPC_DeltaU inline, compute, verify action respects constraints."""
        resp = server("tools/call", {
            "name": "create_controller",
            "arguments": {
                "name": "mpc1",
                "type": "MPC_DeltaU",
                "params": {
                    "dt": 0.02, "horizon": 5,
                    "state_cost": [1.0, 1.0],
                    "control_cost": [0.1, 0.1],
                    "delta_u_penalty": [0.5, 0.5],
                    "constraints": {
                        "upper": [0.5, 0.5, 0.5, 0.5],
                        "lower": [-0.5, -0.5, -0.5, -0.5],
                    },
                },
            },
        })
        assert "Created" in resp["result"]["content"][0]["text"]

        resp = server("tools/call", {
            "name": "controller_compute",
            "arguments": {"name": "mpc1", "state": [10.0, 10.0], "u_prev": [0.0, 0.0]},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert all(abs(v) <= 0.5 + 1e-4 for v in data["action"])

    def test_controller_reset(self, server):
        """Create a PID, compute, reset, verify success."""
        server("tools/call", {
            "name": "create_controller",
            "arguments": {
                "name": "pid2", "type": "PID",
                "params": {"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
            },
        })
        resp = server("tools/call", {
            "name": "controller_reset",
            "arguments": {"name": "pid2"},
        })
        assert "reset" in resp["result"]["content"][0]["text"].lower()

    def test_list_controller_types(self, server):
        """list_controller_types returns registered types."""
        resp = server("tools/call", {
            "name": "list_controller_types",
            "arguments": {},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "PID" in data["controller_types"]
        assert "LQR" in data["controller_types"]

    def test_list_controllers(self, server):
        """list_controllers returns created instances."""
        server("tools/call", {
            "name": "create_controller",
            "arguments": {
                "name": "pid3", "type": "PID",
                "params": {"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
            },
        })
        resp = server("tools/call", {
            "name": "list_controllers",
            "arguments": {},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "pid3" in data["controllers"]


class TestEstimatorFunctional:
    """End-to-end estimator flow through the MCP server."""

    def test_create_kalman_and_estimate(self, server):
        """Create a KalmanFilter inline, estimate, verify state_estimate."""
        resp = server("tools/call", {
            "name": "create_estimator",
            "arguments": {
                "name": "kf1", "type": "KalmanFilter",
                "params": {"dt": 0.02, "process_noise": [0.01, 0.01], "measurement_noise": [0.1, 0.1]},
            },
        })
        assert "Created" in resp["result"]["content"][0]["text"]

        resp = server("tools/call", {
            "name": "estimator_estimate",
            "arguments": {"name": "kf1", "measurement": [0.0, 0.0], "control_input": [0.0, 0.0]},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "state_estimate" in data
        assert len(data["state_estimate"]) == 2

    def test_create_luenberger_and_estimate(self, server):
        """Create a LuenbergerObserver inline, estimate, verify state_estimate."""
        resp = server("tools/call", {
            "name": "create_estimator",
            "arguments": {
                "name": "lo1", "type": "LuenbergerObserver",
                "params": {"dt": 0.02, "observer_gain": [0.8, 0.8]},
            },
        })
        assert "Created" in resp["result"]["content"][0]["text"]

        resp = server("tools/call", {
            "name": "estimator_estimate",
            "arguments": {"name": "lo1", "measurement": [1.0, 2.0], "control_input": [0.0, 0.0]},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "state_estimate" in data

    def test_estimator_reset(self, server):
        """Create an estimator, estimate, reset, verify success."""
        server("tools/call", {
            "name": "create_estimator",
            "arguments": {
                "name": "kf2", "type": "KalmanFilter",
                "params": {"dt": 0.02, "process_noise": [0.01], "measurement_noise": [0.1]},
            },
        })
        resp = server("tools/call", {
            "name": "estimator_reset",
            "arguments": {"name": "kf2"},
        })
        assert "reset" in resp["result"]["content"][0]["text"].lower()

    def test_list_estimator_types(self, server):
        """list_estimator_types returns registered types."""
        resp = server("tools/call", {
            "name": "list_estimator_types",
            "arguments": {},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "KalmanFilter" in data["estimator_types"]
        assert "LuenbergerObserver" in data["estimator_types"]


class TestTrajectoryFunctional:
    """End-to-end trajectory flow through the MCP server."""

    def test_create_cubic_generate_and_position_at(self, server):
        """Create cubic inline, generate, position_at, verify pos/vel/acc."""
        resp = server("tools/call", {
            "name": "create_trajectory",
            "arguments": {"name": "cubic1", "type": "cubic_segments", "params": {"dt": 0.02}},
        })
        assert "Created" in resp["result"]["content"][0]["text"]

        resp = server("tools/call", {
            "name": "trajectory_generate",
            "arguments": {
                "name": "cubic1",
                "start_position": [0.0, 0.0, 0.0],
                "end_position": [0.0, 0.24, 0.15],
                "duration": 3.0,
                "start_vel": [0.0, 0.0, 0.0],
                "end_vel": [0.0, 0.0, 0.0],
            },
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert data["status"] == "generated"

        resp = server("tools/call", {
            "name": "trajectory_position_at",
            "arguments": {"name": "cubic1", "t": 1.5},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "position" in data
        assert "velocity" in data
        assert "acceleration" in data
        assert len(data["position"]) == 3

    def test_create_quintic_generate_and_position_at(self, server):
        """Create quintic inline, generate, position_at, verify pos/vel/acc."""
        resp = server("tools/call", {
            "name": "create_trajectory",
            "arguments": {"name": "quint1", "type": "quintic_segments", "params": {"dt": 0.02}},
        })
        assert "Created" in resp["result"]["content"][0]["text"]

        resp = server("tools/call", {
            "name": "trajectory_generate",
            "arguments": {
                "name": "quint1",
                "start_position": [0.0, 0.0, 0.0],
                "end_position": [0.0, 0.15, 0.10],
                "duration": 2.0,
            },
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert data["status"] == "generated"

        resp = server("tools/call", {
            "name": "trajectory_position_at",
            "arguments": {"name": "quint1", "t": 1.0},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "position" in data

    def test_create_trajectory_from_config(self, server):
        """Create a trajectory from a config file."""
        resp = server("tools/call", {
            "name": "create_trajectory",
            "arguments": {"name": "t1", "config_path": "configs/trajectories/arm_extension.toml"},
        })
        assert "Created" in resp["result"]["content"][0]["text"]

    def test_list_trajectory_types(self, server):
        """list_trajectory_types returns registered types."""
        resp = server("tools/call", {
            "name": "list_trajectory_types",
            "arguments": {},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert "cubic_segments" in data["trajectory_types"]
        assert "quintic_segments" in data["trajectory_types"]


class TestCrossComponentFunctional:
    """Verify multiple component types work together in the same session."""

    def test_controller_and_estimator_together(self, server):
        """Create a controller and estimator, use both, verify they don't interfere."""
        server("tools/call", {
            "name": "create_controller",
            "arguments": {
                "name": "ctrl", "type": "PID",
                "params": {"dt": 0.02, "kp": [2.0], "ki": [0.0], "kd": [0.0]},
            },
        })
        server("tools/call", {
            "name": "create_estimator",
            "arguments": {
                "name": "est", "type": "KalmanFilter",
                "params": {"dt": 0.02, "process_noise": [0.01], "measurement_noise": [0.1]},
            },
        })

        resp = server("tools/call", {
            "name": "controller_compute",
            "arguments": {"name": "ctrl", "state": [1.0], "reference": [0.0]},
        })
        ctrl_data = json.loads(resp["result"]["content"][0]["text"])
        assert "action" in ctrl_data

        resp = server("tools/call", {
            "name": "estimator_estimate",
            "arguments": {"name": "est", "measurement": [0.0], "control_input": [0.0]},
        })
        est_data = json.loads(resp["result"]["content"][0]["text"])
        assert "state_estimate" in est_data

    def test_all_three_component_types(self, server):
        """Create a controller, estimator, and trajectory — use all three."""
        server("tools/call", {
            "name": "create_controller",
            "arguments": {
                "name": "c", "type": "PID",
                "params": {"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
            },
        })
        server("tools/call", {
            "name": "create_estimator",
            "arguments": {
                "name": "e", "type": "LuenbergerObserver",
                "params": {"dt": 0.02, "observer_gain": [0.8]},
            },
        })
        server("tools/call", {
            "name": "create_trajectory",
            "arguments": {"name": "t", "type": "cubic_segments", "params": {"dt": 0.02}},
        })
        server("tools/call", {
            "name": "trajectory_generate",
            "arguments": {
                "name": "t",
                "start_position": [0.0], "end_position": [1.0], "duration": 2.0,
                "start_vel": [0.0], "end_vel": [0.0],
            },
        })

        resp = server("tools/call", {
            "name": "controller_compute",
            "arguments": {"name": "c", "state": [0.5], "reference": [0.0]},
        })
        assert "action" in json.loads(resp["result"]["content"][0]["text"])

        resp = server("tools/call", {
            "name": "estimator_estimate",
            "arguments": {"name": "e", "measurement": [0.5], "control_input": [0.0]},
        })
        assert "state_estimate" in json.loads(resp["result"]["content"][0]["text"])

        resp = server("tools/call", {
            "name": "trajectory_position_at",
            "arguments": {"name": "t", "t": 1.0},
        })
        assert "position" in json.loads(resp["result"]["content"][0]["text"])

    def test_multiple_requests_in_sequence(self, server):
        """Send 5 requests sequentially, verify all succeed."""
        for i in range(5):
            resp = server("tools/call", {
                "name": "create_controller",
                "arguments": {
                    "name": f"pid_seq_{i}", "type": "PID",
                    "params": {"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
                },
            })
            assert "Created" in resp["result"]["content"][0]["text"]

        resp = server("tools/call", {
            "name": "list_controllers",
            "arguments": {},
        })
        data = json.loads(resp["result"]["content"][0]["text"])
        assert len(data["controllers"]) == 5


class TestErrorHandlingFunctional:
    """Verify the server handles errors gracefully at the protocol level."""

    def test_unknown_tool_name_returns_error(self, server):
        """Calling a tool that doesn't exist returns a JSON-RPC error."""
        resp = server("tools/call", {
            "name": "nonexistent_tool",
            "arguments": {},
        })
        assert "error" in resp or "isError" in resp.get("result", {})

    def test_create_controller_unknown_type_returns_error(self, server):
        """Creating a controller with an unknown type returns an error message."""
        resp = server("tools/call", {
            "name": "create_controller",
            "arguments": {"name": "bad", "type": "NonExistent", "params": {}},
        })
        text = resp["result"]["content"][0]["text"]
        assert "Unknown controller type" in text

    def test_compute_nonexistent_controller_returns_error(self, server):
        """Computing with a name that was never created returns an error."""
        resp = server("tools/call", {
            "name": "controller_compute",
            "arguments": {"name": "nonexistent", "state": [1.0]},
        })
        text = resp["result"]["content"][0]["text"]
        assert "No controller named" in text

    def test_estimate_nonexistent_estimator_returns_error(self, server):
        """Estimating with a name that was never created returns an error."""
        resp = server("tools/call", {
            "name": "estimator_estimate",
            "arguments": {"name": "nonexistent", "measurement": [0.0], "control_input": [0.0]},
        })
        text = resp["result"]["content"][0]["text"]
        assert "No estimator named" in text

    def test_position_at_nonexistent_trajectory_returns_error(self, server):
        """position_at with a name that was never created returns an error."""
        resp = server("tools/call", {
            "name": "trajectory_position_at",
            "arguments": {"name": "nonexistent", "t": 0.0},
        })
        text = resp["result"]["content"][0]["text"]
        assert "No trajectory named" in text
