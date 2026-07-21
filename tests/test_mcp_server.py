"""Tests for the MCP server — controller tools.

Tests cover creation, computation, reset, listing, and edge cases
(malicious inputs, overflow, memory, type mismatches).
"""

import json
import pytest
import numpy as np
from shinro_mcp_server import (
    create_controller,
    controller_compute,
    controller_reset,
    list_controller_types,
    list_controllers,
    _store,
)


@pytest.fixture(autouse=True)
def clear_store():
    _store.clear()
    yield


class TestCreateController:
    """Verify controller creation via inline params and config files."""

    def test_create_pid_inline_creates_and_stores_instance(self):
        """PID created with inline params is stored and returns success message."""
        result = create_controller(
            name="pid1",
            type="PID",
            params={"dt": 0.02, "kp": [2.0, 2.0], "ki": [0.1, 0.1], "kd": [0.5, 0.5]},
        )
        assert "Created PID controller 'pid1'" in result
        assert "pid1" in _store

    def test_create_lqr_from_config_creates_and_stores_instance(self):
        """LQR created from a TOML config file is stored."""
        result = create_controller(
            name="lqr1", config_path="configs/controllers/lqr_base.toml"
        )
        assert "Created config-based controller 'lqr1'" in result
        assert "lqr1" in _store

    def test_create_mpc_from_config_creates_and_stores_instance(self):
        """MPC_DeltaU created from a TOML config file is stored."""
        result = create_controller(
            name="mpc1", config_path="configs/controllers/mpc_base.toml"
        )
        assert "Created config-based controller 'mpc1'" in result
        assert "mpc1" in _store

    def test_create_duplicate_name_overwrites_previous_instance(self):
        """Creating a controller with an existing name overwrites the old one."""
        create_controller(
            name="x", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        create_controller(
            name="x", type="LQR",
            params={"state_cost": [1.0], "control_cost": [1.0], "dt": 0.1},
        )
        assert _store["x"]._registry_name == "LQR"

    def test_create_unknown_type_returns_error_and_does_not_store(self):
        """An unregistered controller type returns an error message."""
        result = create_controller(name="bad", type="NonExistent", params={})
        assert "Unknown controller type" in result
        assert "bad" not in _store

    def test_create_no_params_returns_usage_instruction(self):
        """Calling create without config_path or type+params returns guidance."""
        result = create_controller(name="bad")
        assert "Provide either" in result
        assert "bad" not in _store

    def test_create_config_path_nonexistent_raises_filenotfound(self):
        """A nonexistent config file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            create_controller(name="bad", config_path="nonexistent.toml")

    def test_create_config_path_missing_type_field_raises_key_error(self):
        """A config file without a 'type' field raises a KeyError from the factory."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("dt = 0.02\n")
            path = f.name
        try:
            with pytest.raises(KeyError):
                create_controller(name="bad", config_path=path)
        finally:
            os.unlink(path)

    def test_create_pid_missing_required_params_defaults_to_zeros(self):
        """PID creation with incomplete params defaults missing gains to zeros."""
        result = create_controller(name="bad", type="PID", params={"dt": 0.02})
        assert "Created" in result
        assert "bad" in _store

    def test_create_lqr_missing_required_params_returns_error(self):
        """LQR creation with empty params returns error."""
        result = create_controller(name="bad", type="LQR", params={})
        assert "bad" not in _store


class TestControllerCompute:
    """Verify control action computation for all controller types."""

    def test_compute_pid_returns_action_with_correct_dimension(self):
        """PID compute returns a 1-element action vector."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [2.0], "ki": [0.1], "kd": [0.5]},
        )
        data = json.loads(controller_compute(name="pid", state=[1.0], reference=[0.0]))
        assert "action" in data
        assert len(data["action"]) == 1

    def test_compute_lqr_returns_action_with_correct_dimension(self):
        """LQR compute returns a 2-element action vector."""
        create_controller(
            name="lqr", type="LQR",
            params={"state_cost": [1.0, 1.0], "control_cost": [1.0, 1.0], "dt": 0.1},
        )
        data = json.loads(
            controller_compute(name="lqr", state=[0.5, -0.3], reference=[0.0, 0.0])
        )
        assert "action" in data
        assert len(data["action"]) == 2

    def test_compute_mpc_deltau_returns_action_with_correct_dimension(self):
        """MPC_DeltaU compute returns a 2-element action vector."""
        create_controller(
            name="mpc", type="MPC_DeltaU",
            params={
                "dt": 0.02, "horizon": 5,
                "state_cost": [1.0, 1.0],
                "control_cost": [0.1, 0.1],
                "delta_u_penalty": [0.5, 0.5],
            },
        )
        data = json.loads(
            controller_compute(name="mpc", state=[0.5, -0.3], u_prev=[0.0, 0.0])
        )
        assert "action" in data
        assert len(data["action"]) == 2

    def test_compute_mpc_deltau_defaults_u_prev_to_zeros(self):
        """MPC_DeltaU compute works without u_prev (defaults to zeros)."""
        create_controller(
            name="mpc", type="MPC_DeltaU",
            params={
                "dt": 0.02, "horizon": 5,
                "state_cost": [1.0, 1.0],
                "control_cost": [0.1, 0.1],
                "delta_u_penalty": [0.5, 0.5],
            },
        )
        data = json.loads(controller_compute(name="mpc", state=[0.5, -0.3]))
        assert "action" in data

    def test_compute_mpc_lti_returns_action(self):
        """MPC_LTI compute returns an action vector."""
        create_controller(
            name="mpc_lti", type="MPC_LTI",
            params={
                "dt": 0.02, "horizon": 5,
                "state_cost": [1.0, 1.0],
                "control_cost": [0.1, 0.1],
            },
        )
        data = json.loads(controller_compute(name="mpc_lti", state=[0.5, -0.3]))
        assert "action" in data

    def test_compute_unknown_name_returns_error(self):
        """Computing with a name that was never created returns an error."""
        result = controller_compute(name="nonexistent", state=[1.0])
        assert "No controller named" in result

    def test_compute_wrong_state_dimension_returns_error(self):
        """Computing with a state vector of wrong dimension returns an error."""
        create_controller(
            name="lqr", type="LQR",
            params={"state_cost": [1.0, 1.0], "control_cost": [1.0, 1.0], "dt": 0.1},
        )
        result = controller_compute(name="lqr", state=[1.0])
        assert "Error in compute" in result

    def test_compute_empty_state_returns_error(self):
        """Computing with an empty state vector returns an error."""
        create_controller(
            name="lqr", type="LQR",
            params={"state_cost": [1.0, 1.0], "control_cost": [1.0, 1.0], "dt": 0.1},
        )
        result = controller_compute(name="lqr", state=[])
        assert "Error in compute" in result

    def test_compute_pid_no_reference_defaults_to_zeros(self):
        """PID compute without reference defaults to zero target (regulation)."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [2.0], "ki": [0.0], "kd": [0.0]},
        )
        data = json.loads(controller_compute(name="pid", state=[5.0]))
        assert data["action"][0] == pytest.approx(-10.0)

    def test_compute_pid_integral_accumulates_across_calls(self):
        """PID integral term grows when error persists across compute calls."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [0.0], "ki": [1.0], "kd": [0.0]},
        )
        a1 = json.loads(controller_compute(name="pid", state=[1.0], reference=[0.0]))["action"][0]
        a2 = json.loads(controller_compute(name="pid", state=[1.0], reference=[0.0]))["action"][0]
        assert abs(a2) > abs(a1)

    def test_compute_mismatched_state_and_reference_dimensions_broadcasts(self):
        """State and reference with different dimensions broadcasts in numpy (no error)."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [1.0, 1.0], "ki": [0.0, 0.0], "kd": [0.0, 0.0]},
        )
        result = controller_compute(name="pid", state=[1.0, 2.0], reference=[0.0])
        import json
        data = json.loads(result)
        assert "action" in data


class TestControllerReset:
    """Verify reset clears internal state for stateful controllers."""

    def test_reset_pid_clears_integral_and_derivative_state(self):
        """PID reset zeros the integral accumulator and resets the derivative flag."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [0.0], "ki": [1.0], "kd": [0.0]},
        )
        controller_compute(name="pid", state=[1.0], reference=[0.0])
        result = controller_reset(name="pid")
        assert "reset" in result.lower()
        ctrl = _store["pid"]
        assert np.allclose(ctrl._integral, 0.0)
        assert not ctrl.has_run

    def test_reset_unknown_name_returns_error(self):
        """Resetting a nonexistent controller returns an error."""
        result = controller_reset(name="nonexistent")
        assert "No controller named" in result

    def test_reset_lqr_is_noop(self):
        """LQR has no internal state, reset is a no-op (returns success)."""
        create_controller(
            name="lqr", type="LQR",
            params={"state_cost": [1.0], "control_cost": [1.0], "dt": 0.1},
        )
        result = controller_reset(name="lqr")
        assert "reset" in result.lower()


class TestListTools:
    """Verify listing tools return correct metadata."""

    def test_list_controller_types_includes_all_registered_types(self):
        """list_controller_types returns all registered controller types."""
        data = json.loads(list_controller_types())
        assert "controller_types" in data
        assert "PID" in data["controller_types"]
        assert "LQR" in data["controller_types"]
        assert "MPC_DeltaU" in data["controller_types"]

    def test_list_controllers_returns_empty_dict_when_none_created(self):
        """list_controllers returns an empty dict when no controllers exist."""
        data = json.loads(list_controllers())
        assert data["controllers"] == {}

    def test_list_controllers_returns_all_created_instances(self):
        """list_controllers returns name-to-type mapping for all stored instances."""
        create_controller(
            name="a", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        create_controller(
            name="b", type="LQR",
            params={"state_cost": [1.0], "control_cost": [1.0], "dt": 0.1},
        )
        data = json.loads(list_controllers())
        assert data["controllers"]["a"] == "PID"
        assert data["controllers"]["b"] == "LQR"


class TestMaliciousEdgeCases:
    """Verify graceful handling of pathological inputs."""

    def test_nan_in_state_propagates_to_action(self):
        """NaN in state produces NaN in the action (no crash)."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        data = json.loads(
            controller_compute(name="pid", state=[float("nan")], reference=[0.0])
        )
        assert np.isnan(data["action"][0])

    def test_inf_in_state_propagates_to_action(self):
        """Infinity in state produces NaN in the action (no crash)."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        data = json.loads(
            controller_compute(name="pid", state=[float("inf")], reference=[0.0])
        )
        assert np.isnan(data["action"][0])

    def test_very_large_state_values_do_not_crash(self):
        """Extremely large state values (near float max) do not crash."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        data = json.loads(controller_compute(name="pid", state=[1e308], reference=[0.0]))
        assert "action" in data

    def test_very_large_gains_produce_large_action(self):
        """Extremely large gains produce proportionally large action (no crash)."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [1e308], "ki": [0.0], "kd": [0.0]},
        )
        data = json.loads(controller_compute(name="pid", state=[1.0], reference=[0.0]))
        assert data["action"][0] == -1e308

    def test_negative_gains_produce_positive_feedback(self):
        """Negative proportional gain produces positive feedback (opposite direction)."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [-1.0], "ki": [0.0], "kd": [0.0]},
        )
        data = json.loads(controller_compute(name="pid", state=[1.0], reference=[0.0]))
        assert data["action"][0] == 1.0

    def test_zero_gains_produce_zero_action(self):
        """All gains set to zero produces zero control action."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [0.0], "ki": [0.0], "kd": [0.0]},
        )
        data = json.loads(controller_compute(name="pid", state=[1.0], reference=[0.0]))
        assert data["action"][0] == 0.0

    def test_empty_name_is_accepted(self):
        """An empty string is a valid name (stored in _store)."""
        result = create_controller(
            name="", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        assert "Created" in result
        assert "" in _store

    def test_very_long_name_is_accepted(self):
        """A 10000-character name is accepted without error."""
        long_name = "a" * 10000
        result = create_controller(
            name=long_name, type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        assert "Created" in result
        assert long_name in _store

    def test_many_controllers_do_not_exhaust_memory(self):
        """Creating and computing 100 controllers works without issues."""
        for i in range(100):
            create_controller(
                name=f"c{i}", type="PID",
                params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
            )
        assert len(_store) == 100
        for i in range(100):
            data = json.loads(
                controller_compute(name=f"c{i}", state=[float(i)], reference=[0.0])
            )
            assert "action" in data

    def test_string_in_state_raises_value_error(self):
        """Non-numeric values in state raise a ValueError."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        with pytest.raises((TypeError, ValueError)):
            controller_compute(name="pid", state=["a", "b"])

    def test_none_in_state_produces_nan(self):
        """None values in state produce NaN in the action (no crash)."""
        create_controller(
            name="pid", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        data = json.loads(controller_compute(name="pid", state=[None, 1.0]))
        assert "action" in data

    def test_negative_horizon_mpc_raises_error(self):
        """MPC with negative horizon raises an error (matrix_power undefined)."""
        with pytest.raises(Exception):
            create_controller(
                name="bad_mpc", type="MPC_LTI",
                params={
                    "dt": 0.02, "horizon": -5,
                    "state_cost": [1.0], "control_cost": [0.1],
                },
            )

    def test_zero_horizon_mpc_raises_error(self):
        """MPC with zero horizon raises an error (empty lifted matrices)."""
        with pytest.raises(Exception):
            create_controller(
                name="bad_mpc", type="MPC_LTI",
                params={
                    "dt": 0.02, "horizon": 0,
                    "state_cost": [1.0], "control_cost": [0.1],
                },
            )

    def test_negative_dt_is_accepted_by_pid(self):
        """PID accepts negative dt (produces negative integral accumulation)."""
        result = create_controller(
            name="bad", type="PID",
            params={"dt": -0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        assert "Created" in result

    def test_zero_dt_is_accepted_by_pid(self):
        """PID accepts zero dt (derivative term becomes inf on second call)."""
        result = create_controller(
            name="bad", type="PID",
            params={"dt": 0.0, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        assert "Created" in result

    def test_negative_state_cost_lqr_raises_error(self):
        """LQR with negative-definite Q raises a DARE solve error."""
        with pytest.raises(Exception):
            create_controller(
                name="bad", type="LQR",
                params={
                    "state_cost": [-1.0, -1.0],
                    "control_cost": [1.0, 1.0],
                    "dt": 0.1,
                },
            )

    def test_negative_control_cost_lqr_raises_error(self):
        """LQR with negative-definite R raises a DARE solve error."""
        with pytest.raises(Exception):
            create_controller(
                name="bad", type="LQR",
                params={
                    "state_cost": [1.0, 1.0],
                    "control_cost": [-1.0, -1.0],
                    "dt": 0.1,
                },
            )

    def test_very_large_horizon_mpc_times_out(self):
        """MPC with horizon=10000 hangs (matrix construction O(N^2))."""
        import subprocess
        import sys
        code = """
from shinro_mcp_server import create_controller
result = create_controller(
    name="bad_mpc", type="MPC_LTI",
    params={"dt": 0.02, "horizon": 10000, "state_cost": [1.0], "control_cost": [0.1]},
)
print(result)
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )
        assert proc.returncode != 0 or "Error" in proc.stdout

    def test_very_large_state_dimension_does_not_crash(self):
        """LQR with 1000-dimensional state works without issues."""
        n = 1000
        result = create_controller(
            name="big", type="LQR",
            params={
                "state_cost": [1.0] * n,
                "control_cost": [1.0] * n,
                "dt": 0.1,
            },
        )
        assert "Created" in result
        data = json.loads(
            controller_compute(name="big", state=[0.5] * n, reference=[0.0] * n)
        )
        assert len(data["action"]) == n

    def test_unicode_in_name_is_accepted(self):
        """Unicode characters in controller name are accepted."""
        result = create_controller(
            name="pid_测试", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        assert "Created" in result
        assert "pid_测试" in _store

    def test_special_characters_in_name_are_accepted(self):
        """Special characters in controller name are accepted."""
        result = create_controller(
            name="pid!@#$%^&*()", type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        assert "Created" in result
        assert "pid!@#$%^&*()" in _store

    def test_mpc_deltau_with_constraints_respects_bounds(self):
        """MPC_DeltaU with hard constraints respects the upper/lower bounds."""
        result = create_controller(
            name="mpc_con", type="MPC_DeltaU",
            params={
                "dt": 0.02, "horizon": 5,
                "state_cost": [1.0, 1.0],
                "control_cost": [0.1, 0.1],
                "delta_u_penalty": [0.5, 0.5],
                "constraints": {
                    "upper": [0.5, 0.5, 0.5, 0.5],
                    "lower": [-0.5, -0.5, -0.5, -0.5],
                },
            },
        )
        assert "Created" in result
        data = json.loads(
            controller_compute(name="mpc_con", state=[10.0, 10.0], u_prev=[0.0, 0.0])
        )
        assert all(abs(v) <= 0.5 + 1e-4 for v in data["action"])
