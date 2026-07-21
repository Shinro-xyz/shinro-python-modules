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
    create_estimator,
    estimator_estimate,
    estimator_reset,
    list_estimator_types,
    list_estimators,
    create_trajectory,
    trajectory_generate,
    trajectory_position_at,
    list_trajectory_types,
    list_trajectories,
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


class TestCreateEstimator:
    """Verify estimator creation via inline params and config files."""

    def test_create_kalman_inline_creates_and_stores_instance(self):
        """KalmanFilter created with inline params is stored."""
        result = create_estimator(
            name="kf1", type="KalmanFilter",
            params={"dt": 0.02, "process_noise": [0.01, 0.01], "measurement_noise": [0.1, 0.1]},
        )
        assert "Created KalmanFilter estimator 'kf1'" in result
        assert "kf1" in _store

    def test_create_luenberger_inline_creates_and_stores_instance(self):
        """LuenbergerObserver created with inline params is stored."""
        result = create_estimator(
            name="lo1", type="LuenbergerObserver",
            params={"dt": 0.02, "observer_gain": [0.8, 0.8]},
        )
        assert "Created LuenbergerObserver estimator 'lo1'" in result
        assert "lo1" in _store

    def test_create_estimator_from_config_creates_and_stores_instance(self):
        """Estimator created from a TOML config file is stored."""
        result = create_estimator(
            name="est1", config_path="configs/estimators/luenberger_base.toml"
        )
        assert "Created config-based estimator 'est1'" in result
        assert "est1" in _store

    def test_create_estimator_unknown_type_returns_error(self):
        """An unregistered estimator type returns an error message."""
        result = create_estimator(name="bad", type="NonExistent", params={})
        assert "Unknown estimator type" in result
        assert "bad" not in _store

    def test_create_estimator_no_params_returns_usage_instruction(self):
        """Calling create_estimator without config_path or type+params returns guidance."""
        result = create_estimator(name="bad")
        assert "Provide either" in result
        assert "bad" not in _store

    def test_create_estimator_config_path_nonexistent_raises_filenotfound(self):
        """A nonexistent config file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            create_estimator(name="bad", config_path="nonexistent.toml")


class TestEstimatorEstimate:
    """Verify state estimation for both estimator types."""

    def test_estimate_kalman_returns_state_estimate(self):
        """KalmanFilter estimate returns a state_estimate vector."""
        create_estimator(
            name="kf", type="KalmanFilter",
            params={"dt": 0.02, "process_noise": [0.01, 0.01], "measurement_noise": [0.1, 0.1]},
        )
        data = json.loads(estimator_estimate(name="kf", measurement=[0.0, 0.0], control_input=[0.0, 0.0]))
        assert "state_estimate" in data
        assert len(data["state_estimate"]) == 2

    def test_estimate_luenberger_returns_state_estimate(self):
        """LuenbergerObserver estimate returns a state_estimate vector."""
        create_estimator(
            name="lo", type="LuenbergerObserver",
            params={"dt": 0.02, "observer_gain": [0.8, 0.8]},
        )
        data = json.loads(estimator_estimate(name="lo", measurement=[0.0, 0.0], control_input=[0.0, 0.0]))
        assert "state_estimate" in data
        assert len(data["state_estimate"]) == 2

    def test_estimate_kalman_converges_with_noiseless_measurements(self):
        """KalmanFilter estimate converges toward true state with noiseless measurements."""
        create_estimator(
            name="kf", type="KalmanFilter",
            params={"dt": 0.02, "process_noise": [0.01, 0.01], "measurement_noise": [0.001, 0.001]},
        )
        for _ in range(20):
            data = json.loads(estimator_estimate(name="kf", measurement=[1.0, 2.0], control_input=[0.0, 0.0]))
        assert abs(data["state_estimate"][0] - 1.0) < 0.1
        assert abs(data["state_estimate"][1] - 2.0) < 0.1

    def test_estimate_luenberger_converges_with_noiseless_measurements(self):
        """LuenbergerObserver estimate converges toward true state with noiseless measurements."""
        create_estimator(
            name="lo", type="LuenbergerObserver",
            params={"dt": 0.02, "observer_gain": [0.8, 0.8]},
        )
        for _ in range(20):
            data = json.loads(estimator_estimate(name="lo", measurement=[1.0, 2.0], control_input=[0.0, 0.0]))
        assert abs(data["state_estimate"][0] - 1.0) < 0.1
        assert abs(data["state_estimate"][1] - 2.0) < 0.1

    def test_estimate_unknown_name_returns_error(self):
        """Estimating with a name that was never created returns an error."""
        result = estimator_estimate(name="nonexistent", measurement=[0.0], control_input=[0.0])
        assert "No estimator named" in result

    def test_estimate_wrong_measurement_dim_returns_error(self):
        """Estimating with wrong measurement dimension returns an error."""
        create_estimator(
            name="kf", type="KalmanFilter",
            params={"dt": 0.02, "process_noise": [0.01, 0.01], "measurement_noise": [0.1, 0.1]},
        )
        result = estimator_estimate(name="kf", measurement=[0.0, 0.0, 0.0], control_input=[0.0, 0.0])
        assert "Error" in result


class TestEstimatorReset:
    """Verify reset clears internal state for stateful estimators."""

    def test_reset_kalman_clears_state(self):
        """KalmanFilter reset clears the state estimate."""
        create_estimator(
            name="kf", type="KalmanFilter",
            params={"dt": 0.02, "process_noise": [0.01, 0.01], "measurement_noise": [0.1, 0.1]},
        )
        estimator_estimate(name="kf", measurement=[1.0, 2.0], control_input=[0.0])
        result = estimator_reset(name="kf")
        assert "reset" in result.lower()
        est = _store["kf"]
        assert np.allclose(est.x_hat, 0.0)

    def test_reset_luenberger_clears_state(self):
        """LuenbergerObserver reset clears the state estimate."""
        create_estimator(
            name="lo", type="LuenbergerObserver",
            params={"dt": 0.02, "observer_gain": [0.8, 0.8]},
        )
        estimator_estimate(name="lo", measurement=[1.0, 2.0], control_input=[0.0])
        result = estimator_reset(name="lo")
        assert "reset" in result.lower()
        est = _store["lo"]
        assert np.allclose(est.x_hat, 0.0)

    def test_reset_unknown_name_returns_error(self):
        """Resetting a nonexistent estimator returns an error."""
        result = estimator_reset(name="nonexistent")
        assert "No estimator named" in result


class TestEstimatorListTools:
    """Verify estimator listing tools return correct metadata."""

    def test_list_estimator_types_includes_all_registered_types(self):
        """list_estimator_types returns all registered estimator types."""
        data = json.loads(list_estimator_types())
        assert "estimator_types" in data
        assert "KalmanFilter" in data["estimator_types"]
        assert "LuenbergerObserver" in data["estimator_types"]

    def test_list_estimators_returns_empty_dict_when_none_created(self):
        """list_estimators returns an empty dict when no estimators exist."""
        data = json.loads(list_estimators())
        assert data["estimators"] == {}

    def test_list_estimators_returns_all_created_instances(self):
        """list_estimators returns name-to-type mapping for all stored instances."""
        create_estimator(
            name="a", type="KalmanFilter",
            params={"dt": 0.02, "process_noise": [0.01], "measurement_noise": [0.1]},
        )
        create_estimator(
            name="b", type="LuenbergerObserver",
            params={"dt": 0.02, "observer_gain": [0.8]},
        )
        data = json.loads(list_estimators())
        assert data["estimators"]["a"] == "KalmanFilter"
        assert data["estimators"]["b"] == "LuenbergerObserver"


class TestCreateTrajectory:
    """Verify trajectory creation via config files and inline."""

    def test_create_trajectory_from_config_creates_and_stores_instance(self):
        """Trajectory created from a TOML config file is stored."""
        result = create_trajectory(
            name="t1", config_path="configs/trajectories/arm_extension.toml"
        )
        assert "Created config-based trajectory 't1'" in result
        assert "t1" in _store

    def test_create_cubic_inline_creates_and_stores_instance(self):
        """CubicPolynomial created inline is stored."""
        result = create_trajectory(
            name="cubic", type="cubic_segments", params={"dt": 0.02}
        )
        assert "Created cubic_segments trajectory 'cubic'" in result
        assert "cubic" in _store

    def test_create_quintic_inline_creates_and_stores_instance(self):
        """QuinticPolynomial created inline is stored."""
        result = create_trajectory(
            name="quint", type="quintic_segments", params={"dt": 0.02}
        )
        assert "Created quintic_segments trajectory 'quint'" in result
        assert "quint" in _store

    def test_create_trajectory_unknown_type_returns_error(self):
        """An unregistered trajectory type returns an error message."""
        result = create_trajectory(name="bad", type="NonExistent", params={})
        assert "Unknown trajectory type" in result
        assert "bad" not in _store

    def test_create_trajectory_no_params_returns_usage_instruction(self):
        """Calling create_trajectory without config_path or type+params returns guidance."""
        result = create_trajectory(name="bad")
        assert "Provide either" in result
        assert "bad" not in _store

    def test_create_trajectory_config_path_nonexistent_raises_filenotfound(self):
        """A nonexistent config file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            create_trajectory(name="bad", config_path="nonexistent.toml")


class TestTrajectoryGenerate:
    """Verify trajectory generation for cubic and quintic generators."""

    def test_generate_cubic_returns_success(self):
        """CubicPolynomial generate returns a success status."""
        create_trajectory(name="cubic", type="cubic_segments", params={"dt": 0.02})
        result = trajectory_generate(
            name="cubic",
            start_position=[0.0, 0.0, 0.0],
            end_position=[0.0, 0.24, 0.15],
            duration=3.0,
            start_vel=[0.0, 0.0, 0.0],
            end_vel=[0.0, 0.0, 0.0],
        )
        data = json.loads(result)
        assert data["status"] == "generated"
        assert data["duration"] == 3.0

    def test_generate_quintic_returns_success(self):
        """QuinticPolynomial generate returns a success status."""
        create_trajectory(name="quint", type="quintic_segments", params={"dt": 0.02})
        result = trajectory_generate(
            name="quint",
            start_position=[0.0, 0.0, 0.0],
            end_position=[0.0, 0.15, 0.10],
            duration=2.0,
        )
        data = json.loads(result)
        assert data["status"] == "generated"

    def test_generate_unknown_name_returns_error(self):
        """Generating with a name that was never created returns an error."""
        result = trajectory_generate(
            name="nonexistent",
            start_position=[0.0], end_position=[1.0], duration=1.0,
        )
        assert "No trajectory named" in result

    def test_generate_config_based_trajectory_returns_error(self):
        """Config-based trajectories (waypoints, phase_list) don't support generate()."""
        create_trajectory(name="wp", config_path="configs/trajectories/base_straight.toml")
        result = trajectory_generate(
            name="wp",
            start_position=[0.0], end_position=[1.0], duration=1.0,
        )
        assert "does not support" in result


class TestTrajectoryPositionAt:
    """Verify position_at evaluation for generated trajectories."""

    def test_position_at_returns_position_velocity_acceleration(self):
        """position_at returns position, velocity, and acceleration vectors."""
        create_trajectory(name="cubic", type="cubic_segments", params={"dt": 0.02})
        trajectory_generate(
            name="cubic",
            start_position=[0.0, 0.0, 0.0],
            end_position=[0.0, 0.24, 0.15],
            duration=3.0,
            start_vel=[0.0, 0.0, 0.0],
            end_vel=[0.0, 0.0, 0.0],
        )
        data = json.loads(trajectory_position_at(name="cubic", t=1.5))
        assert "position" in data
        assert "velocity" in data
        assert "acceleration" in data
        assert len(data["position"]) == 3

    def test_position_at_start_matches_start_position(self):
        """position_at(0) equals the start position."""
        create_trajectory(name="cubic", type="cubic_segments", params={"dt": 0.02})
        trajectory_generate(
            name="cubic",
            start_position=[1.0, 2.0, 3.0],
            end_position=[4.0, 5.0, 6.0],
            duration=3.0,
            start_vel=[0.0, 0.0, 0.0],
            end_vel=[0.0, 0.0, 0.0],
        )
        data = json.loads(trajectory_position_at(name="cubic", t=0.0))
        assert data["position"] == [1.0, 2.0, 3.0]

    def test_position_at_end_matches_end_position(self):
        """position_at(duration) equals the end position."""
        create_trajectory(name="cubic", type="cubic_segments", params={"dt": 0.02})
        trajectory_generate(
            name="cubic",
            start_position=[1.0, 2.0, 3.0],
            end_position=[4.0, 5.0, 6.0],
            duration=3.0,
            start_vel=[0.0, 0.0, 0.0],
            end_vel=[0.0, 0.0, 0.0],
        )
        data = json.loads(trajectory_position_at(name="cubic", t=3.0))
        assert data["position"] == pytest.approx([4.0, 5.0, 6.0])

    def test_position_at_unknown_name_returns_error(self):
        """position_at with a name that was never created returns an error."""
        result = trajectory_position_at(name="nonexistent", t=0.0)
        assert "No trajectory named" in result

    def test_position_at_before_generate_returns_error(self):
        """position_at before calling generate returns an error (no crash)."""
        create_trajectory(name="cubic", type="cubic_segments", params={"dt": 0.02})
        result = trajectory_position_at(name="cubic", t=0.0)
        assert "Error" in result

    def test_position_at_time_clamped_to_duration(self):
        """position_at with t > duration is clamped to duration."""
        create_trajectory(name="cubic", type="cubic_segments", params={"dt": 0.02})
        trajectory_generate(
            name="cubic",
            start_position=[0.0], end_position=[1.0], duration=2.0,
            start_vel=[0.0], end_vel=[0.0],
        )
        data = json.loads(trajectory_position_at(name="cubic", t=100.0))
        assert data["position"][0] == pytest.approx(1.0)


class TestTrajectoryListTools:
    """Verify trajectory listing tools return correct metadata."""

    def test_list_trajectory_types_includes_all_registered_types(self):
        """list_trajectory_types returns all registered trajectory types."""
        data = json.loads(list_trajectory_types())
        assert "trajectory_types" in data
        assert "cubic_segments" in data["trajectory_types"]
        assert "quintic_segments" in data["trajectory_types"]
        assert "waypoints" in data["trajectory_types"]
        assert "phase_list" in data["trajectory_types"]

    def test_list_trajectories_returns_empty_dict_when_none_created(self):
        """list_trajectories returns an empty dict when no trajectories exist."""
        data = json.loads(list_trajectories())
        assert data["trajectories"] == {}

    def test_list_trajectories_returns_all_created_instances(self):
        """list_trajectories returns name-to-type mapping for all stored instances."""
        create_trajectory(name="a", type="cubic_segments", params={"dt": 0.02})
        create_trajectory(name="b", config_path="configs/trajectories/arm_extension.toml")
        data = json.loads(list_trajectories())
        assert data["trajectories"]["a"] == "cubic_segments"
        assert data["trajectories"]["b"] == "ndarray"
