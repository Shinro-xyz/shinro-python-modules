"""Tests for the MCP codegen tools — trace, compose, inspect, lower, measure.

The zig-free half of the compile pipeline. These pin three properties:

* the tools capture what the components actually compute (interpret matches the
  live component on the same inputs),
* a lowering writes an isolated graph and NEVER the shipped
  ``src/shinro/runtime/graph_data.zig``,
* graph names live in their own store, so a graph and a controller can share a
  name without either tool resolving the wrong object.
"""

import json

import numpy as np
import pytest

from shinro.codegen.runtime_paths import runtime_root
from shinro.mcp._common import _graph_store, _store
from shinro.mcp.codegen_server import (
    build_composed_graph,
    check_component,
    generate_scenario,
    graph_summary,
    interpret_graph,
    list_codegen_ops,
    lower_graph,
    measure_graph,
    trace_component,
)
from shinro.mcp.server import create_controller, list_controllers

BASE_SCENARIO = "tests/integration/scenarios/base_tracking.toml"
KALMAN = "samples/estimators/kalman_base.toml"
LQR = "samples/controllers/lqr_base.toml"

#: The composed base graph's ports (y/x_ref/u_prev are host inputs; the KF's
#: state is threaded back each tick).
BASE_INPUTS = {
    "y": [0.0, 0.0, 0.0],
    "x_ref": [0.0, 0.0, 0.0],
    "u_prev": [0.0, 0.0, 0.0],
    "state_x_hat": [0.0, 0.0, 0.0],
    "state_P": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
}


@pytest.fixture(autouse=True)
def clear_stores():
    """Each test starts with empty component and graph registries."""
    _store.clear()
    _graph_store.clear()
    yield
    _store.clear()
    _graph_store.clear()


def _base_graph(name: str = "base") -> dict:
    return json.loads(
        build_composed_graph(
            name,
            KALMAN,
            LQR,
            3,
            3,
            input_limits_min=[-0.5, -0.5, -1.0],
            input_limits_max=[0.5, 0.5, 1.0],
        )
    )


class TestIntrospection:
    """Verify the op/recipe inventory and the construct-only contract probe."""

    def test_list_codegen_ops_includes_registered_ops_and_recipes(self):
        """Every op and graph recipe is reported, sorted."""
        data = json.loads(list_codegen_ops())
        assert "matmul" in data["ops"]
        assert "input" in data["ops"]
        assert "closed_loop_tracking" in data["recipes"]
        assert "policy_only" in data["recipes"]
        assert data["ops"] == sorted(data["ops"])

    def test_check_component_reports_contract_without_tracing(self):
        """check_component names the traced method and its inputs, storing nothing."""
        data = json.loads(check_component(KALMAN))
        assert data["type"] == "KalmanFilter"
        assert data["category"] == "estimator"
        assert data["method"] == "estimate"
        assert data["input_names"] == ["measurement", "control_input"]
        assert _graph_store == {}

    def test_check_component_missing_file_returns_error(self):
        """A nonexistent config returns an error rather than raising."""
        data = json.loads(check_component("nope.toml"))
        assert "error" in data


class TestTraceComponent:
    """Verify single-component tracing, state discovery, and re-tracing."""

    def test_trace_kalman_discovers_recurrent_state(self):
        """The KF's x_hat/P are detected as recurrent state and reported with shapes."""
        data = json.loads(trace_component("kf", KALMAN, {"measurement": [3], "control_input": [3]}))
        assert data["stored"] == "kf"
        assert data["state_attrs"] == ["x_hat", "P"]
        assert data["state_shapes"] == {"x_hat": [3], "P": [3, 3]}
        assert data["lowerable"] is True
        assert data["unregistered_ops"] == []

    def test_trace_kalman_records_ops_and_ports(self):
        """The trace captures matmul/inv (the gain + covariance update) and one out port."""
        data = json.loads(trace_component("kf", KALMAN, {"measurement": [3], "control_input": [3]}))
        assert data["ops"]["matmul"] > 0
        assert data["ops"]["inv"] == 1
        assert data["inputs"] == ["measurement", "control_input"]
        assert "out" in data["outputs"]

    def test_trace_with_state_shapes_makes_persistent_ports(self):
        """Re-tracing with the reported state shapes yields state_* ports."""
        data = json.loads(
            trace_component(
                "kf",
                KALMAN,
                {"measurement": [3], "control_input": [3]},
                {"x_hat": [3], "P": [3, 3]},
            )
        )
        assert data["state_inputs"] == ["state_x_hat", "state_P"]
        assert data["state_outputs"] == ["state_x_hat", "state_P"]
        # Pre-injected state is an input port now, not a frozen trace-time value.
        assert "state_x_hat" in data["inputs"]

    def test_trace_with_wrong_state_rank_reports_error(self):
        """A state shape of the wrong rank is surfaced as an error, not a crash."""
        data = json.loads(
            trace_component(
                "kf",
                KALMAN,
                {"measurement": [3], "control_input": [3]},
                {"x_hat": [3, 1]},
            )
        )
        assert "error" in data
        assert "ShapeMismatch" in data["error"]

    def test_trace_accepts_string_and_int_shape_specs(self):
        """Shapes may be given as '3x3', a bare int, or a list."""
        data = json.loads(trace_component("kf", KALMAN, {"measurement": "3", "control_input": 3}))
        assert "error" not in data
        assert data["inputs"] == ["measurement", "control_input"]

    def test_trace_unknown_component_type_returns_error(self):
        """An unregistered type returns an error string."""
        data = json.loads(trace_component("bad", "samples/controllers/lqr_base.toml", {}))
        assert "error" in data


class TestComposeAndInterpret:
    """Verify the closed-loop composition and the interpreter oracle."""

    def test_build_composed_graph_exposes_host_and_state_ports(self):
        """The composed base graph has y/x_ref/u_prev plus the KF's recurrent state."""
        data = _base_graph()
        assert data["stored"] == "base"
        assert data["inputs"] == ["y", "x_ref", "u_prev", "state_x_hat", "state_P"]
        assert data["outputs"] == ["u", "state_x_hat", "state_P", "state_u_prev"]
        assert data["lowerable"] is True

    def test_build_composed_graph_requires_both_or_neither_limits(self):
        """A half-specified clip range is a usage error."""
        data = json.loads(build_composed_graph("x", KALMAN, LQR, 3, 3, input_limits_min=[-1.0]))
        assert "error" in data
        assert "both" in data["error"]

    def test_interpret_zero_error_gives_zero_control(self):
        """At zero state error the LQR commands zero (and the KF covariance updates)."""
        _base_graph()
        data = json.loads(interpret_graph("base", BASE_INPUTS))
        assert data["u"] == pytest.approx([0.0, 0.0, 0.0])
        # The covariance is updated even from a zero innovation — proof the
        # recurrent state is threaded, not frozen.
        assert data["state_P"][0][0] != pytest.approx(1.0)

    def test_interpret_reshapes_flat_values_to_port_shapes(self):
        """A flat 9-element state_P list is reshaped to the port's (3, 3)."""
        _base_graph()
        data = json.loads(interpret_graph("base", BASE_INPUTS))
        assert np.asarray(data["state_P"]).shape == (3, 3)

    def test_interpret_missing_port_returns_error_with_expected_ports(self):
        """Omitting an input port reports which ports the graph needs."""
        _base_graph()
        data = json.loads(interpret_graph("base", {"y": [0.0, 0.0, 0.0]}))
        assert "error" in data
        assert "x_ref" in data["expected"]

    def test_interpret_wrong_value_size_returns_error(self):
        """A value that does not fit the declared port shape is rejected."""
        _base_graph()
        inputs = dict(BASE_INPUTS)
        inputs["y"] = [0.0, 0.0]
        data = json.loads(interpret_graph("base", inputs))
        assert "error" in data

    def test_interpret_unknown_graph_returns_error(self):
        """An unknown graph name reports the miss instead of raising."""
        data = json.loads(interpret_graph("nope", {}))
        assert "error" in data

    def test_graph_summary_reports_ports_and_ops(self):
        """graph_summary reads back a stored graph without re-tracing."""
        _base_graph()
        data = json.loads(graph_summary("base"))
        assert data["name"] == "base"
        assert data["kind"] == "composed"
        assert data["nodes_traced"] > 10
        assert "u" in data["outputs"]

    def test_graph_summary_unknown_graph_returns_error(self):
        """An unknown graph name reports the miss."""
        assert "error" in json.loads(graph_summary("nope"))


class TestLowerGraph:
    """Verify lowering writes an isolated graph and refuses the shipped one."""

    def test_lower_graph_writes_graph_and_manifest(self, tmp_path):
        """Lowering emits graph_data.zig + a manifest reporting the buffer sizes."""
        _base_graph()
        out = tmp_path / "base"
        data = json.loads(lower_graph("base", str(out)))
        assert "error" not in data
        assert (out / "graph_data.zig").exists()
        assert (out / "graph_data_manifest.json").exists()
        assert data["nodes_traced"] > 10
        assert data["nodes_lowered"] > 0
        assert data["buf_bytes"] > 0
        assert data["has_solve_qp"] is False

    def test_lower_graph_manifest_matches_disk(self, tmp_path):
        """The manifest summary returned is the one written to disk."""
        _base_graph()
        out = tmp_path / "base"
        data = json.loads(lower_graph("base", str(out)))
        manifest = json.loads((out / "graph_data_manifest.json").read_text())
        assert data["nodes_lowered"] == manifest["nodes_total"]
        assert data["buf_len_f64"] == manifest["buf_len"]

    def test_lower_graph_records_provenance(self, tmp_path):
        """A tool-lowered graph is stamped with generator + tool versions."""
        _base_graph()
        out = tmp_path / "base"
        lower_graph("base", str(out))
        manifest = json.loads((out / "graph_data_manifest.json").read_text())
        assert manifest["provenance"]["generated_by"] == "shinro-mcp"
        assert "numpy_version" in manifest["provenance"]

    def test_lower_graph_refuses_the_packaged_runtime_dir(self):
        """Refuses to clobber the shipped graph, and leaves it byte-identical."""
        shipped = runtime_root() / "graph_data.zig"
        before = shipped.read_bytes()
        _base_graph()
        data = json.loads(lower_graph("base", str(runtime_root())))
        assert "error" in data
        assert "refusing" in data["error"]
        assert shipped.read_bytes() == before

    def test_lower_graph_refuses_a_subdir_of_the_runtime_dir(self):
        """The guard covers subdirectories, not just the runtime root itself."""
        _base_graph()
        data = json.loads(lower_graph("base", str(runtime_root() / "codegen")))
        assert "error" in data
        assert "refusing" in data["error"]

    def test_lower_graph_unknown_graph_returns_error(self, tmp_path):
        """Lowering an unknown name reports the miss."""
        data = json.loads(lower_graph("nope", str(tmp_path)))
        assert "error" in data


class TestGenerateScenario:
    """Verify the gen stage: trace + compose + lower from a scenario TOML."""

    def test_generate_scenario_writes_isolated_graph(self, tmp_path):
        """The gen stage produces graph_data.zig + manifest and stores the graph."""
        out = tmp_path / "base"
        data = json.loads(generate_scenario("base", BASE_SCENARIO, str(out)))
        assert "error" not in data
        assert (out / "graph_data.zig").exists()
        assert (out / "graph_data_manifest.json").exists()
        assert data["nodes_traced"] > 10
        assert data["nodes_lowered"] > 0
        assert data["stored"] == "base"
        assert "base" in _graph_store

    def test_generate_scenario_graph_is_interpretable(self, tmp_path):
        """The stored graph can be interpreted straight away."""
        generate_scenario("base", BASE_SCENARIO, str(tmp_path / "base"))
        data = json.loads(interpret_graph("base", BASE_INPUTS))
        assert "u" in data

    def test_generate_scenario_missing_file_returns_error(self, tmp_path):
        """A nonexistent scenario reports an error."""
        data = json.loads(generate_scenario("x", "nope.toml", str(tmp_path)))
        assert "error" in data

    def test_generate_scenario_refuses_runtime_dir(self):
        """The gen stage shares the runtime-dir guard with lower_graph."""
        data = json.loads(generate_scenario("x", BASE_SCENARIO, str(runtime_root())))
        assert "error" in data
        assert "refusing" in data["error"]


class TestMeasureGraph:
    """Verify the static size metric."""

    def test_measure_graph_reports_buffers_and_blobs(self):
        """Static metrics include node count, host buffers, and the VM stack buffer."""
        _base_graph()
        data = json.loads(measure_graph("base"))
        assert "error" not in data
        assert data["nodes_lowered"] > 10
        assert data["nodes_traced"] > 10
        assert data["buf_len_f64"] > 0
        assert data["buf_bytes"] == data["buf_len_f64"] * 8
        assert data["has_solve_qp"] is False

    def test_measure_graph_unknown_name_returns_error(self):
        """An unknown graph name reports the miss."""
        assert "error" in json.loads(measure_graph("nope"))


class TestGraphStoreIsolation:
    """Verify graphs live in their own registry."""

    def test_graph_and_controller_can_share_a_name(self):
        """A graph named 'x' and a controller named 'x' do not shadow each other."""
        create_controller(
            name="x",
            type="PID",
            params={"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
        )
        build_composed_graph("x", KALMAN, LQR, 3, 3)

        # The controller tools still see the controller...
        assert json.loads(list_controllers())["controllers"]["x"] == "PID"
        # ...and the codegen tools still see the graph.
        assert json.loads(graph_summary("x"))["kind"] == "composed"
        assert "x" in _graph_store
        assert "x" in _store
