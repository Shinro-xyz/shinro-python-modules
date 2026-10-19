"""Tests for the MCP verification tools — gate A, zig build, oracle B, stamp, verify.

Gate A runs live (numpy only). The zig stage builds one real kernel per session
and the remaining tests reuse it, so the suite pays for a single compile while
still covering the oracle, the stamp, and drift detection end to end.

Two things these pin beyond "it returns 0":

* **stdout discipline** — ``build_scenario`` / ``stamp`` / ``verify`` narrate on
  stdout, which on a stdio MCP server is the JSON-RPC channel. The tools must
  capture that narration into the result (asserted via the returned ``log``).
* **the unverified path** — a build without a scenario still stamps, but records
  ``oracle.status == "not_run"`` rather than claiming verification.
"""

import json
import shutil

import pytest

from shinro.mcp._common import _graph_store, _store
from shinro.mcp.codegen_server import (
    build_kernel,
    generate_scenario,
    run_gate_a,
    run_oracle,
    stamp_deployment,
    verify_deployment,
)

BASE_SCENARIO = "tests/integration/scenarios/base_tracking.toml"

needs_zig = pytest.mark.skipif(shutil.which("zig") is None, reason="zig not on PATH")


@pytest.fixture(autouse=True)
def clear_stores():
    _store.clear()
    _graph_store.clear()
    yield
    _store.clear()
    _graph_store.clear()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Generate + build the base kernel once; return the paths and build result."""
    if shutil.which("zig") is None:
        pytest.skip("zig not on PATH")
    out = tmp_path_factory.mktemp("mcp-build") / "base"
    gen = json.loads(generate_scenario("base", BASE_SCENARIO, str(out)))
    assert "error" not in gen, gen
    result = json.loads(build_kernel(str(out), scenario_path=BASE_SCENARIO))
    return {"out": out, "gen": gen, "build": result}


class TestGateA:
    """Verify the interpret-vs-live equivalence gate (no zig required)."""

    def test_gate_a_passes_on_the_base_scenario(self):
        """The composed graph reproduces the live KF+LQR control bit-exactly."""
        data = json.loads(run_gate_a(BASE_SCENARIO, ticks=50, seed=0))
        assert data["passed"] is True
        assert data["max_abs_err"] <= data["tolerance"]
        assert data["ticks"] == 50

    def test_gate_a_reports_the_compared_graph_size(self):
        """The result reports the graph it compared, so a stale graph is visible."""
        data = json.loads(run_gate_a(BASE_SCENARIO, ticks=5))
        assert data["nodes"] > 10

    def test_gate_a_rejects_a_policy_only_scenario(self, tmp_path):
        """A graph with no live estimator/controller pair is refused, not faked."""
        scenario = tmp_path / "policy.toml"
        scenario.write_text(
            "[controller]\n"
            'type = "onnx_rl"\n'
            'config = "samples/controllers/onnx_rl.toml"\n'
            "\n[compile]\n"
            'optimize = "debug"\n'
            'artifact_name = "libpolicy"\n'
        )
        data = json.loads(run_gate_a(str(scenario)))
        assert "error" in data
        assert "closed loop" in data["error"]

    def test_gate_a_missing_scenario_returns_error(self):
        """A nonexistent scenario reports an error rather than raising."""
        assert "error" in json.loads(run_gate_a("nope.toml"))


@needs_zig
class TestBuildKernel:
    """Verify the zig stage: integrity, gate A, build, oracle, stamp, verify."""

    def test_build_succeeds_and_reports_ok(self, built):
        """The e2e build exits 0 with a named artifact."""
        result = built["build"]
        assert result["exit_code"] == 0, result["log"]
        assert result["status"] == "ok"
        assert result["artifact"] is not None
        assert result["artifact_bytes"] > 0

    def test_build_records_a_passed_oracle(self, built):
        """The deployment record carries the oracle outcome it verified."""
        oracle = built["build"]["oracle"]
        assert oracle["status"] == "passed"
        assert oracle["max_abs_err"] < oracle["tolerance"]
        assert oracle["samples"] == 20

    def test_build_writes_a_deployment_record(self, built):
        """Stamp + verify leave a master-hashed record next to the binary."""
        assert built["build"]["master_hash"]
        record = json.loads((built["out"] / "lib" / "libbase.deployment.json").read_text())
        assert record["master_hash"] == built["build"]["master_hash"]
        assert set(record["slots"]) == {"config", "graph", "solver", "binary"}

    def test_build_log_is_captured_not_printed(self, built):
        """The pipeline's stdout narration comes back as a field (never to the transport)."""
        log = built["build"]["log"]
        assert any("oracle B" in line for line in log)
        assert any("gate A" in line for line in log)

    def test_build_reports_zig_missing_when_unavailable(self, tmp_path, monkeypatch):
        """Without zig the tool reports the exit code instead of raising."""
        monkeypatch.setattr("shinro.mcp._verification.shutil.which", lambda _: None)
        data = json.loads(build_kernel(str(tmp_path)))
        assert data["exit_code"] == 5
        assert data["status"] == "zig_missing"
        assert data["artifact"] is None

    def test_build_requires_a_generated_graph(self, tmp_path):
        """Building a dir with no graph is a usage error, not a crash."""
        data = json.loads(build_kernel(str(tmp_path)))
        assert data["exit_code"] == 2
        assert data["status"] == "usage_error"


@needs_zig
class TestOracleB:
    """Verify the .so-vs-interpreter oracle against the built artifact."""

    def test_oracle_passes_on_the_built_kernel(self, built):
        """shinro_step matches interpret() across every port."""
        data = json.loads(
            run_oracle(str(built["out"]), scenario_path=BASE_SCENARIO, samples=10, seed=1)
        )
        assert data["passed"] is True, data
        assert data["max_abs_err"] < data["tolerance"]
        assert data["has_solve_qp"] is False

    def test_oracle_accepts_a_stored_graph_name(self, built, tmp_path):
        """A graph generated through the tool can be verified without a scenario.

        The store is cleared between tests, so re-generate to repopulate it —
        lowering is deterministic, so this is the same graph the kernel was
        built from.
        """
        assert "error" not in json.loads(generate_scenario("base", BASE_SCENARIO, str(tmp_path / "regen")))
        data = json.loads(run_oracle(str(built["out"]), name="base", samples=5))
        assert data["passed"] is True

    def test_oracle_reports_a_missing_binary(self, tmp_path):
        """Pointing at a prefix with no .so reports the path it looked for."""
        data = json.loads(run_oracle(str(tmp_path), scenario_path=BASE_SCENARIO))
        assert "error" in data
        assert "libbase.so" in data["error"]

    def test_oracle_requires_a_graph_source(self, built):
        """Neither a name nor a scenario is a usage error."""
        data = json.loads(run_oracle(str(built["out"])))
        assert "error" in data


@needs_zig
class TestStampAndVerify:
    """Verify the deployment record producer and the independent drift gate."""

    def test_stamp_roundtrip_matches_the_record(self, built):
        """Re-stamping a built prefix reproduces its master hash."""
        data = json.loads(stamp_deployment(str(built["out"])))
        assert "error" not in data
        assert data["master_hash"] == built["build"]["master_hash"]

    def test_stamp_carries_a_supplied_oracle_block(self, built):
        """An oracle outcome passed in is recorded verbatim."""
        data = json.loads(
            stamp_deployment(str(built["out"]), oracle={"status": "passed", "samples": 3, "max_abs_err": 1e-15})
        )
        assert data["oracle"]["status"] == "passed"
        assert data["oracle"]["samples"] == 3

    def test_stamp_missing_binary_returns_error(self, tmp_path):
        """Stamping a prefix with no built artifacts reports the missing file."""
        data = json.loads(stamp_deployment(str(tmp_path)))
        assert "error" in data

    def test_verify_passes_on_a_matching_record(self, built):
        """The drift gate accepts an untouched build."""
        record = str(built["out"] / "lib" / "libbase.deployment.json")
        data = json.loads(verify_deployment(record))
        assert data["ok"] is True
        assert data["failures"] == []

    def test_verify_detects_binary_drift(self, built, tmp_path):
        """A byte appended to the .so is caught as a binary-slot mismatch."""
        drifted = tmp_path / "lib" / "libbase.so"
        drifted.parent.mkdir(parents=True)
        shutil.copy(built["out"] / "lib" / "libbase.so", drifted)
        with open(drifted, "ab") as f:
            f.write(b"\x00")

        record = str(built["out"] / "lib" / "libbase.deployment.json")
        data = json.loads(verify_deployment(record, binary_path=str(drifted)))
        assert data["ok"] is False
        assert any("binary" in failure for failure in data["failures"])

    def test_verify_detects_graph_drift(self, built, tmp_path):
        """A modified graph_data.zig is caught as a graph-slot mismatch."""
        graph_dir = tmp_path / "g"
        shutil.copytree(built["out"], graph_dir)
        graph = graph_dir / "graph_data.zig"
        graph.write_text(graph.read_text() + "\n// drift\n")

        record = str(graph_dir / "lib" / "libbase.deployment.json")
        data = json.loads(verify_deployment(record, graph_path=str(graph)))
        assert data["ok"] is False
        assert any("graph" in failure for failure in data["failures"])

    def test_verify_missing_record_returns_error(self, tmp_path):
        """A nonexistent record is reported, not raised."""
        data = json.loads(verify_deployment(str(tmp_path / "nope.json")))
        assert "error" in data
