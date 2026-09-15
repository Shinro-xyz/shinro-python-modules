"""The kernel-size metric (``shinro.codegen.measure``).

Pins the properties the metric relies on: the byte fields are derived from the
same manifest the VM compiles, they are self-consistent, and the MPPI graph it
measures is the standard one (``epsilon`` port ``(N, K*D_u)``, recurrent
``state_u``). The artifact measurement is exercised only when ``zig`` is
available — the metric is deliberately not part of the default CI gate.
"""

import json
import shutil

import pytest

from shinro.codegen import measure
from shinro.codegen.lower_zig import lower_zig


class TestParseDims:
    """The ``D_x x D_u x N x K`` spec parser."""

    def test_valid(self):
        assert measure.parse_dims("3x3x100x15") == (3, 3, 100, 15)
        assert measure.parse_dims("6X6X10X4") == (6, 6, 10, 4)  # case-insensitive

    def test_rejects_wrong_arity(self):
        with pytest.raises(ValueError, match="D_x x D_u x N x K"):
            measure.parse_dims("3x3x100")

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError, match="four positive integers"):
            measure.parse_dims("3x3xax15")

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            measure.parse_dims("3x3x0x15")


class TestGraphMetrics:
    """Static size derivation from a lowered graph."""

    def test_byte_fields_are_self_consistent(self, tmp_path):
        cg = measure.mppi_lti_graph(3, 3, 6, 3)
        m = measure.graph_metrics(cg, tmp_path)

        assert m["buf_bytes"] == m["buf_len_f64"] * 8
        assert m["input_bytes"] == sum(p["bytes"] for p in m["inputs"])
        assert m["output_bytes"] == sum(p["bytes"] for p in m["outputs"])
        assert m["state_bytes"] == sum(p["bytes"] for p in m["state_outputs"])
        assert m["nodes_total"] > 0
        assert m["has_solve_qp"] is False

    def test_manifest_records_the_memory_metric(self, tmp_path):
        """The byte figures live in the manifest, so every graph self-reports."""
        lower_zig(measure.mppi_lti_graph(3, 3, 6, 3), str(tmp_path / "graph_data.zig"))
        manifest = json.loads((tmp_path / "graph_data_manifest.json").read_text())

        assert manifest["buf_bytes"] == manifest["buf_len"] * 8
        assert manifest["const_blob_bytes"] == manifest["const_blob_len"] * 8
        assert manifest["clip_blob_bytes"] == 2 * manifest["clip_blob_len"] * 8
        assert manifest["clip_blob_len"] > 0  # MPPI clips the sample batch
        assert manifest["input_bytes"] == sum(p["bytes"] for p in manifest["inputs"])
        assert manifest["output_bytes"] == sum(p["bytes"] for p in manifest["outputs"])
        assert manifest["state_bytes"] == sum(p["bytes"] for p in manifest["state_outputs"])
        ports = manifest["inputs"] + manifest["outputs"] + manifest["state_outputs"]
        assert all("bytes" in p for p in ports)

    def test_graph_metrics_reads_the_manifest(self, tmp_path):
        """The metric is a faithful read — no duplicated buffer math."""
        m = measure.graph_metrics(measure.mppi_lti_graph(3, 3, 6, 3), tmp_path)
        manifest = json.loads((tmp_path / "graph_data_manifest.json").read_text())
        for key in ("buf_bytes", "const_blob_bytes", "clip_blob_bytes", "input_bytes", "output_bytes", "state_bytes"):
            assert m[key] == manifest[key], key

    def test_epsilon_dominates_the_input_buffer(self, tmp_path):
        """The sampling port is N*K*D_u f64 — the host-side packing cost."""
        cg = measure.mppi_lti_graph(3, 3, 6, 3)
        m = measure.graph_metrics(cg, tmp_path)
        eps = next(p for p in m["inputs"] if p["name"] == "epsilon")
        assert eps["shape"] == [6, 9]
        assert eps["bytes"] == 6 * 3 * 3 * 8
        assert eps["bytes"] > m["input_bytes"] / 2

    def test_state_port_is_the_nominal_plan(self, tmp_path):
        cg = measure.mppi_lti_graph(4, 2, 8, 4)
        m = measure.graph_metrics(cg, tmp_path)
        assert [p["shape"] for p in m["state_outputs"]] == [[4, 2]]


class TestSweep:
    """The multi-input sweep as a measurement (no compiler)."""

    def test_measure_static_records_every_config(self):
        doc = measure.measure([(3, 3, 6, 3), (6, 6, 10, 4), (8, 4, 12, 5)], "ReleaseFast", build=False)
        assert [c["label"] for c in doc["configs"]] == [
            "D_x=3 D_u=3 N=6 K=3",
            "D_x=6 D_u=6 N=10 K=4",
            "D_x=8 D_u=4 N=12 K=5",
        ]
        assert doc["build"] is False
        # static-only: no artifact measured, but the size metrics are present
        assert all(c["kernel"] is None for c in doc["configs"])
        assert all(c["graph"]["buf_bytes"] > 0 for c in doc["configs"])
        # wider inputs grow the input buffer (D_u=6 > D_u=3 at the same N)
        by_label = {c["label"]: c["graph"]["input_bytes"] for c in doc["configs"]}
        assert by_label["D_x=6 D_u=6 N=10 K=4"] > by_label["D_x=3 D_u=3 N=6 K=3"]


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig not on PATH")
class TestKernelMetrics:
    """The compiled-artifact measurement (opt-in: needs zig)."""

    def test_records_artifact_bytes_and_compile_cost(self, tmp_path):
        cg = measure.mppi_lti_graph(3, 3, 6, 3)
        k = measure.kernel_metrics(cg, tmp_path, "ReleaseFast")
        assert k["optimize"] == "ReleaseFast"
        assert k["so_bytes"] > 0
        assert k["compile_seconds"] > 0
