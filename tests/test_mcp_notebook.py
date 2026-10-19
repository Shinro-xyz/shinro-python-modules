"""Tests for the experiment notebook server — append-only JSONL run records.

The notebook is the one server that persists: a record must survive the session,
carry what ran and what came out, and be robust to a partially-corrupt file. These pin
the log format, the provenance the server fills in, and the read paths.
"""

import json

import pytest

from shinro.mcp.notebook_server import get_run, list_runs, record_run


def _lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


class TestRecordRun:
    """Appending records: one JSON object per line, self-describing."""

    def test_appends_one_parseable_line_and_creates_parents(self, tmp_path):
        """The notebook file is created (with parents) and holds one JSON object."""
        notebook = tmp_path / "runs" / "experiments.jsonl"
        result = json.loads(
            record_run(
                str(notebook),
                "trial-a",
                components={"plant": {"type": "DCMotor"}, "controller": {"type": "PID"}},
                metrics={"final_error": 1e-3},
            )
        )
        assert result["recorded"]
        assert result["path"] == str(notebook)

        lines = _lines(notebook)
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["name"] == "trial-a"
        assert record["components"]["plant"]["type"] == "DCMotor"
        assert record["metrics"]["final_error"] == 1e-3

    def test_records_append_and_ids_are_unique(self, tmp_path):
        """A second record appends rather than rewriting, with a distinct id."""
        notebook = tmp_path / "exp.jsonl"
        first = json.loads(record_run(str(notebook), "same-name", metrics={"e": 1.0}))["recorded"]
        second = json.loads(record_run(str(notebook), "same-name", metrics={"e": 2.0}))["recorded"]
        assert first != second, "ids must be unique even for identical names"
        assert len(_lines(notebook)) == 2

    def test_provenance_is_filled_by_the_server(self, tmp_path):
        """Timestamp, version, platform and cwd are recorded, not supplied by the caller."""
        notebook = tmp_path / "exp.jsonl"
        record_run(str(notebook), "trial")
        record = json.loads(_lines(notebook)[0])
        prov = record["provenance"]
        assert record["created_at"].endswith("Z")
        assert "shinro_version" in prov
        assert "python" in prov and "platform" in prov
        assert prov["cwd"]
        # Inside this repository the git SHA is available; it identifies the code.
        assert "git_sha" in prov

    def test_non_json_serializable_metrics_report_an_error(self, tmp_path):
        """A bad payload is reported and nothing is written."""
        notebook = tmp_path / "exp.jsonl"
        result = json.loads(record_run(str(notebook), "trial", metrics={"bad": object()}))
        assert "error" in result
        assert "JSON-serializable" in result["error"]
        assert not notebook.exists()

    def test_tags_and_notes_round_trip(self, tmp_path):
        """Tags and notes are stored verbatim."""
        notebook = tmp_path / "exp.jsonl"
        record_run(str(notebook), "trial", notes="ran on the Pi", tags=["motor", "baseline"])
        record = json.loads(_lines(notebook)[0])
        assert record["notes"] == "ran on the Pi"
        assert record["tags"] == ["motor", "baseline"]


class TestListRuns:
    """Reading summaries back out of the log."""

    def test_missing_notebook_is_an_empty_list(self, tmp_path):
        """A notebook that does not exist yet is not an error."""
        data = json.loads(list_runs(str(tmp_path / "nope.jsonl")))
        assert data == {"runs": [], "skipped_lines": 0}

    def test_summaries_are_oldest_first_with_metric_keys(self, tmp_path):
        """Summaries keep insertion order and expose the metric names only."""
        notebook = tmp_path / "exp.jsonl"
        record_run(str(notebook), "one", metrics={"a": 1.0})
        record_run(str(notebook), "two", metrics={"b": 2.0, "c": 3.0}, tags=["x"])
        runs = json.loads(list_runs(str(notebook)))["runs"]
        assert [r["name"] for r in runs] == ["one", "two"]
        assert runs[1]["metrics"] == ["b", "c"]
        assert runs[1]["tags"] == ["x"]

    def test_malformed_lines_are_skipped_and_counted(self, tmp_path):
        """One bad append must not make the rest of the notebook unreadable."""
        notebook = tmp_path / "exp.jsonl"
        record_run(str(notebook), "good")
        with notebook.open("a", encoding="utf-8") as f:
            f.write("{not json}\n")
        data = json.loads(list_runs(str(notebook)))
        assert [r["name"] for r in data["runs"]] == ["good"]
        assert data["skipped_lines"] == 1


class TestGetRun:
    """Fetching a full record by id or by name."""

    def test_by_id(self, tmp_path):
        """An exact id returns that record."""
        notebook = tmp_path / "exp.jsonl"
        run_id = json.loads(record_run(str(notebook), "trial", metrics={"e": 0.5}))["recorded"]
        record = json.loads(get_run(str(notebook), run_id))
        assert record["id"] == run_id
        assert record["metrics"]["e"] == 0.5

    def test_by_name_returns_the_most_recent(self, tmp_path):
        """Names are not unique; the latest match wins."""
        notebook = tmp_path / "exp.jsonl"
        record_run(str(notebook), "sweep", metrics={"e": 1.0})
        record_run(str(notebook), "sweep", metrics={"e": 2.0})
        record = json.loads(get_run(str(notebook), "sweep"))
        assert record["metrics"]["e"] == 2.0

    def test_unknown_run_returns_an_error(self, tmp_path):
        """A name or id that is not in the log is reported."""
        notebook = tmp_path / "exp.jsonl"
        record_run(str(notebook), "trial")
        assert "error" in json.loads(get_run(str(notebook), "missing"))


@pytest.mark.parametrize("bad_path", ["", "."])
def test_degenerate_path_reports_an_error(bad_path, tmp_path):
    """A path that is a directory (or empty) fails loudly instead of raising."""
    result = json.loads(record_run(bad_path or ".", "trial"))
    assert "error" in result
