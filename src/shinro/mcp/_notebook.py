"""Experiment notebook — append-only JSONL run records.

A separate server for a separate job: recording what you tried and what came out,
so an agent can compare trials across sessions instead of losing them when the
process exits.

Design rules:

- **The caller names the file.** No hidden write location, the same rule the lowering
  tools follow. Relative paths resolve against the server's working directory.
- **Append-only.** One JSON object per line. Nothing rewrites or deletes records, so
  concurrent readers never see a half-written notebook and a crashed session cannot
  corrupt earlier entries.
- **Self-describing.** A record carries the components, the metrics, and provenance the
  *server* fills (timestamp, version, git SHA, platform) — the part an agent could
  forget or get wrong. It references nothing from a live session, which is what lets
  this live in its own process.
"""

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shinro.mcp._common import notebook_server as server


def _target(path: str) -> Path:
    """Resolve the notebook path (relative to the server's CWD) and expand ~."""
    return Path(path).expanduser()


def _provenance() -> dict:
    """Environment facts the server records for every run."""
    try:
        from importlib.metadata import version

        shinro_version = version("shinro")
    except ImportError:  # PackageNotFoundError subclasses ImportError
        shinro_version = "unknown"

    prov: dict = {
        "shinro_version": shinro_version,
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "cwd": str(Path.cwd()),
    }
    # A git SHA makes a record traceable to the code that produced it. Absent when the
    # working directory is not a repository, which is why it is a best-effort field.
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=2, check=False
        )
        if sha.returncode == 0 and sha.stdout.strip():
            prov["git_sha"] = sha.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=2, check=False
            )
            prov["git_dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git on PATH
        # Best effort: provenance without git fields is still useful.
        return prov
    return prov


def _read(path: Path) -> tuple[list[dict], int]:
    """Read a notebook, returning records plus the count of unparsable lines.

    A malformed line is skipped rather than raised: an agent should still be able to
    read the rest of its notebook after one bad append.
    """
    if not path.exists():
        return [], 0
    records: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                skipped += 1
    return records, skipped


@server.tool()
def record_run(
    path: str,
    name: str,
    components: dict | None = None,
    metrics: dict | None = None,
    notes: str = "",
    tags: list[str] | None = None,
) -> str:
    """Append one run record to a JSONL notebook, creating the file if needed.

    The record is self-describing: `components` should say *what* ran (type plus the
    config path or params that define it), `metrics` what came out. Provenance
    (timestamp, shinro version, git SHA, platform, working directory) is filled by the
    server, so a record stays traceable to the code that produced it.

    Args:
        path: Notebook file (e.g. ``runs/experiments.jsonl``), relative to the server's
            working directory. Parent directories are created.
        name: Human label for the trial, e.g. ``"pid-motor-ref2"``. Not unique — the
            server assigns a unique ``id`` to every record.
        components: What ran, e.g. ``{"plant": {"type": "DCMotor"}, "controller":
            {"type": "PID", "config": "configs/pid.toml"}}``.
        metrics: What came out, e.g. ``{"final_error": 0.0097, "steps": 600,
            "artifact_bytes": 11706317}``.
        notes: Free text.
        tags: Optional labels for filtering, e.g. ``["motor", "baseline"]``.
    """
    target = _target(path)
    now = datetime.now(UTC)
    record: dict[str, Any] = {
        "id": f"{now:%Y%m%dT%H%M%S}-{os.urandom(2).hex()}",
        "name": name,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "components": components or {},
        "metrics": metrics or {},
        "notes": notes,
        "tags": list(tags or []),
        "provenance": _provenance(),
    }
    try:
        line = json.dumps(record, sort_keys=True)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"components/metrics must be JSON-serializable: {e}"})

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        return json.dumps({"error": f"cannot write notebook '{path}': {e}"})

    return json.dumps({"recorded": record["id"], "name": name, "path": str(target)})


@server.tool()
def list_runs(path: str) -> str:
    """List every run in a notebook, oldest first, as compact summaries.

    Use ``get_run`` for the full record. A missing notebook is an empty list, not an
    error, so this is safe to call before anything has been recorded.

    Args:
        path: Notebook file, relative to the server's working directory.
    """
    records, skipped = _read(_target(path))
    return json.dumps(
        {
            "runs": [
                {
                    "id": r.get("id"),
                    "name": r.get("name"),
                    "created_at": r.get("created_at"),
                    "tags": r.get("tags", []),
                    "metrics": sorted((r.get("metrics") or {}).keys()),
                }
                for r in records
            ],
            "skipped_lines": skipped,
        }
    )


@server.tool()
def get_run(path: str, run: str) -> str:
    """Fetch one full record, matched by ``id`` first, then by ``name`` (most recent).

    Args:
        path: Notebook file, relative to the server's working directory.
        run: A record ``id`` (e.g. ``20261019T154518-a1b2``) or a ``name``. When several
            records share a name, the most recent one is returned.
    """
    records, _ = _read(_target(path))
    for record in records:
        if record.get("id") == run:
            return json.dumps(record)
    named = [r for r in records if r.get("name") == run]
    if named:
        return json.dumps(named[-1])
    return json.dumps({"error": f"No run with id or name '{run}' in {path}"})


def _lookup(record: dict, dotted: str):
    """Read a dotted path (``"metrics.final_error"``) from a record, or ``None``."""
    current: Any = record
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def _metric_paths(records: list[dict]) -> list[str]:
    """Every numeric metric path seen in the notebook, for a helpful error."""
    paths: set[str] = set()
    for record in records:
        for key, value in (record.get("metrics") or {}).items():
            if isinstance(value, dict):
                paths.update(f"metrics.{key}.{sub}" for sub, v in value.items() if isinstance(v, (int, float)))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                paths.add(f"metrics.{key}")
    return sorted(paths)


@server.tool()
def compare_runs(
    path: str,
    metric: str = "metrics.final_error",
    direction: str = "min",
    tags: list[str] | None = None,
    name: str | None = None,
    limit: int = 10,
) -> str:
    """Rank the runs in a notebook by one metric — the "which should I keep?" call.

    Reading the whole log to pick the next hypothesis is what makes a search loop
    chatty; this answers it in one call. Records that lack the metric are counted, not
    ranked, so a notebook with mixed experiments still compares like with like.

    Args:
        path: Notebook file, relative to the server's working directory.
        metric: Dotted path into the record — e.g. ``metrics.final_error`` (default),
            ``metrics.artifact_bytes``, or a nested ``metrics.check.steady_state.error``.
        direction: ``"min"`` (default, lower is better) or ``"max"``.
        tags: Only consider records carrying **all** of these tags.
        name: Only consider records whose ``name`` contains this substring.
        limit: How many ranked rows to return (default 10).
    """
    if direction not in ("min", "max"):
        return json.dumps({"error": f"direction must be 'min' or 'max', got '{direction}'"})

    records, skipped = _read(_target(path))
    if not records:
        return json.dumps({"error": f"no runs in {path}", "skipped_lines": skipped})

    wanted = set(tags or [])
    considered = [
        record
        for record in records
        if wanted.issubset(set(record.get("tags") or [])) and (name is None or name in str(record.get("name", "")))
    ]
    if not considered:
        return json.dumps({"error": f"no runs match tags={sorted(wanted)} name~={name!r}", "total_runs": len(records)})

    ranked: list[dict] = []
    missing = 0
    for record in considered:
        value = _lookup(record, metric)
        if value is None:
            missing += 1
            continue
        ranked.append(
            {
                "id": record.get("id"),
                "name": record.get("name"),
                "created_at": record.get("created_at"),
                "value": value,
                "tags": record.get("tags", []),
            }
        )

    if not ranked:
        return json.dumps(
            {
                "error": f"none of the {len(considered)} considered runs has a numeric '{metric}'",
                "available_metrics": _metric_paths(considered),
            }
        )

    ranked.sort(key=lambda row: row["value"], reverse=(direction == "max"))
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position

    return json.dumps(
        {
            "metric": metric,
            "direction": direction,
            "considered": len(considered),
            "missing_metric": missing,
            "skipped_lines": skipped,
            "best": ranked[0],
            "ranked": ranked[: max(1, limit)],
        }
    )
