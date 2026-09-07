"""Verify a deployment record against the artifacts on disk.

Independent re-hash of the deployed artifacts against ``libbase.deployment.json``
(the producer/verifier separation IEC 61508 favors): recompute the binary,
config, graph, and solver hashes and compare to the record. Exits non-zero on
any drift, so it can gate a deploy or a post-deploy audit.

On a robot, pass ``--binary`` pointing at the deployed ``.so``; the config
files are always re-hashed (the drift-detection core). ``--graph`` and
``--solver-dir`` verify the build-environment artifacts that are compiled
into the binary.

Run: ``python3 scripts/verify_deployment.py --record build/lib/libbase.deployment.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from stamp_deployment import _slot_from_dir
except ImportError:  # imported as scripts.verify_deployment (e.g. from tests)
    from scripts.stamp_deployment import _slot_from_dir

from shinro.utils.config_resolver import resolve_config_path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(name: str, expected: str, actual: str, failures: list[str]) -> None:
    if expected != actual:
        failures.append(f"{name}: expected {expected}, got {actual}")


def verify(
    record_path: Path,
    binary_path: Path | None = None,
    graph_path: Path | None = None,
    solver_dir: Path | None = None,
) -> int:
    """Re-hash artifacts and compare against the deployment record.

    Args:
        record_path: Path to ``libbase.deployment.json``.
        binary_path: Deployed ``.so`` (default: the record's binary path).
        graph_path: ``graph_data.zig`` to verify (build-environment check).
        solver_dir: Baked solver dir to verify (build-environment check).

    Returns:
        0 if everything matches, 1 otherwise.
    """
    record = json.loads(record_path.read_text())
    failures: list[str] = []

    so = binary_path or Path(record["binary"]["path"])
    if not so.exists():
        failures.append(f"binary: {so} not found")
    else:
        _check("binary", record["slots"]["binary"], _sha256_file(so), failures)

    for cfg, expected in (record["config"]["files"] or {}).items():
        try:
            resolved = resolve_config_path(cfg)
        except FileNotFoundError:
            failures.append(f"config: {cfg} not found")
            continue
        _check(f"config {cfg}", expected, _sha256_file(resolved), failures)

    if graph_path is not None:
        if not graph_path.exists():
            failures.append(f"graph: {graph_path} not found")
        else:
            _check("graph", record["slots"]["graph"], _sha256_file(graph_path), failures)
    if solver_dir is not None:
        _check("solver", record["slots"]["solver"], _slot_from_dir(solver_dir), failures)

    if failures:
        print("VERIFY FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK: {record_path} matches artifacts (master={record['master_hash']})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a deployment record against artifacts on disk.")
    parser.add_argument("--record", required=True, help="path to libbase.deployment.json")
    parser.add_argument("--binary", help="path to the deployed libbase.so (default: record's binary path)")
    parser.add_argument("--graph", help="path to graph_data.zig to verify (build-environment check)")
    parser.add_argument("--solver-dir", help="path to the baked solver dir to verify (build-environment check)")
    args = parser.parse_args()
    sys.exit(
        verify(
            Path(args.record),
            Path(args.binary) if args.binary else None,
            Path(args.graph) if args.graph else None,
            Path(args.solver_dir) if args.solver_dir else None,
        )
    )


if __name__ == "__main__":
    main()
