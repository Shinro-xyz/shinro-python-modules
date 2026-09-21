"""Stamp a deployment record for a built ``libbase.so``.

Post-compile step: after ``zig build`` produces ``<prefix>/lib/lib<name>.so``,
this module reads the build manifest (``lib<name>.manifest.json``), hashes the
binary and the baked solver tree, and writes a deterministic deployment
record (``lib<name>.deployment.json``) carrying a single **master hash** that
commits to the whole config -> graph -> solver -> binary chain.

The master hash is a pure function of its inputs (no timestamps in the
record; the timestamp lives only in the archive filename), so identical
inputs produce byte-identical records — the diffable audit record for
"which estimator/controller pair is deployed".

The record also carries **build provenance** (``target`` / ``optimize`` /
``stripped`` / ``zig_version``, copied verbatim from the build manifest) and an
**oracle** block (whether the .so was checked against the interpreter, with the
sample count / tolerance / max error). Neither is folded into the master hash:
the binary slot already distinguishes a Debug from a ReleaseFast build, so
those fields explain *why* the hash changed rather than defining it.

Run: ``python3 -m shinro.codegen.stamp --prefix build/``
(wired into ``make zig-build``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from shinro.codegen.runtime_paths import runtime_root

# Fixed sentinel for absent slots (e.g. no solver on LQR/PID graphs, or no
# config provenance). Keeps the master-hash format uniform so LQR/PID and MPC
# deployments stay structurally comparable.
SENTINEL = hashlib.sha256(b"").hexdigest()

#: Oracle block written when ``stamp()`` is called without oracle results
#: (e.g. the Makefile's shipped-graph build, which does not run the ctypes
#: oracle). Copied per record so callers can never mutate the shared default.
_ORACLE_NOT_RUN = {"status": "not_run", "reason": "stamp invoked without oracle results"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _slot_from_configs(configs: dict[str, str]) -> str:
    """Canonical slot over sorted (path, sha256) pairs; sentinel if none."""
    if not configs:
        return SENTINEL
    parts = [f"{path}\x00{configs[path]}" for path in sorted(configs)]
    return _sha256_bytes("\n".join(parts).encode())


def _slot_from_dir(solver_dir: Path | None) -> str:
    """Flat hash over sorted files in a bake dir; sentinel if absent."""
    if solver_dir is None or not solver_dir.is_dir():
        return SENTINEL
    files = sorted(p for p in solver_dir.rglob("*") if p.is_file())
    parts = [f"{p.relative_to(solver_dir).as_posix()}\x00{_sha256_file(p)}" for p in files]
    return _sha256_bytes("\n".join(parts).encode())


def _master(config_slot: str, graph_slot: str, solver_slot: str, binary_slot: str) -> str:
    return _sha256_bytes(
        f"config={config_slot}\ngraph={graph_slot}\nsolver={solver_slot}\nbinary={binary_slot}\n".encode()
    )


def stamp(
    prefix: Path,
    build_root: Path,
    name: str = "libbase",
    oracle: dict | None = None,
    native_record: tuple[Path, dict] | None = None,
) -> dict:
    """Compute and write the deployment record for a built prefix dir.

    Args:
        prefix: The ``zig build --prefix`` directory (contains ``lib/``).
        build_root: The zig build root, used to resolve a relative
            ``solver_dir`` recorded in the manifest (default: the packaged
            ``src/shinro/runtime/``).
        name: Artifact stem — ``<name>.so`` / ``<name>.manifest.json``
            (default ``libbase``). Must match the ``-Dname`` the build used.
        oracle: Oracle outcome to record under ``"oracle"`` — e.g.
            ``{"status": "passed", "samples": 20, ...}`` or
            ``{"status": "not_run", "reason": "cross-compiled"}``. When
            ``None`` the record notes the oracle was not run.
        native_record: Optional ``(path, record_dict)`` of the oracle-verified
            *native* build of the same config/graph/solver — set for a
            cross-compiled build, so the record references the verification it
            cannot run itself. Adds a top-level ``"native_ref"`` block with the
            native master hash and whether the config/graph/solver slots agree
            (only the binary slot legitimately differs across targets).

    Returns:
        The deployment record dict (also written to disk).
    """
    lib_dir = prefix / "lib"
    manifest_path = lib_dir / f"{name}.manifest.json"
    so_path = lib_dir / f"{name}.so"

    if not manifest_path.exists():
        raise FileNotFoundError(f"no build manifest at {manifest_path}")
    if not so_path.exists():
        raise FileNotFoundError(f"no {so_path.name} at {so_path}")

    manifest = json.loads(manifest_path.read_text())

    graph_sha = manifest["provenance"]["graph_sha256"]
    solver_dir_raw = manifest["provenance"]["solver_dir"]

    graph_block = manifest.get("graph") or {}
    graph_prov = graph_block.get("provenance") or {}
    configs = graph_prov.get("configs") or {}

    solver_dir = None
    if solver_dir_raw:
        p = Path(solver_dir_raw)
        solver_dir = p if p.is_absolute() else build_root / p

    config_slot = _slot_from_configs(configs)
    graph_slot = graph_sha
    solver_slot = _slot_from_dir(solver_dir)
    binary_slot = _sha256_file(so_path)
    master = _master(config_slot, graph_slot, solver_slot, binary_slot)

    record = {
        "master_hash": master,
        "slots": {
            "config": config_slot,
            "graph": graph_slot,
            "solver": solver_slot,
            "binary": binary_slot,
        },
        "config": {
            "files": configs,
            "python_version": graph_prov.get("python_version"),
            "numpy_version": graph_prov.get("numpy_version"),
        },
        "build": {
            "target": manifest.get("target"),
            "optimize": manifest.get("optimize"),
            "stripped": manifest.get("stripped"),
            "zig_version": manifest.get("zig_version"),
            "libc": manifest.get("libc"),
            "float_type": manifest.get("float_type"),
        },
        "oracle": dict(oracle) if oracle is not None else dict(_ORACLE_NOT_RUN),
        "graph": {
            "sha256": graph_sha,
            "has_solve_qp": graph_block.get("has_solve_qp"),
        },
        "solver": manifest.get("solver"),
        "binary": {"sha256": binary_slot, "path": str(so_path)},
    }

    # A cross-compiled build cannot run the ctypes oracle here; when the
    # oracle-verified native build of the same config/graph/solver is known,
    # reference it so the record points at the verification that does cover it.
    if native_record is not None:
        npath, nrec = native_record
        nslots = nrec.get("slots") or {}
        matches = all(record["slots"][k] == nslots.get(k) for k in ("config", "graph", "solver"))
        record["native_ref"] = {
            "record": str(npath),
            "master_hash": nrec.get("master_hash"),
            "slots_match": matches,
        }

    record_path = lib_dir / f"{name}.deployment.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    # Archive copy: timestamp in filename only, so the record stays a pure
    # function of its inputs (mirrors the build-manifest convention).
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    archive_dir = prefix / "deployments"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{ts}-{master[:8]}.json"
    archive_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"wrote {record_path} (master={master})")
    print(f"wrote {archive_path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Stamp a deployment record for a built kernel (lib<name>.so).")
    parser.add_argument("--prefix", default="build", help="zig build prefix dir (default: build)")
    parser.add_argument(
        "--build-root",
        default=str(runtime_root()),
        help="zig build root for resolving a relative solver_dir (default: the packaged runtime)",
    )
    parser.add_argument("--name", default="libbase", help="artifact stem, e.g. libbase → libbase.so (default: libbase)")
    args = parser.parse_args()
    stamp(Path(args.prefix), Path(args.build_root), args.name)


if __name__ == "__main__":
    main()
