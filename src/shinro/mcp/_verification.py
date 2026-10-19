"""MCP tools for the verification gates and the zig stage.

The checks that sit between a lowered graph and a deployment
(``shinro.verification`` + the build stage in ``shinro.scenarios.build``):

    run_gate_a          interpret(graph) vs the live estimator+controller
    build_kernel        the zig stage: integrity + gate A + build + oracle + stamp + verify
    run_oracle          .so vs interpret(), on random inputs (oracle B)
    stamp_deployment    write the deployment record (master hash over the slots)
    verify_deployment   re-hash the artifacts against the record (drift gate)

Gate A and oracle B are complementary: gate A proves the *graph* reproduces the
components, oracle B proves the *Zig VM* reproduces the graph. Only both make a
stamp's "verified" claim true.

**stdout discipline.** ``build_scenario``, ``stamp`` and ``verify`` print progress
and failures. stdout is the JSON-RPC channel of this server, so every call into
them runs under a redirect and the captured text is returned inside the JSON
result instead of being written to the transport.
"""

import contextlib
import io
import json
import shutil
from pathlib import Path

from shinro.codegen.runtime_paths import runtime_root
from shinro.mcp._common import _graph_store, server
from shinro.scenarios.build import GATE_A_TICKS, TOL_GATE_A
from shinro.scenarios.build import build_scenario as _build_scenario
from shinro.scenarios.gen import load_scenario
from shinro.scenarios.recipes import available_graphs, build_recipe, live_components_and_plant
from shinro.verification.gate_a import run_gate_a as _run_gate_a
from shinro.verification.oracle import load_so, tol_for
from shinro.verification.oracle import run_oracle as _run_oracle
from shinro.verification.stamp import stamp as _stamp
from shinro.verification.verify import verify as _verify

#: exit code -> status, mirroring :mod:`shinro.scenarios.build`.
_EXIT_STATUS = {
    0: "ok",
    1: "untraceable",
    2: "usage_error",
    3: "oracle_mismatch",
    4: "build_or_verify_failed",
    5: "zig_missing",
}


def _capture(fn, *args, **kwargs):
    """Call ``fn`` with stdout/stderr redirected; return ``(result, log)``.

    The build pipeline narrates on stdout; on a stdio MCP server that would
    corrupt the JSON-RPC stream, so the narration is captured and handed back
    as a result field.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning ``None`` when absent or unparseable."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _graph_for(scenario_path: str | None, name: str | None):
    """Resolve the graph to verify: a stored graph, else a fresh one from the scenario."""
    if name is not None:
        cg = _graph_store.get(name)
        if cg is None:
            raise KeyError(f"No graph named '{name}'. Trace or generate one first.")
        return cg
    if scenario_path is None:
        raise ValueError("provide either 'name' (a stored graph) or 'scenario_path'")
    spec = load_scenario(scenario_path)
    return build_recipe(spec["compile"]["recipe"], spec)


@server.tool()
def run_gate_a(scenario_path: str, ticks: int = GATE_A_TICKS, seed: int = 0) -> str:
    """Gate A: does the composed graph reproduce the live estimator + controller?

    Builds the graph and the live components from the same scenario, drives both
    over ``ticks`` closed-loop steps with shared random sensor readings, and
    compares the control. Runs before any compile time is spent: a mismatch means
    the tracer or composer lost a computation, so the build refuses to stamp.

    Only ``closed_loop_tracking`` scenarios have a live pair to drive; a
    policy-only graph has no closed loop, so this reports that instead.

    Args:
        scenario_path: Scenario TOML path.
        ticks: Closed-loop steps to compare (default 50; > 1 so a frozen
            recurrence only diverges after the first tick).
        seed: RNG seed for the shared y / x_ref / host inputs.
    """
    try:
        spec = load_scenario(scenario_path)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})

    recipe = spec["compile"]["recipe"]
    if recipe not in available_graphs():
        return json.dumps({"error": f"unknown recipe '{recipe}' (registered: {available_graphs()})"})
    if recipe != "closed_loop_tracking":
        return json.dumps(
            {
                "error": (
                    f"gate A needs a live estimator + controller to drive; recipe '{recipe}' "
                    "has no closed loop (policy-only graphs are covered by oracle B)"
                )
            }
        )

    try:
        cg = build_recipe(recipe, spec)
        est, ctrl, n_x, n_u, limits, plant = live_components_and_plant(spec)
        max_err, log = _capture(
            _run_gate_a, cg, est, ctrl, n_x, n_u, input_limits=limits, ticks=ticks, seed=seed, plant=plant
        )
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})
    except NotImplementedError as e:
        return json.dumps({"error": f"not traceable: {e}"})

    return json.dumps(
        {
            "passed": max_err <= TOL_GATE_A,
            "max_abs_err": max_err,
            "tolerance": TOL_GATE_A,
            "ticks": ticks,
            "seed": seed,
            "nodes": len(cg.graph.nodes),
            "log": log.strip(),
        }
    )


@server.tool()
def build_kernel(
    graph_dir: str,
    scenario_path: str | None = None,
    prefix: str | None = None,
    optimize: str | None = None,
    target: str | None = None,
    solver_dir: str | None = None,
    solver: str | None = None,
    artifact_name: str | None = None,
    samples: int = 20,
    seed: int = 0,
) -> str:
    """Run the zig stage: compile a lowered graph into a verified, stamped kernel.

    This is ``make compile`` stage 2 / ``shinro-compile``. In order: zig presence,
    build flags, graph-vs-scenario integrity, gate A, ``zig build``, oracle B
    (``.so`` vs interpreter), stamp, and verify. Requires ``zig`` on PATH.

    Omitting ``scenario_path`` builds and stamps the graph *unverified* (no
    integrity check, no oracles) — the same loud-warning path the CLI takes.

    ``prefix`` defaults to ``graph_dir``, i.e. the artifacts land in
    ``<graph_dir>/lib/<artifact>.so`` (+ ``.manifest.json`` / ``.deployment.json``).
    For a QP (MPC) graph a solver is required: pass ``solver="emosqp"`` to bake
    one on demand (needs ``scenario_path``), or ``solver_dir`` to reuse a bake.

    Args:
        graph_dir: Directory holding ``graph_data.zig`` + its manifest
            (the ``out_dir`` of generate_scenario).
        scenario_path: Scenario TOML — enables the integrity check and oracles.
        prefix: ``zig build --prefix`` dir (default: ``graph_dir``).
        optimize: ``"debug"`` or ``"release"`` (ReleaseFast) — overrides the TOML.
        target: Zig target triple (e.g. ``aarch64-linux-gnu``); a non-native
            target skips the host oracle, which cannot dlopen it.
        solver_dir: Pre-baked OSQP solver dir.
        solver: Solver name to bake on demand (e.g. ``"emosqp"``).
        artifact_name: Kernel stem, installed as ``lib/<name>.so``.
        samples: Random inputs for oracle B (default 20).
        seed: RNG seed for the oracles (default 0).
    """
    prefix_path = Path(prefix) if prefix else Path(graph_dir)
    artifact = artifact_name or "libbase"

    try:
        code, log = _capture(
            _build_scenario,
            graph_dir,
            scenario=scenario_path,
            prefix=prefix,
            optimize=optimize,
            target=target,
            solver_dir=solver_dir,
            solver=solver,
            artifact_name=artifact_name,
            samples=samples,
            seed=seed,
        )
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})

    manifest = _read_json(prefix_path / "lib" / f"{artifact}.manifest.json")
    if manifest and artifact_name is None:
        # The scenario's [compile].artifact_name may differ from the default; the
        # build wrote that name, so prefer the artifact actually on disk.
        for candidate in sorted((prefix_path / "lib").glob("*.manifest.json")):
            manifest = _read_json(candidate) or manifest
            artifact = candidate.name.removesuffix(".manifest.json")
    record = _read_json(prefix_path / "lib" / f"{artifact}.deployment.json")
    so_path = prefix_path / "lib" / f"{artifact}.so"

    return json.dumps(
        {
            "exit_code": code,
            "status": _EXIT_STATUS.get(code, f"unknown({code})"),
            "artifact": str(so_path) if so_path.exists() else None,
            "artifact_bytes": so_path.stat().st_size if so_path.exists() else None,
            "oracle": (record or {}).get("oracle"),
            "master_hash": (record or {}).get("master_hash"),
            "deployment_record": str(prefix_path / "lib" / f"{artifact}.deployment.json")
            if record
            else None,
            "log": log.strip().splitlines()[-40:],
        }
    )


@server.tool()
def run_oracle(
    prefix: str,
    name: str | None = None,
    scenario_path: str | None = None,
    artifact_name: str = "libbase",
    graph_dir: str | None = None,
    samples: int = 20,
    seed: int = 0,
    tolerance: float | None = None,
) -> str:
    """Oracle B: compare the compiled ``.so`` against the interpreter on random inputs.

    dlopens ``<prefix>/lib/<artifact_name>.so``, drives ``shinro_step`` through
    ctypes and compares every output and recurrent-state port against
    ``interpret(graph)``. This proves the Zig VM executes the graph faithfully;
    gate A proves the graph represents the components.

    The tolerance defaults to the tier for the graph (1e-3 when it contains a
    ``.solve_qp`` node, otherwise 1e-12), read from the graph manifest.

    Args:
        prefix: The ``zig build --prefix`` dir containing ``lib/``.
        name: Stored graph to compare (else ``scenario_path`` rebuilds it).
        scenario_path: Scenario TOML to rebuild the graph from.
        artifact_name: Artifact stem — ``lib/<name>.so`` (default ``libbase``).
        graph_dir: Dir holding ``graph_data_manifest.json`` (default ``prefix``).
        samples: Random inputs to compare (default 20).
        seed: RNG seed (default 0).
        tolerance: Explicit tolerance override.
    """
    if shutil.which("zig") is None:
        return json.dumps({"error": "zig is not on PATH — build the kernel first (`build_kernel`)"})

    try:
        cg = _graph_for(scenario_path, name)
    except (KeyError, ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})
    except NotImplementedError as e:
        return json.dumps({"error": f"not traceable: {e}"})

    manifest = _read_json(Path(graph_dir or prefix) / "graph_data_manifest.json") or {}
    tol = tolerance if tolerance is not None else tol_for(manifest)

    try:
        lib = load_so(prefix, artifact_name)
        max_err = _run_oracle(lib, cg, samples, seed)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except (ValueError, NotImplementedError) as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    return json.dumps(
        {
            "passed": max_err < tol,
            "max_abs_err": max_err,
            "tolerance": tol,
            "samples": samples,
            "seed": seed,
            "has_solve_qp": bool(manifest.get("has_solve_qp")),
        }
    )


@server.tool()
def stamp_deployment(
    prefix: str,
    artifact_name: str = "libbase",
    oracle: dict | None = None,
    native_record: str | None = None,
) -> str:
    """Write the deployment record for a built kernel.

    Computes the master hash over the four slots (config, graph, solver, binary)
    and writes ``lib/<name>.deployment.json``, plus the build-env archive. Pass
    the ``oracle`` block returned by ``run_oracle`` so the record carries the
    verification outcome it claims.

    Args:
        prefix: The ``zig build --prefix`` dir containing ``lib/``.
        artifact_name: Artifact stem (must match the build's ``-Dname``).
        oracle: Oracle outcome dict (e.g. from run_oracle: status/max_abs_err/...).
        native_record: For a cross build, path to the oracle-verified native
            deployment record to reference.
    """
    native = None
    if native_record is not None:
        record = _read_json(Path(native_record))
        if record is None:
            return json.dumps({"error": f"cannot read native record at {native_record}"})
        native = (Path(native_record), record)

    try:
        record, log = _capture(_stamp, Path(prefix), runtime_root(), artifact_name, oracle=oracle, native_record=native)
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    return json.dumps(
        {
            "record_path": str(Path(prefix) / "lib" / f"{artifact_name}.deployment.json"),
            "master_hash": record.get("master_hash"),
            "slots": record.get("slots"),
            "oracle": record.get("oracle"),
            "log": log.strip().splitlines(),
        }
    )


@server.tool()
def verify_deployment(
    record_path: str,
    binary_path: str | None = None,
    graph_path: str | None = None,
    solver_dir: str | None = None,
) -> str:
    """Re-hash the deployed artifacts against a deployment record (drift gate).

    Independent verification of the record's slots — the producer/verifier
    separation: recompute the config, graph, solver and binary hashes and compare.
    A mismatch means the artifact on disk is not the one that was verified.

    Args:
        record_path: Path to ``lib<name>.deployment.json``.
        binary_path: The deployed ``.so`` (default: the record's binary path).
        graph_path: ``graph_data.zig`` to re-hash (build-environment check).
        solver_dir: Baked solver dir to re-hash (build-environment check).
    """
    log: str
    try:
        code, log = _capture(
            _verify,
            Path(record_path),
            Path(binary_path) if binary_path else None,
            Path(graph_path) if graph_path else None,
            Path(solver_dir) if solver_dir else None,
        )
    except (OSError, ValueError, KeyError) as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    failures = [
        line.strip().removeprefix("- ").strip()
        for line in log.splitlines()
        if line.strip().startswith("- ")
    ]
    return json.dumps({"ok": code == 0, "exit_code": code, "failures": failures, "log": log.strip()})
