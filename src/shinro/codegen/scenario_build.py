"""Compile a generated scenario graph into a verified, stamped ``libbase.so``.

The zig-required stage of the e2e pipeline. Takes the graph pair produced by
:func:`shinro.codegen.scenario_gen.gen_scenario` (``graph_data.zig`` + its
manifest), compiles the comptime VM via the packaged ``runtime/build.zig``,
and — when ``--scenario`` is given — verifies the result before stamping:

1. **Pre-flight** — ``zig`` must be on PATH (loud error, exit 5).
2. **Build flags** — from the scenario's ``[compile]`` section (``optimize`` /
   ``target`` / ``solver`` / ``solver_dir``), overridable on the CLI.
   ``optimize = "release"`` maps to ``-Doptimize=ReleaseFast`` (the only
   validated release mode); ``target`` cross-compiles (e.g.
   ``aarch64-linux-gnu``). A QP (MPC) graph needs a solver: ``solver =
   "emosqp"`` bakes one on demand into ``<graph_dir>/emosqp`` (reused when
   current), while ``solver_dir`` consumes a pre-baked tree.
3. **Integrity check** — re-runs the gen stage in-process and byte-compares the
   fresh graph manifest against the on-disk one, proving the on-disk graph is
   exactly what this scenario produces (deterministic lowering).
4. **Gate A** — drives the live estimator + controller and ``interpret(composed
   graph)`` over N ticks and compares the control: does the graph reproduce the
   components' math? Mismatch → exit 3, *before* any compile time is spent. Only
   for ``closed_loop_tracking`` — a policy-only graph has no live pair.
5. **Build** — ``zig build -Dgraph=<abs path>`` into an isolated prefix, so the
   shared ``src/shinro/runtime/graph_data.zig`` is never touched.
6. **Oracle B** — loads the ``.so`` via ctypes and compares ``shinro_step``
   against ``interpret()`` on N random inputs across every output and state
   port (tolerance 1e-12; 1e-3 for QP graphs). Mismatch → exit 3. Skipped
   for non-native targets — a cross-compiled ``.so`` cannot be dlopen'd on
   the host, so the oracle only runs when the target matches the host
   architecture (``native`` or an explicit host triple); the integrity
   check still runs either way.
7. **Stamp + verify** — writes the deployment record (master hash over
   config/graph/solver/binary, plus the build provenance and the oracle
   outcome) and re-hashes the artifacts against it.

The two oracles are complementary layers of *equivalence*: **gate A** (pre-
compile, numpy) proves ``interpret(graph) == the live components`` — the
tracer/composer is faithful; **oracle B** (post-compile) proves
``.so == interpret(graph)`` — the Zig VM is faithful. Neither alone certifies
the deployed artifact; only both together, plus the integrity check, make the
stamp's "verified" claim true.

Without ``--scenario`` the build is stamped only — no oracle, no integrity
check — and prints a loud warning that the artifact is unverified.

Run::

    python3 -m shinro.codegen.scenario_build build/base --scenario tests/integration/scenarios/base_tracking.toml

Exit codes: 0 ok · 2 usage/config · 3 oracle mismatch · 4 build/verify failure · 5 zig missing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from shinro.codegen.oracle import load_so, run_oracle, tol_for
from shinro.codegen.runtime_paths import runtime_root
from shinro.codegen.scenario_gen import gen_scenario, load_scenario
from shinro.codegen.stamp import stamp
from shinro.codegen.verify import verify

RUNTIME = runtime_root()

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ORACLE = 3
EXIT_BUILD = 4
EXIT_NO_ZIG = 5


class BuildError(RuntimeError):
    """The zig build failed or produced no artifact."""


# ─── pipeline stages ─────────────────────────────────────────────────────────


def _check_zig() -> bool:
    if shutil.which("zig") is not None:
        return True
    print("ERROR: zig not on PATH — the e2e workflow needs it to compile the kernel.", file=sys.stderr)
    print("Install: https://ziglang.org/download (or your package manager).", file=sys.stderr)
    print("Nothing was built.", file=sys.stderr)
    return False


def _build(graph_path: Path, prefix: Path, optimize: str, target: str, solver_dir: str | None, name: str = "libbase") -> None:
    """Compile the comptime VM against the given graph into an isolated prefix."""
    cmd = [
        "zig",
        "build",
        "--build-file",
        str(RUNTIME / "build.zig"),
        "--prefix",
        str(prefix),
        f"-Dgraph={graph_path}",
        f"-Dname={name}",
    ]
    if optimize == "release":
        cmd += ["-Doptimize=ReleaseFast"]
    if target != "native":
        cmd += [f"-Dtarget={target}"]
    if solver_dir is not None:
        cmd += [f"-Dsolver_dir={Path(solver_dir).resolve()}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(f"zig build failed:\n{result.stderr.strip()[:2000]}")


def _bake_on_demand(solver_name: str, spec: dict, graph_dir: Path, manifest: dict) -> str:
    """Bake the QP solver for the scenario's controller into ``<graph_dir>/emosqp``.

    Reuses an existing bake when its ``n_vars`` and controller-config sha match
    (:func:`shinro.codegen.bake.bake_is_current`), so re-running a build does not
    regenerate the solver. Returns the bake dir to pass as ``solver_dir``.
    """
    from shinro.codegen.bake import SOLVERS, bake_emosqp, bake_is_current

    if solver_name not in SOLVERS:
        raise ValueError(f"unknown solver '{solver_name}' (registered: {sorted(SOLVERS)})")
    expected = (manifest.get("solve_qp") or {}).get("expected_n_vars")
    if expected is None:
        raise ValueError("graph manifest has no solve_qp.expected_n_vars")
    config = spec["controller_config"]
    bake_dir = graph_dir / "emosqp"
    if bake_is_current(str(bake_dir), expected, config):
        print(f"reusing current {solver_name} bake at {bake_dir}")
    else:
        print(f"baking {solver_name} solver for {config} (n_vars={expected}) -> {bake_dir}")
        baked = bake_emosqp(config, str(bake_dir))
        if baked.n_vars != expected:
            raise ValueError(f"baked n_vars={baked.n_vars} != graph expected_n_vars={expected}")
    return str(bake_dir)


def _check_graph_integrity(scenario_path: str, graph_dir: Path):
    """Re-gen the scenario and byte-compare its manifest against the on-disk one.

    Lowering is deterministic, so a mismatch means the ``.so`` was built from a
    stale or different graph than the scenario currently produces. Returns the
    fresh :class:`ComposedGraph` for the oracle.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fresh_cg, fresh_path = gen_scenario(scenario_path, tmp)
        fresh_manifest = Path(fresh_path).with_name("graph_data_manifest.json")
        on_disk = graph_dir / "graph_data_manifest.json"
        if not on_disk.exists():
            raise ValueError(f"no graph manifest at {on_disk} — run gen_scenario.py first")
        if on_disk.read_bytes() != fresh_manifest.read_bytes():
            raise ValueError(
                f"graph at {graph_dir} is stale — it does not match what {scenario_path} "
                f"currently produces. Re-run gen_scenario.py before building."
            )
    return fresh_cg


#: Gate A runs this many ticks; > 1 so a frozen recurrence (identical at tick
#: 0) only passes if the graph actually threads state back correctly.
GATE_A_TICKS = 50
#: Gate A tolerance. ``interpret()`` and the live components run the same numpy
#: ops in the same order, so agreement is bit-exact in practice; a small epsilon
#: leaves room for platform FP without masking a real divergence.
TOL_GATE_A = 1e-9


def _resolve_native_ref(graph_dir: Path, optimize: str, name: str, explicit: str | None) -> tuple[Path, dict] | None:
    """Locate the oracle-verified native record for a cross build (or ``None``).

    Explicit ``--native-record`` wins; otherwise the per-mode convention: the
    sibling ``<graph_dir>/../<optimize>-native/lib/<name>.deployment.json``.
    """
    path = Path(explicit) if explicit else graph_dir.parent / f"{optimize}-native" / "lib" / f"{name}.deployment.json"
    if not path.exists():
        return None
    try:
        return path, json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _run_gate_a(cg, spec: dict, seed: int) -> int | None:
    """Pre-compile gate A: ``interpret(cg)`` vs the live estimator+controller.

    Returns an exit code on mismatch, or ``None`` to proceed. Only the
    ``closed_loop_tracking`` recipe has an estimator+controller pair to drive
    live; a policy-only graph has no live closed loop to compare.
    """
    from shinro.codegen.gate_a import run_gate_a
    from shinro.codegen.recipes import live_components

    try:
        est, ctrl, n_x, n_u, limits = live_components(spec)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_USAGE
    max_err = run_gate_a(cg, est, ctrl, n_x, n_u, input_limits=limits, ticks=GATE_A_TICKS, seed=seed)
    if max_err > TOL_GATE_A:
        print(
            f"GATE A MISMATCH: composed graph diverged from the live components "
            f"(max abs err {max_err:.3e} > {TOL_GATE_A:.1e}); the tracer/composer does "
            f"not reproduce the component math — refusing to stamp.",
            file=sys.stderr,
        )
        return EXIT_ORACLE
    print(f"gate A (interpret vs live): {GATE_A_TICKS} ticks, max abs err {max_err:.3e} ✓")
    return None


def build_scenario(
    graph_dir: str | Path,
    scenario: str | None = None,
    prefix: str | None = None,
    optimize: str | None = None,
    target: str | None = None,
    solver_dir: str | None = None,
    solver: str | None = None,
    native_record: str | None = None,
    artifact_name: str | None = None,
    samples: int = 20,
    seed: int = 0,
) -> int:
    """Compile a generated graph into a verified, stamped kernel (``lib<name>.so``).

    Args:
        graph_dir: Directory containing ``graph_data.zig`` + its manifest
            (the ``--out`` of :func:`shinro.codegen.scenario_gen.gen_scenario`).
        scenario: Scenario TOML path. When given, enables the integrity check
            and the oracle; when omitted, the artifact is built and stamped but
            NOT verified (a loud warning is printed).
        prefix: ``zig build --prefix`` directory (default: ``graph_dir``).
        optimize: Override ``[compile].optimize`` (``"debug"``/``"release"``).
        target: Override ``[compile].target`` (zig triple, e.g.
            ``aarch64-linux-gnu``).
        solver_dir: Override ``[compile].solver_dir`` (pre-baked OSQP solver dir).
        solver: Override ``[compile].solver`` — bake this solver on demand
            (e.g. ``"emosqp"``) into ``<graph_dir>/emosqp``.
        native_record: For a cross-compiled build, path to the oracle-verified
            native record to reference (default: the per-mode sibling
            ``<graph_dir>/../<optimize>-native/lib/<name>.deployment.json``).
        artifact_name: Override ``[compile].artifact_name`` — the kernel is
            installed as ``lib/<name>.so`` (default ``libbase`` → ``libbase.so``).
        samples: Random inputs for the oracle (default 20).
        seed: RNG seed for the oracle (default 0).

    Returns:
        An exit code: 0 ok · 2 usage/config · 3 oracle mismatch · 4
        build/verify failure · 5 zig missing.
    """
    if not _check_zig():
        return EXIT_NO_ZIG

    graph_dir = Path(graph_dir)
    graph_path = (graph_dir / "graph_data.zig").resolve()
    if not graph_path.exists():
        print(f"ERROR: no graph at {graph_path} — run gen_scenario.py first", file=sys.stderr)
        return EXIT_USAGE
    manifest_path = graph_dir / "graph_data_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no graph manifest at {manifest_path} — run gen_scenario.py first", file=sys.stderr)
        return EXIT_USAGE
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: graph manifest at {manifest_path} is corrupt: {e}", file=sys.stderr)
        return EXIT_USAGE

    # Build flags: CLI > [compile] TOML > defaults.
    opt, tgt, sdir, name = "debug", "native", None, "libbase"
    solver_name: str | None = None
    spec: dict | None = None
    if scenario:
        try:
            spec = load_scenario(scenario)
        except (ValueError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return EXIT_USAGE
        opt = spec["compile"]["optimize"]
        tgt = spec["compile"]["target"]
        sdir = spec["compile"]["solver_dir"]
        solver_name = spec["compile"]["solver"]
        name = spec["compile"]["artifact_name"]
    if optimize:
        opt = optimize
    if target:
        tgt = target
    if solver_dir:
        sdir = solver_dir
    if solver:
        solver_name = solver
    if artifact_name:
        name = artifact_name

    # A QP graph links a baked OSQP solver. Either consume a pre-baked
    # solver_dir, or bake one on demand (into the isolated <graph_dir>/emosqp).
    if manifest["has_solve_qp"]:
        if sdir and solver_name:
            print(
                "ERROR: QP graph has both a solver_dir and [compile].solver — they are "
                "mutually exclusive (solver_dir reuses a pre-baked solver; solver bakes one).",
                file=sys.stderr,
            )
            return EXIT_USAGE
        if not sdir and not solver_name:
            print(
                "ERROR: graph contains a .solve_qp node but no solver — set "
                "[compile].solver (bake on demand) or [compile].solver_dir / "
                "--solver-dir (reuse a pre-baked OSQP solver).",
                file=sys.stderr,
            )
            return EXIT_USAGE
        if solver_name:
            if spec is None:
                print(
                    "ERROR: [compile].solver needs --scenario (it bakes from the "
                    "scenario's controller config).",
                    file=sys.stderr,
                )
                return EXIT_USAGE
            try:
                sdir = _bake_on_demand(solver_name, spec, graph_dir, manifest)
            except (ValueError, RuntimeError) as e:
                print(f"BAKE FAILED: {e}", file=sys.stderr)
                return EXIT_BUILD
    elif solver_name:
        print(
            "ERROR: [compile].solver is set but the graph has no .solve_qp node "
            "(the controller is not a QP/MPC controller).",
            file=sys.stderr,
        )
        return EXIT_USAGE

    # Pre-compile verification (scenario required): (1) the on-disk graph is
    # exactly what the scenario produces (integrity), then (2) the composed
    # graph reproduces the live estimator+controller math (gate A). Both run
    # before the zig build so a trace/compose bug fails fast — without spending
    # compile time or stamping a "verified" record for the wrong math.
    fresh_cg = None
    if scenario:
        assert spec is not None  # set whenever scenario is, above
        try:
            fresh_cg = _check_graph_integrity(scenario, graph_dir)
        except (ValueError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return EXIT_USAGE
        except NotImplementedError as e:
            print(f"TRACE FAILED: {e}", file=sys.stderr)
            return EXIT_USAGE
        if spec["compile"]["recipe"] == "closed_loop_tracking":
            rc = _run_gate_a(fresh_cg, spec, seed)
            if rc is not None:
                return rc

    prefix_path = Path(prefix) if prefix else graph_dir

    try:
        _build(graph_path, prefix_path, opt, tgt, sdir, name)
    except BuildError as e:
        print(f"BUILD FAILED: {e}", file=sys.stderr)
        return EXIT_BUILD

    oracle: dict | None = None
    if scenario:
        assert spec is not None  # set whenever scenario is, above
        if tgt == "native":
            # [compile].oracle_tol overrides the tier default for QP graphs
            # whose settling at this problem size is coarser than 1e-3.
            tol = spec["compile"].get("oracle_tol") or tol_for(manifest)
            lib = load_so(prefix_path, name)
            max_err = run_oracle(lib, fresh_cg, samples, seed)
            if max_err >= tol:
                print(
                    f"ORACLE MISMATCH: .so diverged from interpreter (max abs err {max_err:.3e} >= {tol})",
                    file=sys.stderr,
                )
                return EXIT_ORACLE
            print(f"oracle B (.so vs interpret): {samples} random inputs, max abs err {max_err:.3e} ✓")
            oracle = {
                "status": "passed",
                "method": ".so shinro_step vs interpret()",
                "samples": samples,
                "seed": seed,
                "max_abs_err": max_err,
                "tolerance": tol,
            }
        else:
            print(
                f"NOTE: target '{tgt}' is not native — skipping host oracle "
                f"(cannot dlopen a cross-compiled .so); integrity check passed.",
                file=sys.stderr,
            )
            oracle = {
                "status": "not_run",
                "reason": "cross-compiled",
                "note": "cannot dlopen a cross-compiled .so; oracle-verify the native build",
            }
    else:
        print(
            "WARNING: --scenario omitted — skipping oracle + integrity checks; "
            "the artifact is built and stamped but NOT verified against the interpreter.",
            file=sys.stderr,
        )
        oracle = {"status": "not_run", "reason": "unverified build (--scenario omitted)"}

    native_ref = None
    if scenario and tgt != "native":
        native_ref = _resolve_native_ref(graph_dir, opt, name, native_record)
        if native_ref is None:
            print(
                f"NOTE: no native record to reference (looked for <out>/../{opt}-native/lib/{name}.deployment.json; "
                f"pass --native-record <path> to point at the oracle-verified native build)",
                file=sys.stderr,
            )
    stamp(prefix_path, RUNTIME, name, oracle=oracle, native_record=native_ref)
    record = prefix_path / "lib" / f"{name}.deployment.json"
    if verify(record, graph_path=graph_path) != 0:
        print("VERIFY FAILED: deployment record does not match artifacts", file=sys.stderr)
        return EXIT_BUILD
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_dir", help="dir containing graph_data.zig (gen_scenario.py --out)")
    parser.add_argument("--scenario", help="scenario TOML (enables oracle + integrity check)")
    parser.add_argument("--prefix", help="zig build prefix (default: graph_dir)")
    parser.add_argument("--optimize", choices=["debug", "release"], help="override [compile].optimize")
    parser.add_argument("--target", help="override [compile].target (zig triple, e.g. aarch64-linux-gnu)")
    parser.add_argument("--solver-dir", help="override [compile].solver_dir (pre-baked OSQP solver dir)")
    parser.add_argument("--solver", help="override [compile].solver (bake on demand, e.g. emosqp)")
    parser.add_argument("--native-record", help="cross build: path to the oracle-verified native record to reference")
    parser.add_argument("--artifact-name", help="override [compile].artifact_name (kernel installs as lib/<name>.so)")
    parser.add_argument("--samples", type=int, default=20, help="random inputs for the oracle (default 20)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the oracle (default 0)")
    args = parser.parse_args()
    return build_scenario(
        args.graph_dir,
        scenario=args.scenario,
        prefix=args.prefix,
        optimize=args.optimize,
        target=args.target,
        solver_dir=args.solver_dir,
        solver=args.solver,
        native_record=args.native_record,
        artifact_name=args.artifact_name,
        samples=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
