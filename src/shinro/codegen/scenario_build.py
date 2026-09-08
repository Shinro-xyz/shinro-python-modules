"""Compile a generated scenario graph into a verified, stamped ``libbase.so``.

The zig-required stage of the e2e pipeline. Takes the graph pair produced by
:func:`shinro.codegen.scenario_gen.gen_scenario` (``graph_data.zig`` + its
manifest), compiles the comptime VM via the packaged ``runtime/build.zig``,
and — when ``--scenario`` is given — verifies the result before stamping:

1. **Pre-flight** — ``zig`` must be on PATH (loud error, exit 5).
2. **Build flags** — from the scenario's ``[compile]`` section (``optimize`` /
   ``target`` / ``solver_dir``), overridable on the CLI. ``optimize = "release"``
   maps to ``-Doptimize=ReleaseFast`` (the only validated release mode);
   ``target`` cross-compiles (e.g. ``aarch64-linux-gnu``); ``solver_dir`` is
   required for QP (MPC) graphs and ignored otherwise.
3. **Build** — ``zig build -Dgraph=<abs path>`` into an isolated prefix, so the
   shared ``src/shinro/runtime/graph_data.zig`` is never touched.
4. **Integrity check** — re-runs the gen stage in-process and byte-compares the
   fresh graph manifest against the on-disk one, proving the ``.so`` was built
   from exactly the graph this scenario produces (deterministic lowering).
5. **Oracle B** — loads the ``.so`` via ctypes and compares ``shinro_step``
   against ``interpret()`` on N random inputs across every output and state
   port (tolerance 1e-12; 1e-3 for QP graphs). Mismatch → exit 3.
6. **Stamp + verify** — writes the deployment record (master hash over
   config/graph/solver/binary) and re-hashes the artifacts against it.

Without ``--scenario`` the build is stamped only — no oracle, no integrity
check — and prints a loud warning that the artifact is unverified.

Run::

    python3 -m shinro.codegen.scenario_build build/base --scenario tests/integration/scenarios/base_tracking.toml

Exit codes: 0 ok · 2 usage/config · 3 oracle mismatch · 4 build/verify failure · 5 zig missing.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from shinro.codegen import interpret
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

# Tolerance for the .so-vs-interpreter oracle. Non-QP graphs (KF+LQR, KF+PID)
# agree to float-exactness (Zig's LU inv differs from LAPACK only in last
# ulps); QP graphs settle within the solver's eps (same precedent as the
# TestSolveQpOracle / TestMpcComposedOracle tolerances).
TOL_NON_QP = 1e-12
TOL_QP = 1e-3


class BuildError(RuntimeError):
    """The zig build failed or produced no artifact."""


# ─── C-ABI helpers (mirror tests/test_zig_lowering.py) ─────────────────────


def _input_shape(graph, name: str) -> tuple[int, ...]:
    for node in graph.nodes:
        if node.op == "input" and node.attrs["name"] == name:
            return node.shape
    raise KeyError(f"input port '{name}' not found in graph")


def _output_size(graph, name: str) -> int:
    for node in graph.nodes:
        if node.op == "output" and node.attrs["name"] == name:
            return int(np.prod(node.shape))
    raise KeyError(f"output port '{name}' not found in graph")


def _output_split(cg) -> tuple[int, int]:
    n_out = sum(_output_size(cg.graph, name) for name in cg.outputs)
    n_state = sum(_output_size(cg.graph, name) for name in cg.state_outputs)
    return n_out, n_state


def _state_slices(cg) -> dict[str, tuple[int, int]]:
    slices: dict[str, tuple[int, int]] = {}
    off = 0
    for name in cg.state_outputs:
        size = _output_size(cg.graph, name)
        slices[name] = (off, off + size)
        off += size
    return slices


def _pack_arrays(cg, arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(arrays[name], dtype=np.float64).ravel() for name in cg.inputs])


def _step(lib, cg, inputs: np.ndarray, n_out: int, n_state: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(n_out, dtype=np.float64)
    state = np.zeros(n_state, dtype=np.float64)
    lib.shinro_step(
        inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out, state


def _random_inputs(cg, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Random inputs per port; square 2-D state ports are made SPD (covariance-like)."""
    inputs: dict[str, np.ndarray] = {}
    for name in cg.inputs:
        shape = _input_shape(cg.graph, name)
        if len(shape) == 2 and shape[0] == shape[1]:
            a = rng.normal(0.0, 0.1, shape)
            inputs[name] = a @ a.T + 0.1 * np.eye(shape[0])
        else:
            inputs[name] = rng.normal(0.0, 0.1, shape)
    return inputs


def _load_so(prefix: Path):
    so_path = Path(prefix) / "lib" / "libbase.so"
    if not so_path.exists():
        raise BuildError(f"zig build produced no libbase.so at {so_path}")
    lib = ctypes.CDLL(str(so_path))
    lib.shinro_step.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3
    lib.shinro_step.restype = None
    return lib


# ─── pipeline stages ─────────────────────────────────────────────────────────


def _check_zig() -> bool:
    if shutil.which("zig") is not None:
        return True
    print("ERROR: zig not on PATH — the e2e workflow needs it to compile libbase.so.", file=sys.stderr)
    print("Install: https://ziglang.org/download (or your package manager).", file=sys.stderr)
    print("Nothing was built.", file=sys.stderr)
    return False


def _build(graph_path: Path, prefix: Path, optimize: str, target: str, solver_dir: str | None) -> None:
    """Compile the comptime VM against the given graph into an isolated prefix."""
    cmd = [
        "zig",
        "build",
        "--build-file",
        str(RUNTIME / "build.zig"),
        "--prefix",
        str(prefix),
        f"-Dgraph={graph_path}",
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


def _oracle(lib, cg, samples: int, seed: int, tol: float) -> float:
    """Compare the .so's shinro_step against interpret() on random inputs."""
    n_out, n_state = _output_split(cg)
    sl = _state_slices(cg)
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(samples):
        inputs = _random_inputs(cg, rng)
        packed = _pack_arrays(cg, inputs)
        out, state = _step(lib, cg, packed, n_out, n_state)

        traced = interpret(cg.graph, inputs)
        off = 0
        for name in cg.outputs:
            expected = np.asarray(traced[name]).ravel()
            max_err = max(max_err, float(np.max(np.abs(out[off : off + expected.size] - expected))))
            off += expected.size
        for name in cg.state_outputs:
            expected = np.asarray(traced[name]).ravel()
            start, stop = sl[name]
            max_err = max(max_err, float(np.max(np.abs(state[start:stop] - expected))))
    return max_err


def build_scenario(
    graph_dir: str,
    scenario: str | None = None,
    prefix: str | None = None,
    optimize: str | None = None,
    target: str | None = None,
    solver_dir: str | None = None,
    samples: int = 20,
    seed: int = 0,
) -> int:
    """Compile a generated graph into a verified, stamped ``libbase.so``.

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
        solver_dir: Override ``[compile].solver_dir`` (baked OSQP solver dir).
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
    opt, tgt, sdir = "debug", "native", None
    if scenario:
        try:
            spec = load_scenario(scenario)
        except (ValueError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return EXIT_USAGE
        opt = spec["compile"]["optimize"]
        tgt = spec["compile"]["target"]
        sdir = spec["compile"]["solver_dir"]
    if optimize:
        opt = optimize
    if target:
        tgt = target
    if solver_dir:
        sdir = solver_dir

    if manifest["has_solve_qp"] and not sdir:
        print(
            "ERROR: graph contains a .solve_qp node but no solver_dir — set "
            "[compile].solver_dir or pass --solver-dir (a graph must be built "
            "against a matching OSQP bake).",
            file=sys.stderr,
        )
        return EXIT_USAGE

    prefix_path = Path(prefix) if prefix else graph_dir

    try:
        _build(graph_path, prefix_path, opt, tgt, sdir)
    except BuildError as e:
        print(f"BUILD FAILED: {e}", file=sys.stderr)
        return EXIT_BUILD

    if scenario:
        try:
            fresh_cg = _check_graph_integrity(scenario, graph_dir)
        except (ValueError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return EXIT_USAGE
        except NotImplementedError as e:
            print(f"TRACE FAILED: {e}", file=sys.stderr)
            return EXIT_USAGE
        tol = TOL_QP if manifest["has_solve_qp"] else TOL_NON_QP
        lib = _load_so(prefix_path)
        max_err = _oracle(lib, fresh_cg, samples, seed, tol)
        if max_err >= tol:
            print(
                f"ORACLE MISMATCH: .so diverged from interpreter (max abs err {max_err:.3e} >= {tol})",
                file=sys.stderr,
            )
            return EXIT_ORACLE
        print(f"oracle B (.so vs interpret): {samples} random inputs, max abs err {max_err:.3e} ✓")
    else:
        print(
            "WARNING: --scenario omitted — skipping oracle + integrity checks; "
            "the artifact is built and stamped but NOT verified against the interpreter.",
            file=sys.stderr,
        )

    stamp(prefix_path, RUNTIME)
    record = prefix_path / "lib" / "libbase.deployment.json"
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
    parser.add_argument("--solver-dir", help="override [compile].solver_dir (baked OSQP solver dir)")
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
        samples=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
