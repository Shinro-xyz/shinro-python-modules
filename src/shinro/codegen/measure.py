"""Kernel size metrics for lowered graphs — a repeatable measurement.

The three sizes that matter for a deployed kernel differ by orders of
magnitude, so they are reported separately and machine-readably rather than as
a one-off print:

1. **C-ABI host buffers** — what the host packs/unpacks each tick (the input
   buffer is dominated by the ``epsilon`` port for MPPI).
2. **VM stack buffer** — ``buf: [buf_len]f64`` inside ``shinro_step``; this is
   the kernel's memory story (a small-stack RTOS cares about it).
3. **Compiled artifact** — the ``.so`` on disk, plus the compile cost.

``graph_metrics`` is pure derivation from a lowered graph (fast, no compiler).
``kernel_metrics`` adds the build: it lowers, runs ``zig build``, and records
the artifact bytes and the wall-clock compile time.

Unlike the build/deployment manifests (which are deterministic audit records),
these measurements purposely include a wall-clock compile time, so the JSON is
a measurement sample, not a diffable record — no master hash depends on it.

Usage:
    python -m shinro.codegen.measure                          # static only
    python -m shinro.codegen.measure --build --optimize ReleaseFast
    python -m shinro.codegen.measure --dims 3x3x6x3,6x6x10x4 --json out.json

Each ``--dims`` entry is ``D_x x D_u x N x K`` (the MPPI sampling rollout shape).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from shinro.codegen.compose import ComposedGraph
from shinro.codegen.lower_zig import lower_zig
from shinro.codegen.runtime_paths import runtime_root
from shinro.codegen.trace_node import trace_node
from shinro.controllers.mppi import MPPIController
from shinro.utils.array_backend import NumpyBackend

#: Default sweep: the test fixture, a mid-size, and two multi-input systems.
DEFAULT_DIMS = ("3x3x6x3", "3x3x100x15", "6x6x10x4", "8x4x12x5")


def parse_dims(spec: str) -> tuple[int, int, int, int]:
    """Parse ``D_x x D_u x N x K`` (e.g. ``3x3x100x15``) into four ints."""
    parts = spec.lower().split("x")
    if len(parts) != 4:
        raise ValueError(f"--dims entry {spec!r} must be D_x x D_u x N x K")
    try:
        dims = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"--dims entry {spec!r} must be four positive integers") from exc
    if any(d <= 0 for d in dims):
        raise ValueError(f"--dims entry {spec!r} must be positive")
    return dims  # type: ignore[return-value]


def mppi_lti_graph(D_x: int, D_u: int, N: int, K: int, dt: float = 0.05) -> ComposedGraph:
    """The standard MPPI sampling graph for an arbitrary LTI system.

    Dynamics/cost are injected as trace-safe operators (``x @ A.T + u @ B.T``,
    quadratic forms as contractions) rather than through ``attach_plant``, so
    any ``(D_x, D_u)`` can be measured — the plant path fixes the dims to the
    plant's model. ``epsilon`` is a free port (sampling stays on the host) and
    ``u`` recurs as ``state_u``.
    """
    rng = np.random.default_rng(0)
    a = np.diag(rng.uniform(0.8, 1.0, D_x))
    b = rng.normal(0.0, 0.15, (D_x, D_u))
    q = np.abs(rng.normal(1.0, 0.2, D_x))
    r = np.abs(rng.normal(0.1, 0.02, D_u))

    def dynamics(x_batch, u_batch, dt_step):
        return x_batch + dt_step * (x_batch @ a.T + u_batch @ b.T)

    def cost(x_batch, u_batch):
        return (x_batch * x_batch) @ q + (u_batch * u_batch) @ r

    ctrl = MPPIController(
        dynamics_fn=dynamics,
        cost_fn=cost,
        num_samples=N,
        temperature=1.0,
        dt=dt,
        horizon=K,
        noise_sigma=[0.4] * D_u,
        u_min=[-1.0] * D_u,
        u_max=[1.0] * D_u,
        seed=0,
        backend=NumpyBackend(),
    )
    ng = trace_node(
        ctrl,
        input_shapes={
            "current_state": (D_x,),
            "target_state": (D_x,),
            "epsilon": (N, K * D_u),
        },
        state_shapes={"u": (K, D_u)},
    )
    return ComposedGraph(
        graph=ng.graph,
        inputs=["current_state", "target_state", "epsilon", "state_u"],
        outputs=["out", "costs"],
        state_inputs=["state_u"],
        state_outputs=["state_u"],
    )


def graph_metrics(cg: ComposedGraph, workdir: pathlib.Path) -> dict:
    """Static size metrics for a graph — the manifest ``lower_zig`` writes.

    No compiler, and no duplicated buffer math: ``lower_zig`` records the byte
    figures itself (``buf_bytes``, ``const_blob_bytes``, ``clip_blob_bytes``,
    and a ``bytes`` entry per port), so this is a faithful read of the same
    node table the VM compiles.
    """
    graph_path = workdir / "graph_data.zig"
    lower_zig(cg, str(graph_path))
    manifest_path = graph_path.with_name("graph_data_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - build-time artifact
        raise RuntimeError(f"cannot read {manifest_path}: {exc}") from exc

    return {
        "nodes_total": manifest["nodes_total"],
        "buf_len_f64": manifest["buf_len"],
        "buf_bytes": manifest["buf_bytes"],
        "const_blob_bytes": manifest["const_blob_bytes"],
        "clip_blob_bytes": manifest["clip_blob_bytes"],
        "input_bytes": manifest["input_bytes"],
        "output_bytes": manifest["output_bytes"],
        "state_bytes": manifest["state_bytes"],
        "has_solve_qp": manifest["has_solve_qp"],
        "ops": manifest["ops"],
        "inputs": manifest["inputs"],
        "outputs": manifest["outputs"],
        "state_outputs": manifest["state_outputs"],
    }


def kernel_metrics(cg: ComposedGraph, workdir: pathlib.Path, optimize: str) -> dict:
    """Build the kernel and record its artifact bytes and compile cost.

    Fails loudly (raises) when ``zig`` is missing: a measurement tool that
    silently reported "no artifact" would be worse than useless.
    """
    if shutil.which("zig") is None:
        raise RuntimeError("zig is not on PATH — cannot measure the compiled artifact")
    graph_path = workdir / "graph_data.zig"
    lower_zig(cg, str(graph_path))
    cmd = [
        "zig",
        "build",
        "--build-file",
        str(runtime_root() / "build.zig"),
        "--prefix",
        str(workdir),
        f"-Dgraph={graph_path}",
        f"-Doptimize={optimize}",
    ]
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"zig build failed: {result.stderr.strip()[:400]}")
    so_path = workdir / "lib" / "libbase.so"
    if not so_path.exists():
        raise RuntimeError(f"zig build produced no libbase.so under {workdir}")
    return {"optimize": optimize, "so_bytes": so_path.stat().st_size, "compile_seconds": round(elapsed, 2)}


def measure(specs: list[tuple[int, int, int, int]], optimize: str, build: bool) -> dict:
    """Measure each ``(D_x, D_u, N, K)`` spec; returns the metrics document."""
    configs = []
    for d_x, d_u, n, k in specs:
        label = f"D_x={d_x} D_u={d_u} N={n} K={k}"
        with tempfile.TemporaryDirectory(prefix="shinro-measure-") as tmp:
            workdir = pathlib.Path(tmp)
            cg = mppi_lti_graph(d_x, d_u, n, k)
            entry = {
                "label": label,
                "D_x": d_x,
                "D_u": d_u,
                "N": n,
                "K": k,
                "graph": graph_metrics(cg, workdir),
                "kernel": kernel_metrics(cg, workdir, optimize) if build else None,
            }
            configs.append(entry)
            print(_row(entry), flush=True)
    return {"generated_by": "shinro.codegen.measure", "float_type": "f64", "build": build, "optimize": optimize, "configs": configs}


def _row(entry: dict) -> str:
    """One table line: the three sizes side by side (see ``_header``)."""
    g = entry["graph"]
    kernel = entry["kernel"]
    so = f"{kernel['so_bytes'] / 1024:.1f}" if kernel else "-"
    secs = f"{kernel['compile_seconds']:.2f}" if kernel else "-"
    return (
        f"{entry['label']:24s} {g['nodes_total']:6d}"
        f" {g['input_bytes'] / 1024:9.1f} {g['buf_bytes'] / 1024:9.1f}"
        f" {so:>9s} {secs:>10s}"
    )


def _header(optimize: str, build: bool) -> str:
    mode = optimize if build else "static only (no build)"
    return (
        f"{'config':24s} {'nodes':>6s} {'in KiB':>9s} {'VM KiB':>9s} {'so KiB':>9s} {'compile s':>10s}"
        f"   [{mode}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure lowered MPPI kernel sizes.")
    parser.add_argument("--dims", default=",".join(DEFAULT_DIMS), help="comma-separated D_x x D_u x N x K entries")
    parser.add_argument("--optimize", default="ReleaseFast", help="zig optimize mode (production: ReleaseFast)")
    parser.add_argument("--build", action="store_true", help="compile and measure the artifact (slower)")
    parser.add_argument("--json", dest="json_path", help="also write the metrics document here")
    args = parser.parse_args(argv)

    try:
        specs = [parse_dims(s) for s in args.dims.split(",") if s.strip()]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(_header(args.optimize, args.build))
    document = measure(specs, args.optimize, args.build)

    if args.json_path:
        out = pathlib.Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
