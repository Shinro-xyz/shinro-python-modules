"""Benchmark `shinro_step` wall time per control tick for a compiled kernel.

The lowering oracle proves a kernel is *correct*; this measures whether it is
*fast*, so lowering changes can be A/B'd on runtime rather than argued about.
It dlopens a compiled artifact, drives the C ABI with seeded inputs, feeds the
recurrent ``state_*`` outputs back into their matching inputs (a realistic
rollout), and reports ns/tick.

Timing samples are integer nanoseconds and the summary is plain arithmetic, so
nothing here depends on numpy's float types.

Pass ``--onnx-model`` to also time the eager numpy path for the same policy
(``OnnxRLAdapter`` with ``model_path``), which is the honest yardstick for an
NN policy: the compiled kernel must beat it to be worth deploying.

Usage::

    python3 scripts/bench_tick.py --artifact-dir build/fence_before
    python3 scripts/bench_tick.py --artifact-dir build/onnx_scale/H256x4/out \\
        --kernel lib_neural_network --onnx-model build/onnx_scale/H256x4/policy.onnx
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _flat(shape) -> int:
    dims = list(shape or [])
    n = 1
    for d in dims:
        n *= d
    return n


def _port_offsets(ports: list[dict]) -> list[tuple[int, int]]:
    """Flat (start, stop) offsets of each port, in port order."""
    bounds: list[tuple[int, int]] = []
    offset = 0
    for port in ports:
        size = _flat(port["shape"])
        bounds.append((offset, offset + size))
        offset += size
    return bounds


def _load_manifest(path: Path) -> dict:
    """Read a graph manifest, failing with a build hint rather than a traceback."""
    if not path.exists():
        raise SystemExit(f"no graph manifest at {path} — build the artifact first (make compile / zig build)")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"graph manifest at {path} is corrupt: {exc}") from exc


#: Aim for ~0.2 ms of work per timed sample, so clock overhead is <1% of it.
#: (A single bare call on this class of host can be shorter than the timer's
#: effective resolution, which inflates per-call medians by orders of magnitude.)
_TARGET_SAMPLE_NS = 200_000


def _time_ticks(step: Callable[[], object], ticks: int, warmup: int) -> list[float]:
    """Per-call nanosecond samples, timed in batches sized to dwarf timer overhead."""
    for _ in range(warmup):
        step()

    start = time.perf_counter_ns()
    step()
    single = max(1, time.perf_counter_ns() - start)
    batch = max(1, min(ticks, _TARGET_SAMPLE_NS // single))

    samples: list[float] = []
    remaining = ticks
    while remaining > 0:
        n = min(batch, remaining)
        start = time.perf_counter_ns()
        for _ in range(n):
            step()
        samples.append((time.perf_counter_ns() - start) / n)
        remaining -= n
    return samples


def _stats(samples: list[float]) -> dict:
    """median / mean / p99 / rate from per-tick nanosecond samples (ints)."""
    ordered = sorted(samples)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    p99_index = min(n - 1, (n * 99) // 100)
    return {
        "ticks": n,
        # min is the robust estimator on a noisy host: the fastest pass is the
        # one least perturbed by scheduling, so it tracks true cost best.
        "ns_per_tick_min": ordered[0],
        "ns_per_tick_median": median,
        "ns_per_tick_mean": sum(ordered) / n,
        "ns_per_tick_p99": ordered[p99_index],
        "hz_median": 1e9 / median,
    }


def bench_so(artifact_dir: Path, kernel: str, ticks: int, warmup: int, seed: int, manifest_path: Path | None = None) -> dict:
    """Time one tick of ``lib<kernel>.so`` from ``artifact_dir``."""
    manifest = _load_manifest(manifest_path or (artifact_dir / "graph_data_manifest.json"))
    so_path = artifact_dir / "lib" / f"{kernel}.so"
    lib = ctypes.CDLL(str(so_path))
    ptr = ctypes.POINTER(ctypes.c_double)
    lib.shinro_step.argtypes = [ptr, ptr, ptr]
    lib.shinro_step.restype = None

    in_slices = _port_offsets(manifest["inputs"])
    state_slices = _port_offsets(manifest["state_outputs"])
    n_in = sum(stop - start for start, stop in in_slices)
    n_out = sum(_flat(p["shape"]) for p in manifest["outputs"])
    n_state = sum(_flat(p["shape"]) for p in manifest["state_outputs"])

    rng = np.random.default_rng(seed)
    inputs = rng.normal(0.0, 0.1, max(n_in, 1)).copy()
    outputs = np.zeros(max(n_out, 1))
    state = np.zeros(max(n_state, 1))

    # Recurrent feedback: each state_* output is also a state_* input next tick.
    in_names = [p["name"] for p in manifest["inputs"]]
    feedback: list[tuple[int, int, int, int]] = []
    for name, (sstart, sstop) in zip([p["name"] for p in manifest["state_outputs"]], state_slices):
        if name in in_names:
            istart = in_slices[in_names.index(name)][0]
            feedback.append((istart, istart + (sstop - sstart), sstart, sstop))

    def step() -> None:
        lib.shinro_step(
            inputs.ctypes.data_as(ptr),
            outputs.ctypes.data_as(ptr),
            state.ctypes.data_as(ptr),
        )
        for i0, i1, s0, s1 in feedback:
            inputs[i0:i1] = state[s0:s1]

    samples = _time_ticks(step, ticks, warmup)

    return {
        "artifact": f"{artifact_dir}/lib/{kernel}.so",
        "node_count": manifest.get("nodes_total"),
        "buf_bytes": manifest.get("buf_bytes"),
        "const_blob_bytes": manifest.get("const_blob_bytes"),
        **_stats(samples),
    }


def bench_eager(model_path: str, ticks: int, warmup: int, seed: int, n_x: int | None) -> dict:
    """Time one tick of the eager numpy path (the yardstick for an NN policy)."""
    from shinro.controllers.onnx_rl_adapter import OnnxRLAdapter

    cfg: dict = {"model_path": model_path}
    if n_x is not None:
        cfg["n_x"] = n_x
    ctrl = OnnxRLAdapter.from_config(cfg)
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 0.1, ctrl.policy.state_size)

    samples = _time_ticks(lambda: ctrl.compute(state), ticks, warmup)

    return {"artifact": f"eager(interpret) {model_path}", **_stats(samples)}


def _print_row(row: dict) -> None:
    label = Path(row["artifact"]).name if "/" in row["artifact"] else row["artifact"]
    print(
        f"  {label:<30} min {row['ns_per_tick_min']:>11,.0f} ns"
        f"   median {row['ns_per_tick_median']:>11,.0f} ns"
        f"   p99 {row['ns_per_tick_p99']:>11,.0f} ns"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-dir", required=True, help="dir with graph_data_manifest.json + lib/<kernel>.so")
    parser.add_argument("--manifest", help="graph manifest path (default: <artifact-dir>/graph_data_manifest.json)")
    parser.add_argument("--kernel", default="libbase", help="artifact stem (default libbase)")
    parser.add_argument("--onnx-model", help="also time the eager numpy path for this .onnx policy")
    parser.add_argument("--n-x", type=int, help="plant state dim for the eager path (default: derived from the model)")
    parser.add_argument("--ticks", type=int, default=20000, help="timed ticks (default 20000)")
    parser.add_argument("--warmup", type=int, default=2000, help="warmup ticks (default 2000)")
    parser.add_argument("--seed", type=int, default=0, help="input RNG seed")
    parser.add_argument("--json", help="also write the rows to this JSON path")
    args = parser.parse_args()

    rows = [
        bench_so(
            Path(args.artifact_dir),
            args.kernel,
            args.ticks,
            args.warmup,
            args.seed,
            Path(args.manifest) if args.manifest else None,
        )
    ]
    if args.onnx_model:
        rows.append(bench_eager(args.onnx_model, args.ticks, args.warmup, args.seed, args.n_x))

    print(f"=== tick benchmark: {args.ticks} ticks, {args.warmup} warmup ===")
    for row in rows:
        _print_row(row)
    if len(rows) == 2:
        compiled, eager = rows
        ratio = eager["ns_per_tick_min"] / compiled["ns_per_tick_min"]
        print(f"  -> compiled is {ratio:.2f}x {'faster' if ratio > 1 else 'SLOWER'} than eager numpy (min)")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
