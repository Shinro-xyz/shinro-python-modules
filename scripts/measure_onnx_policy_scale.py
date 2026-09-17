"""ONNX policy size sweep: parameter count -> artifact size, compile time, footprint.

Generates realistic RL-actor MLPs (``obs -> [hidden]*depth -> act``) of
increasing parameter count and compiles each through the real pipeline
(``gen_scenario.py`` → ``build_scenario.py --optimize release``), reporting the
compiled ``.so`` size, the stage wall-clock times, and the lowered VM's buffer
sizes (stack buffer, baked constants, node count). A stage that fails or times
out is recorded as data, not raised — the scaling wall is the point.

The ONNX analogue of ``scripts/measure_kernels.py``: a one-off measurement
harness, not part of ``make test``. Everything lands under the gitignored
``build/onnx_scale/``.

    python3 scripts/measure_onnx_policy_scale.py
    python3 scripts/measure_onnx_policy_scale.py --archs 64:2,256:4,512:4
    python3 scripts/measure_onnx_policy_scale.py --json build/onnx_scale.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "build" / "onnx_scale"
#: The zig build's LOCAL cache (relative to the runtime build root). Clearing it
#: forces a cold compile — zig caches comptime work, and a warm cache turns a
#: 3-minute compile into ~2 seconds.
_LOCAL_ZIG_CACHE = REPO / "src" / "shinro" / "runtime" / ".zig-cache"

#: Observation / action dimensions of the fixed policy interface.
OBS, ACT = 64, 16
#: Default ``hidden:depth`` architectures — a realistic shallow-to-deep RL actor.
DEFAULT_ARCHS = "64:2,256:2,256:4,512:4,512:6,512:8"


def params_for(obs: int, act: int, hidden: int, depth: int) -> int:
    """Parameter count of the ``obs -> [hidden]*depth -> act`` MLP."""
    return (obs + 1) * hidden + (depth - 1) * (hidden + 1) * hidden + (hidden + 1) * act


def make_mlp(obs: int, act: int, hidden: int, depth: int, seed: int):
    """Build an ``obs -> [hidden]*depth -> act`` tanh MLP as an onnx ModelProto."""
    import numpy as np
    from onnx import TensorProto, helper

    rng = np.random.default_rng(seed)
    nodes: list = []
    inits: list = []

    def init(name: str, array) -> object:
        a = np.asarray(array, dtype=np.float32)
        return helper.make_tensor(name, TensorProto.FLOAT, a.shape, a.flatten().tolist())

    prev = "obs"
    for layer in range(depth + 1):
        in_dim = obs if layer == 0 else hidden
        out_dim = act if layer == depth else hidden
        out = "action" if layer == depth else f"h{layer}"
        nodes.append(helper.make_node("Gemm", [prev, f"w{layer}", f"b{layer}"], [out], transB=1))
        inits += [
            init(f"w{layer}", rng.normal(0.0, 1.0 / math.sqrt(in_dim), (out_dim, in_dim))),
            init(f"b{layer}", rng.normal(0.0, 0.1, out_dim)),
        ]
        if out != "action":
            nodes.append(helper.make_node("Tanh", [out], [f"a{layer}"]))
            prev = f"a{layer}"

    x = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, obs])
    y = helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, act])
    graph = helper.make_graph(nodes, "sweep", [x], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def _env(extra: dict | None = None) -> dict:
    env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{REPO}"}
    if extra:
        env.update(extra)
    return env


def _run(cmd: list[str], timeout: float, env: dict | None = None) -> tuple[float, str, str]:
    """Run a stage; return ``(seconds, stdout, error)`` — ``stdout`` is "" on failure."""
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env or _env(), timeout=timeout)
    except subprocess.TimeoutExpired:
        return time.perf_counter() - start, "", f"TIMEOUT >{timeout:.0f}s"
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        return elapsed, result.stdout, _summarize_error(result.stderr)
    return elapsed, result.stdout, ""


def _summarize_error(stderr: str) -> str:
    """Pull the most informative line out of a zig/python failure."""
    for line in stderr.splitlines():
        low = line.lower()
        if "error:" in low or "exceeded" in low or "out of memory" in low or "panic" in low:
            return line.strip()[:180]
    lines = stderr.strip().splitlines()
    return lines[-1][:180] if lines else "unknown failure"


def measure_point(hidden: int, depth: int, timeout: float, seed: int) -> dict:
    """Compile one ``hidden x depth`` policy and return its measurements (or error)."""
    import onnx

    n_params = params_for(OBS, ACT, hidden, depth)
    row: dict = {"params": n_params, "hidden": hidden, "depth": depth}
    root = WORK / f"H{hidden}x{depth}"
    root.mkdir(parents=True, exist_ok=True)
    policy = root / "policy.onnx"
    onnx.save(make_mlp(OBS, ACT, hidden, depth, seed), str(policy))

    ctrl = root / "ctrl.toml"
    ctrl.write_text(
        f'type = "onnx_rl"\nmodel_path = "{policy}"\naction_space = "continuous"\n'
        f"\n[observation]\nstate_keys = {list(range(OBS))}\n"
    )
    scenario = root / "scenario.toml"
    scenario.write_text(
        f'[controller]\nconfig = "{ctrl}"\n'
        f'[compile]\nn_x = {OBS}\nn_u = {ACT}\nartifact_name = "lib_neural_network"\n'
    )

    out = root / "out"
    gen_sec, _gen_out, gen_err = _run([sys.executable, "scripts/gen_scenario.py", str(scenario), "--out", str(out)], timeout)
    row["onnx_bytes"] = policy.stat().st_size
    row["gen_s"] = round(gen_sec, 2)
    if gen_err:
        row["error"] = f"gen: {gen_err}"
        return row

    graph_src = out / "graph_data.zig"
    row["graph_src_bytes"] = graph_src.stat().st_size if graph_src.exists() else 0

    # Cold compile: drop the shared local cache so this point is not measured
    # against another point's cached comptime evaluation. The only expected
    # failure is a missing dir (first build); anything else is reported, because
    # an uncleared cache would silently make the timing warm.
    cold = True
    if _LOCAL_ZIG_CACHE.exists():
        try:
            shutil.rmtree(_LOCAL_ZIG_CACHE)
        except OSError as exc:
            cold = False
            print(f"warning: could not clear {_LOCAL_ZIG_CACHE} ({exc}); this point may be warm", file=sys.stderr)
    row["cold_cache"] = cold
    build_sec, build_out, build_err = _run(
        [sys.executable, "scripts/build_scenario.py", str(out), "--scenario", str(scenario), "--optimize", "release"],
        timeout,
    )
    row["build_s"] = round(build_sec, 2)
    if build_err:
        row["error"] = f"build: {build_err}"
        return row

    try:
        manifest = json.loads((out / "graph_data_manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        row["error"] = f"manifest unreadable: {exc}"
        return row
    row.update(
        so_bytes=(out / "lib" / "lib_neural_network.so").stat().st_size,
        nodes=manifest["nodes_total"],
        buf_bytes=manifest["buf_bytes"],
        const_bytes=manifest["const_blob_bytes"],
        oracle=next((ln.strip() for ln in build_out.splitlines() if "oracle B" in ln), ""),
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", default=DEFAULT_ARCHS, help=f"comma-separated hidden:depth (default {DEFAULT_ARCHS})")
    parser.add_argument("--timeout", type=float, default=900.0, help="per-stage timeout in seconds (default 900)")
    parser.add_argument("--seed", type=int, default=0, help="weight RNG seed")
    parser.add_argument("--json", help="also write the rows to this JSON path")
    args = parser.parse_args()

    archs: list = []
    try:
        archs = [tuple(int(p) for p in spec.split(":")) for spec in args.archs.split(",")]
    except ValueError:
        archs = []
    if not archs or any(len(a) != 2 for a in archs):
        parser.error(f"--archs must be comma-separated hidden:depth pairs (got {args.archs!r})")
    hdr = (
        f"{'params':>9} {'arch':>10} {'onnx':>9} {'graph.zig':>10} {'so':>9} "
        f"{'gen s':>6} {'build s':>8} {'nodes':>6} {'vm buf':>8} {'consts':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for hidden, depth in archs:
        row = measure_point(hidden, depth, args.timeout, args.seed)
        rows.append(row)
        arch = f"{hidden}x{depth}"
        if "error" in row:
            src_mb = row.get("graph_src_bytes", 0) / 1024 / 1024
            print(
                f"{row['params']:>9} {arch:>10} {row['onnx_bytes'] / 1024:>8.1f}K {src_mb:>9.1f}M"
                f"  FAILED: {row['error']}"
            )
            continue
        print(
            f"{row['params']:>9} {arch:>10} {row['onnx_bytes'] / 1024:>8.1f}K"
            f" {row['graph_src_bytes'] / 1024 / 1024:>9.1f}M {row['so_bytes'] / 1024:>8.1f}K"
            f" {row['gen_s']:>6.2f} {row['build_s']:>8.2f} {row['nodes']:>6}"
            f" {row['buf_bytes'] / 1024:>7.1f}K {row['const_bytes'] / 1024:>7.1f}K"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
