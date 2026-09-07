"""Generate the graph pair for a scenario TOML (zig-free stage of the e2e pipeline).

Reads a scenario TOML's ``[controller]`` / ``[estimator]`` config paths,
``[scenario].input_limits``, and the ``[compile]`` section (``n_x`` / ``n_u``),
traces + composes the closed-loop step graph via the generic
:func:`shinro.codegen.build.build_composed_graph`, and lowers it to an
**isolated** path (``--out``) — ``runtime/graph_data.zig`` is never touched, so
compiling a custom scenario never clobbers the shipped graph.

The ``[compile]`` section is the build spec: it is validated strictly (unknown
keys and invalid ``optimize`` values are loud errors) and its sha256 is
recorded in the graph manifest's provenance, so the deployment record commits
to the exact build flags that produced the artifact.

Run::

    python3 scripts/gen_scenario.py tests/integration/scenarios/base_tracking.toml --out build/base

Exit codes: 0 ok · 1 untraceable · 2 usage/config error.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np

from shinro.codegen import build_composed_graph, lower_zig
from shinro.utils.config_resolver import resolve_config_path

EXIT_OK = 0
EXIT_UNTRACEABLE = 1
EXIT_USAGE = 2

_COMPILE_KEYS = {"n_x", "n_u", "optimize", "target", "solver_dir"}
_ALLOWED_OPTIMIZE = {"debug", "release"}


def _sha256(path: str) -> str:
    """Return the sha256 of a file's raw bytes (hermetic: no preprocessing)."""
    with open(resolve_config_path(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _validate_compile(compile_cfg: dict | None, scenario_path: str) -> dict:
    """Parse and strictly validate the ``[compile]`` section.

    Returns a dict with ``n_x`` / ``n_u`` (required) and ``optimize`` /
    ``target`` / ``solver_dir`` (optional, with defaults). Unknown keys and
    invalid ``optimize`` values are loud errors — the section is the build
    spec, so a typo must not silently change the build.

    Raises:
        ValueError: On a missing section, missing ``n_x``/``n_u``, unknown
            keys, or an invalid ``optimize`` value.
    """
    if compile_cfg is None:
        raise ValueError(f"{scenario_path}: missing [compile] section (n_x, n_u required)")
    unknown = set(compile_cfg) - _COMPILE_KEYS
    if unknown:
        raise ValueError(f"{scenario_path}: [compile] has unknown key(s): {sorted(unknown)}")
    if "n_x" not in compile_cfg or "n_u" not in compile_cfg:
        raise ValueError(f"{scenario_path}: [compile] requires n_x and n_u")
    optimize = compile_cfg.get("optimize", "debug")
    if optimize not in _ALLOWED_OPTIMIZE:
        raise ValueError(
            f"{scenario_path}: [compile].optimize must be 'debug' or 'release' "
            f"(got '{optimize}'). ReleaseSafe hangs in osqp_solve (Zig integration "
            f"bug) and ReleaseSmall is unvalidated — only ReleaseFast is shippable."
        )
    return {
        "n_x": int(compile_cfg["n_x"]),
        "n_u": int(compile_cfg["n_u"]),
        "optimize": optimize,
        "target": compile_cfg.get("target", "native"),
        "solver_dir": compile_cfg.get("solver_dir"),
    }


def load_scenario(scenario_path: str) -> dict:
    """Parse a scenario TOML into the fields the compile pipeline needs.

    Returns a dict with ``estimator_config`` / ``controller_config`` (config
    TOML paths), ``input_limits`` (``(lo, hi)`` ndarray pair or ``None``), and
    the validated ``compile`` section.

    Raises:
        ValueError: On a missing ``[controller]``/``[estimator]`` section or an
            invalid ``[compile]`` section.
    """
    with open(resolve_config_path(scenario_path), "rb") as f:
        cfg = tomllib.load(f)
    if "controller" not in cfg or "estimator" not in cfg:
        raise ValueError(
            f"{scenario_path}: scenario requires [controller] and [estimator] sections"
        )
    limits = None
    il = cfg.get("scenario", {}).get("input_limits")
    if il:
        limits = (np.array(il["min"], dtype=np.float64), np.array(il["max"], dtype=np.float64))
    return {
        "estimator_config": cfg["estimator"]["config"],
        "controller_config": cfg["controller"]["config"],
        "input_limits": limits,
        "compile": _validate_compile(cfg.get("compile"), scenario_path),
    }


def _provenance(scenario_path: str, spec: dict) -> dict:
    """Build the provenance dict for ``lower_zig``.

    Records the sha256 of the scenario TOML itself (pinning the whole build
    spec, including ``[compile]``) plus the estimator/controller configs, so
    the deployment record's config slot commits to all three.
    """
    return {
        "configs": {
            scenario_path: _sha256(scenario_path),
            spec["estimator_config"]: _sha256(spec["estimator_config"]),
            spec["controller_config"]: _sha256(spec["controller_config"]),
        },
        "python_version": sys.version.split()[0],
        "numpy_version": version("numpy"),
    }


def gen_scenario(scenario_path: str, out_dir: str) -> tuple:
    """Trace + compose + lower a scenario into an isolated graph pair.

    Args:
        scenario_path: Scenario TOML path.
        out_dir: Output directory for ``graph_data.zig`` + its manifest.

    Returns:
        A tuple ``(composed_graph, graph_path)``.

    Raises:
        ValueError: On a config/usage error.
        NotImplementedError: If a component uses an untraceable op.
    """
    spec = load_scenario(scenario_path)
    cg = build_composed_graph(
        spec["estimator_config"],
        spec["controller_config"],
        spec["compile"]["n_x"],
        spec["compile"]["n_u"],
        input_limits=spec["input_limits"],
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / "graph_data.zig"
    lower_zig(cg, str(graph_path), provenance=_provenance(scenario_path, spec))
    return cg, graph_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="scenario TOML path")
    parser.add_argument("--out", default="build/scenario", help="output dir (default: build/scenario)")
    args = parser.parse_args()

    try:
        cg, graph_path = gen_scenario(args.scenario, args.out)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_USAGE
    except NotImplementedError as e:
        print(f"TRACE FAILED: {e}", file=sys.stderr)
        return EXIT_UNTRACEABLE
    except Exception as e:
        print(f"GEN FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNTRACEABLE

    print(f"wrote {graph_path} ({len(cg.graph.nodes)} nodes)")
    print(f"inputs: {cg.inputs}")
    print(f"outputs: {cg.outputs}")
    print(f"state outputs: {cg.state_outputs}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
