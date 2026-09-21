"""Generate the graph pair for a scenario TOML (zig-free stage of the e2e pipeline).

Reads a scenario TOML's ``[controller]`` / ``[estimator]`` config paths,
``[scenario].input_limits``, and the ``[compile]`` section (``n_x`` / ``n_u`` /
``recipe``), dispatches to a registered :mod:`shinro.codegen.recipes` recipe to
compose the step graph, and lowers it to an **isolated** path (``--out``) — the
shipped ``src/shinro/runtime/graph_data.zig`` is never touched, so compiling a
custom scenario never clobbers the shipped graph.

An optional plant-only ``[plant]`` section (``type`` + ``config``) derives
``n_x``/``n_u`` and the ``A_dynamics``/``B_dynamics`` model from the plant (the
same derivation the simulation path uses), so a scenario can be written as
specs + weights with no hand-computed model. The plant is authoritative:
``[compile]`` dims may be omitted, and a declared value that disagrees with the
plant is a loud error.

The ``[compile]`` section is the build spec: it is validated strictly (unknown
keys and invalid ``optimize`` values are loud errors) and its sha256 is
recorded in the graph manifest's provenance, so the deployment record commits
to the exact build flags that produced the artifact. ``recipe`` selects the
compose wiring (``closed_loop_tracking`` by default; ``policy_only`` when
``[estimator]`` is absent).

Run::

    python3 -m shinro.codegen.scenario_gen tests/integration/scenarios/base_tracking.toml --out build/base

Exit codes: 0 ok · 1 untraceable · 2 usage/config error.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from pathlib import Path

import numpy as np

from shinro.codegen import lower_zig
from shinro.codegen.recipes import available_graphs, build_recipe, tool_versions, type_from_component_config
from shinro.utils.config_resolver import resolve_config_path

EXIT_OK = 0
EXIT_UNTRACEABLE = 1
EXIT_USAGE = 2

_COMPILE_KEYS = {"n_x", "n_u", "optimize", "target", "solver_dir", "oracle_tol", "artifact_name", "recipe"}
_ALLOWED_OPTIMIZE = {"debug", "release"}
_DEFAULT_RECIPE = "closed_loop_tracking"
_POLICY_RECIPE = "policy_only"


def _sha256(path: str) -> str:
    """Return the sha256 of a file's raw bytes (hermetic: no preprocessing)."""
    with open(resolve_config_path(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _sha256_raw(path: str) -> str:
    """Return the sha256 of a file by literal path (no config-dir resolution).

    Used for the ``.onnx`` weights, which live wherever the controller config
    points and are not one of the resolved config locations.
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _validate_compile(compile_cfg: dict | None, scenario_path: str) -> dict:
    """Parse and strictly validate the ``[compile]`` section.

    Returns a dict with ``n_x`` / ``n_u`` (optional — derived from ``[plant]``
    when absent), ``recipe`` (optional — inferred from the sections when
    absent), and ``optimize`` / ``target`` / ``solver_dir`` / ``artifact_name``
    (optional, with defaults). Unknown keys and invalid ``optimize`` values are
    loud errors — the section is the build spec, so a typo must not silently
    change the build.

    Raises:
        ValueError: On a missing section, unknown keys, or an invalid
            ``optimize`` value.
    """
    if compile_cfg is None:
        raise ValueError(f"{scenario_path}: missing [compile] section")
    unknown = set(compile_cfg) - _COMPILE_KEYS
    if unknown:
        raise ValueError(f"{scenario_path}: [compile] has unknown key(s): {sorted(unknown)}")
    optimize = compile_cfg.get("optimize", "debug")
    if optimize not in _ALLOWED_OPTIMIZE:
        raise ValueError(
            f"{scenario_path}: [compile].optimize must be 'debug' or 'release' "
            f"(got '{optimize}'). ReleaseSafe hangs in osqp_solve (Zig integration "
            f"bug) and ReleaseSmall is unvalidated — only ReleaseFast is shippable."
        )
    # The artifact stem becomes lib/<name>.so + lib/<name>.{manifest,deployment}.json;
    # a stray separator or space would silently write outside the prefix.
    artifact_name = compile_cfg.get("artifact_name", "libbase")
    if not isinstance(artifact_name, str) or not artifact_name or any(c in artifact_name for c in "/\\ \t"):
        raise ValueError(
            f"{scenario_path}: [compile].artifact_name must be a simple file-name stem "
            f"like 'libbase' or 'lib_neural_network' (got {artifact_name!r})"
        )
    recipe = compile_cfg.get("recipe")
    if recipe is not None and not isinstance(recipe, str):
        raise ValueError(f"{scenario_path}: [compile].recipe must be a string (got {recipe!r})")
    return {
        "n_x": int(compile_cfg["n_x"]) if "n_x" in compile_cfg else None,
        "n_u": int(compile_cfg["n_u"]) if "n_u" in compile_cfg else None,
        "optimize": optimize,
        "target": compile_cfg.get("target", "native"),
        "solver_dir": compile_cfg.get("solver_dir"),
        "artifact_name": artifact_name,
        # Optional oracle-B override for QP graphs whose realistic solution
        # settles more coarsely than the tier default at this problem size
        # (two independent OSQP runs — C-baked vs Python — agree to solver
        # settling, which grows with n_vars and constraint activity).
        "oracle_tol": float(compile_cfg["oracle_tol"]) if "oracle_tol" in compile_cfg else None,
        "recipe": recipe,
    }


def load_scenario(scenario_path: str) -> dict:
    """Parse a scenario TOML into the fields the compile pipeline needs.

    Returns a dict with ``scenario_path`` / ``estimator_config`` /
    ``controller_config`` (config TOML paths), ``input_limits`` (``(lo, hi)``
    ndarray pair or ``None``), the validated ``compile`` section (with
    ``recipe`` resolved), and ``plant`` (the ``[plant]`` section or ``None``).

    The ``recipe`` defaults to ``closed_loop_tracking``, or ``policy_only``
    when ``[estimator]`` is absent; an explicit ``[compile].recipe`` wins but
    must be registered.

    Raises:
        ValueError: On a missing ``[controller]`` section, an invalid
            ``[compile]`` section, an unknown recipe, or a missing
            ``[estimator]`` on anything other than a policy-only scenario.
    """
    with open(resolve_config_path(scenario_path), "rb") as f:
        cfg = tomllib.load(f)
    if "controller" not in cfg:
        raise ValueError(f"{scenario_path}: scenario requires a [controller] section")
    controller = cfg["controller"]
    estimator = cfg.get("estimator")
    limits = None
    il = cfg.get("scenario", {}).get("input_limits")
    if il:
        limits = (np.array(il["min"], dtype=np.float64), np.array(il["max"], dtype=np.float64))

    compile_spec = _validate_compile(cfg.get("compile"), scenario_path)
    recipe = compile_spec["recipe"]
    if recipe is None:
        recipe = _POLICY_RECIPE if estimator is None else _DEFAULT_RECIPE
    if recipe not in available_graphs():
        raise ValueError(f"{scenario_path}: unknown [compile].recipe '{recipe}' (registered: {available_graphs()})")
    if estimator is None:
        # Only a policy (onnx_rl) controller can stand alone — a classical
        # controller with no [estimator] is a scenario error, not a policy.
        ctype = controller.get("type") or type_from_component_config(controller["config"])
        if recipe != _POLICY_RECIPE or ctype != "onnx_rl":
            raise ValueError(
                f"{scenario_path}: missing [estimator] section — only a policy-only 'onnx_rl' "
                f"scenario may omit it (recipe {recipe!r}, controller type {ctype!r})"
            )
    compile_spec["recipe"] = recipe

    return {
        "scenario_path": scenario_path,
        "estimator_config": estimator["config"] if estimator else None,
        "controller_config": controller["config"],
        "estimator_type": estimator.get("type") if estimator else None,
        "controller_type": controller.get("type"),
        "input_limits": limits,
        "compile": compile_spec,
        "plant": cfg.get("plant"),
    }


def _provenance(scenario_path: str, spec: dict, model_path: str | None = None) -> dict:
    """Build the provenance dict for ``lower_zig``.

    Records the sha256 of the scenario TOML itself (pinning the whole build
    spec, including ``[compile]``) plus the estimator/controller configs — the
    plant config when derivation was used, and the ``.onnx`` weights for a
    policy-only scenario — so the deployment record's config slot commits to
    every file the artifact was built from.
    """
    configs = {
        scenario_path: _sha256(scenario_path),
        spec["controller_config"]: _sha256(spec["controller_config"]),
    }
    if spec["estimator_config"]:
        configs[spec["estimator_config"]] = _sha256(spec["estimator_config"])
    plant = spec.get("plant")
    if plant and "config" in plant:
        configs[plant["config"]] = _sha256(plant["config"])
    if model_path:
        configs[model_path] = _sha256_raw(model_path)
    return {"configs": configs, **tool_versions()}


def gen_scenario(scenario_path: str, out_dir: str) -> tuple:
    """Trace + compose + lower a scenario into an isolated graph pair.

    Dispatches to the scenario's resolved ``[compile].recipe`` (a registered
    :mod:`shinro.codegen.recipes` graph recipe), so the compose wiring is chosen
    explicitly rather than by which sections the scenario happens to declare.

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
    cg = build_recipe(spec["compile"]["recipe"], spec)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / "graph_data.zig"
    lower_zig(
        cg,
        str(graph_path),
        provenance=_provenance(scenario_path, spec, model_path=spec.get("policy_model_path")),
    )
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
