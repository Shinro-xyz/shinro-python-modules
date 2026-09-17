"""Generate the graph pair for a scenario TOML (zig-free stage of the e2e pipeline).

Reads a scenario TOML's ``[controller]`` / ``[estimator]`` config paths,
``[scenario].input_limits``, and the ``[compile]`` section (``n_x`` / ``n_u``),
traces + composes the closed-loop step graph via the generic
:func:`shinro.codegen.build.build_composed_graph`, and lowers it to an
**isolated** path (``--out``) — the shipped ``src/shinro/runtime/graph_data.zig``
is never touched, so compiling a custom scenario never clobbers the shipped
graph.

An optional ``[plant]`` section derives ``n_x``/``n_u`` and the
``A_dynamics``/``B_dynamics`` model from the plant (the same derivation the
simulation path uses), so a scenario can be written as specs + weights with no
hand-computed model. Explicit ``[compile]`` dims and config ``A/B`` win over
the derived values.

The ``[compile]`` section is the build spec: it is validated strictly (unknown
keys and invalid ``optimize`` values are loud errors) and its sha256 is
recorded in the graph manifest's provenance, so the deployment record commits
to the exact build flags that produced the artifact.

Run::

    python3 -m shinro.codegen.scenario_gen tests/integration/scenarios/base_tracking.toml --out build/base

Exit codes: 0 ok · 1 untraceable · 2 usage/config error.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from shinro.codegen import build_composed_graph, lower_zig
from shinro.factories.registry import _CONTROLLER_REGISTRY, _ESTIMATOR_REGISTRY, _PLANT_REGISTRY
from shinro.utils.array_backend import NumpyBackend
from shinro.utils.config_resolver import resolve_config_path
from shinro.utils.linearization import derive_model

if TYPE_CHECKING:
    from shinro.codegen.compose import ComposedGraph

EXIT_OK = 0
EXIT_UNTRACEABLE = 1
EXIT_USAGE = 2

_COMPILE_KEYS = {"n_x", "n_u", "optimize", "target", "solver_dir", "oracle_tol", "artifact_name"}
_ALLOWED_OPTIMIZE = {"debug", "release"}


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
    when absent) and ``optimize`` / ``target`` / ``solver_dir`` /
    ``artifact_name`` (optional, with defaults). Unknown keys and invalid
    ``optimize`` values are loud errors — the section is the build spec, so a
    typo must not silently change the build.

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
    }


def load_scenario(scenario_path: str) -> dict:
    """Parse a scenario TOML into the fields the compile pipeline needs.

    Returns a dict with ``estimator_config`` / ``controller_config`` (config
    TOML paths), ``input_limits`` (``(lo, hi)`` ndarray pair or ``None``), the
    validated ``compile`` section, and ``plant`` (the ``[plant]`` section or
    ``None``). When ``[plant]`` is present, ``gen_scenario`` derives
    ``n_x``/``n_u`` and the ``A_dynamics``/``B_dynamics`` model from it.

    Raises:
        ValueError: On a missing ``[controller]`` section, an invalid
            ``[compile]`` section, or a missing ``[estimator]`` on anything
            other than a policy-only (``onnx_rl``) scenario.
    """
    with open(resolve_config_path(scenario_path), "rb") as f:
        cfg = tomllib.load(f)
    if "controller" not in cfg:
        raise ValueError(f"{scenario_path}: scenario requires a [controller] section")
    controller = cfg["controller"]
    estimator = cfg.get("estimator")
    if estimator is None:
        # A policy-only scenario: the ONNX policy is a standalone graph with no
        # estimator to compose with. Any other controller needs [estimator].
        ctype = controller.get("type") or _type_from_component_config(controller["config"])
        if ctype != "onnx_rl":
            raise ValueError(
                f"{scenario_path}: missing [estimator] section — only a policy-only 'onnx_rl' "
                f"scenario may omit it (got controller type {ctype!r})"
            )
    limits = None
    il = cfg.get("scenario", {}).get("input_limits")
    if il:
        limits = (np.array(il["min"], dtype=np.float64), np.array(il["max"], dtype=np.float64))
    return {
        "estimator_config": estimator["config"] if estimator else None,
        "controller_config": controller["config"],
        "estimator_type": estimator.get("type") if estimator else None,
        "controller_type": controller.get("type"),
        "input_limits": limits,
        "compile": _validate_compile(cfg.get("compile"), scenario_path),
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
    return {
        "configs": configs,
        "python_version": sys.version.split()[0],
        "numpy_version": version("numpy"),
    }


def _policy_graph(spec: dict, scenario_path: str) -> ComposedGraph:
    """Build the composed graph for a policy-only scenario (no estimator).

    The controller config is read directly rather than through the factory —
    the importer wants the raw ONNX/observation tables, and the policy graph is
    standalone (nothing to compose it with). The ``.onnx`` path is recorded on
    ``spec`` so the provenance can pin the weights.

    Args:
        spec: The loaded scenario spec (``estimator_config`` is None).
        scenario_path: Scenario TOML path, for error messages.

    Returns:
        The imported, memoryless :class:`ComposedGraph`.

    Raises:
        ValueError: If the controller config has no ``model_path``.
    """
    from shinro.codegen.onnx_import import import_onnx_policy

    with open(resolve_config_path(spec["controller_config"]), "rb") as f:
        ctrl = tomllib.load(f)
    model_path = ctrl.get("model_path")
    if model_path is None:
        raise ValueError(
            f"{scenario_path}: a policy-only scenario needs the controller's model_path "
            f"(artifact_dir is the *result* of this compile, not an input)"
        )
    action_cfg = {
        "action_space": ctrl.get("action_space", "continuous"),
        "deterministic": ctrl.get("deterministic", True),
        "action_scale": ctrl.get("action_scale", 1.0),
        "action_bias": ctrl.get("action_bias", 0.0),
    }
    for key in ("action_clip_low", "action_clip_high"):
        if key in ctrl:
            action_cfg[key] = ctrl[key]
    spec["policy_model_path"] = model_path
    return import_onnx_policy(
        model_path,
        n_x=spec["compile"]["n_x"],
        obs_cfg=ctrl.get("observation", {}),
        action_cfg=action_cfg,
        output_name=ctrl.get("output_name"),
    )


def _type_from_component_config(config_path: str) -> str:
    """Read the component's registered name from its own config TOML.

    Scenario TOMLs may omit ``type`` in ``[controller]``/``[estimator]`` (only
    a ``config`` path); the plant-derived injection needs the registry class,
    so the type is read from the component file when not declared.
    """
    with open(resolve_config_path(config_path), "rb") as f:
        t = tomllib.load(f).get("type")
    if t is None:
        raise ValueError(f"{config_path}: missing 'type' — required for plant-derived injection")
    return t


def gen_scenario(scenario_path: str, out_dir: str) -> tuple:
    """Trace + compose + lower a scenario into an isolated graph pair.

    When the scenario declares a ``[plant]`` section, ``n_x``/``n_u`` and the
    ``A_dynamics``/``B_dynamics`` model are derived from the plant (explicit
    ``[compile]`` dims and config ``A/B`` win over the derived values).

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

    # A policy-only scenario (onnx_rl) is standalone: no estimator to compose
    # with, so it bypasses the plant-derivation + build_composed_graph path and
    # goes straight from the ONNX graph to the lowered table.
    if spec["estimator_config"] is None:
        cg = _policy_graph(spec, scenario_path)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        graph_path = out / "graph_data.zig"
        lower_zig(
            cg,
            str(graph_path),
            provenance=_provenance(scenario_path, spec, model_path=spec["policy_model_path"]),
        )
        return cg, graph_path

    n_x = spec["compile"]["n_x"]
    n_u = spec["compile"]["n_u"]
    est_cfg = spec["estimator_config"]
    ctrl_cfg = spec["controller_config"]

    # When the scenario declares a plant-only [plant] section (type + config),
    # derive the dims and the A_dynamics/B_dynamics model from it — the same
    # derivation the simulation path (ScenarioFactory) uses, so sim and
    # compile can never disagree about the model. Sim-backed [plant] sections
    # (name only) are sim-only and ignored here. Explicit [compile] n_x/n_u and
    # config A/B win over derived.
    plant_cfg = spec["plant"]
    plant = None
    if plant_cfg is not None and "type" in plant_cfg and "config" in plant_cfg:
        with open(resolve_config_path(plant_cfg["config"]), "rb") as f:
            plant = _PLANT_REGISTRY[plant_cfg["type"]].from_config(
                tomllib.load(f), backend=NumpyBackend()
            )
        n_x = plant.get_state().shape[0]
        A_d, B_d = derive_model(plant)
        n_u = B_d.shape[1]
        # Strict-parse + plant-derived injection (dt filled/checked, model
        # derived when omitted), then flatten back to the dict shape
        # build_composed_graph's factories dispatch on ("type" key).
        est_type = spec["estimator_type"] or _type_from_component_config(spec["estimator_config"])
        ctrl_type = spec["controller_type"] or _type_from_component_config(spec["controller_config"])
        est_cfg = {
            **dataclasses.asdict(_ESTIMATOR_REGISTRY[est_type].load_config(
                spec["estimator_config"], plant=plant, derive_model=True
            )),
            "type": est_type,
        }
        ctrl_cfg = {
            **dataclasses.asdict(_CONTROLLER_REGISTRY[ctrl_type].load_config(
                spec["controller_config"], plant=plant, derive_model=True
            )),
            "type": ctrl_type,
        }

    if n_x is None or n_u is None:
        raise ValueError(
            f"{scenario_path}: cannot determine n_x/n_u — set [compile] n_x/n_u "
            "or add a [plant] section"
        )

    cg = build_composed_graph(
        est_cfg,
        ctrl_cfg,
        n_x,
        n_u,
        input_limits=spec["input_limits"],
        plant=plant,
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
