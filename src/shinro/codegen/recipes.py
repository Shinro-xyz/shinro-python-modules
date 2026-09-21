"""Registered graph recipes — the named compose wirings a scenario selects.

A *recipe* turns a loaded scenario spec into a :class:`ComposedGraph`. The e2e
pipeline dispatches through the registry, so the compose wiring is explicit data
(a scenario's ``[compile].recipe``) rather than an implicit branch on which
sections the scenario happens to declare.

Two recipes ship:

- ``closed_loop_tracking`` (default) — estimator + controller composed into the
  fixed ABC dataflow (``y → estimator → x̂ → controller → u → [clip]``), with
  plant-derived dims/model when the scenario declares a plant-only ``[plant]``
  section. This is the generic :func:`shinro.codegen.build.build_composed_graph`
  path.
- ``policy_only`` — a standalone ONNX policy graph (no estimator to compose).

This module also owns the two **shipped default graphs** (KF+LQR and KF+MPC_LTI
on the 3-DOF holonomic base) that ``make zig-gen`` / ``make zig-build-mpc``
write to the shared ``src/shinro/runtime/graph_data.zig``. ``scripts/gen_base.py``
and ``scripts/gen_mpc.py`` are shims over them. Each graph's config set is
declared once (:data:`LQR_CONFIGS` / :data:`DEFAULT_MPC_CONTROLLER`) and the
manifest provenance is derived from it, so a config cannot be silently left out
of the record.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
import tomllib
from collections.abc import Callable, Sequence
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

from shinro.codegen import lower_zig
from shinro.codegen.build import build_composed_graph, instantiate
from shinro.codegen.runtime_paths import runtime_root
from shinro.factories.registry import _CONTROLLER_REGISTRY, _ESTIMATOR_REGISTRY, _PLANT_REGISTRY
from shinro.utils.array_backend import NumpyBackend
from shinro.utils.config_resolver import resolve_config_path
from shinro.utils.linearization import derive_model

if TYPE_CHECKING:
    from shinro.codegen.compose import ComposedGraph

# ─── registry ───────────────────────────────────────────────────────────────

_GRAPH_RECIPES: dict[str, Callable[[dict], ComposedGraph]] = {}


def register_graph(name: str) -> Callable[[Callable[[dict], ComposedGraph]], Callable[[dict], ComposedGraph]]:
    """Register a recipe under ``name`` (a scenario's ``[compile].recipe``).

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def deco(fn: Callable[[dict], ComposedGraph]) -> Callable[[dict], ComposedGraph]:
        if name in _GRAPH_RECIPES:
            raise ValueError(f"graph recipe '{name}' is already registered")
        _GRAPH_RECIPES[name] = fn
        return fn

    return deco


def available_graphs() -> list[str]:
    """Return the registered recipe names, sorted."""
    return sorted(_GRAPH_RECIPES)


def build_recipe(name: str, spec: dict) -> ComposedGraph:
    """Build the graph for ``name`` from a loaded scenario spec.

    Args:
        name: A registered recipe name (see :func:`available_graphs`).
        spec: The dict returned by
            :func:`shinro.codegen.scenario_gen.load_scenario` (must carry
            ``scenario_path`` for error messages).

    Returns:
        The composed graph for one closed-loop step.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _GRAPH_RECIPES:
        raise ValueError(f"unknown graph recipe '{name}' (registered: {available_graphs()})")
    return _GRAPH_RECIPES[name](spec)


def tool_versions() -> dict[str, str]:
    """Tool versions recorded in a lowering's provenance."""
    return {"python_version": sys.version.split()[0], "numpy_version": version("numpy")}


def _sha256(config_path: str) -> str:
    """Return the sha256 of a config file's raw bytes (hermetic: no preprocessing)."""
    with open(resolve_config_path(config_path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def type_from_component_config(config_path: str) -> str:
    """Read a component's registered name from its own config TOML.

    Scenario TOMLs may omit ``type`` in ``[controller]``/``[estimator]`` (only a
    ``config`` path); plant-derived injection needs the registry class, so the
    type is read from the component file when not declared.
    """
    with open(resolve_config_path(config_path), "rb") as f:
        t = tomllib.load(f).get("type")
    if t is None:
        raise ValueError(f"{config_path}: missing 'type' — required for plant-derived injection")
    return t


# ─── recipes ────────────────────────────────────────────────────────────────


@register_graph("closed_loop_tracking")
def closed_loop_tracking(spec: dict) -> ComposedGraph:
    """Estimator + controller closed loop (the generic composed-step graph).

    When the scenario declares a plant-only ``[plant]`` section (``type`` +
    ``config``), the plant is the single source of truth: ``n_x``/``n_u`` and the
    estimator/controller model are derived from it (the same derivation
    :class:`ScenarioFactory` uses, so sim and compile cannot disagree), and it
    **wins** — a ``[compile]`` ``n_x``/``n_u`` that disagrees with the plant is a
    loud error, mirroring the ``[scenario].dt`` check, so a stale dimension
    cannot silently bake the wrong graph size.
    """
    est_cfg, ctrl_cfg, n_x, n_u, limits, plant = closed_loop_configs(spec)
    return build_composed_graph(est_cfg, ctrl_cfg, n_x, n_u, input_limits=limits, plant=plant)


def closed_loop_configs(spec: dict) -> tuple:
    """Resolve ``(est_cfg, ctrl_cfg, n_x, n_u, input_limits, plant)`` for a spec.

    The shared front half of the ``closed_loop_tracking`` recipe. A plant-only
    ``[plant]`` section (``type`` + ``config``) is authoritative for the dims and
    model; a declared ``[compile]`` dim that disagrees is a loud error.
    :func:`live_components` reuses this so gate A drives exactly the components
    the graph was traced from.
    """
    n_x = spec["compile"]["n_x"]
    n_u = spec["compile"]["n_u"]
    est_cfg = spec["estimator_config"]
    ctrl_cfg = spec["controller_config"]

    plant_cfg = spec["plant"]
    plant = None
    if plant_cfg is not None and "type" in plant_cfg and "config" in plant_cfg:
        ptype = plant_cfg["type"]
        with open(resolve_config_path(plant_cfg["config"]), "rb") as f:
            plant = _PLANT_REGISTRY[ptype].from_config(tomllib.load(f), backend=NumpyBackend())
        derived_n_x = plant.get_state().shape[0]
        _, B_d = derive_model(plant)
        derived_n_u = B_d.shape[1]
        # The plant is authoritative: a declared [compile] dim must agree, or a
        # stale n_x/n_u would silently bake the wrong graph size.
        if n_x is not None and n_x != derived_n_x:
            raise ValueError(
                f"{spec['scenario_path']}: [compile].n_x={n_x} disagrees with plant "
                f"'{ptype}' state dim {derived_n_x} — drop n_x or set it to {derived_n_x}"
            )
        if n_u is not None and n_u != derived_n_u:
            raise ValueError(
                f"{spec['scenario_path']}: [compile].n_u={n_u} disagrees with plant "
                f"'{ptype}' input dim {derived_n_u} — drop n_u or set it to {derived_n_u}"
            )
        n_x, n_u = derived_n_x, derived_n_u
        # Strict-parse + plant-derived injection (dt filled/checked, model
        # derived when omitted), then flatten back to the dict shape
        # build_composed_graph's factories dispatch on ("type" key).
        est_type = spec["estimator_type"] or type_from_component_config(spec["estimator_config"])
        ctrl_type = spec["controller_type"] or type_from_component_config(spec["controller_config"])
        est_cfg = {
            **dataclasses.asdict(_ESTIMATOR_REGISTRY[est_type].load_config(spec["estimator_config"], plant=plant, derive_model=True)),
            "type": est_type,
        }
        ctrl_cfg = {
            **dataclasses.asdict(_CONTROLLER_REGISTRY[ctrl_type].load_config(spec["controller_config"], plant=plant, derive_model=True)),
            "type": ctrl_type,
        }

    if n_x is None or n_u is None:
        raise ValueError(
            f"{spec['scenario_path']}: cannot determine n_x/n_u — set [compile] n_x/n_u or add a [plant] section"
        )
    return est_cfg, ctrl_cfg, n_x, n_u, spec["input_limits"], plant


def live_components(spec: dict) -> tuple:
    """Instantiate the live estimator/controller for gate A.

    Returns ``(est, ctrl, n_x, n_u, input_limits)`` — the same instances
    :func:`closed_loop_tracking` traces, rebuilt from the same configs.
    """
    est_cfg, ctrl_cfg, n_x, n_u, limits, plant = closed_loop_configs(spec)
    est, ctrl = instantiate(est_cfg, ctrl_cfg, plant)
    return est, ctrl, n_x, n_u, limits


@register_graph("policy_only")
def policy_only(spec: dict) -> ComposedGraph:
    """A standalone ONNX policy graph (no estimator to compose).

    The controller config is read directly rather than through the factory — the
    importer wants the raw ONNX/observation tables, and the policy graph has
    nothing to compose with. The ``.onnx`` path is recorded on ``spec`` so the
    provenance can pin the weights.
    """
    from shinro.codegen.onnx_import import import_onnx_policy

    with open(resolve_config_path(spec["controller_config"]), "rb") as f:
        ctrl = tomllib.load(f)
    model_path = ctrl.get("model_path")
    if model_path is None:
        raise ValueError(
            f"{spec['scenario_path']}: a policy-only scenario needs the controller's model_path "
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


# ─── shipped default graphs (make zig-gen / make zig-build-mpc) ─────────────

#: The 3-DOF holonomic base's controller-output clip, shared by both shipped graphs.
_BASE_LIMITS = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))

LQR_CONFIGS = ("samples/estimators/kalman_base.toml", "samples/controllers/lqr_base.toml")
DEFAULT_MPC_CONTROLLER = "samples/controllers/mpc_lti_base.toml"


def build_base_graph() -> ComposedGraph:
    """Trace KF + LQR and compose the shipped base_tracking step graph.

    The KF's covariance ``P`` is a recurrent state port (pre-injected by the
    two-pass trace), so the deployed graph runs the full live predict-update
    Riccati recursion — the host seeds ``P0 = 0.1*I`` at tick 0 and feeds
    ``state_P`` back each tick.
    """
    return build_composed_graph(LQR_CONFIGS[0], LQR_CONFIGS[1], n_x=3, n_u=3, input_limits=_BASE_LIMITS)


def build_mpc_composed_graph(controller_config: str = DEFAULT_MPC_CONTROLLER) -> ComposedGraph:
    """Trace KF + MPC and compose the closed-loop step graph.

    Same estimator side as :func:`build_base_graph`; the controller is a
    regulator (``MPC_LTI`` by default, or ``MPC_DeltaU`` via
    ``controller_config``). Compose feeds the tracking error ``x0 = x̂ − x_ref``
    to the regulator, so regulating it to zero tracks ``x_ref``.
    """
    return build_composed_graph(LQR_CONFIGS[0], controller_config, n_x=3, n_u=3, input_limits=_BASE_LIMITS)


def _shipped_provenance(configs: Sequence[str]) -> dict:
    """Provenance dict for a shipped graph: config sha256s + tool versions."""
    return {"configs": {path: _sha256(path) for path in configs}, **tool_versions()}


def _write_shipped(graph: ComposedGraph, configs: Sequence[str]) -> None:
    path = runtime_root() / "graph_data.zig"
    lower_zig(graph, str(path), provenance=_shipped_provenance(configs))
    print(f"wrote {path} ({len(graph.graph.nodes)} nodes, inputs={graph.inputs})")


def main_lqr() -> None:
    """Entry point for ``scripts/gen_base.py`` (wired to ``make zig-gen``)."""
    _write_shipped(build_base_graph(), LQR_CONFIGS)


def main_mpc() -> None:
    """Entry point for ``scripts/gen_mpc.py`` (wired to ``make zig-mpc-gen``)."""
    _write_shipped(build_mpc_composed_graph(), (LQR_CONFIGS[0], DEFAULT_MPC_CONTROLLER))
