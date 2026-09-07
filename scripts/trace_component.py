"""Trace a registered component from its config TOML and verify the captured graph.

Three modes:

- **Inventory** (no arguments): list every registered controller / estimator /
  trajectory / plant, cross-referenced with the config TOMLs that exist for
  them. Answers "what do we have?".
- **Contract** (``--list <config>``): print the inferred trace contract —
  which method and input names the tracer will call. Answers "what shapes do
  I need to provide?".
- **Trace + oracle** (``<config> --shape ... [--state ...]``): trace the
  component, report graph stats + op coverage, then run the oracle check:
  re-execute the captured graph with the Python interpreter on the same
  random inputs as a live call of the component and assert bit-exact
  agreement. This is the "does my component trace?" gate — it proves the
  tracer captured the math faithfully. It does *not* prove the Zig lowering
  preserves it (that's the ctypes oracle in ``tests/test_zig_lowering.py``).

Exit codes: 0 all green · 1 untraceable op · 2 usage/config error · 3 oracle mismatch.

Run::

    python3 scripts/trace_component.py
    python3 scripts/trace_component.py --list src/shinro/configs/estimators/kalman_base.toml
    python3 scripts/trace_component.py src/shinro/configs/controllers/lqr_base.toml \\
        --shape current_state=3 --shape target_state=3
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from shinro.codegen import has_op, interpret, trace_node
from shinro.codegen.infer_contract import infer_contract
from shinro.components import Controller, Plant, StateEstimator, TrajectoryGenerator
from shinro.factories import (
    _CONTROLLER_REGISTRY,
    _ESTIMATOR_REGISTRY,
    _PLANT_REGISTRY,
    _TRAJECTORY_REGISTRY,
    ControllerFactory,
    EstimatorFactory,
    TrajectoryFactory,
)
from shinro.utils.array_backend import NumpyBackend
from shinro.utils.config_resolver import resolve_config_path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "src" / "shinro" / "configs"

REGISTRIES: dict[str, dict[str, type]] = {
    "controller": _CONTROLLER_REGISTRY,
    "estimator": _ESTIMATOR_REGISTRY,
    "trajectory": _TRAJECTORY_REGISTRY,
    "plant": _PLANT_REGISTRY,
}

FACTORIES = {
    "controller": ControllerFactory,
    "estimator": EstimatorFactory,
    "trajectory": TrajectoryFactory,
}

EXIT_OK = 0
EXIT_UNTRACEABLE = 1
EXIT_USAGE = 2
EXIT_ORACLE = 3

_ABC_TYPES = (Controller, StateEstimator, TrajectoryGenerator, Plant)


def parse_shape(spec: str) -> tuple[int, ...]:
    """Parse ``3``, ``3x1``, ``3x3``, or ``scalar`` into a shape tuple."""
    spec = spec.strip().lower()
    if spec == "scalar":
        return ()
    try:
        return tuple(int(d) for d in spec.split("x"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid shape '{spec}' (use e.g. 3, 3x1, 3x3, scalar)") from e


def _parse_pairs(specs: list[str]) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"expected NAME=SPEC, got '{spec}'")
        name, shape = spec.split("=", 1)
        out[name] = parse_shape(shape)
    return out


def load_component(config_path: str) -> tuple[Any, str, str]:
    """Instantiate a registered component from its config TOML.

    Returns:
        ``(component, category, type_name)``.
    """
    resolved = resolve_config_path(config_path)
    with open(resolved, "rb") as f:
        cfg = tomllib.load(f)
    ctype = cfg.get("type")
    if not ctype:
        raise ValueError(f"{config_path}: no 'type' field")
    for category, registry in REGISTRIES.items():
        if ctype in registry:
            if category == "plant":
                component = registry[ctype].from_config(cfg, backend=NumpyBackend())
            else:
                component = FACTORIES[category](config_path).create(backend=NumpyBackend())
            if not isinstance(component, _ABC_TYPES):
                raise ValueError(
                    f"{config_path}: type '{ctype}' from_config returned "
                    f"{type(component).__name__}, not a component instance — not standalone-traceable"
                )
            return component, category, ctype
    known = sorted(t for r in REGISTRIES.values() for t in r)
    raise ValueError(f"{config_path}: type '{ctype}' is not registered. Registered: {known}")


def cmd_inventory() -> int:
    """List registered components cross-referenced with available configs."""
    configs_by_type: dict[str, list[str]] = {}
    if CONFIG_DIR.is_dir():
        for toml in sorted(CONFIG_DIR.rglob("*.toml")):
            try:
                with open(toml, "rb") as f:
                    cfg = tomllib.load(f)
            except tomllib.TOMLDecodeError:
                continue
            ctype = cfg.get("type")
            if ctype:
                rel = toml.relative_to(CONFIG_DIR).as_posix()
                configs_by_type.setdefault(ctype, []).append(rel)

    for category, registry in REGISTRIES.items():
        if not registry:
            continue
        label = "trajectories" if category == "trajectory" else f"{category}s"
        print(f"{label} ({len(registry)}):")
        for name in sorted(registry):
            cfgs = configs_by_type.get(name, [])
            if cfgs:
                print(f"  {name:<22} {', '.join(cfgs)}")
            else:
                print(f'  {name:<22} (no config TOML yet — write one with type = "{name}")')
        print()
    return EXIT_OK


def _load_or_die(config_path: str) -> tuple[Any, str, str] | None:
    try:
        return load_component(config_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return None


def cmd_contract(config_path: str) -> int:
    """Print the inferred trace contract for a config."""
    loaded = _load_or_die(config_path)
    if loaded is None:
        return EXIT_USAGE
    component, category, ctype = loaded
    contract = infer_contract(component)
    args = ", ".join(contract.input_names)
    print(f"{ctype}.{contract.method_name}({args})")
    print(f"category: {category}")
    print("state attrs are auto-detected at trace time; pass --state name=shape to")
    print("pre-inject them as recurrent ports (recommended for estimators).")
    return EXIT_OK


def _random_state(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Random state value; square 2-D states are made SPD (covariance-like)."""
    if len(shape) == 2 and shape[0] == shape[1]:
        a = rng.normal(0.0, 0.1, shape)
        return a @ a.T + 0.1 * np.eye(shape[0])
    return rng.normal(0.0, 0.1, shape)


def cmd_trace(
    config_path: str,
    shapes: dict[str, tuple[int, ...]],
    state_shapes: dict[str, tuple[int, ...]],
    samples: int,
    seed: int,
) -> int:
    """Trace a component and run the interpret-vs-live oracle check."""
    loaded = _load_or_die(config_path)
    if loaded is None:
        return EXIT_USAGE
    component, category, ctype = loaded
    contract = infer_contract(component)

    missing = [n for n in contract.input_names if n not in shapes]
    if missing:
        print(f"missing shapes for input(s): {', '.join(missing)}", file=sys.stderr)
        print(f"contract: {ctype}.{contract.method_name}({', '.join(contract.input_names)})", file=sys.stderr)
        print("pass --shape name=SPEC for each (SPEC = 3, 3x1, 3x3, scalar)", file=sys.stderr)
        return EXIT_USAGE

    try:
        ng = trace_node(component, input_shapes=shapes, state_shapes=state_shapes)
    except NotImplementedError as e:
        print(f"TRACE FAILED: {e}", file=sys.stderr)
        return EXIT_UNTRACEABLE
    except Exception as e:  # component code that tracers can't execute
        print(
            f"TRACE FAILED: {type(component).__name__} is not traceable: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return EXIT_UNTRACEABLE

    ops = Counter(node.op for node in ng.graph.nodes)
    unknown = [op for op in ops if not has_op(op)]
    coverage = ", ".join(f"{op}×{n}" for op, n in sorted(ops.items()))
    if unknown:
        print(f"traced: {len(ng.graph.nodes)} nodes | ops: {coverage}")
        print(f"NOT LOWERABLE: {', '.join(unknown)}", file=sys.stderr)
        return EXIT_UNTRACEABLE
    print(f"traced: {len(ng.graph.nodes)} nodes | ops: {coverage} — all registered ✓")

    rng = np.random.default_rng(seed)
    state_attrs = list(state_shapes)
    failures: list[str] = []
    for i in range(samples):
        live_inputs = {name: rng.normal(0.0, 0.1, shape) for name, shape in shapes.items()}
        state_values = {name: _random_state(rng, shape) for name, shape in state_shapes.items()}
        for name, value in state_values.items():
            setattr(component, name, value)
        result = getattr(component, contract.method_name)(*[live_inputs[n] for n in contract.input_names])
        live_state = {name: getattr(component, name) for name in state_attrs}

        graph_inputs = dict(live_inputs)
        for name, value in state_values.items():
            graph_inputs[f"state_{name.lstrip('_')}"] = value
        traced = interpret(ng.graph, graph_inputs)

        checks = [("out", result)]
        checks += [(f"state_{name.lstrip('_')}", live_state[name]) for name in state_attrs]
        for port, expected in checks:
            actual = traced.get(port)
            if actual is None:
                failures.append(f"sample {i}: graph has no output '{port}'")
                continue
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                diff = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
                failures.append(f"sample {i}: port '{port}' diverged (max abs diff {diff:.3e})")

    if failures:
        print(f"oracle A (interpret vs live): {samples - len(failures)}/{samples} passed — MISMATCH")
        for f in failures[:10]:
            print(f"  - {f}", file=sys.stderr)
        return EXIT_ORACLE
    print(f"oracle A (interpret vs live): {samples}/{samples} random inputs bit-exact ✓")
    return EXIT_OK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="component config TOML (omit for inventory)")
    parser.add_argument("--list", action="store_true", help="print the inferred trace contract and exit")
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="NAME=SPEC",
        help="input shape, repeatable (SPEC = 3, 3x1, 3x3, scalar)",
    )
    parser.add_argument(
        "--state",
        action="append",
        default=[],
        metavar="NAME=SPEC",
        help="recurrent state attr shape, repeatable (pre-injected as a state_* port)",
    )
    parser.add_argument("--samples", type=int, default=20, help="random inputs for the oracle (default 20)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the oracle (default 0)")
    args = parser.parse_args()

    if args.config is None:
        return cmd_inventory()

    shapes = _parse_pairs(args.shape)
    state_shapes = _parse_pairs(args.state)

    if args.list:
        return cmd_contract(args.config)

    return cmd_trace(args.config, shapes, state_shapes, args.samples, args.seed)


if __name__ == "__main__":
    sys.exit(main())
