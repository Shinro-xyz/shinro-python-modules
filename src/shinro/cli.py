"""``shinro`` — one CLI over the scenario TOML.

The scenario TOML is the single artifact a developer authors; these verbs cover
the four gates (behavior → trace → lowering → drift):

    shinro check  <cfg.toml>       construct a component (or scenario); print the trace contract
    shinro run    <scenario.toml>  run the closed loop and apply [scenario.tolerance]
    shinro trace  <cfg.toml>       trace a component: op coverage + interpret-vs-live oracle
    shinro build  <scenario.toml>  trace → compose → lower → zig → oracle → stamp → verify
    shinro verify <scenario.toml>  re-hash the stamped artifacts (drift gate)

``build``/``verify`` default their output dir to
``build/<scenario-stem>/<optimize>-<target>`` (or ``[compile].out`` when set),
so different scenarios, build modes, and targets never clobber each other; pass
``--out`` to override.

Each verb is a thin dispatcher over the existing package APIs
(:mod:`shinro.codegen.component_cli`, :mod:`shinro.codegen.cli`,
:mod:`shinro.codegen.verify`, :class:`shinro.factories.ScenarioFactory`) — this
module owns no pipeline logic of its own.

Third-party components: ``--import MODULE`` (repeatable) imports a module
before the verb runs, so a ``@register_*`` component defined outside shinro is
visible to the registry. Accepted both before and after the verb name.

Exit codes: 0 ok · 1 gate failed (untraceable · behavior · drift) · 2
usage/config · 3 oracle mismatch · 4 build/verify failure · 5 zig missing.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from shinro.codegen.cli import compile_scenario
from shinro.codegen.component_cli import (
    EXIT_OK,
    EXIT_USAGE,
    _parse_pairs,
    cmd_check,
    cmd_contract,
    cmd_inventory,
    cmd_trace,
)
from shinro.codegen.scenario_gen import load_scenario
from shinro.codegen.verify import verify
from shinro.utils.config_resolver import resolve_config_path
from shinro.utils.plugin_loader import PluginImportError, import_modules

EXIT_GATE = 1


def _resolve_out(
    scenario_path: str,
    spec: dict,
    out_flag: str | None,
    optimize_flag: str | None,
    target_flag: str | None,
) -> str:
    """Resolve the build/verify output dir.

    Precedence: ``--out`` > ``[compile].out`` > ``build/<scenario-stem>/<optimize>-<target>``.
    The default is keyed on the scenario *and* the build mode, so a Debug and a
    ReleaseFast build (or a native and a cross build) of the same scenario land
    in separate dirs instead of overwriting each other's graph and record.
    """
    if out_flag:
        return out_flag
    compile_spec = spec["compile"]
    if compile_spec.get("out"):
        return compile_spec["out"]
    optimize = optimize_flag or compile_spec["optimize"]
    target = target_flag or compile_spec["target"]
    return str(Path("build") / Path(scenario_path).stem / f"{optimize}-{target}")


def _config_kind(path: str) -> str | None:
    """Classify a TOML as ``"component"`` (top-level ``type``), ``"scenario"``, or ``None``.

    ``None`` covers both an unreadable file and a config that is neither shape;
    the caller reports the path either way.
    """
    try:
        with open(resolve_config_path(path), "rb") as f:
            cfg = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    if "type" in cfg:
        return "component"
    if "controller" in cfg or "scenario" in cfg:
        return "scenario"
    return None


def _check_scenario(path: str) -> int:
    """Construct a full scenario and report its composed roles (no run)."""
    from shinro.factories import ScenarioFactory

    try:
        scenario = ScenarioFactory(path).build()
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as e:  # sim-backed scenarios need MuJoCo; report cleanly
        print(f"BUILD FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_USAGE
    roles = [
        role
        for role, present in (
            ("plant", scenario.plant is not None),
            ("controller", scenario.controller is not None),
            ("estimator", scenario.estimator is not None),
            ("trajectory", scenario.trajectory is not None),
            ("sim", scenario.sim is not None),
        )
        if present
    ]
    name = (scenario.config.get("scenario") or {}).get("name", Path(path).stem)
    print(f"OK: scenario '{name}' constructs — roles: {', '.join(roles)}")
    return EXIT_OK


def _cmd_check(args: argparse.Namespace) -> int:
    kind = _config_kind(args.config)
    if kind == "component":
        return cmd_check(args.config)
    if kind == "scenario":
        return _check_scenario(args.config)
    print(
        f"ERROR: {args.config}: not a readable component config (top-level 'type') or a scenario ([controller])",
        file=sys.stderr,
    )
    return EXIT_USAGE


def _cmd_run(args: argparse.Namespace) -> int:
    from shinro.factories import ScenarioFactory

    try:
        scenario = ScenarioFactory(args.scenario).build()
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as e:
        print(f"BUILD FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_USAGE

    try:
        result = scenario.run(steps=args.steps, seed=args.seed)
    except Exception as e:
        print(f"RUN FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_USAGE

    report = result.check()
    for metric in ("steady_state", "estimator"):
        if metric not in report:
            continue
        m = report[metric]
        if m.get("ok") is None:
            print(f"{metric}: not computable for this run kind")
        else:
            print(f"{metric}: {m['error']:.4f} (tol {m['tol']}) {'✓' if m['ok'] else '✗'}")
    gate = report.get("ok")
    if gate is not None and not gate:
        print("BEHAVIOR GATE FAILED — run exceeds [scenario.tolerance]", file=sys.stderr)
        return EXIT_GATE
    print(f"run OK: {len(result)} steps")
    return EXIT_OK


def _cmd_trace(args: argparse.Namespace) -> int:
    if args.config is None:
        return cmd_inventory()
    if args.list:
        return cmd_contract(args.config)
    return cmd_trace(args.config, _parse_pairs(args.shape), _parse_pairs(args.state), args.samples, args.seed)


def _cmd_build(args: argparse.Namespace) -> int:
    try:
        spec = load_scenario(args.scenario)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_USAGE
    out = _resolve_out(args.scenario, spec, args.out, args.optimize, args.target)
    return compile_scenario(
        args.scenario,
        out,
        optimize=args.optimize,
        target=args.target,
        solver_dir=args.solver_dir,
        solver=args.solver,
        native_record=args.native_record,
        artifact_name=args.artifact_name,
        samples=args.samples,
        seed=args.seed,
    )


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        spec = load_scenario(args.scenario)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_USAGE
    name = spec["compile"]["artifact_name"]
    out = Path(_resolve_out(args.scenario, spec, args.out, args.optimize, args.target))
    record = out / "lib" / f"{name}.deployment.json"
    if not record.exists():
        print(f"ERROR: no deployment record at {record} — run `shinro build {args.scenario}` first", file=sys.stderr)
        return EXIT_USAGE
    if args.graph:
        graph = Path(args.graph)
    else:
        default_graph = out / "graph_data.zig"
        graph = default_graph if default_graph.exists() else None
    solver_dir = Path(args.solver_dir) if args.solver_dir else None
    return verify(record, Path(args.binary) if args.binary else None, graph, solver_dir)


_IMPORT_HELP = "import a module before the verb runs so its @register_* components are visible (repeatable)"


def _add_import_flag(parser: argparse.ArgumentParser, default: object) -> None:
    """Add ``--import MODULE``. Accepted both before and after the verb name.

    On subparsers the default is ``argparse.SUPPRESS`` so a flag given before the
    verb is not clobbered; when given after the verb it appends to the same list.
    """
    parser.add_argument(
        "--import",
        action="append",
        default=default,
        dest="import_modules",
        metavar="MODULE",
        help=_IMPORT_HELP,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shinro",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_import_flag(parser, default=[])
    sub = parser.add_subparsers(dest="verb", required=True)

    c = sub.add_parser("check", help="construct a component (or scenario) and print the trace contract")
    c.add_argument("config", help="component config TOML, or a scenario TOML")

    r = sub.add_parser("run", help="run a scenario and apply [scenario.tolerance]")
    r.add_argument("scenario", help="scenario TOML path")
    r.add_argument("--steps", type=int, help="step count (default: [scenario].duration / dt)")
    r.add_argument("--seed", type=int, help="noise RNG seed (overrides [noise.measurement].seed)")

    t = sub.add_parser("trace", help="trace a component: op coverage + interpret-vs-live oracle")
    t.add_argument("config", nargs="?", help="component config TOML (omit for the inventory)")
    t.add_argument("--list", action="store_true", help="print the inferred trace contract and exit")
    t.add_argument("--shape", action="append", default=[], metavar="NAME=SPEC", help="input shape, repeatable (3, 3x1, 3x3, scalar)")
    t.add_argument("--state", action="append", default=[], metavar="NAME=SPEC", help="recurrent state attr shape, repeatable")
    t.add_argument("--samples", type=int, default=20, help="random inputs for the oracle (default 20)")
    t.add_argument("--seed", type=int, default=0, help="RNG seed for the oracle (default 0)")

    b = sub.add_parser("build", help="trace -> compose -> lower -> zig -> oracle -> stamp -> verify")
    b.add_argument("scenario", help="scenario TOML path")
    b.add_argument("--out", help="output dir (default: [compile].out or build/<scenario-stem>/<optimize>-<target>)")
    b.add_argument("--optimize", choices=["debug", "release"], help="override [compile].optimize")
    b.add_argument("--target", help="override [compile].target (zig triple)")
    b.add_argument("--solver-dir", help="override [compile].solver_dir (pre-baked OSQP solver dir)")
    b.add_argument("--solver", help="override [compile].solver (bake on demand, e.g. emosqp)")
    b.add_argument("--native-record", help="cross build: path to the oracle-verified native record to reference")
    b.add_argument("--artifact-name", help="override [compile].artifact_name")
    b.add_argument("--samples", type=int, default=20, help="random inputs for the oracle (default 20)")
    b.add_argument("--seed", type=int, default=0, help="RNG seed for the oracle (default 0)")

    v = sub.add_parser("verify", help="re-hash the stamped artifacts against the deployment record")
    v.add_argument("scenario", help="scenario TOML path")
    v.add_argument("--out", help="output dir holding the record (default: [compile].out or build/<scenario-stem>/<optimize>-<target>)")
    v.add_argument("--optimize", choices=["debug", "release"], help="build mode to locate (default: [compile].optimize)")
    v.add_argument("--target", help="build target to locate (default: [compile].target)")
    v.add_argument("--binary", help="deployed .so (default: the record's binary path)")
    v.add_argument("--graph", help="graph_data.zig to verify (default: <out>/graph_data.zig)")
    v.add_argument("--solver-dir", help="baked solver dir to verify")

    # Also accepted after the verb (`shinro build x.toml --import my_pkg`);
    # SUPPRESS keeps a pre-verb flag from being overwritten.
    for p in (c, r, t, b, v):
        _add_import_flag(p, default=argparse.SUPPRESS)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        import_modules(args.import_modules)
    except PluginImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_USAGE
    handlers = {
        "check": _cmd_check,
        "run": _cmd_run,
        "trace": _cmd_trace,
        "build": _cmd_build,
        "verify": _cmd_verify,
    }
    return handlers[args.verb](args)


if __name__ == "__main__":
    sys.exit(main())
