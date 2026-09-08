"""``shinro-compile`` — compile a scenario TOML into a verified, stamped ``libbase.so``.

The external-team entry point for the e2e pipeline. Chains the two stages
that ``make compile`` runs in a checkout:

1. **gen** — trace + compose + lower the scenario's estimator + controller
   into an isolated ``graph_data.zig`` + manifest (zig-free).
2. **build** — ``zig build`` the comptime VM against that graph, then
   integrity-check, oracle-verify, stamp, and verify the deployment record.

Requires ``zig`` on PATH for the build stage. Exit codes: 0 ok · 1
untraceable · 2 usage/config · 3 oracle mismatch · 4 build/verify failure ·
5 zig missing.
"""

from __future__ import annotations

import argparse
import sys

from shinro.codegen.scenario_build import build_scenario
from shinro.codegen.scenario_gen import EXIT_UNTRACEABLE, EXIT_USAGE, gen_scenario


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="scenario TOML path")
    parser.add_argument("--out", default="build/scenario", help="output dir for the graph (default: build/scenario)")
    parser.add_argument("--optimize", choices=["debug", "release"], help="override [compile].optimize")
    parser.add_argument("--target", help="override [compile].target (zig triple, e.g. aarch64-linux-gnu)")
    parser.add_argument("--solver-dir", help="override [compile].solver_dir (baked OSQP solver dir)")
    parser.add_argument("--samples", type=int, default=20, help="random inputs for the oracle (default 20)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the oracle (default 0)")
    args = parser.parse_args()

    # Stage 1: gen (zig-free).
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

    # Stage 2: build + oracle + stamp + verify.
    return build_scenario(
        args.out,
        scenario=args.scenario,
        optimize=args.optimize,
        target=args.target,
        solver_dir=args.solver_dir,
        samples=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
