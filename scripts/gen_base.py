"""Generate ``src/shinro/runtime/graph_data.zig`` for the base_tracking closed-loop step.

Entry point: ``python scripts/gen_base.py`` (wired to ``make zig-gen``).

Lives in ``scripts/`` (not ``src/shinro/``) because it's LeKiwi-specific —
one fixed graph for a single scenario. The framework-side pieces it uses
(tracing, compose, lowering) are generic and stay in ``shinro.codegen``.

Builds the base_tracking composed graph (KalmanFilter + LQR on the 3-DOF
holonomic base, with the scenario's input clip) via the generic
:func:`shinro.codegen.build.build_composed_graph`, serializes it through
:func:`shinro.codegen.lower_zig.lower_zig`, and writes the Zig data table that
``src/shinro/runtime/lower.zig`` compiles against.

Note: ``src/shinro/runtime/graph_data.zig`` is a shared generated path — this
script, ``scripts/gen_mpc.py``, and the pytest fixtures in
``tests/test_zig_lowering.py`` all overwrite it, so the shipped graph is
whichever ran last. Re-run this script (or ``make zig-gen``) to restore the
shipped KF + LQR base graph.
"""

from __future__ import annotations

import hashlib
import sys
from importlib.metadata import version

import numpy as np

from shinro.codegen import build_composed_graph, lower_zig
from shinro.utils.config_resolver import resolve_config_path


def _sha256(config_path: str) -> str:
    """Return the sha256 of a config file's raw bytes (hermetic: no preprocessing)."""
    with open(resolve_config_path(config_path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_base_graph():
    """Trace KF + LQR and compose the base_tracking step graph.

    Delegates to the generic :func:`shinro.codegen.build.build_composed_graph`
    with the shipped base configs and the 3-DOF holonomic base dims. The KF's
    covariance P is a recurrent state port (pre-injected by the two-pass
    trace), so the deployed graph runs the full live predict-update Riccati
    recursion — the host seeds P0 = 0.1*I at tick 0 and feeds state_P back
    each tick.

    Returns:
        A :class:`~shinro.codegen.compose.ComposedGraph` for the base_tracking
        closed-loop step.
    """
    limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))
    return build_composed_graph(
        "configs/estimators/kalman_base.toml",
        "configs/controllers/lqr_base.toml",
        n_x=3,
        n_u=3,
        input_limits=limits,
    )


def main() -> None:
    """Generate ``src/shinro/runtime/graph_data.zig`` from the base_tracking graph.

    Entry point for ``python scripts/gen_base.py`` (wired to
    ``make zig-gen``). Serializes the composed graph and prints a summary of
    the node count and input ports.
    """
    composed = build_base_graph()
    lower_zig(
        composed,
        "src/shinro/runtime/graph_data.zig",
        provenance={
            "configs": {
                "configs/estimators/kalman_base.toml": _sha256("configs/estimators/kalman_base.toml"),
                "configs/controllers/lqr_base.toml": _sha256("configs/controllers/lqr_base.toml"),
            },
            "python_version": sys.version.split()[0],
            "numpy_version": version("numpy"),
        },
    )
    n = len(composed.graph.nodes)
    print(f"wrote src/shinro/runtime/graph_data.zig ({n} nodes, inputs={composed.inputs})")


if __name__ == "__main__":
    main()
