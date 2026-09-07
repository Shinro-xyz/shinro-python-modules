"""Generate ``runtime/graph_data.zig`` for the KF + MPC closed-loop step.

Alternative deployment target to ``scripts/gen_base.py`` (KF + LQR): swaps the
controller for an MPC regulator — ``MPC_LTI`` from ``mpc_lti_base.toml`` by
default, or ``MPC_DeltaU`` from ``mpc_base.toml`` via the ``controller_config``
argument. Compose feeds the tracking error ``x0 = x_hat - x_ref`` to the
regulator, so regulating it to zero tracks ``x_ref`` (exact for the base
plant, A = I; general A would need an ``(A - I) x_ref`` feedforward — see
``docs/codegen.md``).

The emosqp bake in ``runtime/codegen/emosqp/`` is generated for the MPC_LTI
config (``mpc_lti_base.toml``, n_vars=30), so no solver regeneration is
needed for the default. A DeltaU graph (n_vars=45) must be built against a
DeltaU bake via ``zig build -Dgraph=... -Dsolver_dir=...`` — see
``runtime/README.md``. Both gen scripts write the same
``runtime/graph_data.zig`` — the shipped graph is whichever ran last (re-run
``make zig-gen`` for the KF + LQR base graph).

Run: ``python3 scripts/gen_mpc.py``
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


def build_mpc_composed_graph(controller_config: str = "configs/controllers/mpc_lti_base.toml"):
    """Trace KF + MPC and compose the closed-loop step graph.

    Same estimator side as :func:`scripts.gen_base.build_base_graph` (the
    covariance P is a recurrent port). The controller is an MPC regulator
    (``MPC_LTI`` by default, or ``MPC_DeltaU`` via ``controller_config``),
    traced on its ``compute`` signature; compose maps the state input to the
    error state ``x_hat - x_ref`` and inserts the input clip on the controller
    output. For ``MPC_DeltaU`` the controller's ``u_prev`` input shares the
    estimator's previous-control recurrent port.

    Args:
        controller_config: Controller TOML config path (default:
            ``mpc_lti_base.toml``).

    Returns:
        A :class:`~shinro.codegen.compose.ComposedGraph` for the KF + MPC
        closed-loop step.
    """
    limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))
    return build_composed_graph(
        "configs/estimators/kalman_base.toml",
        controller_config,
        n_x=3,
        n_u=3,
        input_limits=limits,
    )


def main() -> None:
    """Generate ``runtime/graph_data.zig`` from the KF + MPC_LTI graph.

    Entry point for ``python scripts/gen_mpc.py``. Serializes the composed
    graph and prints a summary of the node count and input ports.
    """
    controller_config = "configs/controllers/mpc_lti_base.toml"
    composed = build_mpc_composed_graph(controller_config)
    lower_zig(
        composed,
        "runtime/graph_data.zig",
        provenance={
            "configs": {
                "configs/estimators/kalman_base.toml": _sha256("configs/estimators/kalman_base.toml"),
                controller_config: _sha256(controller_config),
            },
            "python_version": sys.version.split()[0],
            "numpy_version": version("numpy"),
        },
    )
    n = len(composed.graph.nodes)
    print(f"wrote runtime/graph_data.zig ({n} nodes, inputs={composed.inputs})")


if __name__ == "__main__":
    main()
