"""Generate ``runtime/graph_data.zig`` for the KF + MPC_LTI closed-loop step.

Alternative deployment target to ``scripts/gen_base.py`` (KF + LQR): swaps the
controller for MPC_LTI from ``mpc_lti_base.toml``. Compose feeds the tracking
error ``x0 = x_hat - x_ref`` to the regulator, so regulating it to zero tracks
``x_ref`` (exact for the base plant, A = I; general A would need an
``(A - I) x_ref`` feedforward — see ``docs/codegen.md``).

The emosqp bake in ``runtime/codegen/emosqp/`` is generated for this same
config (``mpc_lti_base.toml``, n_vars=30), so no solver regeneration is
needed. Both gen scripts write the same ``runtime/graph_data.zig`` — the
shipped graph is whichever ran last (re-run ``make zig-gen`` for the KF + LQR
base graph).

Run: ``python3 scripts/gen_mpc.py``
"""

from __future__ import annotations

import numpy as np

from shinro.codegen import trace_node
from shinro.codegen.compose import compose
from shinro.codegen.lower_zig import lower_zig
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.utils.array_backend import NumpyBackend


def build_mpc_composed_graph():
    """Trace KF + MPC_LTI and compose the closed-loop step graph.

    Same estimator side as :func:`scripts.gen_base.build_base_graph` (the
    covariance P is a recurrent port). The controller is MPC_LTI, traced on
    its ``compute(x0)`` signature; compose maps ``x0`` to the error state
    ``x_hat - x_ref`` and inserts the input clip on the controller output.

    Returns:
        A :class:`~shinro.codegen.compose.ComposedGraph` for the KF + MPC_LTI
        closed-loop step.
    """
    kf = EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())
    mpc = ControllerFactory("configs/controllers/mpc_lti_base.toml").create(backend=NumpyBackend())

    # Seed for live numpy use only — in the traced graph both x_hat and P are
    # pre-injected tracers (recurrent ports); the host supplies their initial
    # values at tick 0 (P0 = 0.1*I matches KalmanFilter.reset()).
    kf.P = np.eye(3) * 0.1
    kf.x_hat = np.zeros((3, 1))
    kf_graph = trace_node(
        kf,
        input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
        state_shapes={"x_hat": (3, 1), "P": (3, 3)},
    )
    mpc_graph = trace_node(mpc, input_shapes={"x0": (3,)})
    limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))
    return compose(
        kf_graph,
        mpc_graph,
        plant_dims={"n_x": 3, "n_u": 3},
        input_limits=limits,
    )


def main() -> None:
    """Generate ``runtime/graph_data.zig`` from the KF + MPC_LTI graph.

    Entry point for ``python scripts/gen_mpc.py``. Serializes the composed
    graph and prints a summary of the node count and input ports.
    """
    composed = build_mpc_composed_graph()
    lower_zig(composed, "runtime/graph_data.zig")
    n = len(composed.graph.nodes)
    print(f"wrote runtime/graph_data.zig ({n} nodes, inputs={composed.inputs})")


if __name__ == "__main__":
    main()
