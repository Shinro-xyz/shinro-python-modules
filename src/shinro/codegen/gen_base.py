"""Generate ``runtime/graph_data.zig`` for the base_tracking closed-loop step.

Entry point: ``python -m shinro.codegen.gen_base`` (wired to ``make zig-gen``).

Builds the base_tracking composed graph (KalmanFilter + LQR on the 3-DOF
holonomic base, with the scenario's input clip), serializes it via
:func:`shinro.codegen.lower_zig.lower_zig`, and writes the Zig data table that
``runtime/lower.zig`` compiles against.

This is the MVP bring-up target: one fixed graph, proven bit-exact against the
interpreter. Swap the graph here to lower any other ComposedGraph.
"""

from __future__ import annotations

import numpy as np

from shinro.codegen import lower_zig, trace_node
from shinro.codegen.compose import compose
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.utils.array_backend import NumpyBackend


def build_base_graph():
    """Trace KF + LQR and compose the base_tracking step graph."""
    kf = EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())
    lqr = ControllerFactory("configs/controllers/lqr_base.toml").create(backend=NumpyBackend())

    kf.P = np.eye(3) * 0.1
    kf.x_hat = np.zeros((3, 1))
    kf_graph = trace_node(
        kf,
        input_shapes={"measurement": (3, 1), "control_input": (3, 1)},
        state_shapes={"x_hat": (3, 1)},
    )
    lqr_graph = trace_node(
        lqr,
        input_shapes={"current_state": (3,), "target_state": (3,)},
    )
    limits = (np.array([-0.5, -0.5, -1.0]), np.array([0.5, 0.5, 1.0]))
    return compose(
        kf_graph,
        lqr_graph,
        plant_dims={"n_x": 3, "n_u": 3},
        input_limits=limits,
    )


def main() -> None:
    composed = build_base_graph()
    lower_zig(composed, "runtime/graph_data.zig")
    n = len(composed.graph.nodes)
    print(f"wrote runtime/graph_data.zig ({n} nodes, inputs={composed.inputs})")


if __name__ == "__main__":
    main()
