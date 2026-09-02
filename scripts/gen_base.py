"""Generate ``runtime/graph_data.zig`` for the base_tracking closed-loop step.

Entry point: ``python scripts/gen_base.py`` (wired to ``make zig-gen``).

Lives in ``scripts/`` (not ``src/shinro/``) because it's LeKiwi-specific —
one fixed graph for a single scenario. The framework-side pieces it uses
(tracing, compose, lowering) are generic and stay in ``shinro.codegen``.

Builds the base_tracking composed graph (KalmanFilter + LQR on the 3-DOF
holonomic base, with the scenario's input clip), serializes it via
:func:`shinro.codegen.lower_zig.lower_zig`, and writes the Zig data table that
``runtime/lower.zig`` compiles against.

This is the MVP bring-up target: one fixed graph, proven bit-exact against the
interpreter. Swap the graph here to build any other ComposedGraph.
"""

from __future__ import annotations

import numpy as np

from shinro.codegen import trace_node
from shinro.codegen.compose import compose
from shinro.codegen.lower_zig import lower_zig
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.utils.array_backend import NumpyBackend


def build_base_graph():
    """Trace KF + LQR and compose the base_tracking step graph.

    Instantiates the Kalman filter and LQR from their TOML configs with a
    :class:`NumpyBackend`, seeds the KF's covariance and state, traces both
    components, and composes them into the closed-loop graph for the 3-DOF
    holonomic base (with the scenario's input clip). The KF's covariance P is
    pre-injected as a recurrent state port, so the deployed graph runs the
    full live predict-update Riccati recursion (not a frozen gain) — the
    host seeds P0 = 0.1*I at tick 0 and feeds state_P back each tick. This
    is the MVP bring-up target graph that ``runtime/graph_data.zig`` is
    generated from.

    Returns:
        A :class:`~shinro.codegen.compose.ComposedGraph` for the base_tracking
        closed-loop step.
    """
    kf = EstimatorFactory("configs/estimators/kalman_base.toml").create(backend=NumpyBackend())
    lqr = ControllerFactory("configs/controllers/lqr_base.toml").create(backend=NumpyBackend())

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
    """Generate ``runtime/graph_data.zig`` from the base_tracking graph.

    Entry point for ``python scripts/gen_base.py`` (wired to
    ``make zig-gen``). Serializes the composed graph and prints a summary of
    the node count and input ports.
    """
    composed = build_base_graph()
    lower_zig(composed, "runtime/graph_data.zig")
    n = len(composed.graph.nodes)
    print(f"wrote runtime/graph_data.zig ({n} nodes, inputs={composed.inputs})")


if __name__ == "__main__":
    main()
