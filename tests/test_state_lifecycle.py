"""State-lifecycle hardening: multi-tick threading, first-tick gate, determinism.

The base-dims live multitick tests in ``test_zig_lowering.py`` cover
KF+LQR / KF+PID / Luenberger / MPC closed loops at (3, 3). This module closes
the remaining state-lifecycle gaps:

1. **Multi-tick at every compile-scan case** — ``shinro_step`` vs
   ``interpret_step`` over 100 sequential ticks with state outputs fed back
   as next-tick inputs. Both sides see identical y/x_ref sequences (no plant
   in the loop), so any divergence is pure kernel-vs-interpreter *state
   handling*: PID anti-windup accumulation, KF P recursion, ``u_prev``
   threading, ``has_run`` gate transitions — at every (plant, controller,
   estimator, dim) combo, not just base.
2. **First-tick gate contract** — the PID D-term is gated on tick 0
   (``has_run=0``) and active from tick 1. Asserted against the live
   component, with the gate transition checked explicitly.
3. **Determinism** — same ``.so``, identical (inputs, state) buffers ->
   bit-identical outputs and state across repeated calls: no hidden runtime
   state, no runtime dispatch.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from test_zig_lowering import ALL_SCAN_CASES, NumpyBackend, _build_so, _plant_graph

from shinro.codegen import interpret
from shinro.codegen.compose import ComposedGraph
from shinro.codegen.oracle import output_split, pack_arrays, state_slices, step_so
from shinro.codegen.trace_node import trace_node

N_TICKS = 100
TOL = 1e-12


@pytest.fixture(scope="module")
def lifecycle_so(tmp_path_factory, request):
    """A compiled scan .so for one matrix case (same graphs as the scan)."""
    case_name, plant_name, plant_cfg, n_x, n_u, dt, controller, estimator = request.param
    d = tmp_path_factory.mktemp(f"lifecycle-{case_name}")
    cg = _plant_graph(d, case_name, plant_name, plant_cfg, n_x, n_u, dt, controller, estimator)
    lib, cg = _build_so(cg, d / "build", graph_path=d / "graph_data.zig")
    return case_name, lib, cg, n_x, n_u


def _state_port_map(cg):
    """Map each recurrent input port to (kernel slice key, ref state key).

    Kernel slices are keyed by full state-output name ("state_x_hat",
    "state_u_prev"); ``interpret_step`` returns state keyed by the stripped
    name ("x_hat", "u_prev"). Non-state ports (y, x_ref) map to None.
    """
    sl = state_slices(cg)
    mapping = {}
    for port in cg.inputs:
        if port in sl:
            mapping[port] = (port, port.removeprefix("state_"))
        elif f"state_{port}" in sl:
            mapping[port] = (f"state_{port}", port)
        else:
            mapping[port] = (None, None)
    return sl, mapping


@pytest.mark.parametrize(
    "lifecycle_so", ALL_SCAN_CASES, indirect=True, ids=[c[0] for c in ALL_SCAN_CASES]
)
def test_multitick_state_threading(lifecycle_so):
    """100 sequential ticks with fed-back state: .so == interpret_step."""
    case, lib, cg, n_x, n_u = lifecycle_so
    rng = np.random.default_rng(zlib.crc32(case.encode()))
    y_seq = rng.normal(0.0, 0.1, (N_TICKS, n_x))
    xr_seq = rng.normal(0.0, 0.05, (N_TICKS, n_x))

    n_out, n_state = output_split(cg)
    sl, port_map = _state_port_map(cg)
    in_shapes = {n.attrs["name"]: n.shape for n in cg.graph.nodes if n.op == "input"}

    # both sides start from the zero state (the correct first tick)
    kstate = np.zeros(n_state)
    rstate = {
        port.removeprefix("state_"): np.zeros(in_shapes[port])
        for port, (skey, _rkey) in port_map.items()
        if skey is not None
    }
    u_prev = np.zeros(n_u)

    max_err = 0.0
    for t in range(N_TICKS):
        # kernel side
        kports = {"y": y_seq[t], "x_ref": xr_seq[t], "u_prev": u_prev}
        for port, (skey, _rkey) in port_map.items():
            if skey is not None:
                a, b = sl[skey]
                kports[port] = kstate[a:b].reshape(in_shapes[port])
        out, kstate = step_so(lib, pack_arrays(cg, {k: np.asarray(v).ravel() for k, v in kports.items()}), n_out, n_state)

        # reference side (own state copy)
        rports = {"y": y_seq[t], "x_ref": xr_seq[t], "u_prev": u_prev.copy()}
        for port, (_skey, rkey) in port_map.items():
            if rkey is not None:
                rports[port] = rstate[rkey].copy()
        traced = interpret(cg.graph, rports)
        out_r = np.asarray(traced["u"]).ravel()
        rstate = {
            port.removeprefix("state_"): np.asarray(traced[port]).copy()
            for port in cg.state_outputs
        }

        max_err = max(max_err, float(np.max(np.abs(out - out_r))))
        for port in cg.state_outputs:
            a, b = sl[port]
            got = kstate[a:b]
            exp = np.asarray(traced[port]).ravel()
            max_err = max(max_err, float(np.max(np.abs(got - exp))))

        u_prev = out.copy()

    assert max_err < TOL, f"{case}: state threading diverged over {N_TICKS} ticks: {max_err:.3e}"


def test_pid_first_tick_gate(tmp_path):
    """The PID D-term is gated on tick 0 (has_run=0) and active from tick 1.

    Tick-0 and tick-1 kernel outputs must match the live component with
    explicitly-threaded state, and has_run must transition 0 -> 1. A live
    control run with has_run forced to 1 proves the gate changes the output
    (otherwise the gating would be untestable dead logic).
    """
    from shinro.controllers.pid import PIDController as PID

    pid = PID(
        kp=np.array([2.0, 2.0, 2.0]),
        ki=np.array([0.5, 0.5, 0.5]),
        kd=np.array([0.5, 0.5, 0.5]),
        dt=0.02,
        output_limits=(np.array([-0.3, -0.3, -0.6]), np.array([0.3, 0.3, 0.6])),
        backend=NumpyBackend(),
    )
    ng = trace_node(
        pid,
        input_shapes={"current_state": (3,), "target_state": (3,)},
        state_shapes={"_integral": (3,), "_prev_error": (3,), "_has_run": (3,)},
    )
    g = ng.graph
    in_names = [n.attrs["name"] for n in g.nodes if n.op == "input"]
    out_names = [n.attrs["name"] for n in g.nodes if n.op == "output"]
    state_outs = [n for n in out_names if n != "out"]
    cg = ComposedGraph(graph=g, inputs=in_names, outputs=["out"], state_outputs=state_outs)

    d = tmp_path / "pid-gate"
    d.mkdir()
    lib, cg = _build_so(cg, d / "build", graph_path=d / "graph_data.zig")
    n_out, n_state = output_split(cg)
    sl = state_slices(cg)

    rng = np.random.default_rng(97)
    cur = rng.normal(0, 0.2, 3)
    tgt = rng.normal(0, 0.1, 3)

    def live_state():
        return {"_integral": np.zeros(3), "_prev_error": np.zeros(3), "_has_run": np.zeros(3)}

    # gate-matters control (live only): has_run=0 vs has_run=1 on the same
    # inputs must differ, else the gate is dead logic.
    pid._integral = np.zeros(3)
    pid._prev_error = np.zeros(3)
    pid._has_run = np.zeros(3)
    u_gated = pid.compute(cur.copy(), tgt.copy())
    pid._integral = np.zeros(3)
    pid._prev_error = np.zeros(3)
    pid._has_run = np.ones(3)
    u_ungated = pid.compute(cur.copy(), tgt.copy())
    assert not np.allclose(u_gated, u_ungated), "has_run gate has no effect on tick-0 output"

    # tick 0: zero state everywhere — kernel must match the gated live output
    kstate = np.zeros(n_state)
    ports0 = {"current_state": cur, "target_state": tgt}
    for port in cg.state_outputs:
        ports0[port] = np.zeros(sl[port][1] - sl[port][0])
    inp = pack_arrays(cg, {k: np.asarray(v).ravel() for k, v in ports0.items()})
    out0, kstate = step_so(lib, inp, n_out, n_state)
    pid._integral = np.zeros(3)
    pid._prev_error = np.zeros(3)
    pid._has_run = np.zeros(3)
    u0_live = pid.compute(cur.copy(), tgt.copy())
    assert np.max(np.abs(out0 - u0_live)) < TOL, "tick-0 kernel != live gated PID"
    has_run_after0 = kstate[sl["state_has_run"][0] : sl["state_has_run"][1]]
    assert np.all(has_run_after0 == 1.0), "has_run did not transition 0 -> 1 on tick 0"

    # tick 1: thread both sides; D-term must now be active and still agree
    cur1 = rng.normal(0, 0.2, 3)
    tgt1 = rng.normal(0, 0.1, 3)
    ports1 = {"current_state": cur1, "target_state": tgt1}
    for port in cg.state_outputs:
        a, b = sl[port]
        ports1[port] = kstate[a:b]
    inp1 = pack_arrays(cg, {k: np.asarray(v).ravel() for k, v in ports1.items()})
    out1, _ = step_so(lib, inp1, n_out, n_state)
    u1_live = pid.compute(cur1.copy(), tgt1.copy())  # pid state evolved live above
    assert np.max(np.abs(out1 - u1_live)) < TOL, "tick-1 kernel != live PID"


def test_deterministic_repeat_calls(tmp_path):
    """Same .so, identical (inputs, state) -> bit-identical results.

    Pins the "fixed at compile time" guarantee: no hidden runtime state, no
    runtime dispatch, no allocation jitter — repeated calls are exact.
    """
    case = ALL_SCAN_CASES[0]
    case_name, plant_name, plant_cfg, n_x, n_u, dt, controller, estimator = case
    d = tmp_path / "determinism"
    d.mkdir()
    cg = _plant_graph(d, case_name, plant_name, plant_cfg, n_x, n_u, dt, controller, estimator)
    lib, cg = _build_so(cg, d / "build", graph_path=d / "graph_data.zig")
    n_out, n_state = output_split(cg)
    sl = state_slices(cg)

    rng = np.random.default_rng(5)
    ports = {"y": rng.normal(0, 0.1, n_x), "x_ref": rng.normal(0, 0.05, n_x), "u_prev": np.zeros(n_u)}
    for port in cg.state_outputs:
        ports[port] = rng.normal(0, 0.05, sl[port][1] - sl[port][0])
    inp = pack_arrays(cg, {k: np.asarray(v).ravel() for k, v in ports.items()})

    results = []
    for _ in range(3):
        out, state = step_so(lib, inp.copy(), n_out, n_state)
        results.append((out.copy(), state.copy()))

    for out, state in results[1:]:
        assert np.array_equal(out, results[0][0]), "outputs not bit-identical across calls"
        assert np.array_equal(state, results[0][1]), "state not bit-identical across calls"
