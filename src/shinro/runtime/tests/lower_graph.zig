// Zig-native C-ABI oracle for the comptime graph VM, driven by a frozen fixture.
//
// One copy of this file is compiled per fixture: build.zig wires each fixture's
// `entry` module (rooted at entry.zig, with that fixture's graph) and its
// expected vectors in as `vectors`. The test calls the *exported* C-ABI entry
// `entry.shinro_step` — the same symbol the host dlopen-s — on the recorded
// inputs and checks the outputs and recurrent state against `interpret()`
// within `tol`. No `.so`, no ctypes, no Python: a VM/ABI regression fails here
// in the plain `zig build test` step.
//
// The QP fixture's `entry` module carries the baked OSQP solver too, so this
// same driver covers the `.solve_qp` op natively.
//
// The graphs and vectors are generated once by scripts/gen_lower_fixtures.py
// (see its docstring) and committed; this driver is handwritten and never
// regenerated.

const std = @import("std");
const entry = @import("entry");
const v = @import("vectors");

test "shinro_step matches interpret() on the frozen fixture" {
    for (0..v.n_samples) |s| {
        var inputs: [v.n_in]f64 = undefined;
        @memcpy(&inputs, v.inputs[s * v.n_in ..][0..v.n_in]);

        var outputs = [_]f64{0.0} ** v.n_out;
        var state = [_]f64{0.0} ** v.n_state;
        entry.shinro_step(&inputs, &outputs, &state);

        for (0..v.n_out) |j| {
            try std.testing.expectApproxEqAbs(v.outputs[s * v.n_out + j], outputs[j], v.tol);
        }
        for (0..v.n_state) |j| {
            try std.testing.expectApproxEqAbs(v.states[s * v.n_state + j], state[j], v.tol);
        }
    }
}
