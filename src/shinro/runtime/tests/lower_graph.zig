// Zig-native oracle for the comptime graph VM, driven by a frozen fixture.
//
// One copy of this file is compiled per fixture: build.zig wires each fixture's
// graph module in as `graph` and its expected vectors in as `vectors`. The VM
// (`lower.Vm(Ctx)`) is instantiated over that graph and run on the recorded
// inputs; the outputs and recurrent state must match `interpret()` within
// `tol`. No `.so`, no ctypes, no Python — a VM regression fails here in the
// plain `zig build test` step.
//
// The graphs and vectors are generated once by scripts/gen_lower_fixtures.py
// (see its docstring) and committed; this driver is handwritten and never
// regenerated.

const std = @import("std");
const lower = @import("lower");
const g = @import("graph");
const v = @import("vectors");

/// The fixture's VM context: a graph, no baked QP solver.
const Ctx = struct {
    pub const graph = g;
    pub const sm = struct {};
    pub const qp = struct {};
};

test "VM matches interpret() on the frozen fixture" {
    for (0..v.n_samples) |s| {
        var inputs: [v.n_in]f64 = undefined;
        @memcpy(&inputs, v.inputs[s * v.n_in ..][0..v.n_in]);

        var outputs = [_]f64{0.0} ** v.n_out;
        var state = [_]f64{0.0} ** v.n_state;
        lower.Vm(Ctx).step(&inputs, &outputs, &state);

        for (0..v.n_out) |j| {
            try std.testing.expectApproxEqAbs(v.outputs[s * v.n_out + j], outputs[j], v.tol);
        }
        for (0..v.n_state) |j| {
            try std.testing.expectApproxEqAbs(v.states[s * v.n_state + j], state[j], v.tol);
        }
    }
}
