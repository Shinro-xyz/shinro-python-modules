// Zig-native C-ABI tests for the comptime graph VM (runtime/lower.zig).
//
// The VM is graph-agnostic — `lower.Vm(Ctx).step(...)` runs whichever graph the
// context carries. This test binds the small committed fixture graph
// (tests/lower_fixture_graph.zig, wired by build.zig as this module's
// `graph_data`) and drives `step` directly, in-process. Unlike the Python
// ctypes oracle in tests/test_zig_lowering.py — which builds a `.so` and calls
// `shinro_step` from Python — this needs no shared-library build, no ctypes,
// and no Python, so a VM regression surfaces in the plain `zig build test`
// step alone.
//
// The production binding (the deployed graph + the `shinro_step` export) lives
// in entry.zig; this file is the test-side equivalent.
//
// Fixture semantics: x = [1, 2] -> named output [3, 5]; r -> tanh(r) on the
// recurrent state output. See the fixture header for the exact node table.

const std = @import("std");
const lower = @import("lower");
const g = @import("graph_data");

/// The fixture's VM context: a graph, no baked QP solver.
const Ctx = struct {
    pub const graph = g;
    pub const sm = struct {};
    pub const qp = struct {};
};

test "VM: input packing, const matmul, add, clip, named + state outputs" {
    // The host packs inputs in cg.inputs order, flat and contiguous. The
    // fixture's two input ports live at aux offsets 0 (x) and 2 (r), so the
    // packed buffer is [x0, x1, r0, r1].
    var inputs = [_]f64{ 1, 2, 0.5, -0.5 };
    var outputs = [_]f64{ 0, 0 };
    var state = [_]f64{ 0, 0 };

    lower.Vm(Ctx).step(&inputs, &outputs, &state);

    // y = A@x = [2, 6]; z = y + b = [3, 8]; clip into [-10, 5] -> [3, 5].
    try std.testing.expectEqualSlices(f64, &[_]f64{ 3, 5 }, &outputs);
    // The second `out` node (aux >= n_outputs) routes to state_out as tanh(r).
    // Both sides call std.math.tanh on the same input, so this is exact.
    try std.testing.expectEqual(std.math.tanh(@as(f64, 0.5)), state[0]);
    try std.testing.expectEqual(std.math.tanh(@as(f64, -0.5)), state[1]);
}

test "VM: recurrent state output feeds back as the next tick's input" {
    var inputs = [_]f64{ 1, 2, 0.5, -0.5 };
    var outputs = [_]f64{ 0, 0 };
    var state = [_]f64{ 0, 0 };

    lower.Vm(Ctx).step(&inputs, &outputs, &state);

    // Tick 2: feed the state output back into the state input slot (aux 2/3),
    // exactly as the host's control loop does between ticks. The named output
    // is independent of r, so it must be unchanged.
    inputs[2] = state[0];
    inputs[3] = state[1];
    var outputs2 = [_]f64{ 0, 0 };
    var state2 = [_]f64{ 0, 0 };
    lower.Vm(Ctx).step(&inputs, &outputs2, &state2);

    try std.testing.expectEqualSlices(f64, &[_]f64{ 3, 5 }, &outputs2);
    try std.testing.expectEqual(std.math.tanh(std.math.tanh(@as(f64, 0.5))), state2[0]);
    try std.testing.expectEqual(std.math.tanh(std.math.tanh(@as(f64, -0.5))), state2[1]);
}
