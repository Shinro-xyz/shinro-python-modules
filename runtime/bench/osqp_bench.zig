// runtime/bench/osqp_bench.zig — standalone OSQP benchmark.
//
// Measures solve_time across the MPC problem sizes using OSQP's own
// info.solve_time. Run with:  zig build bench --build-file runtime/build.zig
//
// Kept separate from the unit tests because std.debug.print output corrupts
// the `zig build test` --listen protocol.

const std = @import("std");
const c = @cImport({ @cInclude("osqp.h"); });

/// A box-constrained QP: min ½ uᵀ H u + qᵀ u, H = h_diag·I, -b ≤ u ≤ b.
/// Holds the CSC arrays AND the solver so the arrays outlive the solve.
fn BoxQp(comptime n: usize) type {
    return struct {
        solver: ?*c.OSQPSolver = null,
        P_p: [n + 1]c.OSQPInt = undefined,
        P_i: [n]c.OSQPInt = undefined,
        P_x: [n]c.OSQPFloat = undefined,
        A_p: [n + 1]c.OSQPInt = undefined,
        A_i: [n]c.OSQPInt = undefined,
        A_x: [n]c.OSQPFloat = undefined,
        q: [n]c.OSQPFloat = undefined,
        l: [n]c.OSQPFloat = undefined,
        u: [n]c.OSQPFloat = undefined,

        fn init(h_diag: f64, q_val: f64, bound: f64) BoxQp(n) {
            var self: BoxQp(n) = .{};
            for (0..n + 1) |j| self.P_p[j] = @intCast(j);
            for (0..n) |j| {
                self.P_i[j] = @intCast(j);
                self.P_x[j] = h_diag;
            }
            for (0..n + 1) |j| self.A_p[j] = @intCast(j);
            for (0..n) |j| {
                self.A_i[j] = @intCast(j);
                self.A_x[j] = 1.0;
            }
            for (0..n) |j| {
                self.q[j] = q_val;
                self.l[j] = -bound;
                self.u[j] = bound;
            }
            var P = c.OSQPCscMatrix{ .m = @intCast(n), .n = @intCast(n), .p = &self.P_p, .i = &self.P_i, .x = &self.P_x, .nzmax = @intCast(n), .nz = @intCast(n), .owned = 0 };
            var A = c.OSQPCscMatrix{ .m = @intCast(n), .n = @intCast(n), .p = &self.A_p, .i = &self.A_i, .x = &self.A_x, .nzmax = @intCast(n), .nz = @intCast(n), .owned = 0 };
            var settings: c.OSQPSettings = undefined;
            c.osqp_set_default_settings(&settings);
            settings.verbose = 0;
            const rc = c.osqp_setup(&self.solver, &P, &self.q, &A, &self.l, &self.u, @intCast(n), @intCast(n), &settings);
            std.debug.assert(rc == 0);
            return self;
        }

        fn deinit(self: *BoxQp(n)) void {
            _ = c.osqp_cleanup(self.solver.?);
        }
    };
}

fn bench(comptime n: usize, iters: usize) void {
    var qp = BoxQp(n).init(2.0, 0.0, 1.0);
    defer qp.deinit();

    _ = c.osqp_solve(qp.solver.?); // warm-up

    var total_solve_s: f64 = 0.0;
    for (0..iters) |_| {
        _ = c.osqp_solve(qp.solver.?);
        total_solve_s += qp.solver.?.info.*.solve_time;
    }
    const per_solve_us = total_solve_s / @as(f64, @floatFromInt(iters)) * 1_000_000.0;

    const info = qp.solver.?.info.*;
    const status_str = std.mem.sliceTo(&info.status, 0);
    std.debug.print("n={d}: {d:.2} us/solve, iter={d}, status={s}\n", .{ n, per_solve_us, info.iter, status_str });
}

pub fn main() void {
    bench(3, 1000); // minimal (N=1, n_u=3)
    bench(30, 1000); // base (N=10, n_u=3)
    bench(600, 10); // max (N=200, n_u=3)
}
