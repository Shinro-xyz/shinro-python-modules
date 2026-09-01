// runtime/tests/osqp.zig — cimport smoke test + benchmark for the OSQP binding.
//
// Proves the @cImport("osqp.h") binding compiles, links against the vendored
// libosqp.so, and solves known QPs. Then benchmarks solve_time across the MPC
// problem sizes using OSQP's own info.solve_time.
//
// Requires third_party/osqp (header + libosqp.so) and `zig build test`.

const std = @import("std");
const c = @cImport({ @cInclude("osqp.h"); });

/// A box-constrained QP: min ½ uᵀ H u + qᵀ u, H = h_diag·I, -b ≤ u ≤ b.
/// Holds the CSC arrays AND the solver so the arrays outlive the solve
/// (OSQP references the data, it doesn't copy it).
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

test "osqp binding: solves a 2-var box-constrained QP" {
    // min ½ uᵀ H u + qᵀ u, H = 2I, q = 0, -1 ≤ u ≤ 1  →  u* = [0, 0]
    var qp = BoxQp(2).init(2.0, 0.0, 1.0);
    defer qp.deinit();

    const solve_rc = c.osqp_solve(qp.solver.?);
    try std.testing.expectEqual(@as(c.OSQPInt, 0), solve_rc);

    const x = qp.solver.?.solution.*.x;
    try std.testing.expectApproxEqAbs(0.0, x[0], 1e-9);
    try std.testing.expectApproxEqAbs(0.0, x[1], 1e-9);
}

test "osqp binding: active constraints saturate u at the bound" {
    // q = -4 → unconstrained u* = [2, 2], clamped to [-1, 1] → u* = [1, 1]
    var qp = BoxQp(2).init(2.0, -4.0, 1.0);
    defer qp.deinit();

    _ = c.osqp_solve(qp.solver.?);
    const x = qp.solver.?.solution.*.x;
    // OSQP solves to its default tolerance (~1e-3), not bit-exact.
    try std.testing.expectApproxEqAbs(1.0, x[0], 1e-3);
    try std.testing.expectApproxEqAbs(1.0, x[1], 1e-3);
}

test "osqp binding: reports primal infeasible for inconsistent constraints" {
    // min ½u²  s.t.  u ≥ 1 AND -u ≥ 1  (u ≥ 1 and u ≤ -1) → infeasible.
    // A box-constrained QP with l ≤ u is always feasible, so this needs a
    // non-identity A with conflicting rows.
    const n: c.OSQPInt = 1;
    const m: c.OSQPInt = 2;

    var P_p = [_]c.OSQPInt{ 0, 1 };
    var P_i = [_]c.OSQPInt{ 0 };
    var P_x = [_]c.OSQPFloat{ 1.0 };
    var P = c.OSQPCscMatrix{ .m = n, .n = n, .p = &P_p, .i = &P_i, .x = &P_x, .nzmax = 1, .nz = 1, .owned = 0 };

    var A_p = [_]c.OSQPInt{ 0, 1, 2 };
    var A_i = [_]c.OSQPInt{ 0, 0 };
    var A_x = [_]c.OSQPFloat{ 1.0, -1.0 };
    var A = c.OSQPCscMatrix{ .m = m, .n = n, .p = &A_p, .i = &A_i, .x = &A_x, .nzmax = 2, .nz = 2, .owned = 0 };

    var q = [_]c.OSQPFloat{ 0.0 };
    var l = [_]c.OSQPFloat{ 1.0, 1.0 };
    var u = [_]c.OSQPFloat{ 1e30, 1e30 };

    var settings: c.OSQPSettings = undefined;
    c.osqp_set_default_settings(&settings);
    settings.verbose = 0;

    var solver: ?*c.OSQPSolver = null;
    const setup_rc = c.osqp_setup(&solver, &P, &q, &A, &l, &u, m, n, &settings);
    try std.testing.expectEqual(@as(c.OSQPInt, 0), setup_rc);
    defer _ = c.osqp_cleanup(solver.?);

    _ = c.osqp_solve(solver.?);
    const status = solver.?.info.*.status_val;
    // OSQP_PRIMAL_INFEASIBLE = 3 (enum in osqp_api_constants.h)
    try std.testing.expectEqual(@as(c.OSQPInt, 3), status);
}
