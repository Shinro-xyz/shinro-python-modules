// runtime/qp.zig — wraps the OSQP codegen static solver for the VM.
//
// The generated code (codegen/emosqp/) bakes the base MPC problem into a
// statically-allocated OSQPSolver global — no malloc, no libosqp dependency.
// For MPC only the linear cost q changes per tick, so the VM's .solve_qp op
// updates q, solves, and copies the full solution u into its output slot.
//
// The solver is problem-specific: it is regenerated for the MPC config via
// scripts/gen_emosqp_test.py. The .solve_qp op's output size must match the
// baked problem's n_vars.

const c = @cImport({ @cInclude("osqp.h"); });

// The statically-allocated solver from codegen/emosqp/workspace.c.
extern var solver: c.OSQPSolver;

/// Solve the baked MPC QP with the given linear cost `q`, writing the full
/// solution `u` into `out`. `q` and `out` must both be length n_vars.
pub fn solve_qp(q: []const f64, out: []f64) void {
    _ = c.osqp_update_data_vec(&solver, q.ptr, null, null);
    _ = c.osqp_solve(&solver);
    const x = solver.solution.*.x;
    for (0..out.len) |j| out[j] = x[j];
}
