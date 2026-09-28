// runtime/entry.zig — the deployment entry point.
//
// This is the only file that binds the generated graph to the VM and the only
// one that declares the `shinro_step` C-ABI symbol. `lower.zig` holds the
// graph-agnostic VM (`Vm(Ctx)`); here we assemble the context from the graph
// selected by `-Dgraph` (plus the bake selected by `-Dsolver_dir`) and expose
// its `step` under the stable C name.
//
// Why the split: the graph binding and the ABI surface change with deployment
// (which graph, which bake), while the VM must not. Keeping the binding here
// lets the VM be compiled against any graph — tests instantiate `Vm(Ctx)`
// directly with their own fixture — and keeps every build option
// (`-Dgraph`, `-Dsolver_dir`) touching exactly one small file.

const lower = @import("lower.zig");
const g = @import("graph_data");
// OSQP is deliberately conditional: the generated graph declares whether it
// contains a .solve_qp node. Non-QP graphs (LQR, PID, ...) build without the
// OSQP C sources, headers, or solver_meta module at all.
const solver_meta = if (g.has_solve_qp) @import("solver_meta") else struct {};
const qp_mod = if (g.has_solve_qp) @import("qp.zig") else struct {};

/// The VM context for the deployed graph: the graph plus the (optional) bake.
pub const Ctx = struct {
    pub const graph = g;
    pub const sm = solver_meta;
    pub const qp = qp_mod;
};

/// One tick of the deployed closed-loop step (see lower.zig's `Vm(Ctx).step`
/// for the buffer contract). `pub` so the Zig-native fixture tests can call the
/// real C-ABI entry directly; the exported symbol name is unchanged.
pub export fn shinro_step(inputs: [*]const f64, outputs: [*]f64, state_out: [*]f64) void {
    lower.Vm(Ctx).step(inputs, outputs, state_out);
}
