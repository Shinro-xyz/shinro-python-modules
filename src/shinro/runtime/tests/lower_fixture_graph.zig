// Fixture `graph_data.zig` for the Zig-native lower.zig test (tests/lower.zig).
//
// The graph is injected into the VM via `Vm(Ctx)`. The deployed binding
// (entry.zig) points `Ctx.graph` at runtime/graph_data.zig; the Zig-native VM
// test (tests/lower.zig) points it at this tiny, hand-authored graph so the
// `step` path can be exercised natively — `zig build test`, no `.so`, no ctypes,
// no Python. See build.zig's VM-test module.
//
// Schema contract: the `Op` enum and `Node` struct MUST mirror
// src/shinro/runtime/graph_data.zig exactly. lower.zig `switch`es on `node.op`
// with no `else`, so the enum must stay exhaustive — adding a VM op fails this
// fixture's compile, which is deliberate (it forces the fixture to track the op
// surface). Only the constants the fixture's own ops read need to exist: the
// per-op tables (`gemm_alpha`, `elu_alpha`, `layernorm_eps`, ...) are only
// referenced from the switch arms of ops present in the node table, and those
// arms are comptime-skipped otherwise.
//
// The fixture computes, for a 2-vector x, a 2-vector recurrent input r, and
// A = [[2,0],[0,3]], b = [1,2]:
//   y  = A @ x          (matmul; A read in place from const_blob)
//   z  = y + b          (add)
//   o  = clip(z, -10, 5)  → the named output port (output 0)
//   r' = tanh(r)        → the recurrent state output port
// so with x = [1,2] the output is [3,5] and the state is tanh([0.5,-0.5]).

pub const Op = enum {
    cst,
    cst_f32,
    inp,
    out,
    matmul,
    add,
    sub,
    mul,
    div,
    ne,
    neg,
    transpose,
    inv,
    cholesky,
    reshape,
    clip,
    where_op,
    any,
    copy,
    tanh,
    relu,
    exp,
    argmax,
    one_hot,
    slice,
    sin,
    cos,
    stack,
    solve_qp,
    abs,
    sign,
    pow,
    lt,
    min,
    gemm,
    sigmoid,
    softmax,
    gelu,
    elu,
    layernorm,
    lstm,
    gru,
    rnn,
    concat,
    gather,
    sqrt,
    log,
    mod,
    leaky_relu,
};

pub const Node = struct {
    op: Op,
    inputs: []const usize,
    rows: usize,
    cols: usize,
    aux: usize,
    vec: bool,
};

pub const buf_len = 16;
pub const has_solve_qp = false;
pub const n_outputs = 1;

// One entry per node. Const nodes own no workspace slot, so their offset is
// unused (0); every other node gets a distinct two-f64 slot.
pub const offsets = [_]usize{
    0, // 0 inp x
    2, // 1 inp r
    0, // 2 cst A (no slot)
    4, // 3 matmul
    0, // 4 cst b (no slot)
    6, // 5 add
    8, // 6 clip
    10, // 7 out  -> named output
    12, // 8 tanh
    14, // 9 out  -> state output
};

pub const nodes = [_]Node{
    .{ .op = .inp, .inputs = &.{}, .rows = 2, .cols = 1, .aux = 0, .vec = true },
    .{ .op = .inp, .inputs = &.{}, .rows = 2, .cols = 1, .aux = 2, .vec = true },
    .{ .op = .cst, .inputs = &.{}, .rows = 2, .cols = 2, .aux = 0, .vec = false },
    .{ .op = .matmul, .inputs = &.{ 2, 0 }, .rows = 2, .cols = 1, .aux = 0, .vec = false },
    .{ .op = .cst, .inputs = &.{}, .rows = 2, .cols = 1, .aux = 4, .vec = false },
    .{ .op = .add, .inputs = &.{ 3, 4 }, .rows = 2, .cols = 1, .aux = 0, .vec = false },
    .{ .op = .clip, .inputs = &.{5}, .rows = 2, .cols = 1, .aux = 0, .vec = false },
    .{ .op = .out, .inputs = &.{6}, .rows = 2, .cols = 1, .aux = 0, .vec = false },
    .{ .op = .tanh, .inputs = &.{1}, .rows = 2, .cols = 1, .aux = 0, .vec = false },
    .{ .op = .out, .inputs = &.{8}, .rows = 2, .cols = 1, .aux = 1, .vec = false },
};

// A (2x2, offset 0) then b (2x1, offset 4): A = [[2,0],[0,3]], b = [1,2].
pub const const_blob = [_]f64{ 2, 0, 0, 3, 1, 2 };
pub const const_blob_f32 = [_]f32{};

// clip's bounds are indexed per element by node.aux: clamp each element of
// z = [3,8] into [-10, 5], so only the second element moves (8 -> 5).
pub const clip_lo = [_]f64{ -10, -10 };
pub const clip_hi = [_]f64{ 5, 5 };

pub const output_offsets = [_]usize{0};
pub const state_offsets = [_]usize{0};
