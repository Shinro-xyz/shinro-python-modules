// Zig unit tests for runtime/linalg.zig.
//
// Coverage: matmul / matvec / vecmat against hand-computed numpy-verified
// cases, and inv round-trips (inv(A) @ A == I) matching the tolerance the
// linalg.zig header promises (atol 1e-12).
//
// The `linalg` module is exposed by runtime/build.zig via the test module's
// `imports`; we don't @import("../linalg.zig") because Zig forbids imports
// outside a module's root path.

const std = @import("std");
const la = @import("linalg");

test "matmul 2x2" {
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 5, 6, 7, 8 };
    const r = la.matmul(2, 2, 2, &a, &b);
    try std.testing.expectEqual([_]f64{ 19, 22, 43, 50 }, r);
}

test "matmul 3x3" {
    const a = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 10 };
    const b = [_]f64{ 2, 0, 1, 1, 3, 0, 0, 1, 4 };
    const r = la.matmul(3, 3, 3, &a, &b);
    try std.testing.expectEqual([_]f64{ 4, 9, 13, 13, 21, 28, 22, 34, 47 }, r);
}

test "matvec: (m,k) @ (k,) -> (m,)" {
    const m = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const v = [_]f64{ 2, 1, 3 };
    const r = la.matvec(2, 3, &m, &v);
    try std.testing.expectEqual([_]f64{ 13, 31 }, r);
}

test "vecmat: (k,) @ (k,n) -> (n,)" {
    const v = [_]f64{ 1, 2, 3 };
    const m = [_]f64{ 1, 2, 3, 4, 2, 3, 1, 0, 1, 0, 2, 1 };
    const r = la.vecmat(3, 4, &v, &m);
    try std.testing.expectEqual([_]f64{ 8, 8, 11, 7 }, r);
}

test "inv round-trips 2x2: inv(A) @ A == I" {
    const a = [_]f64{ 4, 7, 2, 6 };
    const ai = la.inv(2, &a);
    const r = la.matmul(2, 2, 2, &ai, &a);
    for (0..2) |i| {
        for (0..2) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, r[i * 2 + j], 1e-12);
        }
    }
}

test "inv round-trips 3x3: inv(A) @ A == I" {
    const a = [_]f64{ 2, 1, 0, 1, 3, 1, 0, 1, 2 };
    const ai = la.inv(3, &a);
    const r = la.matmul(3, 3, 3, &ai, &a);
    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, r[i * 3 + j], 1e-12);
        }
    }
}

test "relu clips negatives to 0" {
    const a = [_]f64{ -1, 0, 0.5, 2 };
    const r = la.relu(4, &a);
    try std.testing.expectEqual([_]f64{ 0, 0, 0.5, 2 }, r);
}

test "exp matches e^x on 0, 1, -1" {
    const a = [_]f64{ 0, 1, -1 };
    const r = la.elementwise_exponential(3, &a);
    try std.testing.expectApproxEqAbs(1.0, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(2.718281828459045, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.36787944117144233, r[2], 1e-12);
}

test "tanh saturates at +-1" {
    const a = [_]f64{ 0, 1, -1 };
    const r = la.tanh(3, &a);
    try std.testing.expectApproxEqAbs(0.0, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.7615941559557649, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(-0.7615941559557649, r[2], 1e-12);
}

test "sigmoid matches the logistic function" {
    const a = [_]f64{ 0, 1, -1 };
    const r = la.sigmoid(3, &a);
    try std.testing.expectApproxEqAbs(0.5, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.7310585786300049, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.2689414213699951, r[2], 1e-12);
}

test "softmax_rows normalizes each row over the last axis" {
    // Row 0: [0, ln3] -> [1/4, 3/4]; row 1: [0, 0] -> [1/2, 1/2].
    const a = [_]f64{ 0, @log(3.0), 0, 0 };
    const r = la.softmax_rows(2, 2, &a);
    try std.testing.expectApproxEqAbs(0.25, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.75, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.5, r[2], 1e-12);
    try std.testing.expectApproxEqAbs(0.5, r[3], 1e-12);
}

test "softmax_rows is stable on large values" {
    // [1000, 999] shifts to [0, -1]; the ratio is that of [1, e^-1].
    const a = [_]f64{ 1000, 999 };
    const r = la.softmax_rows(1, 2, &a);
    try std.testing.expectApproxEqAbs(0.7310585786300049, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.2689414213699951, r[1], 1e-12);
}

test "gelu tanh approximation matches known values" {
    // 0.5x(1+tanh(sqrt(2/pi)(x + 0.044715 x^3))).
    const a = [_]f64{ 0, 1, -1, 4, 8 };
    const r = la.gelu(5, &a);
    try std.testing.expectApproxEqAbs(0.0, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.8411919906082768, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(-0.1588080093917233, r[2], 1e-12);
    try std.testing.expectApproxEqAbs(3.9999297540518075, r[3], 1e-12);
    // Large positive input saturates to identity.
    try std.testing.expectApproxEqAbs(8.0, r[4], 1e-12);
}

test "elu matches the definition" {
    const a = [_]f64{ -2, -1, 0, 1, 2 };
    const r = la.elu(5, 1.0, &a);
    try std.testing.expectApproxEqAbs(-0.8646647167633873, r[0], 1e-12); // e^-2 - 1
    try std.testing.expectApproxEqAbs(-0.6321205588285577, r[1], 1e-12); // e^-1 - 1
    try std.testing.expectApproxEqAbs(0.0, r[2], 1e-12);
    try std.testing.expectApproxEqAbs(1.0, r[3], 1e-12);
    try std.testing.expectApproxEqAbs(2.0, r[4], 1e-12);
}

test "elu honors alpha on the negative branch" {
    const a = [_]f64{ -1, 1 };
    const r = la.elu(2, 0.5, &a);
    try std.testing.expectApproxEqAbs(0.5 * (std.math.exp(-1.0) - 1.0), r[0], 1e-12);
    try std.testing.expectApproxEqAbs(1.0, r[1], 1e-12);
}

test "argmax returns index of max" {
    const a = [_]f64{ 0.2, 0.7, 0.1 };
    try std.testing.expectEqual(@as(usize, 1), la.argmax(3, &a));
}

test "argmax breaks ties to first occurrence" {
    const a = [_]f64{ 1.5, 1.5, 0.3 };
    try std.testing.expectEqual(@as(usize, 0), la.argmax(3, &a));
}

test "onehot places 1.0 at idx" {
    const r = la.onehot(5, 2);
    try std.testing.expectEqual([_]f64{ 0, 0, 1, 0, 0 }, r);
}

test "sin_vec matches sin of known angles" {
    const a = [_]f64{ 0, 1.5707963267948966, 3.141592653589793 };
    const r = la.sin_vec(3, &a);
    try std.testing.expectApproxEqAbs(0.0, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(1.0, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.0, r[2], 1e-12);
}

test "cos_vec matches cos of known angles" {
    const a = [_]f64{ 0, 1.5707963267948966, 3.141592653589793 };
    const r = la.cos_vec(3, &a);
    try std.testing.expectApproxEqAbs(1.0, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.0, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(-1.0, r[2], 1e-12);
}

// gemm: Y = alpha*(A@B') + beta*C. A is (m,k); B is (k,n), or (n,k) with
// transB (torch nn.Linear layout, read by striding). C broadcasts per c_len:
// 1 = scalar, n = (n,) row, otherwise full (m,n). Values below are exact in
// f64, so expectEqual is safe.

test "gemm transB=1 with (n,) bias" {
    const a = [_]f64{ 1, 2, 3 }; // (m=1, k=3)
    const w = [_]f64{ 1, 2, 3, 4, 5, 6 }; // (n=2, k=3)
    const c = [_]f64{ 0.5, -0.5 };
    const r = la.gemm(1, 3, 2, 1.0, 1.0, true, &a, &w, &c, 2);
    try std.testing.expectEqual([_]f64{ 14.5, 31.5 }, r);
}

test "gemm transB=0 with scalar (zero) bias" {
    const a = [_]f64{ 1, 2, 3 };
    const w = [_]f64{ 1, 2, 3, 4, 5, 6 }; // (k=3, n=2)
    const c = [_]f64{0.0};
    const r = la.gemm(1, 3, 2, 1.0, 1.0, false, &a, &w, &c, 1);
    try std.testing.expectEqual([_]f64{ 22, 28 }, r);
}

test "gemm applies alpha and beta" {
    const a = [_]f64{ 1, 2, 3 };
    const w = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const c = [_]f64{ 0.5, -0.5 };
    const r = la.gemm(1, 3, 2, 0.5, 2.0, true, &a, &w, &c, 2);
    try std.testing.expectEqual([_]f64{ 8, 15 }, r);
}

test "gemm 2-D activation (m,k) @ (n,k)'.T" {
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 }; // (m=2, k=3)
    const w = [_]f64{ 1, 2, 3, 4, 5, 6 }; // (n=2, k=3)
    const c = [_]f64{0.0};
    const r = la.gemm(2, 3, 2, 1.0, 1.0, true, &a, &w, &c, 1);
    try std.testing.expectEqual([_]f64{ 14, 32, 32, 77 }, r);
}

test "gemm broadcasts a full (m,n) bias" {
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 }; // (m=2, k=3)
    const w = [_]f64{ 1, 2, 3, 4, 5, 6 }; // (n=2, k=3)
    const c = [_]f64{ 1, 1, 2, 2 }; // (m=2, n=2)
    const r = la.gemm(2, 3, 2, 1.0, 1.0, true, &a, &w, &c, 4);
    try std.testing.expectEqual([_]f64{ 15, 33, 34, 79 }, r);
}

// layernorm_rows: last-axis normalization over (rows, cols) = (samples,
// features). Biased variance, rstd = 1/sqrt(var + eps).

test "layernorm_rows normalizes each row over the last axis" {
    // Rows [1,2,3] and [4,5,6] share the same spread, so both rows map to
    // [-rstd, 0, rstd] with rstd = 1/sqrt(2/3 + eps). The second row is the
    // regression guard for the old in-loop `return out;`, which left every
    // row after the first undefined.
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const scale = [_]f64{ 1, 1, 1 };
    const bias = [_]f64{ 0, 0, 0 };
    const r = la.layernorm_rows(2, 3, &a, &scale, &bias, 1e-5);
    const rstd = 1.0 / std.math.sqrt(2.0 / 3.0 + 1e-5);
    try std.testing.expectApproxEqAbs(-rstd, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.0, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(rstd, r[2], 1e-12);
    try std.testing.expectApproxEqAbs(-rstd, r[3], 1e-12);
    try std.testing.expectApproxEqAbs(0.0, r[4], 1e-12);
    try std.testing.expectApproxEqAbs(rstd, r[5], 1e-12);
}

test "layernorm_rows applies scale and bias to a single row" {
    // A 1-D vector is one row: rows=1, cols=n. out = (x - mean) * rstd * scale + bias.
    const a = [_]f64{ 1, 2, 3 };
    const scale = [_]f64{ 2, 3, 4 };
    const bias = [_]f64{ 0.5, -0.5, 1.0 };
    const r = la.layernorm_rows(1, 3, &a, &scale, &bias, 1e-5);
    const rstd = 1.0 / std.math.sqrt(2.0 / 3.0 + 1e-5);
    try std.testing.expectApproxEqAbs(-1.0 * rstd * 2.0 + 0.5, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(0.0 * rstd * 3.0 - 0.5, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(1.0 * rstd * 4.0 + 1.0, r[2], 1e-12);
}

test "layernorm_rows of a constant row is exactly the bias (eps floor)" {
    // var = 0, so (x - mean) = 0 annihilates the rstd term and eps keeps rstd
    // finite: the normalized value is exactly the bias.
    const a = [_]f64{ 5, 5, 5 };
    const scale = [_]f64{ 2, 3, 4 };
    const bias = [_]f64{ 0.5, -0.5, 1.0 };
    const r = la.layernorm_rows(1, 3, &a, &scale, &bias, 1e-5);
    try std.testing.expectEqual([_]f64{ 0.5, -0.5, 1.0 }, r);
}

// ─── recurrent cells ──────────────────────────────────────────────────────
// Reference vectors are the ONNX formulas evaluated in f64 (and validated
// against onnxruntime to ~1e-7 on real HF policies + synthetic nodes):
// H=2, I=3, random weights from seed 5. Values are irrational (sigmoid/tanh),
// so expectApproxEqAbs at 1e-12.

const rc_x = [_]f64{ -0.4009657126267237, -0.66217949781407248, -0.12418081104762427 };
const rc_h = [_]f64{ 0.21022261903276074, 0.56802326624482136 };
const rc_c = [_]f64{ 0.054853199660904094, -0.27632366026811622 };

const rc_Wl = [_]f64{
    -0.39239017767213918, 0.37437288536729557,  0.81739152147928873,
    0.13638438792236088,  -0.61666433201538584, -0.47913260271804436,
    0.80000954449955575,  0.1014412202543042,   -0.86606742121979241,
    -0.041848096408512905, -0.58161298672237427, -0.31464404703077725,
    -0.24400291163842872, -0.35665668581612181, 0.27668923517664473,
    -0.031542985962644578, -0.29471562901630238, 0.20481891327855847,
    0.41492765353066197,  -0.8215116857028385,  -0.12836506318274701,
    -0.49037367802200627, -0.086577612431016027, -0.64470937337692935,
};
const rc_Rl = [_]f64{
    0.010345197018795599,  -0.018942870522034114, -0.15216887547924449, -0.52396325256012311,
    -0.19809516523654636,  -0.54566445084785453, -0.67760437310236976, 0.11239286622994657,
    -0.55467496894568302,  0.58514805058914665,  0.35829382793691805,  -0.99890834622486058,
    0.13606443470624399,   -0.55085831379052241, 0.016528610079134597, 0.021815996284710804,
};
const rc_Bl = [_]f64{
    -0.99421489411556041,  -0.11671126188288501,  -0.1278950156996955,  0.48100026592154721,
    -0.59072340397810785,  0.36902094892284204,   -0.54948638151820317, -0.16564544634995837,
    -0.42023658421110555,  0.72436564446083596,   0.28410654989414663,  1.2158662514226062,
    0.32095818954116023,   0.42249636685958769,   0.42034143813267005,  -0.30330576795477582,
};

const rc_Wg = [_]f64{
    -0.035014223319191039, 0.67519443387231304,  -0.1982753825864858,
    0.094399765645549333,  -0.010611730208644898, 0.30460824641637035,
    -0.1824543709736863,   -0.07618094437842414,  0.12119071433607521,
    0.051511569243403846,  -0.43248637314091004,  0.4478915215947219,
    -0.64924060412345597,  -0.60055777440176328,  -0.64124589737023097,
    0.4834861394666074,    -0.180304184449635,    -0.48551818926053275,
};
const rc_Rg = [_]f64{
    -0.56801069709482332,  0.21056556873120308,   -0.52742033128891752, -0.63603910504882111,
    0.30699653123443044,   -0.59835386359628528,  -0.16121907537569863, -0.0033807701732608926,
    -0.22266768346490615,  -0.027046980797724138, 0.66938674952738575,  -0.25844707629463748,
};
const rc_Bg = [_]f64{
    -0.62965357943911382, -0.91837285259683976, -0.10238305841693353, -0.17612850520561524,
    0.13254545346516175,  -0.2321222958187408,  -0.23931923152636225, -0.36065787392430232,
    -0.25987988618508939, 0.080113353158712153, -0.1901763636075694,  0.050220926416253253,
};

const rc_Wr = [_]f64{ 0.9505936000166284, 0.23955907539847174, -0.78804492271761306, 0.86676479280655339, 0.17390518499426771, -0.47070664322549233 };
const rc_Rr = [_]f64{ 0.45352447884290509, 0.0088278054549194851, -0.30760926651753456, -0.31675446271680818 };
const rc_Br = [_]f64{ -0.49671589512592867, 0.024059491994919288, 0.53440834687478289, -0.16252604940318935 };

test "rnn_cell matches the ONNX Elman recurrence" {
    const r = la.rnn_cell(2, 3, &rc_x, &rc_Wr, &rc_Rr, &rc_Br, &rc_h);
    try std.testing.expectApproxEqAbs(-0.29485798096491211, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(-0.65687879956822692, r[1], 1e-12);
}

test "lstm_cell matches the ONNX cell, emitting [h ; c]" {
    const r = la.lstm_cell(2, 3, &rc_x, &rc_Wl, &rc_Rl, &rc_Bl, &rc_h, &rc_c);
    // h_next in the first H slots, c_next in the second H.
    try std.testing.expectApproxEqAbs(0.010529203082467157, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(-0.21702325008521592, r[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.027480478533079628, r[2], 1e-12);
    try std.testing.expectApproxEqAbs(-0.24977230441630663, r[3], 1e-12);
}

test "gru_cell lbr=1 resets after the recurrent matmul" {
    const r = la.gru_cell(2, 3, true, &rc_x, &rc_Wg, &rc_Rg, &rc_Bg, &rc_h);
    try std.testing.expectApproxEqAbs(0.55471399880159933, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(-0.10987029168348694, r[1], 1e-12);
}

test "gru_cell lbr=0 resets before the recurrent matmul" {
    const r = la.gru_cell(2, 3, false, &rc_x, &rc_Wg, &rc_Rg, &rc_Bg, &rc_h);
    try std.testing.expectApproxEqAbs(0.49531383719362443, r[0], 1e-12);
    try std.testing.expectApproxEqAbs(-0.10725928371552693, r[1], 1e-12);
}

test "gru_cell lbr is not a no-op (the two variants differ)" {
    // Guards against the classic silent bug: implementing lbr as only a bias
    // placement makes both branches identical. They must differ.
    const a = la.gru_cell(2, 3, true, &rc_x, &rc_Wg, &rc_Rg, &rc_Bg, &rc_h);
    const b = la.gru_cell(2, 3, false, &rc_x, &rc_Wg, &rc_Rg, &rc_Bg, &rc_h);
    try std.testing.expect(@abs(a[0] - b[0]) > 1e-3);
}

// ─── gather ───────────────────────────────────────────────────────────────
// ONNX Gather: index-select along one axis. Indices are f64 buffers (the VM
// has no integer tensor type); negatives count from the end, and an
// out-of-range index clamps rather than panicking in a deployed kernel.

test "gather_index selects in order, including negative indices" {
    const x = [_]f64{ 10, 20, 30, 40 };
    const idx = [_]f64{ 0, 2, -1, -2 };
    const r = la.gather_index(4, 4, &x, &idx);
    try std.testing.expectEqual([_]f64{ 10, 30, 40, 30 }, r);
}

test "gather_index clamps an out-of-range index" {
    const x = [_]f64{ 10, 20, 30, 40 };
    try std.testing.expectEqual([_]f64{40}, la.gather_index(1, 4, &x, &[_]f64{99}));
    try std.testing.expectEqual([_]f64{10}, la.gather_index(1, 4, &x, &[_]f64{-99}));
}

test "gather_rows picks whole rows of a row-major matrix" {
    // x = [[1, 2], [3, 4], [5, 6]] -> rows [2, 0]
    const x = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const idx = [_]f64{ 2, 0 };
    try std.testing.expectEqual([_]f64{ 5, 6, 1, 2 }, la.gather_rows(2, 2, 3, &x, &idx));
}

test "gather_cols picks whole columns of a row-major matrix" {
    // x = [[1, 2, 3], [4, 5, 6]] -> columns [2, 0]
    const x = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const idx = [_]f64{ 2, 0 };
    try std.testing.expectEqual([_]f64{ 3, 1, 6, 4 }, la.gather_cols(2, 2, 3, &x, &idx));
}

test "gather_cols with a negative index" {
    const x = [_]f64{ 1, 2, 3, 4, 5, 6 };
    try std.testing.expectEqual([_]f64{ 3, 6 }, la.gather_cols(2, 1, 3, &x, &[_]f64{-1}));
}

test "gather_index sanitizes non-finite and huge indices before converting" {
    // `@intFromFloat` is UB for these; the kernel must stay defined because the
    // index operand can be a host-fed port rather than a baked constant.
    const x = [_]f64{ 10, 20, 30, 40 };
    try std.testing.expectEqual([_]f64{10}, la.gather_index(1, 4, &x, &[_]f64{std.math.nan(f64)}));
    try std.testing.expectEqual([_]f64{40}, la.gather_index(1, 4, &x, &[_]f64{std.math.inf(f64)}));
    try std.testing.expectEqual([_]f64{10}, la.gather_index(1, 4, &x, &[_]f64{-std.math.inf(f64)}));
    try std.testing.expectEqual([_]f64{40}, la.gather_index(1, 4, &x, &[_]f64{1e300}));
    try std.testing.expectEqual([_]f64{10}, la.gather_index(1, 4, &x, &[_]f64{-1e300}));
}

test "gather_index keeps the valid negative wrap boundary" {
    // -len is a legal index (the first element); only below that does it saturate.
    const x = [_]f64{ 10, 20, 30, 40 };
    try std.testing.expectEqual([_]f64{10}, la.gather_index(1, 4, &x, &[_]f64{-4}));
    try std.testing.expectEqual([_]f64{10}, la.gather_index(1, 4, &x, &[_]f64{-5}));
}

test "gather_index on a length-1 axis is total" {
    const x = [_]f64{7};
    try std.testing.expectEqual([_]f64{7}, la.gather_index(1, 1, &x, &[_]f64{-1}));
    try std.testing.expectEqual([_]f64{7}, la.gather_index(1, 1, &x, &[_]f64{5}));
}
