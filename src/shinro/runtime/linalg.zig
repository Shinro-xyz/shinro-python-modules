// runtime/linalg.zig — fixed-size, comptime-shaped linear algebra for lowered graphs.
//
// All sizes are comptime constants derived from the graph's node shapes, so every
// array is a fixed-size stack value: no allocator, no heap traffic. This is the
// "fixed at compile time" guarantee the lowered .so is built on.
//
// The math mirrors numpy as the reference: matmul follows numpy's 1D conventions
// (2D@2D, 2D@1D as matvec, 1D@2D as vecmat); inv is Gauss-Jordan with partial
// pivoting. For the small, well-conditioned matrices in the control graphs (2x2-4x4
// innovation covariances) it agrees with numpy.linalg.inv to well inside the
// oracle's atol=1e-12, so LAPACK is unnecessary here.
//
// Inputs are slices; the VM hands out fixed-length slices of its buffer.
// 

const std= @import("std");

/// Matrix multiply: (m, k) @ (k, n) -> (m, n), row-major, flat output.
///
/// Mirrors numpy's 2D @ 2D convention. All dimensions are comptime so the
/// output is a fixed-size stack array `[m * n]f64`.
///
/// Args:
///     m: Rows of `a` and of the result.
///     k: Columns of `a` / rows of `b` (the contraction axis).
///     n: Columns of `b` and of the result.
///     a: Flat row-major `m*k` matrix.
///     b: Flat row-major `k*n` matrix.
///
/// Returns:
///     The flat `m*n` row-major result.
pub fn matmul(comptime m: usize, comptime k: usize, comptime n: usize, a: []const f64, b: []const f64) [m * n]f64 {
    var out: [m * n]f64 = undefined;
    for (0..m) |i| {
        for (0..n) |j| {
            var s: f64 = 0.0;
            for (0..k) |p| {
                s += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = s;
        }
    }
    return out;
}

/// Matrix-vector multiply: (m, k) @ (k,) -> (m,), flat output.
///
/// Mirrors numpy's 2D @ 1D convention, where the 1D operand is treated as a
/// column vector. Used when the VM's matmul node has `cols == 1` and the
/// left operand is 2D.
///
/// Args:
///     m: Rows of `a` and of the result.
///     k: Columns of `a` / length of `v` (the contraction axis).
///     a: Flat row-major `m*k` matrix.
///     v: Flat `k`-vector.
///
/// Returns:
///     The flat `m`-vector result.
///
/// Row `i` of `a` and all of `v` are contiguous along the contraction, so each
/// output element is a vector `dot`. This is the recurrent cells' hot path
/// (`x·Wᵀ` and `h·Rᵀ`), where `k` is the input/hidden width — far above
/// `SIMD_MIN_K`.
pub fn matvec(comptime m: usize, comptime k: usize, a: []const f64, v: []const f64) [m]f64 {
    var out: [m]f64 = undefined;
    for (0..m) |i| out[i] = dot(k, a[i * k ..][0..k], v);
    return out;
}

/// Vector-matrix multiply: (k,) @ (k, n) -> (n,), flat output.
///
/// Mirrors numpy's 1D @ 2D convention, where the 1D operand is treated as a
/// row vector. Used when the VM's matmul node has `cols == 1` and the left
/// operand is 1D.
///
/// Args:
///     k: Length of `v` / rows of `b` (the contraction axis).
///     n: Columns of `b` and of the result.
///     v: Flat `k`-vector.
///     b: Flat row-major `k*n` matrix.
///
/// Returns:
///     The flat `n`-vector result.
pub fn vecmat(comptime k: usize, comptime n: usize, v: []const f64, b: []const f64) [n]f64 {
    var out: [n]f64 = undefined;
    for (0..n) |j| {
        var s: f64 = 0.0;
        for (0..k) |p| {
            s += v[p] * b[p * n + j];
        }
        out[j] = s;
    }
    return out;
}

/// Gauss-Jordan inverse with partial pivoting, flat output.
///
/// Inverts the `n`×`n` row-major matrix at `a` in place on a copy, building
/// the inverse in a companion identity matrix. Partial pivoting (choosing the
/// largest-magnitude pivot in each column) keeps small well-conditioned
/// matrices — the 2×2–4×4 innovation covariances in the control graphs —
/// accurate to well inside the oracle's `atol=1e-12`, so LAPACK is
/// unnecessary here.
///
/// Args:
///     n: The matrix dimension (rows == cols).
///     a: Flat row-major `n*n` matrix.
///
/// Returns:
///     The flat `n*n` row-major inverse.
///
/// Panics:
///     If a pivot is exactly zero (singular matrix).
pub fn inv(comptime n: usize, a: []const f64) [n * n]f64 {
    var m: [n * n]f64 = undefined;
    for (0..(n * n)) |i| m[i] = a[i];
    var out: [n * n]f64 = undefined;
    for (0..n) |i| {
        for (0..n) |j| {
            out[i * n + j] = if (i == j) 1.0 else 0.0;
        }
    }

    for (0..n) |k| {
        var p = k;
        for (k + 1..n) |i| {
            if (@abs(m[i * n + k]) > @abs(m[p * n + k])) p = i;
        }
        if (p != k) {
            for (0..n) |j| {
                const t = m[k * n + j];
                m[k * n + j] = m[p * n + j];
                m[p * n + j] = t;
            }
            for (0..n) |j| {
                const t = out[k * n + j];
                out[k * n + j] = out[p * n + j];
                out[p * n + j] = t;
            }
        }
        const piv = m[k * n + k];
        if (piv == 0.0) @panic("singular matrix in inv");
        for (0..n) |j| {
            m[k * n + j] /= piv;
            out[k * n + j] /= piv;
        }
        for (0..n) |i| {
            if (i != k) {
                const f = m[i * n + k];
                if (f == 0.0) continue;
                for (0..n) |j| {
                    m[i * n + j] -= f * m[k * n + j];
                    out[i * n + j] -= f * out[k * n + j];
                }
            }
        }
    }
    return out;
}


/// Elementwise sine: `out[i] = sin(a[i])`. 1-D input, same-length flat output.
pub fn sin_vec (comptime m: usize, a:[]const f64) [m]f64{
    var out: [m]f64= undefined;
    for (0..m) |i| {
        out[i]=std.math.sin(a[i]);
    }
    return out;
}

/// Elementwise cosine: `out[i] = cos(a[i])`. 1-D input, same-length flat output.
pub fn cos_vec(comptime m: usize, a:[]const f64) [m]f64{
    var out: [m]f64 = undefined;
    for (0..m) |i| {
        out[i]=std.math.cos(a[i]);
    }
    return out;
}

/// Elementwise ReLU: `out[i] = max(a[i], 0)`. Shape-preserving.
pub fn relu(comptime m:usize, a:[]const f64) [m]f64{
    var out: [m]f64=undefined;
    for (0..m) |i| {
        out[i]= @max(0.0,a[i]);
    }
    return out;
}

/// Elementwise exponential: `out[i] = exp(a[i])`. Shape-preserving.
pub fn elementwise_exponential (comptime m:usize,a:[]const f64) [m]f64{
    var out: [m]f64=undefined;

    for (0..m) |i| {
        out[i]= std.math.exp(a[i]);
    }
    return out;
}

/// Integer matrix power `A^p` for an `n`x`n` row-major matrix: repeated
/// `matmul` starting from the identity (`p = 0` yields the identity).
pub fn matrix_power (comptime n:usize, a:[]const f64, comptime p: usize) [n*n]f64{
    var result: [n*n]f64 = undefined;

    // making the identuity matrix to make the matmul operations work

    for (0..n) |i| {
        for (0..n) |j| {
            if (i==j) {
                result[i*n+j]=1.0;
            } else {
                result[i*n+j]=0.0;
            }
        }
    }

    // completing the matrix power mults
    for (0..p) |_| {
        result= matmul(n, n, n, &result, a);
    }
    return result;
}

/// Elementwise hyperbolic tangent: `out[i] = tanh(a[i])`. Shape-preserving.
pub fn tanh (comptime m: usize, a: []const f64) [m]f64 {
    var result: [m]f64= undefined;

    for (0..m) |i| {
        result[i]= std.math.tanh(a[i]);
    }
    return result;
}

/// Index of the maximum element (`np.argmax` semantics: ties resolve to the
/// first/lowest index).
pub fn argmax (comptime m: usize, a:[]const f64) usize {
    var best: usize = 0;
    for (0..m) |i| {
        if (a[i]>a[best]) {
            best=i;
        }
    }
    return best;
}

//minimum reductions--> numpy's np.min with axis=None / 0 / 1

/// Minimum over the whole flat array. numpy propagates NaN, so a NaN operand
/// wins the comparison and poisons the result — mirror that (a plain
/// `if (v < best)` would silently ignore NaNs, diverging from the oracle on
/// NaN feeds).
pub fn min_all (comptime m: usize, a: []const f64) [1]f64 {
    var best: f64 = a[0];
    for (1..m) |i| {
        const v = a[i];
        if (std.math.isNan(v) or v < best) best = v;
    }
    return .{best};
}

/// Minimum down each column: (rows, cols) -> (cols,), numpy `min(axis=0)`.
pub fn min_axis0 (comptime rows: usize, comptime cols: usize, a: []const f64) [cols]f64 {
    var out: [cols]f64 = undefined;
    for (0..cols) |j| {
        var best = a[j];
        for (1..rows) |i| {
            const v = a[i * cols + j];
            if (std.math.isNan(v) or v < best) best = v;
        }
        out[j] = best;
    }
    return out;
}

/// Minimum across each row: (rows, cols) -> (rows,), numpy `min(axis=1)`.
pub fn min_axis1 (comptime rows: usize, comptime cols: usize, a: []const f64) [rows]f64 {
    var out: [rows]f64 = undefined;
    for (0..rows) |i| {
        var best = a[i * cols];
        for (1..cols) |j| {
            const v = a[i * cols + j];
            if (std.math.isNan(v) or v < best) best = v;
        }
        out[i] = best;
    }
    return out;
}

/// One-hot vector of length `depth`: 1.0 at `idx`, 0.0 elsewhere. If
/// `idx >= depth` every entry is 0.0 (no bounds check).
pub fn onehot (comptime depth: usize, idx:usize) [depth]f64{
     var out: [depth]f64= undefined;

     for (0..depth) |i| {
         if (i==idx) {
             out[i]=1.0;
         } else {
             out[i]=0.0;
         }
     }
     return out;
 }

/// The target's suggested f64 vector width, in lanes: 2 on NEON (the Pi),
/// 4 on AVX, and 1 (scalar) on a target with no vector suggestion. Computed
/// once at comptime from the target; every use is the same value.
const SIMD_VL = std.simd.suggestVectorLength(f64) orelse 1;

/// Contraction length below which the vector dot product is not worth its
/// horizontal reduce. The floor of 8, independent of the target's vector
/// width, keeps every small/classical-sized contraction on the exact scalar
/// order on *every* target — so a result never depends on the build target's
/// SIMD width — while every policy-sized contraction still takes the vector
/// path.
const SIMD_MIN_K = @max(2 * SIMD_VL, 8);

/// Lane-parallel dot product of two contiguous length-`len` vectors.
///
/// The scalar form is a serial `s += a[p]*b[p]` chain: Zig's strict FP model
/// forbids reassociation, so it compiles to one scalar FMA per element on a
/// `len`-long dependency chain — and `len` is small enough (see SIMD_MIN_K)
/// that nothing else hides it. Accumulating into a `@Vector(VL, f64)` splits
/// that chain into VL independent ones and drives the FP units at full width,
/// with one horizontal reduce at the end; a scalar tail handles `len % VL`.
/// The result differs from the scalar order only in the final reduction
/// (relative ~1e-15 — well inside the oracle's 1e-12), so `len < SIMD_MIN_K`
/// deliberately stays on the scalar loop and remains bit-identical.
fn dot(comptime len: usize, a: []const f64, b: []const f64) f64 {
    if (len < SIMD_MIN_K or SIMD_VL <= 1) {
        var s: f64 = 0.0;
        for (0..len) |p| s += a[p] * b[p];
        return s;
    }
    var acc: @Vector(SIMD_VL, f64) = @splat(0.0);
    var p: usize = 0;
    while (p + SIMD_VL <= len) : (p += SIMD_VL) {
        const av: @Vector(SIMD_VL, f64) = a[p..][0..SIMD_VL].*;
        const bv: @Vector(SIMD_VL, f64) = b[p..][0..SIMD_VL].*;
        acc = @mulAdd(@Vector(SIMD_VL, f64), av, bv, acc);
    }
    var s: f64 = @reduce(.Add, acc);
    while (p < len) : (p += 1) s += a[p] * b[p];
    return s;
}

/// Fused dense layer: `out = alpha * (A @ B') + beta * C`, flat row-major.
///
/// `A` is `(m, k)`; `B` is `(n, k)` when `transB` (torch `nn.Linear` weight
/// layout, read by striding) and `(k, n)` otherwise. `C` broadcasts like
/// numpy: `c_len == 1` a scalar, `c_len == n` a row, else a full `(m, n)`
/// bias. `alpha`, `beta`, and `transB` are comptime and constant-fold.
///
/// `transB` makes both operands contiguous along the contraction (A row `i`
/// and B row `j`), so the inner product routes through the vector `dot`; a
/// non-transB weight is read with stride `n` and stays scalar. This is where a
/// policy's matmuls are: an ONNX `nn.Linear` export is transB, and a policy
/// runs at m = 1.
pub fn gemm (
     comptime m: usize,comptime k: usize,comptime n: usize,comptime alpha:f64,comptime beta:f64,comptime transB: bool,
     a:[]const f64,
     b:[]const f64,
     c:[]const f64,
     comptime c_len:usize
 ) [m*n]f64 {
     var out: [m*n]f64 = undefined;
     for (0..m) |i| {
         for (0..n) |j| {
             const s: f64 = if (transB)
                 dot(k, a[i * k ..][0..k], b[j * k ..][0..k])
             else blk: {
                 // transB=false walks B with stride n (column-major over a
                 // row-major weight), so the contraction is not contiguous and
                 // stays scalar.
                 var t: f64 = 0.0;
                 for (0..k) |p| t += a[i * k + p] * b[p * n + j];
                 break :blk t;
             };
             const cj= if (c_len==1) c[0] else if (c_len==n) c[j] else c[i*n+j];
             out[i*n+j]= alpha*s+beta*cj;
         }
     }
     return out;
 }

/// Elementwise logistic sigmoid, ``f(x) = 1 / (1 + exp(-x))``.
///
/// This is the ONNX ``Sigmoid`` activation. The negation folds into the exp
/// argument, so the kernel needs no intermediate buffer and stays a
/// fixed-size stack array like every other linalg kernel.
pub fn sigmoid(comptime m: usize, a: []const f64) [m]f64 {
    var out: [m]f64 = undefined;
    for (0..m) |i| {
        out[i] = 1.0 / (1.0 + std.math.exp(-a[i]));
    }
    return out;
}

/// Numerically-stable softmax over the last axis of a flat row-major matrix.
///
/// For each row ``i``: ``out[i*cols + j] = exp(a[i*cols+j] - m_i) /
/// sum_k exp(a[i*cols+k] - m_i)``, with ``m_i = max_k a[i*cols+k]``. The
/// per-row max-shift keeps every exponent <= 0 (no overflow; the denominator
/// is >= 1). Shapes are comptime and the output is the full ``[rows*cols]``
/// array, so the op is shape-preserving (numpy ``axis=-1`` semantics).
///
/// A 1-D vector is a single row: callers pass ``rows = 1, cols = n``. Inputs
/// are expected to be pre-scaled by temperature (the kernel carries none). An
/// all-(-inf) row yields NaN, matching numpy.
pub fn softmax_rows(comptime rows: usize, comptime cols: usize, a: []const f64) [rows * cols]f64 {
    var out: [rows * cols]f64 = undefined;
    for (0..rows) |i| {
        const base = i * cols;
        var m: f64 = a[base];
        for (1..cols) |j| m = @max(m, a[base + j]);

        var sum: f64 = 0.0;
        for (0..cols) |j| {
            const e = std.math.exp(a[base + j] - m);
            out[base + j] = e;
            sum += e;
        }
        for (0..cols) |j| out[base + j] /= sum;
    }
    return out;
}

/// GELU, tanh approximation (the GPT-style ``gelu_new``):
/// ``f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.
///
/// The exact erf-based GELU is not implemented; this approximation deviates
/// from it by at most ~4.7e-4 absolute (worst near |x| ~ 2.7) and the same
/// formula is used by ``numpy``/``torch`` (``approximate="tanh"``), so oracle
/// parity is exact.
pub fn gelu(comptime m: usize, a: []const f64) [m]f64 {
    const c: f64 = std.math.sqrt(2.0 / std.math.pi); // sqrt(2/pi)
    var out: [m]f64 = undefined;
    for (0..m) |i| {
        const x = a[i];
        out[i] = 0.5 * x * (1.0 + std.math.tanh(c * (x + 0.044715 * std.math.pow(f64, x, 3.0))));
    }
    return out;
}

/// Elementwise ELU: `out[i] = a[i]` if `a[i] > 0` else `alpha *
/// (exp(a[i]) - 1)`. `alpha` comes from the ONNX op's attribute (default
/// 1.0) and is comptime-baked per node by the lowerer, so it folds into the
/// node's switch arm. Shape-preserving.
pub fn elu(comptime m: usize, comptime alpha: f64, a: []const f64) [m]f64 {
    var out: [m]f64 = undefined;
    for (0..m) |i| {
        const x = a[i];
        out[i] = if (x > 0.0) x else alpha * (std.math.exp(x) - 1.0);
    }
    return out;
}

// ─── recurrent cells ──────────────────────────────────────────────────────
//
// One fused step of an ONNX-style recurrent cell (batch 1, one timestep).
// Each cell folds the two affine maps (`x·Wᵀ` and `h·Rᵀ`), the gate split, the
// gate activations, and the state update into a single kernel — the same
// arithmetic the ONNX `LSTM`/`GRU`/`RNN` ops define. A policy therefore costs
// one graph node per cell instead of the ~17 nodes the equivalent
// gemm/slice/sigmoid/tanh decomposition would emit.
//
// Layouts are ONNX's: `W` is `(rows, input)` (torch `nn.Linear`/RNN layout),
// read as `x·Wᵀ` by striding its rows through `matvec`; `R` is `(rows, H)`;
// `B` is `Wb ‖ Rb` with the input biases first. All dimensions are comptime,
// every buffer is a flat fixed-size stack array, and the inner loops are
// runtime `for` (never `inline for`) per the VM's code-size rule.
//
// Gate order, activation placement, and the GRU reset-gate placement follow
// the ONNX spec exactly; every formula is validated against onnxruntime
// (max|err| ~1e-7) on real HF robotics policies and synthetic per-arch nodes.

/// LSTM cell, one step. Gate order is ONNX's `i, o, f, c`, with Sigmoid on
/// `i`/`o`/`f` and Tanh on the cell candidate and the hidden output.
///
/// Args:
///     x: Flat input `(I,)`.
///     w: `(4H, I)` input weights (ONNX `W`).
///     r: `(4H, H)` recurrent weights (ONNX `R`).
///     b: `(8H,)` biases, `Wb(4H) ‖ Rb(4H)` (ONNX `B`); both halves are added.
///     h_prev: `(H,)` previous hidden state.
///     c_prev: `(H,)` previous cell state.
///
/// Returns:
///     `(2H,)` = `h_next(0..H) ‖ c_next(H..2H)` — the whole next state in one
///     port, which is exactly the ML-Agents `recurrent_in`/`recurrent_out`
///     layout (`[h ‖ c]`, h first).
pub fn lstm_cell(
    comptime H: usize,
    comptime I: usize,
    x: []const f64,
    w: []const f64,
    r: []const f64,
    b: []const f64,
    h_prev: []const f64,
    c_prev: []const f64,
) [2 * H]f64 {
    const gx = matvec(4 * H, I, w, x); // x·Wᵀ
    const gh = matvec(4 * H, H, r, h_prev); // h·Rᵀ

    var out: [2 * H]f64 = undefined;
    for (0..H) |j| {
        const i_g = sigmoid1(gx[j] + gh[j] + b[j] + b[4 * H + j]);
        const o_g = sigmoid1(gx[H + j] + gh[H + j] + b[H + j] + b[5 * H + j]);
        const f_g = sigmoid1(gx[2 * H + j] + gh[2 * H + j] + b[2 * H + j] + b[6 * H + j]);
        const g_g = std.math.tanh(gx[3 * H + j] + gh[3 * H + j] + b[3 * H + j] + b[7 * H + j]);
        const c_next = f_g * c_prev[j] + i_g * g_g;
        out[H + j] = c_next;
        out[j] = o_g * std.math.tanh(c_next);
    }
    return out;
}

/// GRU cell, one step. Gate order is ONNX's `z, r, h`, with Sigmoid on `z`/`r`
/// and Tanh on the candidate.
///
/// `lbr` (ONNX `linear_before_reset`) moves the reset gate *across* the
/// recurrent matmul, not merely the bias:
///   - `lbr = true`  → `h = g(X·Whᵀ + r ⊙ (H·Rhᵀ + Rbh) + Wbh)`
///   - `lbr = false` → `h = g(X·Whᵀ + (r ⊙ H)·Rhᵀ + Rbh + Wbh)`
/// Getting this backwards silently diverges ~1e-1 (no error, just wrong
/// numbers), so it is a comptime parameter, not a runtime branch.
///
/// Args:
///     x: Flat input `(I,)`.
///     w: `(3H, I)` input weights (ONNX `W`).
///     r: `(3H, H)` recurrent weights (ONNX `R`).
///     b: `(6H,)` biases, `Wbz‖Wbr‖Wbh‖Rbz‖Rbr‖Rbh` (ONNX `B`).
///     h_prev: `(H,)` previous hidden state.
///
/// Returns:
///     `(H,)` the next hidden state.
pub fn gru_cell(
    comptime H: usize,
    comptime I: usize,
    comptime lbr: bool,
    x: []const f64,
    w: []const f64,
    r: []const f64,
    b: []const f64,
    h_prev: []const f64,
) [H]f64 {
    const gx = matvec(3 * H, I, w, x); // x·Wᵀ
    const gh = matvec(3 * H, H, r, h_prev); // h·Rᵀ

    var z: [H]f64 = undefined;
    var r_g: [H]f64 = undefined;
    for (0..H) |j| {
        z[j] = sigmoid1(gx[j] + gh[j] + b[j] + b[3 * H + j]);
        r_g[j] = sigmoid1(gx[H + j] + gh[H + j] + b[H + j] + b[4 * H + j]);
    }

    var out: [H]f64 = undefined;
    if (lbr) {
        for (0..H) |j| {
            const n = std.math.tanh(gx[2 * H + j] + r_g[j] * (gh[2 * H + j] + b[5 * H + j]) + b[2 * H + j]);
            out[j] = (1.0 - z[j]) * n + z[j] * h_prev[j];
        }
    } else {
        // (r ⊙ H)·Rhᵀ — the reset is applied to h BEFORE the recurrent
        // matmul, so this is not the same product the lbr=1 branch uses.
        // Rh is R's third H-row block: offset (2H)·H into the (3H, H) matrix.
        var rh: [H]f64 = undefined;
        for (0..H) |j| rh[j] = r_g[j] * h_prev[j];
        var ghr: [H]f64 = undefined;
        for (0..H) |j| {
            var s: f64 = 0.0;
            for (0..H) |p| s += r[(2 * H + j) * H + p] * rh[p];
            ghr[j] = s;
        }
        for (0..H) |j| {
            const n = std.math.tanh(gx[2 * H + j] + ghr[j] + b[5 * H + j] + b[2 * H + j]);
            out[j] = (1.0 - z[j]) * n + z[j] * h_prev[j];
        }
    }
    return out;
}

/// Vanilla (Elman) RNN cell, one step: `h_next = tanh(x·Wᵀ + h·Rᵀ + Wb + Rb)`.
///
/// Args:
///     x: Flat input `(I,)`.
///     w: `(H, I)` input weights (ONNX `W`).
///     r: `(H, H)` recurrent weights (ONNX `R`).
///     b: `(2H,)` biases, `Wb(H) ‖ Rb(H)` (ONNX `B`); both halves are added.
///     h_prev: `(H,)` previous hidden state.
///
/// Returns:
///     `(H,)` the next hidden state.
pub fn rnn_cell(
    comptime H: usize,
    comptime I: usize,
    x: []const f64,
    w: []const f64,
    r: []const f64,
    b: []const f64,
    h_prev: []const f64,
) [H]f64 {
    const gx = matvec(H, I, w, x); // x·Wᵀ
    const gh = matvec(H, H, r, h_prev); // h·Rᵀ
    var out: [H]f64 = undefined;
    for (0..H) |j| out[j] = std.math.tanh(gx[j] + gh[j] + b[j] + b[H + j]);
    return out;
}

/// Scalar logistic sigmoid `1 / (1 + exp(-x))` — the gate activation the
/// recurrent cells apply elementwise (the vector `sigmoid` above is the same
/// map over a buffer; this avoids materialising one).
fn sigmoid1(x: f64) f64 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

// ─── gather (ONNX Gather: index-select along one axis) ────────────────────
//
// The index tensor is a flat f64 buffer (the VM has no integer tensor type), so
// each index is converted once per element. Negative indices count from the end,
// matching ONNX/numpy. `concat` needs no kernel here: it is a pure buffer copy
// and lives in the VM's switch, next to `stack`.

/// Index-select a 1-D vector: `out[j] = x[normalize(idx[j])]`.
///
/// Args:
///     n: Number of selected elements (the index buffer's length).
///     src: Length of `x` (comptime, for negative-index normalization).
///     x: Flat source vector.
///     idx: Flat index buffer; values are truncated toward zero.
pub fn gather_index(comptime n: usize, comptime src: usize, x: []const f64, idx: []const f64) [n]f64 {
    var out: [n]f64 = undefined;
    for (0..n) |j| out[j] = x[normalize_index(idx[j], src)];
    return out;
}

/// Row-select a row-major `(src_rows, cols)` matrix: `out[i, :] = x[idx[i], :]`.
///
/// Args:
///     n_rows: Rows in the index buffer / in the result.
///     cols: Columns per row (unchanged by the selection).
///     src_rows: Rows in `x` (comptime, for negative-index normalization).
pub fn gather_rows(comptime n_rows: usize, comptime cols: usize, comptime src_rows: usize, x: []const f64, idx: []const f64) [n_rows * cols]f64 {
    var out: [n_rows * cols]f64 = undefined;
    for (0..n_rows) |i| {
        const r = normalize_index(idx[i], src_rows);
        for (0..cols) |j| out[i * cols + j] = x[r * cols + j];
    }
    return out;
}

/// Column-select a row-major `(rows, src_cols)` matrix: `out[:, j] = x[:, idx[j]]`.
///
/// Args:
///     rows: Rows in `x` / in the result.
///     n_cols: Columns in the index buffer / in the result.
///     src_cols: Columns in `x` (comptime, for negative-index normalization).
pub fn gather_cols(comptime rows: usize, comptime n_cols: usize, comptime src_cols: usize, x: []const f64, idx: []const f64) [rows * n_cols]f64 {
    var out: [rows * n_cols]f64 = undefined;
    for (0..rows) |i| {
        for (0..n_cols) |j| {
            out[i * n_cols + j] = x[i * src_cols + normalize_index(idx[j], src_cols)];
        }
    }
    return out;
}

/// ONNX/numpy index normalization: a negative index counts from the end.
///
/// The value is sanitized *before* the f64 → int conversion, because
/// `@intFromFloat` is undefined behaviour for NaN, ±inf, and anything outside
/// `i64` range — and the index operand can be a host-fed port, not only a baked
/// constant. Clamping the float into `[-len, len-1]` first makes the conversion
/// provably in range; an out-of-range index then saturates, because defined
/// behaviour beats a panic in a deployed kernel (the importer range-checks
/// baked indices, so the saturation path is not the intended one).
fn normalize_index(raw: f64, comptime len: usize) usize {
    if (comptime len == 0) return 0; // no axis to select from
    const n: i64 = @intCast(len);
    const nf: f64 = @floatFromInt(n);
    if (std.math.isNan(raw)) return 0; // NaN has no integer part; define it as the first element
    const bounded = std.math.clamp(raw, -nf, nf - 1.0); // also maps ±inf to the ends
    const k: i64 = @intFromFloat(bounded); // now provably within [-len, len-1]
    const wrapped = if (k < 0) k + n else k;
    return @intCast(wrapped); // provably within [0, len-1]
}


/// Layer normalization over the last axis of a flat row-major matrix.
///
/// Treats the input as ``(rows, cols)`` = ``(samples, features)``: for each row
/// ``i``, ``out[i][j] = (a[i][j] - mean_i) * rstd_i * scale[j] + bias[j]`` with
/// ``mean_i = (1/cols) Σ_j a[i][j]``, ``var_i = (1/cols) Σ_j (a[i][j] - mean_i)²``
/// and ``rstd_i = 1 / sqrt(var_i + eps)``. Shapes and ``eps`` are comptime, so
/// every array is a fixed-size stack value and ``eps`` constant-folds.
///
/// This is the canonical torch ``nn.LayerNorm`` / ONNX ``LayerNormalization``
/// form: the **biased** (population) variance — divide by ``cols``, not
/// ``cols - 1`` — and ``eps`` added under the square root. ``scale``/``bias``
/// are the ONNX ``Scale``/``B`` operands, one entry per feature (``(cols,)``).
/// A 1-D vector is a single row: callers pass ``rows = 1, cols = n``.
///
/// The numpy mirror in ``shinro.codegen.ops._layernorm`` must evaluate the same
/// expression in the same order (``(x - mean) * (1/sqrt(var + eps)) * scale +
/// bias``) so the three-way oracle (numpy / interpreter / .so) agrees.
pub fn layernorm_rows(comptime rows:usize, comptime cols:usize, a:[]const f64, scale:[]const f64, bias:[]const f64, comptime eps:f64) [rows*cols]f64 {
    var out:[rows*cols]f64 = undefined;
    for (0..rows) |i| {
        //stride is the same as column lengths since matrices are row major
        // finding the mean= sum(a)/n, where n is the col size
        var sum: f64=0.0;
        for (0..cols) |j| {
            sum+=a[i*cols+j];
        }
        const mean:f64= sum/@as(f64, @floatFromInt(cols));

        // finding the variance from the selected columns
        var variance: f64=0.0;
        for (0..cols) |j| {
            const dev=a[i*cols+j]-mean;
            variance+=dev*dev;
        }
        const rstd= 1/(std.math.sqrt(variance/@as(f64, @floatFromInt(cols))+eps));

        // normalize maths done, now time to account for scale and bias

        for (0..cols) |j| {
            out[i*cols+j]=(a[i*cols+j]-mean)*rstd*scale[j]+bias[j];
        }
    }
    return out;
}