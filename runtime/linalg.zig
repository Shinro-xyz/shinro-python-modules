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
pub fn matvec(comptime m: usize, comptime k: usize, a: []const f64, v: []const f64) [m]f64 {
    var out: [m]f64 = undefined;
    for (0..m) |i| {
        var s: f64 = 0.0;
        for (0..k) |p| {
            s += a[i * k + p] * v[p];
        }
        out[i] = s;
    }
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


// sine for an array of values

pub fn sin_vec (comptime m: usize, a:[]const f64) [m]f64{
    var out: [m]f64= undefined;
    for (0..m) |i| {
        out[i]=std.math.sin(a[i]);
    }
    return out;
}

///cosine for an array of values

pub fn cos_vec(comptime m: usize, a:[]const f64) [m]f64{
    var out: [m]f64 = undefined;
    for (0..m) |i| {
        out[i]=std.math.cos(a[i]);
    }
    return out;
}

/// ReLU

pub fn relu(comptime m:usize, a:[]const f64) [m]f64{
    var out: [m]f64=undefined;
    for (0..m) |i| {
        out[i]= @max(0.0,a[i]);
    }
    return out;
}

/// elementwise exponential
pub fn elementwise_exponential (comptime m:usize,a:[]const f64) [m]f64{
    var out: [m]f64=undefined;

    for (0..m) |i| {
        out[i]= std.math.exp(a[i]);
    }
    return out;
}

/// matrix exponential-> only allows for nxn square matrices
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

pub fn tanh (comptime m: usize, a: []const f64) [m]f64 {
    var result: [m]f64= undefined;

    for (0..m) |i| {
        result[i]= std.math.tanh(a[i]);
    }
    return result;
}

//argmax--> what arg is the highest value

pub fn argmax (comptime m: usize, a:[]const f64) usize {
    var best: usize = 0;
    for (0..m) |i| {
        if (a[i]>a[best]) {
            best=i;
        }
    }
    return best;
}

//one hot
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
