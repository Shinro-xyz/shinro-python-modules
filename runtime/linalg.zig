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

pub fn matmul(comptime m: usize, comptime k: usize, comptime n: usize, a: []const f64, b: []const f64) [m * n]f64 {
    var out: [m * n]f64 = undefined;
    for (0..m) |i| {
        for (0..n) |j| {
            var s: f64 = 0.0;
            for (0..k) |p| s += a[i * k + p] * b[p * n + j];
            out[i * n + j] = s;
        }
    }
    return out;
}

pub fn matvec(comptime m: usize, comptime k: usize, a: []const f64, v: []const f64) [m]f64 {
    var out: [m]f64 = undefined;
    for (0..m) |i| {
        var s: f64 = 0.0;
        for (0..k) |p| s += a[i * k + p] * v[p];
        out[i] = s;
    }
    return out;
}

pub fn vecmat(comptime k: usize, comptime n: usize, v: []const f64, b: []const f64) [n]f64 {
    var out: [n]f64 = undefined;
    for (0..n) |j| {
        var s: f64 = 0.0;
        for (0..k) |p| s += v[p] * b[p * n + j];
        out[j] = s;
    }
    return out;
}

/// Gauss-Jordan inverse with partial pivoting. Returns the inverse of the n×n
/// row-major matrix at `a` as a flat [n*n]f64.
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
