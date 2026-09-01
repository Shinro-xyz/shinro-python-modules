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
