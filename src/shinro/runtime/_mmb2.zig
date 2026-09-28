const std = @import("std");
const la = @import("linalg.zig");
var sink: f64 = 0;
fn now_ns() u64 { var ts: std.os.linux.timespec = undefined; _ = std.os.linux.clock_gettime(.MONOTONIC, &ts); return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec)); }
fn scalarMatmul(comptime m: usize, comptime k: usize, comptime n: usize, a: []const f64, b: []const f64) [m * n]f64 {
    var out: [m * n]f64 = undefined;
    for (0..m) |i| for (0..n) |j| { var s: f64 = 0.0; for (0..k) |p| s += a[i * k + p] * b[p * n + j]; out[i * n + j] = s; };
    return out;
}
fn bench(comptime m: usize, comptime k: usize, comptime n: usize, iters: usize) !void {
    var a: [m * k]f64 = undefined; var b: [k * n]f64 = undefined; var s: f64 = 0.1;
    for (&a) |*x| { s += 0.7; x.* = s; } for (&b) |*x| { s += 0.3; x.* = s; } var seed: f64 = 1.0;
    const t0 = now_ns();
    for (0..iters) |_| { a[0] = @as(*volatile f64, &seed).*; const r = scalarMatmul(m, k, n, &a, &b); sink += r[(m * n) / 2]; std.mem.doNotOptimizeAway(&r); }
    const ns_s = now_ns() - t0;
    const t1 = now_ns();
    for (0..iters) |_| { a[0] = @as(*volatile f64, &seed).*; const r = la.matmul(m, k, n, &a, &b); sink += r[(m * n) / 2]; std.mem.doNotOptimizeAway(&r); }
    const ns_v = now_ns() - t1;
    const per = @as(f64, @floatFromInt(iters));
    std.debug.print("{d:>3}x{d:<4}x{d:<4} scalar {d:>10.1} ns | simd {d:>10.1} ns | speedup {d:>5.2}x\n", .{ m, k, n, @as(f64, @floatFromInt(ns_s)) / per, @as(f64, @floatFromInt(ns_v)) / per, @as(f64, @floatFromInt(ns_s)) / @as(f64, @floatFromInt(ns_v)) });
}
pub fn main() !void {
    for (0..2) |_| {
        try bench(1, 128, 128, 1_000_000);
        try bench(128, 128, 128, 100_000);
    }
    std.mem.doNotOptimizeAway(sink);
}
