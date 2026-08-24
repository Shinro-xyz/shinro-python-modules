const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // linalg as a reusable module — imported by both the shared lib and tests.
    const linalg_mod = b.createModule(.{
        .root_source_file = b.path("linalg.zig"),
        .target = target,
        .optimize = optimize,
    });

    // base.so — the C-ABI graph VM the Python host dlopen-s via ctypes.
    // lower.zig is the module root; its relative @import("graph_data.zig")
    // and @import("linalg.zig") resolve next to it automatically.
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("lower.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const lib = b.addLibrary(.{
        .name = "base",
        .root_module = lib_mod,
        .linkage = .dynamic,
    });
    b.installArtifact(lib);

    // Zig-side unit tests. Only linalg.zig for now; future test files slot
    // in as additional test modules under runtime/tests/.
    const test_mod = b.createModule(.{
        .root_source_file = b.path("tests/linalg.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "linalg", .module = linalg_mod }},
    });
    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run Zig unit tests");
    test_step.dependOn(&run_tests.step);
}
