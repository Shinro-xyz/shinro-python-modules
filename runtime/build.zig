const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Generated OSQP codegen static solver (runtime/codegen/emosqp/). The
    // deployment path: the generated C is compiled straight into the binaries
    // (no libosqp.so, no malloc). Regenerated for a specific MPC problem by
    // scripts/gen_emosqp_test.py — the `.solve_qp` VM op's q length must match
    // the baked n_vars.
    const emosqp_include_public = b.path("codegen/emosqp/inc/public");
    const emosqp_include_private = b.path("codegen/emosqp/inc/private");
    const emosqp_inc = b.path("codegen/emosqp");
    const emosqp_srcs = [_][]const u8{
        "codegen/emosqp/workspace.c",
        "codegen/emosqp/src/algebra_libs.c",
        "codegen/emosqp/src/auxil.c",
        "codegen/emosqp/src/csc_math.c",
        "codegen/emosqp/src/csc_utils.c",
        "codegen/emosqp/src/error.c",
        "codegen/emosqp/src/kkt.c",
        "codegen/emosqp/src/matrix.c",
        "codegen/emosqp/src/osqp_api.c",
        "codegen/emosqp/src/qdldl.c",
        "codegen/emosqp/src/qdldl_interface.c",
        "codegen/emosqp/src/scaling.c",
        "codegen/emosqp/src/util.c",
        "codegen/emosqp/src/vector.c",
    };

    // linalg as a reusable module — imported by both the shared lib and tests.
    const linalg_mod = b.createModule(.{
        .root_source_file = b.path("linalg.zig"),
        .target = target,
        .optimize = optimize,
    });

    // base.so — the C-ABI graph VM the Python host dlopen-s via ctypes.
    // lower.zig is the module root; its relative @import("graph_data.zig"),
    // @import("linalg.zig"), and @import("qp.zig") resolve next to it. The
    // `.solve_qp` op compiles the codegen static solver into the library.
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("lower.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    lib_mod.addIncludePath(emosqp_include_public);
    lib_mod.addIncludePath(emosqp_include_private);
    lib_mod.addIncludePath(emosqp_inc);
    lib_mod.addCSourceFiles(.{ .files = &emosqp_srcs, .flags = &.{} });
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

    // OSQP codegen static solver test — compiles the generated C
    // (runtime/codegen/emosqp/) into the test and drives the static solver.
    const emosqp_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/emosqp.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    emosqp_test_mod.addIncludePath(emosqp_include_public);
    emosqp_test_mod.addIncludePath(emosqp_include_private);
    emosqp_test_mod.addIncludePath(emosqp_inc);
    emosqp_test_mod.addCSourceFiles(.{ .files = &emosqp_srcs, .flags = &.{} });
    const emosqp_tests = b.addTest(.{ .root_module = emosqp_test_mod });
    const run_emosqp_tests = b.addRunArtifact(emosqp_tests);

    const test_step = b.step("test", "Run Zig unit tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_emosqp_tests.step);
}
