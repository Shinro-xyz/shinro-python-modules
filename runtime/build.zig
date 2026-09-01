const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Vendored OSQP (built from the v1.0.0 source; header + libosqp.so).
    const osqp_include = b.path("../third_party/osqp/inc/public");
    const osqp_lib = b.path("../third_party/osqp");

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
    lib_mod.addIncludePath(osqp_include);
    lib_mod.addLibraryPath(osqp_lib);
    lib_mod.linkSystemLibrary("osqp", .{});
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

    // OSQP binding smoke test + benchmark.
    const osqp_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/osqp.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    osqp_test_mod.addIncludePath(osqp_include);
    osqp_test_mod.addLibraryPath(osqp_lib);
    osqp_test_mod.linkSystemLibrary("osqp", .{});
    const osqp_tests = b.addTest(.{ .root_module = osqp_test_mod });
    const run_osqp_tests = b.addRunArtifact(osqp_tests);

    // MPC QP oracle test — the base MPC problem solved via OSQP, compared
    // against the Python interpreter.
    const mpc_qp_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/mpc_qp.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mpc_qp_test_mod.addIncludePath(osqp_include);
    mpc_qp_test_mod.addLibraryPath(osqp_lib);
    mpc_qp_test_mod.linkSystemLibrary("osqp", .{});
    const mpc_qp_tests = b.addTest(.{ .root_module = mpc_qp_test_mod });
    const run_mpc_qp_tests = b.addRunArtifact(mpc_qp_tests);

    // OSQP codegen static solver test — compiles the generated C
    // (runtime/codegen/emosqp/) into the test and drives the static solver.
    const emosqp_test_mod = b.createModule(.{
        .root_source_file = b.path("tests/emosqp.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    emosqp_test_mod.addIncludePath(b.path("codegen/emosqp/inc/public"));
    emosqp_test_mod.addIncludePath(b.path("codegen/emosqp/inc/private"));
    emosqp_test_mod.addIncludePath(b.path("codegen/emosqp"));
    emosqp_test_mod.addCSourceFiles(.{
        .files = &.{
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
        },
        .flags = &.{},
    });
    const emosqp_tests = b.addTest(.{ .root_module = emosqp_test_mod });
    const run_emosqp_tests = b.addRunArtifact(emosqp_tests);

    // OSQP benchmark — a standalone executable (not a test) so its stdout
    // doesn't corrupt the `zig build test` --listen protocol.
    const osqp_bench_mod = b.createModule(.{
        .root_source_file = b.path("bench/osqp_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    osqp_bench_mod.addIncludePath(osqp_include);
    osqp_bench_mod.addLibraryPath(osqp_lib);
    osqp_bench_mod.linkSystemLibrary("osqp", .{});
    const osqp_bench = b.addExecutable(.{ .name = "osqp_bench", .root_module = osqp_bench_mod });
    const run_osqp_bench = b.addRunArtifact(osqp_bench);
    const bench_step = b.step("bench", "Run the OSQP benchmark");
    bench_step.dependOn(&run_osqp_bench.step);

    const test_step = b.step("test", "Run Zig unit tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_osqp_tests.step);
    test_step.dependOn(&run_mpc_qp_tests.step);
    test_step.dependOn(&run_emosqp_tests.step);
}
