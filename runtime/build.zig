const std = @import("std");

/// Resolve a path for the build: relative paths are rooted at the build root
/// (runtime/), absolute paths (e.g. pytest tmp dirs) are used as-is.
fn lazyPath(b: *std.Build, p: []const u8) std.Build.LazyPath {
    if (std.fs.path.isAbsolute(p)) {
        return .{ .cwd_relative = p };
    }
    return b.path(p);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options: which generated graph and which baked OSQP solver to
    // compile in. Defaults reproduce the shipped single-instance layout
    // (runtime/graph_data.zig + runtime/codegen/emosqp/); passing either lets
    // a build consume a different graph/solver pair without clobbering the
    // shared paths (e.g. the MPC_DeltaU bake, n_vars=45).
    const graph_path = b.option(
        []const u8,
        "graph",
        "Path to the generated graph_data.zig (default: graph_data.zig)",
    ) orelse "graph_data.zig";
    const solver_dir = b.option(
        []const u8,
        "solver_dir",
        "Directory of the baked OSQP codegen solver (default: codegen/emosqp)",
    ) orelse "codegen/emosqp";

    // Generated OSQP codegen static solver (runtime/codegen/emosqp/ by
    // default). The deployment path: the generated C is compiled straight
    // into the binaries (no libosqp.so, no malloc). Regenerated for a
    // specific MPC problem by scripts/gen_emosqp_test.py — the `.solve_qp`
    // VM op's q length must match the baked n_vars (enforced at comptime via
    // solver_meta.zig).
    const emosqp_include_public = lazyPath(b, b.pathJoin(&.{ solver_dir, "inc", "public" }));
    const emosqp_include_private = lazyPath(b, b.pathJoin(&.{ solver_dir, "inc", "private" }));
    const emosqp_inc = lazyPath(b, solver_dir);
    const emosqp_srcs = [_][]const u8{
        b.pathJoin(&.{ solver_dir, "workspace.c" }),
        b.pathJoin(&.{ solver_dir, "src", "algebra_libs.c" }),
        b.pathJoin(&.{ solver_dir, "src", "auxil.c" }),
        b.pathJoin(&.{ solver_dir, "src", "csc_math.c" }),
        b.pathJoin(&.{ solver_dir, "src", "csc_utils.c" }),
        b.pathJoin(&.{ solver_dir, "src", "error.c" }),
        b.pathJoin(&.{ solver_dir, "src", "kkt.c" }),
        b.pathJoin(&.{ solver_dir, "src", "matrix.c" }),
        b.pathJoin(&.{ solver_dir, "src", "osqp_api.c" }),
        b.pathJoin(&.{ solver_dir, "src", "qdldl.c" }),
        b.pathJoin(&.{ solver_dir, "src", "qdldl_interface.c" }),
        b.pathJoin(&.{ solver_dir, "src", "scaling.c" }),
        b.pathJoin(&.{ solver_dir, "src", "util.c" }),
        b.pathJoin(&.{ solver_dir, "src", "vector.c" }),
    };

    // The Zig-side emosqp oracle test is pinned to the SHIPPED default bake
    // (runtime/codegen/emosqp/): its oracle vectors (tests/emosqp_data.zig)
    // are generated for that problem, so it must not follow -Dsolver_dir.
    const default_emosqp_srcs = [_][]const u8{
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
    // lower.zig is the module root; @import("linalg.zig") and @import("qp.zig")
    // resolve next to it, while the generated graph and the bake's n_vars
    // arrive as anonymous imports selected by -Dgraph / -Dsolver_dir. The
    // `.solve_qp` op compiles the codegen static solver into the library.
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("lower.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    lib_mod.addAnonymousImport("graph_data", .{ .root_source_file = lazyPath(b, graph_path) });
    lib_mod.addAnonymousImport("solver_meta", .{ .root_source_file = lazyPath(b, b.pathJoin(&.{ solver_dir, "solver_meta.zig" })) });
    lib_mod.addIncludePath(emosqp_include_public);
    lib_mod.addIncludePath(emosqp_include_private);
    lib_mod.addIncludePath(emosqp_inc);
    // addCSourceFile (not addCSourceFiles) so an absolute solver_dir (e.g. a
    // pytest tmp bake) is accepted — addCSourceFiles requires relative paths.
    for (emosqp_srcs) |src| {
        lib_mod.addCSourceFile(.{ .file = lazyPath(b, src), .flags = &.{} });
    }
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
    emosqp_test_mod.addIncludePath(b.path("codegen/emosqp/inc/public"));
    emosqp_test_mod.addIncludePath(b.path("codegen/emosqp/inc/private"));
    emosqp_test_mod.addIncludePath(b.path("codegen/emosqp"));
    emosqp_test_mod.addCSourceFiles(.{ .files = &default_emosqp_srcs, .flags = &.{} });
    const emosqp_tests = b.addTest(.{ .root_module = emosqp_test_mod });
    const run_emosqp_tests = b.addRunArtifact(emosqp_tests);

    const test_step = b.step("test", "Run Zig unit tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_emosqp_tests.step);
}
