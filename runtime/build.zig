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
    // Default is Debug; pass -Doptimize for a production build, e.g.:
    //   zig build -Doptimize=ReleaseSafe --build-file runtime/build.zig --prefix build/release/
    //   zig build -Doptimize=ReleaseFast --build-file runtime/build.zig --prefix build/release/
    // The manifest (libbase.manifest.json) records the optimize mode either way.
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
        // Release builds ship without DWARF: debug info is ~80% of the .so
        // (measured: 2.9-3.2 MB of 3.5-3.8 MB). Keep it in Debug for dev.
        .strip = optimize != .Debug,
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

    // Build manifest (audit trail): a deterministic report of what this .so
    // contains, written next to the artifact after every build, plus a
    // timestamped archive copy under <prefix>/manifests/ so teams can browse
    // which controller combinations were built and when.
    writeManifest(b, target, optimize, graph_path, solver_dir);
}

// ─── build manifest (audit trail) ─────────────────────────────────────────

/// Resolve a path to a real filesystem path (absolute as-is, relative against
/// the build root) for reading at build time.
fn resolvePath(b: *std.Build, p: []const u8) []const u8 {
    if (std.fs.path.isAbsolute(p)) return p;
    return std.fs.path.join(b.allocator, &.{ b.build_root.path.?, p }) catch @panic("OOM");
}

/// Read a file at build time; returns "" (with a warning) if unreadable.
fn readFile(b: *std.Build, path: []const u8) []const u8 {
    return std.Io.Dir.cwd().readFileAlloc(b.graph.io, path, b.allocator, .limited(1 << 20)) catch |err| {
        std.debug.print("warning: could not read {s}: {s}\n", .{ path, @errorName(err) });
        return "";
    };
}

/// Lowercase hex sha256 of a file's bytes ("" if unreadable).
fn sha256Hex(b: *std.Build, path: []const u8) []const u8 {
    const bytes = readFile(b, path);
    if (bytes.len == 0) return "";
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(&digest, .lower);
    return b.allocator.dupe(u8, &hex) catch @panic("OOM");
}

/// JSON-escape a string (quotes, backslash, control chars).
fn jsonEscape(b: *std.Build, s: []const u8) []const u8 {
    var out = std.ArrayList(u8).empty;
    for (s) |c| {
        switch (c) {
            '"' => out.appendSlice(b.allocator, "\\\"") catch @panic("OOM"),
            '\\' => out.appendSlice(b.allocator, "\\\\") catch @panic("OOM"),
            '\n' => out.appendSlice(b.allocator, "\\n") catch @panic("OOM"),
            '\r' => out.appendSlice(b.allocator, "\\r") catch @panic("OOM"),
            '\t' => out.appendSlice(b.allocator, "\\t") catch @panic("OOM"),
            else => out.append(b.allocator, c) catch @panic("OOM"),
        }
    }
    return out.items;
}

/// Parse the bake's solver_meta.zig into a JSON object string fragment
/// (n_vars, n_cons, eps, config). Missing fields are simply absent.
fn solverMetaJson(b: *std.Build, solver_dir: []const u8) []const u8 {
    const meta_path = std.fs.path.join(b.allocator, &.{ resolvePath(b, solver_dir), "solver_meta.zig" }) catch @panic("OOM");
    const text = readFile(b, meta_path);
    var out = std.ArrayList(u8).empty;
    var first = true;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");
        if (!std.mem.startsWith(u8, trimmed, "pub const ")) continue;
        const rest = trimmed["pub const ".len..];
        const eq = std.mem.indexOfScalar(u8, rest, '=') orelse continue;
        // name is the identifier before the first ':' (e.g. "n_vars" in
        // "n_vars: usize = 30;").
        const name_end = std.mem.indexOfScalar(u8, rest, ':') orelse eq;
        const name = std.mem.trim(u8, rest[0..name_end], " ");
        const val = std.mem.trim(u8, rest[eq + 1 ..], " ;");
        if (!first) out.append(b.allocator, ',') catch @panic("OOM");
        first = false;
        out.appendSlice(b.allocator, "\"") catch @panic("OOM");
        out.appendSlice(b.allocator, name) catch @panic("OOM");
        out.appendSlice(b.allocator, "\": ") catch @panic("OOM");
        if (std.mem.startsWith(u8, val, "\"") and val.len >= 2) {
            out.appendSlice(b.allocator, "\"") catch @panic("OOM");
            out.appendSlice(b.allocator, jsonEscape(b, val[1 .. val.len - 1])) catch @panic("OOM");
            out.appendSlice(b.allocator, "\"") catch @panic("OOM");
        } else {
            out.appendSlice(b.allocator, val) catch @panic("OOM");
        }
    }
    return out.items;
}

/// Load the graph manifest emitted by lower_zig next to the graph file
/// (<stem>_manifest.json) as a raw JSON object string. "" if absent.
fn graphManifestJson(b: *std.Build, graph_path: []const u8) []const u8 {
    const resolved = resolvePath(b, graph_path);
    const dir = std.fs.path.dirname(resolved) orelse ".";
    const stem = std.fs.path.stem(resolved);
    const manifest_path = std.fmt.allocPrint(b.allocator, "{s}/{s}_manifest.json", .{ dir, stem }) catch @panic("OOM");
    return readFile(b, manifest_path);
}

/// UTC timestamp for the archive filename (YYYY-MM-DDTHHMMSSZ).
fn utcTimestamp(b: *std.Build) []const u8 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.REALTIME, &ts);
    const secs: u64 = @intCast(ts.sec);
    const epoch = std.time.epoch.EpochSeconds{ .secs = secs };
    const yd = epoch.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = epoch.getDaySeconds();
    return std.fmt.allocPrint(
        b.allocator,
        "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}{d:0>2}{d:0>2}Z",
        .{ yd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour(), ds.getSecondsIntoMinute() },
    ) catch @panic("OOM");
}

/// Write the deterministic build report next to the artifact and a
/// timestamped archive copy under <prefix>/manifests/.
fn writeManifest(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    graph_path: []const u8,
    solver_dir: []const u8,
) void {
    const target_triple = target.result.zigTriple(b.allocator) catch @panic("OOM");
    const optimize_name = @tagName(optimize);
    const zig_version = @import("builtin").zig_version_string;

    const graph_sha = sha256Hex(b, resolvePath(b, graph_path));
    const ws = std.fs.path.join(b.allocator, &.{ resolvePath(b, solver_dir), "workspace.c" }) catch @panic("OOM");
    const solver_sha = sha256Hex(b, ws);
    const graph_json = graphManifestJson(b, graph_path);
    const solver_json = solverMetaJson(b, solver_dir);
    const graph_field = if (graph_json.len > 0) graph_json else "null";

    const json_text = std.fmt.allocPrint(
        b.allocator,
        "{{\n" ++
            "  \"target\": \"{s}\",\n" ++
            "  \"optimize\": \"{s}\",\n" ++
            "  \"stripped\": {s},\n" ++
            "  \"zig_version\": \"{s}\",\n" ++
            "  \"libc\": true,\n" ++
            "  \"float_type\": \"f64\",\n" ++
            "  \"provenance\": {{\n" ++
            "    \"graph_path\": \"{s}\",\n" ++
            "    \"solver_dir\": \"{s}\",\n" ++
            "    \"graph_sha256\": \"{s}\",\n" ++
            "    \"solver_sha256\": \"{s}\"\n" ++
            "  }},\n" ++
            "  \"solver\": {{ {s} }},\n" ++
            "  \"graph\": {s}\n" ++
            "}}\n",
        .{
            jsonEscape(b, target_triple),
            jsonEscape(b, optimize_name),
            if (optimize == .Debug) "false" else "true",
            jsonEscape(b, zig_version),
            jsonEscape(b, graph_path),
            jsonEscape(b, solver_dir),
            graph_sha,
            solver_sha,
            solver_json,
            graph_field,
        },
    ) catch @panic("OOM");

    const cwd = std.Io.Dir.cwd();
    const lib_dir = std.fs.path.join(b.allocator, &.{ b.install_prefix, "lib" }) catch @panic("OOM");
    cwd.createDirPath(b.graph.io, lib_dir) catch {};
    const report_path = std.fs.path.join(b.allocator, &.{ lib_dir, "libbase.manifest.json" }) catch @panic("OOM");
    cwd.writeFile(b.graph.io, .{ .sub_path = report_path, .data = json_text }) catch |err| {
        std.debug.print("warning: could not write manifest {s}: {s}\n", .{ report_path, @errorName(err) });
    };

    const manifests_dir = std.fs.path.join(b.allocator, &.{ b.install_prefix, "manifests" }) catch @panic("OOM");
    cwd.createDirPath(b.graph.io, manifests_dir) catch {};
    const sha8 = if (graph_sha.len >= 8) graph_sha[0..8] else "nosha";
    const archive_path = std.fmt.allocPrint(b.allocator, "{s}/{s}-{s}.json", .{ manifests_dir, utcTimestamp(b), sha8 }) catch @panic("OOM");
    cwd.writeFile(b.graph.io, .{ .sub_path = archive_path, .data = json_text }) catch |err| {
        std.debug.print("warning: could not write archive manifest {s}: {s}\n", .{ archive_path, @errorName(err) });
    };
}
