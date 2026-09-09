"""Exhaustive (op x shape-class) differential matrix: .so vs numpy.

The Zig VM is a handwritten dispatch with a *finite* set of code paths
(vecmat/matvec/matmul; same-shape/scalar/row/col broadcast; square/non-square
transpose; 1-D/2-D slice; ...). Every bug found in it lived in a path no test
visited — so this suite visits ALL of them: every op in the vocabulary, each
in every shape class its numpy semantics distinguish, grouped into four graphs
(one zig compile each) with every cell exposed as a named output.

Two contracts per cell:

- **Values** — 10 seeded random feeds; each named output must match
  ``interpret()`` to 1e-12. A wrong shape on the .so side corrupts the packed
  buffers, so value agreement pins the runtime layout too.
- **Shapes** — the manifest's declared output shape for each cell must equal
  the numpy result's shape (the .so has no runtime shapes; flat buffers are
  its contract, so shapes are pinned on the manifest axis).

Level: raw ``Graph`` emission (direct op control). The composed component path
is covered separately by the plant scan and the bench's closed-loop suite.
``solve_qp`` is excluded (bake-coupled; oracle-tested at base dims by the MPC
tests).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_zig_lowering import _build_so, _output_split, _step

from shinro.codegen import interpret
from shinro.codegen.compose import ComposedGraph
from shinro.codegen.tracing import Graph

TOL = 1e-12
N_FEEDS = 10


# ─── graph builders: each returns (outputs, port_specs) ─────────────────────
# port_specs: input port name -> (shape, kind); kind ∈ {free, pos, spd, idx}
#   free  — N(0, 1)
#   pos   — U(0.5, 2) (safe divisors)
#   spd   — random SPD (safe inv operands)
#   idx   — integer-valued float in [0, depth) (one_hot index)


def _linalg_graph(g: Graph):
    """matmul dispatch classes, transpose square/non-square, inv sizes, reshape."""
    outs = {}

    a22 = g.input("a22", (2, 2))
    b22 = g.input("b22", (2, 2))
    a21 = g.input("a21", (2, 1))
    s11 = g.input("s11", (1, 1))
    v2 = g.input("v2", (2,))
    r12 = g.input("r12", (1, 2))
    a31 = g.input("a31", (3, 1))
    b14 = g.input("b14", (1, 4))
    outs["matmul_2d2d"] = g.emit("matmul", [a22, b22], (2, 2))
    outs["matmul_2dcol"] = g.emit("matmul", [a22, a21], (2, 1))
    outs["matmul_vec"] = g.emit("matmul", [v2, a22], (2,))
    outs["matmul_col_scalar"] = g.emit("matmul", [a21, s11], (2, 1))
    outs["matmul_row_vec"] = g.emit("matmul", [r12, v2], (1,))
    outs["matmul_k1_2d"] = g.emit("matmul", [a31, b14], (3, 4))

    t2 = g.input("t2", (2, 2))
    t23 = g.input("t23", (2, 3))
    t32 = g.input("t32", (3, 2))
    outs["transpose_square"] = g.emit("transpose", [t2], (2, 2))
    outs["transpose_23"] = g.emit("transpose", [t23], (3, 2))
    outs["transpose_32"] = g.emit("transpose", [t32], (2, 3))

    i1 = g.input("i1", (1, 1))
    i2 = g.input("i2", (2, 2))
    i4 = g.input("i4", (4, 4))
    outs["inv_1x1"] = g.emit("inv", [i1], (1, 1))
    outs["inv_2x2"] = g.emit("inv", [i2], (2, 2))
    outs["inv_4x4"] = g.emit("inv", [i4], (4, 4))

    r6 = g.input("r6", (6,))
    r23 = g.input("r23", (2, 3))
    # the interpreter's reshape handler reads attrs["target_shape"] (the
    # tracer records it); without it numpy's .reshape(None) flattens — which
    # the shape contract below would catch.
    outs["reshape_1d_to_2d"] = g.emit("reshape", [r6], (2, 3), target_shape=(2, 3))
    outs["reshape_2d_to_1d"] = g.emit("reshape", [r23], (6,), target_shape=(6,))

    specs = {
        "a22": ((2, 2), "free"), "b22": ((2, 2), "free"),
        "a21": ((2, 1), "free"), "s11": ((1, 1), "free"),
        "v2": ((2,), "free"), "r12": ((1, 2), "free"),
        "a31": ((3, 1), "free"), "b14": ((1, 4), "free"),
        "t2": ((2, 2), "free"), "t23": ((2, 3), "free"), "t32": ((3, 2), "free"),
        "i1": ((1, 1), "spd"), "i2": ((2, 2), "spd"), "i4": ((4, 4), "spd"),
        "r6": ((6,), "free"), "r23": ((2, 3), "free"),
    }
    return outs, specs


def _elementwise_graph(g: Graph):
    """Binary ops x operand classes (same/scalar/row/col/vec-aligned/1-D),
    neg, clip bounds variants."""
    outs = {}
    specs: dict = {}

    x = g.input("x32", (3, 2))
    specs["x32"] = ((3, 2), "free")
    same = g.input("same32", (3, 2))
    specs["same32"] = ((3, 2), "free")
    scalar = g.emit("const", [], (), value=np.float64(1.5))
    row = g.emit("const", [], (1, 2), value=np.array([[0.3, -0.7]]))
    col = g.emit("const", [], (3, 1), value=np.array([[0.2], [-0.4], [0.6]]))
    vec = g.input("vec2", (2,))
    specs["vec2"] = ((2,), "free")
    v4 = g.input("v4", (4,))
    specs["v4"] = ((4,), "free")
    v4c = g.input("v4c", (4,))
    specs["v4c"] = ((4,), "free")
    v4b = g.input("v4b", (4, 1))
    specs["v4b"] = ((4, 1), "free")

    for op, oname in (("add", "add"), ("sub", "sub"), ("mul", "mul"), ("div", "div"), ("ne", "ne")):
        outs[f"{oname}_same"] = g.emit(op, [x, same], (3, 2))
        outs[f"{oname}_scalar"] = g.emit(op, [x, scalar], (3, 2))
        outs[f"{oname}_row"] = g.emit(op, [x, row], (3, 2))
        outs[f"{oname}_col"] = g.emit(op, [x, col], (3, 2))
        outs[f"{oname}_vec"] = g.emit(op, [x, vec], (3, 2))
        outs[f"{oname}_1d"] = g.emit(op, [v4, v4c], (4,))
        # (4,) vs (4,1): numpy right-aligns the 1-D operand as a row -> (4,4)
        outs[f"{oname}_1d_col"] = g.emit(op, [v4, v4b], (4, 4))

    neg2 = g.input("neg2", (3, 2))
    specs["neg2"] = ((3, 2), "free")
    outs["neg_2d"] = g.emit("neg", [neg2], (3, 2))
    outs["neg_1d"] = g.emit("neg", [v4], (4,))

    c4 = g.input("c4", (4,))
    specs["c4"] = ((4,), "free")
    c23 = g.input("c23", (2, 3))
    specs["c23"] = ((2, 3), "free")
    outs["clip_array"] = g.emit("clip", [c4], (4,), lo=np.full(4, -0.5), hi=np.full(4, 0.5))
    outs["clip_scalar"] = g.emit("clip", [c4], (4,), lo=np.float64(-0.5), hi=np.float64(0.5))
    outs["clip_2d_scalar"] = g.emit("clip", [c23], (2, 3), lo=np.float64(-0.4), hi=np.float64(0.4))

    return outs, specs


def _selection_graph(g: Graph):
    """where variants, slice 1-D/2-D, stack, copy, any."""
    outs = {}

    x32 = g.input("x32", (3, 2))
    x3 = g.input("x3", (3,))
    one = g.emit("const", [], (), value=np.float64(1.0))
    zero = g.emit("const", [], (), value=np.float64(0.0))
    cond32 = g.emit("ne", [x32, g.emit("const", [], (3, 2), value=np.full((3, 2), 0.25))], (3, 2))
    bias12 = g.emit("const", [], (1, 2), value=np.array([[1.0, -2.0]]))

    outs["where_same"] = g.emit("where", [cond32, x32, g.emit("const", [], (3, 2), value=np.full((3, 2), 9.0))], (3, 2))
    outs["where_scalar_branch"] = g.emit("where", [g.emit("ne", [x3, zero], (3,)), one, x3], (3,))
    outs["where_row_broadcast"] = g.emit("where", [cond32, bias12, x32], (3, 2))

    s6 = g.input("s6", (6,))
    s42 = g.input("s42", (4, 2))
    outs["slice_1d"] = g.emit("slice", [s6], (3,), start=2, stop=5)
    outs["slice_2d_rows"] = g.emit("slice", [s42], (2, 2), start=1, stop=3)

    sa = g.input("stack_a", (2,))
    sb = g.input("stack_b", (2,))
    sc = g.input("stack_c", (2,))
    outs["stack3"] = g.emit("stack", [sa, sb, sc], (3, 2))

    cp = g.input("cp32", (3, 2))
    outs["copy_2d"] = g.emit("copy", [cp], (3, 2))

    a4 = g.input("any4", (4,))
    a23 = g.input("any23", (2, 3))
    outs["any_1d"] = g.emit("any", [a4], ())
    outs["any_2d"] = g.emit("any", [a23], ())

    specs = {
        "x32": ((3, 2), "free"), "x3": ((3,), "free"),
        "s6": ((6,), "free"), "s42": ((4, 2), "free"),
        "stack_a": ((2,), "free"), "stack_b": ((2,), "free"), "stack_c": ((2,), "free"),
        "cp32": ((3, 2), "free"), "any4": ((4,), "free"), "any23": ((2, 3), "free"),
    }
    return outs, specs


def _pointwise_graph(g: Graph):
    """tanh/relu/exp/sin/cos, argmax, one_hot."""
    outs = {}
    x1 = g.input("p1", (1,))
    x8 = g.input("p8", (8,))
    for op in ("tanh", "relu", "exp", "sin", "cos"):
        outs[f"{op}_1"] = g.emit(op, [x1], (1,))
        outs[f"{op}_8"] = g.emit(op, [x8], (8,))

    am5 = g.input("am5", (5,))
    am23 = g.input("am23", (2, 3))
    outs["argmax_1d"] = g.emit("argmax", [am5], ())
    outs["argmax_2d"] = g.emit("argmax", [am23], ())

    oh = g.input("oh_idx", (1,))
    outs["one_hot_4"] = g.emit("one_hot", [oh], (4,), depth=4)

    specs = {
        "p1": ((1,), "free"), "p8": ((8,), "free"),
        "am5": ((5,), "free"), "am23": ((2, 3), "free"),
        "oh_idx": ((1,), "idx"),
    }
    return outs, specs


GRAPHS = {
    "linalg": (_linalg_graph, 11),
    "elementwise": (_elementwise_graph, 23),
    "selection": (_selection_graph, 31),
    "pointwise": (_pointwise_graph, 41),
}


def _feeds(specs: dict, n: int, seed: int) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    feeds = []
    for _ in range(n):
        feed = {}
        for name, (shape, kind) in specs.items():
            if kind == "pos":
                feed[name] = rng.uniform(0.5, 2.0, shape)
            elif kind == "spd":
                a = rng.normal(0.0, 0.5, shape)
                feed[name] = a @ a.T + np.eye(shape[0]) * shape[0]
            elif kind == "idx":
                feed[name] = np.asarray(rng.integers(0, 4, shape), dtype=np.float64)
            else:
                feed[name] = rng.normal(0.0, 1.0, shape)
        feeds.append(feed)
    return feeds


def _boundary_feeds(specs: dict) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Deterministic value-dependent-branch feeds (complement the random ones).

    Zeros stress div's NaN path (0/0 — both engines must agree on NaN),
    ±0.5 lands exactly on the scalar-clip bounds, ±1e3 saturates tanh and
    overflows exp (inf agreement), ±1e-6 probes denormal-adjacent scaling.
    Bit-equal twins (same32 == x32, v4c == v4) exercise ne's equality branch.
    """
    ext = [0.0, 0.5, -0.5, 1e3, -1e3, 1e-6, -1e-6, 1.0]
    feeds = []

    def build(free_fill):
        feed = {}
        for name, (shape, kind) in specs.items():
            if kind == "pos":
                feed[name] = np.full(shape, 1.0)
            elif kind == "spd":
                feed[name] = np.eye(shape[0]) * shape[0]
            elif kind == "idx":
                feed[name] = np.zeros(shape)  # idx 0 (lower edge)
            else:
                feed[name] = np.full(shape, free_fill) if np.isscalar(free_fill) else np.tile(free_fill, shape)
        return feed

    feeds.append(("zeros", build(0.0)))
    feeds.append(("ones", build(1.0)))
    feeds.append(("clip_bounds", build(0.5)))
    feeds.append(("large", build(1e3)))
    feeds.append(("tiny", build(1e-6)))
    twin = build(0.7)
    if "same32" in twin:
        twin["same32"] = twin["x32"].copy()  # bit-equal -> ne must be 0
    if "v4c" in twin:
        twin["v4c"] = twin["v4"].copy()
    if "b22" in twin:
        twin["b22"] = twin["a22"].copy()
    feeds.append(("twins", twin))
    return feeds


@pytest.fixture(scope="module")
def matrix_so(tmp_path_factory, request):
    """Compile one grouped matrix graph; return (lib, cg, feeds, outs_manifest)."""
    name = request.param
    builder, seed = GRAPHS[name]
    d = tmp_path_factory.mktemp(f"matrix-{name}")
    g = Graph()
    outs, specs = builder(g)
    for oname, src in outs.items():
        g.output(oname, src)
    cg = ComposedGraph(graph=g, inputs=list(specs), outputs=list(outs))
    lib, cg2 = _build_so(cg, d / "build", graph_path=d / "graph_data.zig")
    manifest = json.loads((d / "graph_data_manifest.json").read_text())
    feeds = [(_feeds(specs, N_FEEDS, seed)[i], f"random-{i}") for i in range(N_FEEDS)]
    feeds += [(feed, tag) for (tag, feed) in _boundary_feeds(specs)]
    return name, lib, cg2, feeds, manifest


class TestOpShapeMatrix:
    """Every (op, shape-class) cell in the vocabulary: values AND shapes."""

    @pytest.mark.parametrize("matrix_so", list(GRAPHS), indirect=True, ids=list(GRAPHS))
    def test_so_matches_numpy(self, matrix_so):
        name, lib, cg, feeds, manifest = matrix_so
        n_out, n_state = _output_split(cg)
        declared = {o["name"]: tuple(o["shape"]) for o in manifest["outputs"]}

        for feed, feed_tag in feeds:
            inp = np.concatenate([np.asarray(feed[k]).ravel() for k in cg.inputs])
            out, _ = _step(lib, cg, inp, n_out, n_state)
            traced = interpret(cg.graph, dict(feed))
            off = 0
            for oname in cg.outputs:
                exp = np.asarray(traced[oname])
                # shape contract: manifest declaration == numpy result
                assert declared[oname] == exp.shape, (
                    f"{name}/{oname}: manifest shape {declared[oname]} != numpy {exp.shape}"
                )
                got = out[off : off + exp.size]
                # NaN-aware: 0/0 in the boundary feeds must agree as NaN
                ok = np.isclose(got, exp.ravel(), rtol=0.0, atol=TOL, equal_nan=True)
                if not ok.all():
                    bad = np.abs(got - exp.ravel())[~np.isnan(np.abs(got - exp.ravel()))]
                    worst = float(np.max(bad)) if bad.size else float("nan")
                    pytest.fail(f"{name}/{oname} (feed {feed_tag}): max abs err {worst:.3e}")
                off += exp.size


class TestRankGuard:
    """The lowerer refuses rank-3+ shapes loudly (they would otherwise
    silently collapse to (1, 1) and lower to garbage)."""

    def test_rank3_raises(self):
        from shinro.codegen.lower_zig import lower_zig

        g = Graph()
        v = g.emit("const", [], (2, 2, 2), value=np.zeros((2, 2, 2)))
        g.output("bad", v)
        cg = ComposedGraph(graph=g, inputs=[], outputs=["bad"])
        with pytest.raises(ValueError, match="rank-3.*1-D and 2-D"):
            lower_zig(cg, "/tmp/opencode/rank3-guard-graph.zig")
