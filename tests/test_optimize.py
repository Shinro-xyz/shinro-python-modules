"""Tests for :mod:`shinro.codegen.optimize` — compile-time constant folding.

The invariant: folding is **bit-exact**. A folded graph must interpret to the
same arrays as the original, because the fold evaluates intrinsic nodes with the
same numpy handlers ``interpret()`` uses. These tests pin folding, dedup, DCE,
the non-foldable ops, and determinism.
"""

from __future__ import annotations

import numpy as np

from shinro.codegen.compose import ComposedGraph
from shinro.codegen.interpreter import interpret
from shinro.codegen.optimize import canonicalize, fold_constants
from shinro.codegen.tracing import Graph


def _cg(graph: Graph, inputs: list[str], outputs: list[str]) -> ComposedGraph:
    return ComposedGraph(graph=graph, inputs=inputs, outputs=outputs)


def _ops(cg: ComposedGraph) -> list[str]:
    return [n.op for n in cg.graph.nodes]


class TestFoldConstants:
    def test_folds_all_constant_subgraph(self):
        g = Graph()
        c1 = g.const(np.array([[1.0, 2.0], [3.0, 4.0]]))
        c2 = g.const(np.array([[5.0], [6.0]]))
        prod = g.emit("matmul", [c1, c2], (2, 1))
        x = g.input("x", (2, 1))
        s = g.emit("add", [prod, x], (2, 1))
        g.output("y", s)

        folded = fold_constants(_cg(g, ["x"], ["y"]))
        # c1, c2 become dead (their only consumer was folded) and are dropped.
        assert len(folded.graph.nodes) < len(g.nodes)
        assert _ops(folded).count("const") == 1
        assert "matmul" not in _ops(folded)

        feed = {"x": np.array([[10.0], [20.0]])}
        np.testing.assert_array_equal(interpret(folded.graph, feed)["y"], interpret(g, feed)["y"])

    def test_bit_exact_with_transcendentals(self):
        # `inv`/`tanh` fold through the same numpy handlers interpret uses, so
        # even non-trivial arithmetic is byte-identical.
        g = Graph()
        c = g.const(np.array([[2.0, 0.0], [0.0, 3.0]]))
        inv = g.emit("inv", [c], (2, 2))
        t = g.emit("tanh", [inv], (2, 2))
        g.output("t", t)

        folded = fold_constants(_cg(g, [], ["t"]))
        assert _ops(folded) == ["const", "output"]
        assert np.array_equal(interpret(folded.graph, {})["t"], interpret(g, {})["t"])

    def test_dedupes_identical_constants(self):
        g = Graph()
        a = g.const(np.array([1.0, 2.0]))
        b = g.const(np.array([1.0, 2.0]))
        x = g.input("x", (2,))
        g.output("y1", g.emit("add", [a, x], (2,)))
        g.output("y2", g.emit("add", [b, x], (2,)))

        folded = fold_constants(_cg(g, ["x"], ["y1", "y2"]))
        assert _ops(folded).count("const") == 1

        feed = {"x": np.array([5.0, 6.0])}
        np.testing.assert_array_equal(interpret(folded.graph, feed)["y1"], interpret(g, feed)["y1"])

    def test_dedup_keeps_distinct_shapes_separate(self):
        # Same flat bytes, different shape: must not merge (rows/cols differ).
        g = Graph()
        a = g.const(np.array([[1.0, 2.0, 3.0, 4.0]]))  # (1, 4)
        b = g.const(np.array([[1.0], [2.0], [3.0], [4.0]]))  # (4, 1)
        x = g.input("x", (4, 4))
        g.output("y", g.emit("add", [g.emit("add", [a, x], (4, 4)), b], (4, 4)))

        folded = fold_constants(_cg(g, ["x"], ["y"]))
        assert _ops(folded).count("const") == 2

    def test_dce_drops_unreachable_const(self):
        g = Graph()
        g.const(np.array([9.0, 9.0]))  # never consumed
        x = g.input("x", (2,))
        g.output("y", g.emit("add", [x, x], (2,)))

        folded = fold_constants(_cg(g, ["x"], ["y"]))
        assert _ops(folded) == ["input", "add", "output"]

    def test_input_and_solve_qp_are_not_folded(self):
        g = Graph()
        x = g.input("x", (2,))
        q = g.const(np.array([1.0, 2.0]))
        sol = g.emit(
            "solve_qp",
            [q],
            (2,),
            H=np.eye(2),
            A=np.eye(2),
            lb=np.array([-1.0, -1.0]),
            ub=np.array([1.0, 1.0]),
        )
        g.output("u", sol)
        g.output("xout", x)

        folded = fold_constants(_cg(g, ["x"], ["u", "xout"]))
        ops = _ops(folded)
        assert "input" in ops and "solve_qp" in ops

    def test_recurrent_state_ports_are_not_folded(self):
        # Recurrent state enters as an `input` port and leaves as an `output`
        # port. Neither may fold, so a state-dependent subgraph stays live and
        # the recurrence is never frozen into a constant.
        g = Graph()
        w = g.const(np.array([[2.0, 0.0], [0.0, 3.0]]))
        s = g.input("state_x_hat", (2,))
        y = g.emit("matmul", [w, s], (2,))  # depends on the state input
        g.output("y", y)
        g.output("state_x_hat", y)  # recurrent output

        cg = ComposedGraph(
            graph=g,
            inputs=["state_x_hat"],
            outputs=["y"],
            state_inputs=["state_x_hat"],
            state_outputs=["state_x_hat"],
        )
        folded = fold_constants(cg)
        ops = _ops(folded)
        assert "input" in ops and "matmul" in ops  # state input + its consumer survive
        assert folded.inputs == ["state_x_hat"]
        assert folded.state_inputs == ["state_x_hat"]
        assert folded.state_outputs == ["state_x_hat"]

        feed = {"state_x_hat": np.array([1.0, 1.0])}
        out = interpret(folded.graph, feed)
        np.testing.assert_array_equal(out["y"], interpret(g, feed)["y"])
        np.testing.assert_array_equal(out["state_x_hat"], out["y"])

    def test_unused_state_input_is_kept(self):
        # An `input` with no consumer is still a DCE root: the lowerer resolves
        # every name in ComposedGraph.inputs by scanning the node table.
        g = Graph()
        g.input("state_unused", (3,))
        x = g.input("x", (3,))
        g.output("y", g.emit("add", [x, x], (3,)))

        cg = ComposedGraph(graph=g, inputs=["state_unused", "x"], outputs=["y"], state_inputs=["state_unused"])
        folded = fold_constants(cg)
        assert any(n.op == "input" for n in folded.graph.nodes)
        assert "state_unused" in folded.inputs
        out = interpret(folded.graph, {"state_unused": np.zeros(3), "x": np.ones(3)})
        np.testing.assert_array_equal(out["y"], np.array([2.0, 2.0, 2.0]))

    def test_port_lists_preserved(self):
        g = Graph()
        a = g.const(np.array([1.0, 2.0]))
        x = g.input("x", (2,))
        g.output("y", g.emit("mul", [a, x], (2,)))

        cg = ComposedGraph(graph=g, inputs=["x"], outputs=["y"], state_inputs=["s"], state_outputs=["o"])
        folded = fold_constants(cg)
        assert folded.inputs == ["x"]
        assert folded.outputs == ["y"]
        assert folded.state_inputs == ["s"]
        assert folded.state_outputs == ["o"]

    def test_deterministic(self):
        def build():
            g = Graph()
            a = g.const(np.array([1.0, 2.0]))
            b = g.const(np.array([1.0, 2.0]))
            x = g.input("x", (2,))
            g.output("y", g.emit("add", [g.emit("add", [a, x], (2,)), b], (2,)))

            return fold_constants(_cg(g, ["x"], ["y"]))

        f1, f2 = build(), build()
        assert [(n.op, n.inputs, n.shape) for n in f1.graph.nodes] == [(n.op, n.inputs, n.shape) for n in f2.graph.nodes]
        np.testing.assert_array_equal(
            interpret(f1.graph, {"x": np.zeros(2)})["y"],
            interpret(f2.graph, {"x": np.zeros(2)})["y"],
        )


class TestCanonicalize:
    """Identity-op simplification: bit-exact, no arithmetic evaluation."""

    def _run(self, g: Graph, feed: dict[str, np.ndarray]) -> ComposedGraph:
        can = canonicalize(_cg(g, list(feed), ["y"]))
        np.testing.assert_array_equal(
            interpret(can.graph, feed)["y"],
            interpret(g, feed)["y"],  # oracle: identical output
        )
        return can

    def test_mul_by_ones_both_sides(self):
        for first in (True, False):
            g = Graph()
            ones = g.const(np.ones(3))
            x = g.input("x", (3,))
            ins = [ones, x] if first else [x, ones]
            g.output("y", g.emit("mul", ins, (3,)))
            can = self._run(g, {"x": np.array([1.0, -2.0, 3.0])})
            assert "mul" not in _ops(can) and "const" not in _ops(can)  # operand now dead

    def test_div_by_ones(self):
        g = Graph()
        ones = g.const(np.ones(2))
        x = g.input("x", (2,))
        g.output("y", g.emit("div", [x, ones], (2,)))
        assert "div" not in _ops(self._run(g, {"x": np.array([4.0, 5.0])}))

    def test_add_zeros_both_sides(self):
        for first in (True, False):
            g = Graph()
            z = g.const(np.zeros(2))
            x = g.input("x", (2,))
            ins = [z, x] if first else [x, z]
            g.output("y", g.emit("add", ins, (2,)))
            assert "add" not in _ops(self._run(g, {"x": np.array([7.0, 8.0])}))

    def test_sub_zeros_and_pow_one(self):
        g = Graph()
        z = g.const(np.zeros(2))
        one = g.const(np.ones(2))
        x = g.input("x", (2,))
        s = g.emit("sub", [x, z], (2,))
        p = g.emit("pow", [s, one], (2,))
        g.output("y", p)
        assert set(_ops(self._run(g, {"x": np.array([1.5, 2.5])}))) == {"input", "output"}

    def test_copy_and_double_neg(self):
        g = Graph()
        x = g.input("x", (2,))
        n1 = g.emit("neg", [x], (2,))
        n2 = g.emit("neg", [n1], (2,))
        c = g.emit("copy", [n2], (2,))
        g.output("y", c)
        assert "neg" not in _ops(self._run(g, {"x": np.array([3.0, -4.0])}))

    def test_double_transpose(self):
        g = Graph()
        x = g.input("x", (2, 3))
        t = g.emit("transpose", [g.emit("transpose", [x], (3, 2))], (2, 3))
        g.output("y", t)
        assert "transpose" not in _ops(self._run(g, {"x": np.arange(6.0).reshape(2, 3)}))

    def test_nested_reshape_collapses(self):
        g = Graph()
        x = g.input("x", (2, 3))
        r1 = g.emit("reshape", [x], (6,), target_shape=(6,))
        r2 = g.emit("reshape", [r1], (3, 2), target_shape=(3, 2))
        g.output("y", r2)
        can = self._run(g, {"x": np.arange(6.0).reshape(2, 3)})
        assert _ops(can).count("reshape") == 1  # inner reshape collapsed away

    def test_broadcast_identity_is_not_removed(self):
        # add((1,), zeros((3,))) -> (3,); dropping the add would change the
        # shape, so the shape guard must keep it.
        g = Graph()
        z = g.const(np.zeros(3))
        x = g.input("x", (1,))
        g.output("y", g.emit("add", [x, z], (3,)))
        assert "add" in _ops(self._run(g, {"x": np.array([2.0])}))

    def test_non_identity_is_preserved(self):
        g = Graph()
        two = g.const(np.full(2, 2.0))
        x = g.input("x", (2,))
        g.output("y", g.emit("mul", [two, x], (2,)))
        can = self._run(g, {"x": np.array([1.0, 1.0])})
        assert "mul" in _ops(can)  # 2 is not an identity

    def test_port_lists_preserved(self):
        g = Graph()
        x = g.input("x", (2,))
        g.output("y", g.emit("copy", [x], (2,)))
        cg = ComposedGraph(graph=g, inputs=["x"], outputs=["y"], state_inputs=["s"], state_outputs=["o"])
        can = canonicalize(cg)
        assert (can.inputs, can.outputs, can.state_inputs, can.state_outputs) == (["x"], ["y"], ["s"], ["o"])
