"""MCP tools for the codegen pipeline: trace a component, compose, inspect, lower.

These register on the **codegen server** (``shinro-mcp-codegen``), not on the
component server (``shinro-mcp``), so a client designing controllers does not
carry their schemas in context.

The zig-free half of the compile pipeline (``shinro.codegen`` +
:func:`shinro.scenarios.gen.gen_scenario`), exposed as tools:

    check_component      construct a config -> inferred trace contract (no trace)
    trace_component      trace ONE component into a graph (stored by name)
    build_composed_graph trace an estimator + controller into a closed-loop step graph
    generate_scenario    the full gen stage: trace + compose + lower to an isolated dir
    graph_summary        ports / ops / node count for a stored graph
    measure_graph        lowered static size metrics (host buffers, VM stack, blobs)
    interpret_graph      run a stored graph on numpy inputs (the Python oracle)
    lower_graph          serialize a stored graph to a Zig data table + manifest
    list_codegen_ops     registered ops and graph recipes

Two hazards this module guards against:

* **Clobbering the shipped graph.** :func:`lower_zig` defaults to
  ``src/shinro/runtime/graph_data.zig`` — the file the package ships and
  ``make zig-gen`` regenerates. Every lowering tool here takes an explicit
  output directory and refuses one inside the packaged runtime dir.
* **Corrupting the stdio transport.** Nothing in this module prints: tool
  results are returned as JSON strings, because stdout *is* the JSON-RPC
  channel.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from shinro.cli.components import load_component
from shinro.codegen.compose import ComposedGraph
from shinro.codegen.interpreter import interpret
from shinro.codegen.lower_zig import lower_zig
from shinro.codegen.measure import graph_metrics
from shinro.codegen.ops import OP_HANDLERS, has_op
from shinro.codegen.runtime_paths import runtime_root
from shinro.codegen.trace_node import NodeGraph, trace_node
from shinro.mcp._common import _graph_store
from shinro.mcp._common import codegen_server as server
from shinro.scenarios.gen import gen_scenario
from shinro.scenarios.recipes import available_graphs, tool_versions

# ── helpers ───────────────────────────────────────────────────────────────────


def _ports(graph) -> tuple[list[str], list[str]]:
    """The graph's declared input and output port names, in node order."""
    inputs: list[str] = []
    outputs: list[str] = []
    for node in graph.nodes:
        if node.op == "input":
            inputs.append(node.attrs["name"])
        elif node.op == "output":
            outputs.append(node.attrs["name"])
    return inputs, outputs


def _input_shapes(graph) -> dict[str, tuple[int, ...]]:
    """Map each input port name to its declared shape."""
    return {node.attrs["name"]: node.shape for node in graph.nodes if node.op == "input"}


def _to_composed(ng: NodeGraph) -> ComposedGraph:
    """Adapt a single-component :class:`NodeGraph` to a one-step ComposedGraph.

    The composed form is what ``lower_zig`` consumes (it reads ``.inputs`` /
    ``.outputs`` / ``.state_outputs``), so wrapping here keeps one code path for
    summary / interpret / lower regardless of how the graph was captured.
    """
    inputs, outputs = _ports(ng.graph)
    return ComposedGraph(
        graph=ng.graph,
        inputs=inputs,
        outputs=[name for name in outputs if not name.startswith("state_")],
        state_inputs=[name for name in inputs if name.startswith("state_")],
        state_outputs=[name for name in outputs if name.startswith("state_")],
    )


def _summary(name: str, cg: ComposedGraph) -> dict[str, Any]:
    """Ports, op histogram, and lowerability for a stored graph."""
    inputs, outputs = _ports(cg.graph)
    ops: dict[str, int] = {}
    for node in cg.graph.nodes:
        ops[node.op] = ops.get(node.op, 0) + 1
    unknown = sorted(op for op in ops if not has_op(op))
    return {
        "name": name,
        "kind": "composed" if isinstance(cg, ComposedGraph) else "node",
        # The captured graph, BEFORE lowering folds constants/canonicalizes —
        # this is what interpret() (and gate A) run on. The lowered table the VM
        # compiles can be smaller; lowering tools report that as nodes_lowered.
        "nodes_traced": len(cg.graph.nodes),
        "ops": dict(sorted(ops.items())),
        "inputs": inputs,
        "outputs": outputs,
        "state_inputs": [n for n in inputs if n.startswith("state_")],
        "state_outputs": [n for n in outputs if n.startswith("state_")],
        "lowerable": not unknown,
        "unregistered_ops": unknown,
    }


def _parse_shape(spec: list[int] | int | str) -> tuple[int, ...]:
    """Normalize a shape given as a JSON list, a bare int, or ``"3x3"``/``"scalar"``."""
    if isinstance(spec, str):
        text = spec.strip().lower()
        if text == "scalar" or text == "":
            return ()
        try:
            return tuple(int(part) for part in text.split("x"))
        except ValueError as e:
            raise ValueError(f"invalid shape {spec!r} (use [3], [3, 3], 3, or '3x3')") from e
    if isinstance(spec, int):
        return (spec,)
    try:
        return tuple(int(part) for part in spec)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid shape {spec!r} (use [3], [3, 3], 3, or '3x3')") from e


def _parse_shapes(specs: dict[str, list[int] | int | str] | None) -> dict[str, tuple[int, ...]]:
    """Normalize a mapping of port name -> shape spec."""
    return {name: _parse_shape(spec) for name, spec in (specs or {}).items()}


def _array_for(shape: tuple[int, ...], value: Any) -> np.ndarray:
    """Coerce a JSON value into an ndarray of the port's declared shape."""
    arr = np.asarray(value, dtype=np.float64)
    try:
        return arr.reshape(shape)
    except ValueError as e:
        raise ValueError(f"value of size {arr.size} does not fit declared port shape {shape}") from e


def _inside_runtime_dir(target: Path) -> bool:
    """True when ``target`` would write into the packaged Zig runtime dir.

    ``src/shinro/runtime/graph_data.zig`` is generated by ``make zig-gen`` and
    ships in the package; a lowering that overwrote it would silently replace
    the shipped KF+LQR graph for every consumer of this checkout.
    """
    runtime = runtime_root().resolve()
    try:
        return target.resolve().is_relative_to(runtime)
    except OSError:  # pragma: no cover - unreadable path
        return False


def _get_graph(name: str) -> ComposedGraph | None:
    """Fetch a stored graph, or ``None`` when the name was never registered."""
    return _graph_store.get(name)


def _store_graph(name: str, cg: ComposedGraph) -> None:
    _graph_store[name] = cg


# ── introspection ─────────────────────────────────────────────────────────────


@server.tool()
def list_codegen_ops() -> str:
    """List the registered graph ops and the registered scenario graph recipes.

    Every op returned in ``ops`` has both a numpy handler (the interpreter) and a
    Zig VM lowering schema, so a graph built from them lowers without edits.
    """
    import json

    return json.dumps({"ops": sorted(OP_HANDLERS), "recipes": available_graphs()})


@server.tool()
def check_component(config_path: str) -> str:
    """Construct a component from its config TOML and report its trace contract.

    Does NOT trace: this is the "loads? constructs?" gate, plus the method and
    input names the tracer will call. Use it to discover the shapes that
    ``trace_component`` requires.

    Args:
        config_path: Path to a component config TOML (controller, estimator,
            trajectory, or plant).
    """
    import json

    try:
        component, category, ctype = load_component(config_path)
        from shinro.codegen.infer_contract import infer_contract

        contract = infer_contract(component)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    return json.dumps(
        {
            "type": ctype,
            "category": category,
            "method": contract.method_name,
            "input_names": list(contract.input_names),
        }
    )


# ── tracing ───────────────────────────────────────────────────────────────────


@server.tool()
def trace_component(
    name: str,
    config_path: str,
    input_shapes: dict[str, list[int] | int | str],
    state_shapes: dict[str, list[int] | int | str] | None = None,
) -> str:
    """Trace one component call into a graph and store it under ``name``.

    The component is swapped to a trace backend, called once with tracers, and
    every ``bk.*`` / operator it performs is captured as a node — no source
    changes and no per-component registration. Recurrent instance attributes
    (a PID's integral, a Kalman filter's covariance) are detected by attr-diff
    and reported as ``state_attrs`` with their live shapes in ``state_shapes``;
    pass those back via the ``state_shapes`` argument to re-trace with them as
    persistent ``state_*`` ports.

    Args:
        name: Name to store the traced graph under (used by interpret_graph /
            lower_graph / measure_graph).
        config_path: Component config TOML.
        input_shapes: Shape per input name, e.g. {"measurement": [3, 1],
            "control_input": [3]}. Shapes may be lists, ints, or "3x3".
        state_shapes: Optional shapes for known recurrent attrs (e.g. the
            ``state_shapes`` reported by a first trace such as
            {"x_hat": [3], "P": [3, 3]}); pre-injected as state_* ports.
    """
    import json

    try:
        component, category, ctype = load_component(config_path)
        shapes = _parse_shapes(input_shapes)
        states = _parse_shapes(state_shapes)
        ng = trace_node(component, input_shapes=shapes, state_shapes=states or None)
    except NotImplementedError as e:
        return json.dumps({"error": f"not traceable: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    cg = _to_composed(ng)
    _store_graph(name, cg)
    # The attrs were restored by trace_node, so their live shapes are readable —
    # report them so a caller can re-trace with exactly the right state_shapes
    # instead of guessing (a wrong rank fails later as a ShapeMismatchError).
    detected = {attr: list(getattr(component, attr).shape) for attr in ng.state_attrs if hasattr(component, attr)}
    return json.dumps(
        {
            "stored": name,
            "type": ctype,
            "category": category,
            "method": ng.contract.method_name,
            "state_attrs": list(ng.state_attrs),
            "state_shapes": detected,
            **_summary(name, cg),
        }
    )


@server.tool()
def build_composed_graph(
    name: str,
    estimator_config: str,
    controller_config: str,
    n_x: int,
    n_u: int,
    input_limits_min: list[float] | None = None,
    input_limits_max: list[float] | None = None,
) -> str:
    """Trace an estimator + controller and compose the closed-loop step graph.

    Runs the two-pass trace (pass 1 discovers recurrent state by attr-diff, pass
    2 re-traces with it pre-injected) and wires the fixed ABC dataflow
    ``y -> estimator -> x_hat -> controller -> u -> [clip] -> u``. The result is
    one ``shinro_step`` graph, stored under ``name``.

    Args:
        name: Name to store the composed graph under.
        estimator_config: Estimator config TOML.
        controller_config: Controller config TOML.
        n_x: Plant state dimension.
        n_u: Plant input dimension.
        input_limits_min: Optional per-channel lower clip bound (n_u,).
        input_limits_max: Optional per-channel upper clip bound (n_u,).
    """
    import json

    from shinro.codegen.build import build_composed_graph as _build

    if (input_limits_min is None) != (input_limits_max is None):
        return json.dumps({"error": "provide both input_limits_min and input_limits_max, or neither"})
    limits = None
    if input_limits_min is not None:
        limits = (np.asarray(input_limits_min, dtype=np.float64), np.asarray(input_limits_max, dtype=np.float64))

    try:
        cg = _build(estimator_config, controller_config, n_x, n_u, input_limits=limits)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except NotImplementedError as e:
        return json.dumps({"error": f"not traceable: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    _store_graph(name, cg)
    return json.dumps({"stored": name, **_summary(name, cg)})


@server.tool()
def graph_summary(name: str) -> str:
    """Report the ports, op histogram, and lowerability of a stored graph.

    Args:
        name: Name of a stored graph (from trace_component / build_composed_graph
            / generate_scenario).
    """
    import json

    cg = _get_graph(name)
    if cg is None:
        return json.dumps({"error": f"No graph named '{name}'. Trace or generate one first."})
    return json.dumps(_summary(name, cg))


@server.tool()
def list_graphs() -> str:
    """List the graphs held in this server session, compactly.

    Stored graphs hold a full copy of the traced graph, so a long session can
    accumulate them. Use this to see what is live before ``delete_graph``.
    """
    import json

    return json.dumps(
        {
            name: {
                "kind": "composed" if isinstance(cg, ComposedGraph) else "node",
                "nodes_traced": len(cg.graph.nodes),
                "inputs": list(cg.inputs),
                "outputs": list(cg.outputs),
            }
            for name, cg in _graph_store.items()
        }
    )


@server.tool()
def delete_graph(name: str) -> str:
    """Delete a stored graph, freeing it for the rest of the session.

    Artifacts already lowered or built from it are untouched — this only drops the
    in-memory graph (and therefore the ability to ``interpret_graph`` it by name).

    Args:
        name: Name of a stored graph.
    """
    import json

    if name not in _graph_store:
        return json.dumps({"error": f"No graph named '{name}'."})
    del _graph_store[name]
    return json.dumps({"deleted": name})


# ── numeric / size inspection ────────────────────────────────────────────────


@server.tool()
def interpret_graph(name: str, inputs: dict[str, list[float]]) -> str:
    """Run a stored graph on numpy inputs — the Python (interpreter) oracle.

    Walks the captured nodes through the registered numpy handlers. This is the
    reference the compiled kernel is checked against, so it is the cheapest way
    to see what a graph computes without a compiler.

    Args:
        name: Name of a stored graph.
        inputs: Value per input port name, in row-major order (a flat list is
            reshaped to the port's declared shape), e.g. {"y": [0.1, 0.0, 0.0],
            "x_ref": [0, 0, 0]}.
    """
    import json

    cg = _get_graph(name)
    if cg is None:
        return json.dumps({"error": f"No graph named '{name}'. Trace or generate one first."})

    shapes = _input_shapes(cg.graph)
    missing = [port for port in shapes if port not in inputs]
    if missing:
        return json.dumps({"error": f"missing input port(s): {missing}", "expected": list(shapes)})

    try:
        arrays = {port: _array_for(shape, inputs[port]) for port, shape in shapes.items()}
        outputs = interpret(cg.graph, arrays)
    except (ValueError, KeyError) as e:
        return json.dumps({"error": str(e)})
    except NotImplementedError as e:
        return json.dumps({"error": f"not interpretable: {e}"})

    return json.dumps({port: value.tolist() for port, value in outputs.items()})


@server.tool()
def measure_graph(name: str) -> str:
    """Report the lowered static size metrics for a stored graph.

    No compiler and no duplicated buffer math: the graph is lowered into a
    throwaway directory and the manifest it writes is read back, so these are
    the same byte figures the Zig VM compiles — the C-ABI host buffers, the
    internal ``buf: [buf_len]f64`` stack, and the baked constant/weight blobs.

    Args:
        name: Name of a stored graph.
    """
    import json

    cg = _get_graph(name)
    if cg is None:
        return json.dumps({"error": f"No graph named '{name}'. Trace or generate one first."})

    try:
        with tempfile.TemporaryDirectory(prefix="shinro-mcp-measure-") as tmp:
            metrics = graph_metrics(cg, Path(tmp))
    except (ValueError, RuntimeError, NotImplementedError) as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    # Same vocabulary as the other codegen tools, so the two counts are never
    # confused: traced = pre-optimisation (what interpret() runs), lowered =
    # post-fold (what the VM compiles).
    metrics["nodes_traced"] = len(cg.graph.nodes)
    metrics["nodes_lowered"] = metrics.pop("nodes_total")
    return json.dumps(metrics)


# ── lowering ─────────────────────────────────────────────────────────────────


@server.tool()
def lower_graph(name: str, out_dir: str) -> str:
    """Serialize a stored graph to a Zig data table (``graph_data.zig``) + manifest.

    This is the lowering half of the pipeline: it emits data only — the node
    table, packed constants, and port layout — which the handwritten comptime VM
    (``runtime/lower.zig``) specializes over at compile time. No compiler runs
    here; use ``build_kernel`` to turn the result into a ``.so``.

    Refuses to write anywhere inside the packaged ``shinro/runtime/`` directory,
    because ``runtime/graph_data.zig`` is the shipped graph that ``make zig-gen``
    regenerates — a lowering there would silently replace it for every consumer.

    Args:
        name: Name of a stored graph.
        out_dir: Directory to write ``graph_data.zig`` + the manifest (and the
            constant sidecars) into, e.g. "build/mcp/base".
    """
    import json

    cg = _get_graph(name)
    if cg is None:
        return json.dumps({"error": f"No graph named '{name}'. Trace or generate one first."})

    out = Path(out_dir)
    if _inside_runtime_dir(out):
        return json.dumps(
            {
                "error": (
                    f"refusing to lower into {out.resolve()} — that is the packaged runtime dir; "
                    "src/shinro/runtime/graph_data.zig is the shipped graph and must not be "
                    "overwritten from a tool. Pass an isolated out_dir such as build/mcp/<name>."
                )
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / "graph_data.zig"
    try:
        lower_zig(cg, str(graph_path), provenance={"generated_by": "shinro-mcp", **tool_versions()})
    except (ValueError, NotImplementedError) as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    manifest_path = graph_path.with_name("graph_data_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return json.dumps({"error": f"lowered but could not read {manifest_path}: {e}"})

    return json.dumps(
        {
            "graph_path": str(graph_path),
            "manifest_path": str(manifest_path),
            "nodes_traced": len(cg.graph.nodes),
            "nodes_lowered": manifest.get("nodes_total"),
            "buf_len_f64": manifest.get("buf_len"),
            "buf_bytes": manifest.get("buf_bytes"),
            "const_blob_bytes": manifest.get("const_blob_bytes"),
            "has_solve_qp": manifest.get("has_solve_qp"),
            "inputs": manifest.get("inputs"),
            "outputs": manifest.get("outputs"),
            "state_outputs": manifest.get("state_outputs"),
        }
    )


@server.tool()
def generate_scenario(name: str, scenario_path: str, out_dir: str) -> str:
    """Run the gen stage: trace + compose + lower a scenario TOML into an isolated dir.

    The zig-free half of the e2e pipeline (``make compile`` / ``shinro-compile``
    stage 1). Reads the scenario's ``[controller]`` / ``[estimator]`` /
    ``[scenario].input_limits`` / ``[compile]`` sections, dispatches to the
    scenario's registered recipe, and lowers to ``out_dir`` — never to the
    shipped runtime graph.

    The resulting graph is also stored under ``name``, so ``graph_summary`` /
    ``interpret_graph`` / ``measure_graph`` / ``build_kernel`` can consume it.

    Args:
        name: Name to store the composed graph under.
        scenario_path: Scenario TOML path.
        out_dir: Output directory for ``graph_data.zig`` + its manifest.
    """
    import json

    out = Path(out_dir)
    if _inside_runtime_dir(out):
        return json.dumps({"error": f"refusing to lower into {out.resolve()} — that is the packaged runtime dir"})

    try:
        cg, graph_path = gen_scenario(scenario_path, str(out))
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})
    except NotImplementedError as e:
        return json.dumps({"error": f"not traceable: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    _store_graph(name, cg)
    manifest_path = Path(graph_path).with_name("graph_data_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        manifest = {}

    return json.dumps(
        {
            "stored": name,
            "graph_path": str(graph_path),
            "manifest_path": str(manifest_path),
            "nodes_traced": len(cg.graph.nodes),
            "nodes_lowered": manifest.get("nodes_total", len(cg.graph.nodes)),
            "buf_bytes": manifest.get("buf_bytes"),
            "has_solve_qp": manifest.get("has_solve_qp"),
            **_summary(name, cg),
        }
    )
