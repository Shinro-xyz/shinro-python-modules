# Codegen: Compiling the Control Loop to a Native Library

The `codegen` pipeline turns a closed-loop control step (estimator + controller
on a plant) into a static computation graph that can be verified to
float-exactness against numpy and, in a future slice, lowered to native code.

The motivation is deployment: the shipped `.so` must *provably* agree with the
Python loop. Rather than reimplementing the control math by hand in a compiled
language (and risk subtle drift), the graph **is** the Python execution,
transcribed. The only things that can be wrong are the ~20 primitive op
lowerings, which are verified once and reused for every component.

This is an XLA/JAX-style tracing model. Components run once with abstract
`Tracer` values instead of real arrays; every operation they perform is
captured as a node in a `Graph`. The graph is then replayed on real inputs by
the interpreter (the correctness oracle) or, in slice (b), lowered to Zig.

The pipeline lives in `src/shinro/codegen/`. The design narrative is in
`lab-notes/daily/2026-08-24.md`.

## Pipeline overview

```
                    ┌─────────────────────────────────────────────┐
                    │             shinro.codegen                 │
  component ───────▶│ trace_node()   ──▶ NodeGraph (per component)│
                    │ compose()      ──▶ ComposedGraph (one tick) │
                    │ interpret()    ──▶ numpy arrays             │
                    │ lower_zig()    ──▶ Zig → base.so  (slice b) │
                    └─────────────────────────────────────────────┘
```

1. **Trace** each component once (`trace_node`). A component is run with
   `Tracer` values instead of real arrays; its `self.bk` is swapped to a
   `TraceBackend`, so every `bk.*` call emits a graph node. Concrete
   parameters (gain `K`, matrices `A`/`B`) are lifted to `const` nodes the
   moment they touch a traced operation — this is the "fixed as compiled"
   property: shapes and parameters are baked at trace time.
2. **Compose** (`compose`). Stitch the per-component graphs into a single
   graph representing one tick of the closed loop, wiring them per the fixed
   ABC dataflow. Shape mismatches (e.g. the KF's `(n,1)` column vector vs the
   LQR's `(n,)` flat) are bridged by auto-inserted `reshape` nodes — the XLA
   approach.
3. **Interpret** (`interpret`). Replay the graph on real numpy inputs as a
   correctness oracle. If the interpreter's output matches a live
   `NumpyBackend` run to float-exactness, the tracer is sound.
4. **Lower** (slice b, future). Walk the graph and emit a Zig module exposing
   a `shinro_step` C-ABI function with baked constants, compiled to a `.so`.

## Module map

| Module | Role |
|--------|------|
| `codegen/tracing.py` | `Tracer` (abstract value), `Graph` / `Node` (graph records), shape checking. Operator overloads (`@`, `+`, `-`, `*`, `.T`) record nodes. |
| `codegen/trace_backend.py` | `TraceBackend` — a recording `ArrayBackend` that emits nodes for the named `bk.*` methods components call. |
| `codegen/trace_node.py` | `trace_node` / `trace_node_with_state` — run one component call under a `TraceBackend` and return a `NodeGraph`. |
| `codegen/infer_contract.py` | Auto-infer a component's I/O contract from its ABC (`Controller` → `compute`, `StateEstimator` → `estimate`) and `inspect.signature`; detect recurrent state via attr-diff. |
| `codegen/ops.py` | Op-handler registry (`OP_HANDLERS`). The data-driven "switch" — adding a new op is one `@register_op` decorator. |
| `codegen/interpreter.py` | `interpret` / `interpret_step` — replay a graph on real numpy inputs. |
| `codegen/compose.py` | `compose` — merge per-component graphs into one closed-loop step graph, auto-inserting `reshape`/`clip`. |
| `demo_codegen.py` (repo root) | Runnable demo: traces KF+LQR for the base and cartpole plants, composes, and verifies each stage against a live numpy loop. |

## The tracing model

### `Tracer` — the abstract value

A `Tracer` stands in for an ndarray during tracing. It carries only its
concrete shape and its graph node id — **no data**. Operations on tracers
(`@`, `+`, `-`, `*`, `-`, `.T`) emit nodes into the graph and return new
tracers.

`__array_ufunc__ = None` on `Tracer` is critical: it tells numpy to defer to
the tracer's reflected operators (`__rmatmul__` etc.) when a numpy array
interacts with a tracer, instead of coercing the tracer via `np.asarray`.
Without it, `A @ tracer` would silently compute with garbage instead of
recording a `matmul` node.

### `_lift` — freezing constants

When a concrete numpy array (a precomputed gain `K`, a config-baked matrix)
touches a tracer, `_lift` bakes it into a `const` node. This is the
"fixed as compiled" freezing: the graph carries literal parameter values, so a
deployed `.so` needs no runtime configuration — everything is baked in.

### The backend swap

`trace_node` temporarily replaces the component's `self.bk` with a
`TraceBackend` that records into a fresh `Graph`. Named `ArrayBackend` methods
(`eye`, `zeros_like`, `inv`, `clip`, `where`, `copy`, ...) emit nodes; bare
operators (`@`, `+`, ...) are handled by the `Tracer` overloads. Any method the
backend doesn't implement raises a `NotImplementedError` naming the op to
register — a loud, actionable signal, so the op set grows incrementally,
driven by real components.

After the call, the original backend and all array instance attrs are restored
in a `finally`, so a traced call never leaves the component polluted with
tracers (which would reference dead node ids on the next trace).

### Auto-inferred contracts

A component needs no per-component tracer metadata. The contract is derived
from:

1. **Which method to call** — from the ABC the component implements
   (`Controller` → `compute`, `StateEstimator` → `estimate`).
2. **Input names and count** — from `inspect.signature` of that method.
3. **Input shapes** — supplied by the caller, from the scenario's plant
   dimensions.
4. **Constants** — auto-lifted the moment `self.K @ tracer` executes.
5. **State** — auto-detected at trace time via attr-diff: the tracer snapshots
   the `id()` of every array-valued instance attr before the call; any attr
   whose `id()` changed after is a recurrent edge (e.g. `KalmanFilter.x_hat`,
   `KalmanFilter.P`).

The only time `codegen/` needs an edit for a new component is if it uses a new
`ArrayBackend` op — a one-decorator change in `ops.py`.

## The composition pass

`compose(estimator, controller, plant_dims, input_limits)` wires the fixed ABC
dataflow:

```
y (measurement) ──▶ Estimator ──x_hat──▶ Controller ──u──▶ [clip] ──▶ output
x_ref ──────────────────────────────▶ Controller
state_x_hat (recurrent) ─▶ Estimator
```

- **Input ports:** `y`, `x_ref`, `u_prev`, and the recurrent `state_x_hat`.
- **Output ports:** `u`, plus the recurrent state outputs `state_x_hat` and
  `state_u_prev` (fed back as inputs next tick).
- **`clip`**: if `input_limits` is provided (from `[scenario.input_limits]`),
  a `clip` node is inserted on the controller output.
- **Auto-reshape:** where shapes mismatch (KF `(n,1)` → LQR `(n,)`), `reshape`
  nodes are inserted. The estimator's `(n,1)` state output is flattened to the
  controller's `(n,)` expectation; `(n,)` feeds are reshaped to the estimator's
  `(n,1)`.
- **`_merge_and_rewire`**: subgraph `input` nodes are placeholders, not copied —
  consumers are rewired directly to combined-graph source nodes. Subgraph
  `output` nodes are markers and are skipped; `compose` declares the combined
  outputs itself.

The wiring is **not** a per-scenario edge dict — it's the fixed ABC dataflow,
the same for every scenario. What's scenario-specific (clip limits, vector
dims) comes from the scenario config.

## The interpreter

`interpret(graph, inputs)` walks the graph in execution order (the nodes are
already topologically sorted — emitted in execution order), dispatches each
through `OP_HANDLERS[node.op]`, and collects named outputs. It's a 5-line loop
over the registry. If its output matches a live `NumpyBackend` run to
float-exactness, the tracer is sound. The interpreter is the correctness
oracle: every test verifies `interpret(graph, inputs)` against a live numpy
computation.

`interpret_step` is a convenience that splits the outputs into non-state
outputs and recurrent `state_*` outputs (which feed back as inputs next tick).

## Ops

The op registry lives in `ops.py`. Register a handler:

```python
@register_op("matmul")
def _matmul(node, values, inputs):
    return values[node.inputs[0]] @ values[node.inputs[1]]
```

An unsupported op raises `NotImplementedError` naming the op to add and
listing available ops. The current set:

`const`, `input`, `output`, `matmul`, `add`, `sub`, `mul`, `neg`, `transpose`,
`inv`, `reshape`, `clip`, `where`, `copy`, `any`, `solve`, `diag`, `stack`.

## Graph invariants

A captured graph is a flat, topologically ordered list of nodes. Each node
holds an op name, input node ids, a concrete shape, and an opaque `attrs`
dict (baked ndarray for `const`, target shape for `reshape`, `lo`/`hi` for
`clip`, name for `input`/`output`). Inputs/outputs are named ports so the
interpreter and composition pass can refer to them symbolically.

## Writing a new component

Add the component as usual (`@register_controller("Foo")` +
`from_config` + a standard `compute(self, current, target)` signature) and it
traces with zero tracer-side code. If its compute path uses a new
`ArrayBackend` op, register a handler in `ops.py`.

## Running the demo

```bash
python demo_codegen.py
```

Four stages:

1. Trace a `KalmanFilter.estimate()` alone; show the graph; verify the
   interpreter matches live numpy.
2. Compose KF + LQR into one closed-loop step graph (with auto-reshape and
   clip); verify the composed step matches a live numpy loop.
3. Swap the estimator (KF → Luenberger) and re-compose, reusing the LQR graph
   without re-tracing — the modularity proof.
4. The cartpole system (4-state, 1-input): define the system, build the LQR
   gain and Kalman filter from the linearized model, trace both, compose, and
   verify.

Each stage prints `PASS` / `FAIL` based on the max abs error vs a live
`NumpyBackend` reference.

## Tests

- `tests/test_codegen.py` — single-component tracing (KF, LQR), graph-structure
  assertions, tracer primitive unit tests.
- `tests/test_codegen_compose.py` — composition, KF+LQR composed step vs numpy
  loop, estimator/controller swap tests.

## Slice (b) — lowering to Zig (future work)

The next vertical slice: `lower_zig.py` walks the captured graph and emits a
Zig module (a `shinro_step` C-ABI function with baked constants), `zig
build-lib` produces `base.so`, and a `ctypes` cross-check asserts the `.so`'s
output matches the interpreter to float-exactness. The tracer is proven sound;
the lowering is the remaining deployment half.
