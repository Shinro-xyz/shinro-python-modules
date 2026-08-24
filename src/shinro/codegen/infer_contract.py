"""Auto-infer a component's trace I/O contract from its ABC and signature.

This is the "out of the box" mechanism: a new component added with
``@register_controller("Foo")`` + ``from_config`` + a standard
``compute(self, current, target)`` signature traces and composes with zero
tracer-side code. The contract is inferred from:

1. **Which method to call** — from the ABC the component implements
   (``Controller`` → ``compute``, ``StateEstimator`` → ``estimate``).
2. **Input names and count** — from ``inspect.signature`` of that method.
3. **Input shapes** — provided by the caller (from the scenario's plant
   dimensions, the same way ``ScenarioFactory._validate_dimensions`` gets
   them).
4. **Constants** — auto-lifted: the moment ``self.K @ tracer`` executes,
   ``self.K`` (a real ndarray) becomes a ``const`` node. No declaration.
5. **State (recurrent attrs like ``x_hat``)** — auto-detected at trace time
   via attr-diff: snapshot all array-valued instance attrs before the call;
   after the call, any attr whose ``id()`` changed is state.

This means the tracer has no per-component registry to maintain. The only
time ``codegen/`` needs an edit for a new component is if it uses a new
``ArrayBackend`` op — and that's a ``@register_op`` decorator in
``ops.py``, not per-component code.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from shinro.components import Controller, Plant, StateEstimator, TrajectoryGenerator

# Maps each ABC to the method that runs one step of the component's compute.
# The tracer calls this method with Tracer inputs.
ABC_TO_METHOD: dict[type, str] = {
    Controller: "compute",
    StateEstimator: "estimate",
    Plant: "step",
    TrajectoryGenerator: "position_at",
}


@dataclass
class InferredContract:
    """The auto-inferred I/O surface of a component.

    Args:
        method_name: The method to call (``compute``, ``estimate``, ...).
        input_names: Ordered parameter names from the method signature
            (excluding ``self``). E.g. ``["current_state", "target_state"]``
            for ``LQR.compute(self, current_state, target_state)``.
        input_shapes: Maps each input name to a concrete shape, provided by
            the caller from the scenario's plant dimensions.
        state_attrs: Discovered at trace time (via attr-diff), not here.
            Listed for completeness; populated by ``trace_node``.
    """

    method_name: str
    input_names: list[str]
    input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    state_attrs: list[str] = field(default_factory=list)


def infer_contract(component: Any, input_shapes: dict[str, tuple[int, ...]] | None = None) -> InferredContract:
    """Infer a component's trace contract from its ABC and method signature.

    Args:
        component: An instance of a registered Controller / StateEstimator /
            Plant / TrajectoryGenerator.
        input_shapes: Optional dict mapping input names to concrete shapes.
            The caller (e.g. the composition pass) knows these from the
            scenario's plant dimensions. If omitted, shapes are left empty
            and filled in by the caller before tracing.

    Returns:
        An :class:`InferredContract` naming the method to call and its
        input parameter names.

    Raises:
        TypeError: If the component doesn't implement a known ABC.
    """

    method_name = _method_for(component)
    method = getattr(component, method_name)
    sig = inspect.signature(method)
    input_names = [p for p in sig.parameters if p != "self"]
    return InferredContract(
        method_name=method_name,
        input_names=input_names,
        input_shapes=dict(input_shapes or {}),
    )


def _method_for(component: Any) -> str:
    """Return the compute-method name for a component by checking its ABCs."""
    for abc, name in ABC_TO_METHOD.items():
        if isinstance(component, abc):
            return name
    raise TypeError(
        f"component {type(component).__name__} does not implement any of "
        f"the known ABCs ({', '.join(a.__name__ for a in ABC_TO_METHOD)})"
    )


def snapshot_instance_attrs(component: Any) -> dict[str, int]:
    """Snapshot the ``id()`` of every array-valued instance attribute.

    Used by ``trace_node`` to detect state mutations: after the traced call,
    any attr whose ``id()`` changed is flagged as live state (a recurrent
    edge in the composed graph). We snapshot ``id()`` rather than the value
    so the check is O(1) and unambiguous — a reassignment is a new object,
    even if the new value equals the old.

    Only attributes whose current value is a numpy array or a Tracer are
    snapshotted; scalars and other types are ignored (they're not array
    state in the control-law sense).

    Args:
        component: The component about to be traced.

    Returns:
        A dict mapping attribute name → ``id(value)`` for every array-valued
        instance attr.
    """
    from shinro.codegen.tracing import Tracer  # delayed to avoid cycles

    snap: dict[str, int] = {}
    # Walk the instance __dict__ (not the class) — we only care about per-
    # instance state, not class-level defaults.
    for name, value in vars(component).items():
        if _is_array_like(value):
            snap[name] = id(value)
    return snap


def _is_array_like(value: Any) -> bool:
    """Return True for ndarrays and Tracers (the values we track as state)."""
    import numpy as np

    from shinro.codegen.tracing import Tracer

    return isinstance(value, (np.ndarray, Tracer))


def detect_state(component: Any, before: dict[str, int]) -> list[str]:
    """Return the instance attrs whose ``id()`` changed since the snapshot.

    A changed ``id()`` means the component reassigned the attr during the
    traced call — that's a state mutation, and the attr is a recurrent edge
    in the composed graph (its value feeds back as an input next tick).

    Args:
        component: The component after the traced call.
        before: The snapshot from :func:`snapshot_instance_attrs`.

    Returns:
        A list of attribute names that were reassigned during the call.
    """
    from shinro.codegen.tracing import Tracer  # delayed to avoid cycles

    state: list[str] = []
    for name, id_before in before.items():
        current = getattr(component, name, None)
        if not _is_array_like(current):
            continue
        if id(current) != id_before:
            state.append(name)
    return state


# silence unused-import linters
import numpy as np  # noqa: E402

_ = np