"""Backend-aware invocation of user model callables (dynamics / measurement).

Shared by the nonlinear estimators. A model callable may accept an optional
``bk`` (as :meth:`shinro.components.Plant.dynamics` does); the filters pass
their own backend when the signature accepts one, which is what lets a traced
component route the plant's ops through its
:class:`~shinro.codegen.trace_backend.TraceBackend` instead of the plant's
concrete backend. A plain two-argument user lambda keeps working.
"""

import inspect
from functools import cache
from typing import Any


@cache
def accepts_backend(fn: Any) -> bool:
    """Whether a model callable takes a backend (a ``bk`` parameter or ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / C callables expose no signature
        return False
    if "bk" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def call_model(fn: Any, *args: Any, bk: Any) -> Any:
    """Call a model callable, passing the backend when its signature accepts one.

    Passing ``bk`` only when accepted keeps both conventions working: a model
    written for the compiled path (which takes ``bk``, e.g.
    :meth:`shinro.components.Plant.dynamics`) and a plain user lambda that takes
    only the model arguments.
    """
    if accepts_backend(fn):
        return fn(*args, bk=bk)
    return fn(*args)
