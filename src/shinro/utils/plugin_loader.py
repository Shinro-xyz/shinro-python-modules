"""Import third-party component modules so their registered types become visible.

shinro ships only its built-in components: :mod:`shinro.factories` imports
``shinro.controllers`` / ``estimators`` / ``plants`` / ``trajectories`` and
nothing else. A ``@register_controller`` (etc.) decorator in a third-party
package therefore never runs until that package is imported, and the registry
lookup (``_CONTROLLER_REGISTRY[type]``) fails with a bare :class:`KeyError`.

The CLIs expose ``--import MODULE`` (repeatable) for exactly this, and call
:func:`import_modules` before dispatching any verb.
"""

from __future__ import annotations

import importlib


class PluginImportError(Exception):
    """A module named by ``--import`` could not be imported."""


def import_modules(names: list[str]) -> None:
    """Import each dotted module name so its ``@register_*`` components register.

    Args:
        names: Dotted module names, e.g. ``"my_pkg.components"``.

    Raises:
        PluginImportError: If a module cannot be imported. The message names the
            module and the underlying cause, so a typo is a clean CLI error
            instead of a traceback.
    """
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as e:  # surface any import-time failure (not just ImportError)
            raise PluginImportError(f"cannot import {name!r}: {type(e).__name__}: {e}") from e
