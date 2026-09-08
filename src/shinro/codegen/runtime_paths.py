"""Locate the packaged Zig runtime directory.

The Zig comptime VM (``build.zig``, ``lower.zig``, ``linalg.zig``, ``qp.zig``)
ships inside the installed package so ``shinro-compile`` can build a
``libbase.so`` from an installed wheel, not just a source checkout. This
mirrors :func:`shinro.utils.config_resolver.resolve_config_path`'s
packaged-vs-CWD resolution: in a source checkout the runtime is
``src/shinro/runtime/``; in an installed wheel it is the package's
``runtime/`` directory.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def runtime_root() -> Path:
    """Return the packaged Zig runtime directory on disk.

    Returns:
        A :class:`pathlib.Path` pointing at the runtime directory
        (``src/shinro/runtime`` in a checkout, the installed package's
        ``runtime`` otherwise).
    """
    root = resources.files("shinro").joinpath("runtime")
    if isinstance(root, Path):  # pragma: no cover - depends on importlib version
        return root
    with resources.as_file(root) as p:  # pragma: no cover
        return p
