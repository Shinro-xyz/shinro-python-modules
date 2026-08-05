"""Config path resolution for the shinro package.

Factories load TOML configs by path string. This module resolves those
paths so that relative names like ``configs/controllers/lqr_base.toml``
work both from a source checkout (CWD = repo root) and from an installed
wheel (where the packaged configs live inside ``shinro/configs/``).

Resolution order for a given ``path``:

1. If it is an absolute path, use it as-is.
2. If it starts with ``configs/``, resolve against the packaged
   ``shinro/configs`` directory (falling back to a CWD-relative lookup
   for source checkouts).
3. If it names a file that exists relative to the CWD, use it.
4. Otherwise resolve against the packaged ``shinro/configs`` directory.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def _package_config_root() -> Path:
    """Return the packaged ``shinro/configs`` directory on disk."""
    root = resources.files("shinro").joinpath("configs")
    if isinstance(root, Path):  # pragma: no cover - depends on importlib version
        return root
    # importlib.resources.Traversable without an on-disk path: materialize via as_file.
    with resources.as_file(root) as p:  # pragma: no cover
        return p


def resolve_config_path(path: str) -> Path:
    """Resolve a TOML config path to an existing file.

    Args:
        path: Absolute path, a ``configs/...`` package-relative name, or a
            bare filename relative to the CWD / packaged configs.

    Returns:
        A :class:`pathlib.Path` pointing at an existing config file.

    Raises:
        FileNotFoundError: If no candidate location exists.
    """
    p = Path(path)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return p

    candidates: list[Path] = []
    if path.startswith("configs/"):
        packaged = _package_config_root().parent.joinpath(path)
        candidates.append(packaged)
        candidates.append(p)
    elif p.exists():
        candidates.append(p)
    else:
        packaged = _package_config_root().joinpath(p.name)
        candidates.append(packaged)
        candidates.append(p)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Config not found: {path} (tried: {', '.join(str(c) for c in candidates)})")


def get_config_path(name: str) -> Path:
    """Public helper: resolve a config name (without the ``configs/`` prefix).

    Args:
        name: Package-relative config name, e.g. ``controllers/lqr_base.toml``.

    Returns:
        A :class:`pathlib.Path` pointing at the packaged config file.
    """
    return resolve_config_path(f"configs/{name.lstrip('/')}")
