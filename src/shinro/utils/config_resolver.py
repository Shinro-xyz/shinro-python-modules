"""Config path resolution for TOML configs.

Factories load TOML configs by path string. This module resolves those paths
against the filesystem so callers can pass either an absolute path or one
relative to their project root.

shinro ships **no** config files. The TOMLs under ``samples/`` in the
repository are documentation — they show what a config looks like and how to
write one — and are deliberately not installed. Your project owns its configs:

Resolution order for a given ``path``:

1. If it is an absolute path, use it as-is.
2. Otherwise resolve it relative to the current working directory.

A path that resolves to neither raises a loud :class:`FileNotFoundError`.
There is deliberately **no** fallback into the shinro package and no
basename-based search. An earlier implementation fell back to a packaged
``shinro/configs`` copy, which silently substituted shinro's own config for a
caller's misspelled or wrong-CWD path — the exact drift this contract exists to
prevent. Copy the sample you need into your project and run from your project
root.
"""

from __future__ import annotations

from pathlib import Path


def resolve_config_path(path: str) -> Path:
    """Resolve a TOML config path to an existing file.

    Args:
        path: Absolute path, or a path relative to the current working
            directory (e.g. ``configs/controllers/lqr_base.toml`` for a
            config in your own project).

    Returns:
        A :class:`pathlib.Path` pointing at an existing config file.

    Raises:
        FileNotFoundError: If the path does not exist. Resolution never falls
            back to a packaged shinro config — a missing or wrong-CWD path is a
            loud error, not a silent substitution.
    """
    path = str(path)
    p = Path(path)
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return p
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Config not found: {path} (resolved relative to CWD {Path.cwd()}; "
        f"shinro ships no configs — copy the sample you need from the repo's samples/)"
    )
