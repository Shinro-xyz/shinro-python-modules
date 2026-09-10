"""Strict TOML→dataclass config parsing for components.

Components declare a frozen dataclass whose fields ARE the config schema.
:func:`strict_from_dict` parses a raw TOML dict into that dataclass, rejecting
unknown keys, wrong ``type`` values, and missing required fields loudly — so
authoring typos surface at parse time instead of mid-simulation.
"""

from __future__ import annotations

import tomllib
from typing import Any, ClassVar, Protocol

from shinro.utils.config_resolver import resolve_config_path


class DataclassProtocol(Protocol):
    """Minimal structural type for config dataclasses."""

    __dataclass_fields__: ClassVar[dict[str, Any]]


def strict_from_dict[T: DataclassProtocol](cls: type[T], raw: dict[str, Any], type_name: str) -> T:
    """Parse a raw TOML dict into a config dataclass, strictly.

    Args:
        cls: The config dataclass to construct.
        raw: Raw TOML dict. A ``type`` key is validated against ``type_name``
            and dropped; it may be absent (e.g. MCP inline params).
        type_name: The registered component name, used in error messages and
            to validate the ``type`` key.

    Returns:
        A ``cls`` instance.

    Raises:
        ValueError: On unknown keys, a mismatched ``type`` key, or missing
            required fields.
    """
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known - {"type"}
    if unknown:
        raise ValueError(
            f"{type_name} config: unknown key(s) {sorted(unknown)} — valid keys: {sorted(known)}"
        )
    if (t := raw.get("type")) is not None and t != type_name:
        raise ValueError(f"{type_name} config: type = {t!r} — wrong file?")
    try:
        return cls(**{k: v for k, v in raw.items() if k != "type"})  # type: ignore[call-arg]
    except TypeError as e:
        raise ValueError(f"{type_name} config: {e}") from e


def config_from_toml[T: DataclassProtocol](cls: type[T], path: str, type_name: str) -> T:
    """Load a TOML file and strictly parse it into a config dataclass.

    Args:
        cls: The config dataclass to construct.
        path: TOML path (resolved via
            :func:`shinro.utils.config_resolver.resolve_config_path`).
        type_name: The registered component name.

    Returns:
        A ``cls`` instance.
    """
    with open(resolve_config_path(path), "rb") as f:
        return strict_from_dict(cls, tomllib.load(f), type_name)
