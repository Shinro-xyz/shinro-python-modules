"""Strict TOML→dataclass config parsing for components.

Components declare a frozen dataclass whose fields ARE the config schema.
:func:`strict_from_dict` parses a raw TOML dict into that dataclass, rejecting
unknown keys, wrong ``type`` values, and missing required fields loudly — so
authoring typos surface at parse time instead of mid-simulation.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
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


def strict_from_list[T: DataclassProtocol](cls: type[T], raw: list, type_name: str) -> list[T]:
    """Strict-parse a list of TOML dicts into a list of config dataclasses.

    Args:
        cls: The nested config dataclass to construct per entry.
        raw: List of raw TOML dicts (e.g. ``[[segments]]`` entries).
        type_name: The component name, used in error messages.

    Returns:
        A list of ``cls`` instances.
    """
    return [strict_from_dict(cls, item, type_name) for item in raw]


def strip_runtime_keys(raw: dict[str, Any], keys: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pop runtime-injected keys from a raw TOML config dict.

    Sim-backed builds inject non-TOML values into plant config dicts
    (``engine`` objects, ``joint_groups`` tables) before
    :meth:`~shinro.components.ConfigDriven.parse_config` runs; strict parsing
    must not see them.

    Args:
        raw: Raw TOML config dict.
        keys: Runtime-injected key names to pop.

    Returns:
        Tuple of (clean dict to strict-parse, dict of popped runtime values).
    """
    popped = {k: raw[k] for k in keys if k in raw}
    return {k: v for k, v in raw.items() if k not in keys}, popped


@dataclass(frozen=True)
class BoundsConfig:
    """Nested ``[state_bounds]`` table shared by analytical plants."""

    min: list[float] | None = None
    max: list[float] | None = None
