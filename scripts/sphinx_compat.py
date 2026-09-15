# FILE: scripts/sphinx_compat.py
"""Griffe extension: make Sphinx-flavored docstrings render under mkdocstrings.

The shinro source uses a hybrid convention:

* Google ``Args:`` / ``Returns:`` sections — griffe parses these natively.
* Sphinx roles (``:class:``, ``:meth:``, ``:func:``, ``:mod:``) — NOT handled,
  so they leak into the rendered page as literal text.
* Sphinx math (``:math:`` inline, ``.. math::`` blocks) — NOT handled, so
  formulas appear as raw markup instead of equations.

This rewrites the three leaking forms as griffe walks each object, so
mkdocstrings parses already-normalized text. It runs on every build, meaning a
docstring edit needs no manual follow-up.

NOTE: the equivalent of this file does not exist for the Sphinx toolchain, which
understands these forms natively — that path's trade-off is RST config surface
and a less polished theme.
"""

from __future__ import annotations

import re

from griffe import Docstring, Extension

# Inline math: `:math:`expr`` -> `$expr$`. Must run BEFORE the generic role
# stripper, otherwise `:math:` would match the `:role:` pattern first.
_RE_MATH_INLINE = re.compile(r":math:`([^`]+)`")

# Block math: an indented `.. math::` directive -> a fenced `$$` block.
_RE_MATH_BLOCK = re.compile(
    r"^[ \t]*\.\.\s*math::[ \t]*\n(?P<body>(?:[ \t]+\S.*\n?|\n)*)",
    re.MULTILINE,
)

# Remaining Sphinx roles: `:class:`Foo`` -> `` `Foo` `` (keep text, drop role).
_RE_SPHINX_ROLE = re.compile(r":(?:class|meth|func|mod|attr|data|obj|ref|term|exc|doc):`~?([^`]+)`")

# `:func:`target`` leaves empty inline literals behind after the rewrite.
_RE_EMPTY_LITERAL = re.compile(r"``\s*``")


def _math_block_sub(match: re.Match[str]) -> str:
    body = [ln.strip() for ln in match.group("body").splitlines() if ln.strip()]
    if not body:
        return ""
    return "\n\n$$\n" + " ".join(body) + "\n$$\n\n"


def normalize_text(text: str) -> str:
    """Rewrite Sphinx-only markup into mkdocstrings/MathJax-friendly markdown."""
    if not text:
        return text
    text = _RE_MATH_BLOCK.sub(_math_block_sub, text)
    text = _RE_MATH_INLINE.sub(lambda m: f"${m.group(1)}$", text)
    text = _RE_SPHINX_ROLE.sub(lambda m: f"`{m.group(1)}`", text)
    text = _RE_EMPTY_LITERAL.sub("", text)
    return text


class SphinxCompat(Extension):
    """Rewrite Sphinx-only markup so it renders instead of leaking as text."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.normalized = 0

    def on_instance(self, *, node=None, obj=None, agent=None, **kwargs: object) -> None:
        """Fires once per parsed object as griffe walks the tree."""
        if obj is None:
            return
        doc = getattr(obj, "docstring", None)
        if not isinstance(doc, Docstring) or not doc.value:
            return
        new_value = normalize_text(doc.value)
        if new_value != doc.value:
            doc.value = new_value
            self.normalized += 1
