# FILE: scripts/gen_api.py
"""Generate the API-reference pages from the package's own ``__all__``.

The package source is the single source of truth. Each subpackage's ``__all__``
becomes one page; a subpackage without ``__all__`` falls back to an AST scan of
its source files. Adding an export therefore makes it appear in the docs on the
next build with no config, nav, or symbol-list edit.

Run directly (``python scripts/gen_api.py``) or from CI before ``mkdocs build``.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
import warnings

# Import noise (registration warnings) is expected and irrelevant here.
warnings.filterwarnings("ignore")

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
DOCS = REPO / "docs"
OUT = DOCS / "reference"

# Pages to generate, in nav order. `shinro.components` is a module, the rest are
# subpackages. Anything not listed simply doesn't get a page.
PACKAGES: list[tuple[str, str]] = [
    ("shinro.components", "Core ABCs"),
    ("shinro.trajectories", "Trajectories"),
    ("shinro.controllers", "Controllers"),
    ("shinro.estimators", "Estimators"),
    ("shinro.plants", "Plants"),
    ("shinro.factories", "Factories"),
    ("shinro.utils", "Utilities"),
    ("shinro.simulation", "Simulation"),
]


def _slug(modname: str) -> str:
    return modname.replace("shinro.", "").replace(".", "-").replace("_", "-")


def fallback_exports(modname: str) -> list[str]:
    """Fully-qualified public names for a package/module with no ``__all__``.

    Handles both a package directory (``shinro/utils/``) and a single module
    (``shinro/components.py``). Returns dotted identifiers resolving to the real
    defining module, which is what mkdocstrings needs.
    """
    candidates: list[str] = []

    pkg_dir = SRC / modname.replace(".", "/")
    if pkg_dir.is_dir():
        sources = sorted(p for p in pkg_dir.glob("*.py") if p.name != "__init__.py")
        prefix = modname
    else:
        as_file = SRC / (modname.replace(".", "/") + ".py")
        if not as_file.is_file():
            return []
        sources = [as_file]
        prefix = modname

    for py in sources:
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        submod = prefix if not pkg_dir.is_dir() else f"{prefix}.{py.stem}"
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                candidates.append(f"{submod}.{node.name}")

    seen: set[str] = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def main() -> int:
    if not SRC.is_dir():
        print(f"error: source tree not found at {SRC}", file=sys.stderr)
        return 1

    # Import from the repo source directly — no install step required.
    sys.path.insert(0, str(SRC))

    OUT.mkdir(parents=True, exist_ok=True)
    api_nav: list[str] = []
    total = 0
    pages = 0

    for modname, title in PACKAGES:
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # a broken import shouldn't kill the whole build
            print(f"  SKIP {modname}: import failed ({exc})", file=sys.stderr)
            continue

        exported = list(getattr(mod, "__all__", []))
        if exported:
            ids = [f"{modname}.{s}" for s in exported]
            labels = exported
            source = "__all__"
        else:
            ids = fallback_exports(modname)
            labels = [i.rsplit(".", 1)[-1] for i in ids]
            source = "AST fallback (no __all__)"

        if not ids:
            print(f"  SKIP {modname}: no exports found", file=sys.stderr)
            continue

        lines = ["---", f"title: {title}", "---", "", f"# {title}", "", f"`{modname}`", ""]
        if mod.__doc__:
            lines += [mod.__doc__.strip(), ""]

        for label, dotted in zip(labels, ids):
            lines += ["---", "", f"## `{label}`", "", f"::: {dotted}", ""]

        (OUT / f"{_slug(modname)}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        api_nav.append(f"    * [{title}](reference/{_slug(modname)}.md)")
        pages += 1
        total += len(ids)
        print(f"  wrote reference/{_slug(modname)}.md  ({len(ids)} exports, via {source})")

    # Compose the full nav in mkdocs-literate-nav markdown format.
    # Comment lines in the prose file are stripped so they don't reach the nav.
    prose_nav = DOCS / "_nav_prose.md"
    nav = ""
    if prose_nav.is_file():
        keep = [ln for ln in prose_nav.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        nav = "\n".join(keep)
    else:
        print(f"  note: {prose_nav.name} not found — prose pages omitted from nav", file=sys.stderr)

    api_section = "* API Reference\n" + "\n".join(api_nav)
    (DOCS / "SUMMARY.md").write_text(nav + "\n" + api_section + "\n", encoding="utf-8")
    (DOCS / "_nav_api.yml").write_text(api_section + "\n", encoding="utf-8")

    print(f"\n{pages} pages, {total} exported symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
