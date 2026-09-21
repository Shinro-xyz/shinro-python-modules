"""Tests for :mod:`shinro.codegen.bake` — the OSQP codegen solver bake.

``bake_emosqp`` generates the static solver + ``solver_meta.zig``;
``bake_is_current`` is the reuse check the on-demand build path relies on.
These run the real OSQP codegen (seconds), so they are not marked slow — the
graph↔bake wiring itself is covered by ``tests/test_compile_scenario.py``.
"""

from __future__ import annotations

from shinro.codegen.bake import SOLVERS, bake_emosqp, bake_is_current

CONFIG = "samples/controllers/mpc_lti_base.toml"


def test_registry_has_emosqp():
    assert "emosqp" in SOLVERS


def test_bake_then_current(tmp_path):
    out = str(tmp_path / "emosqp")
    assert bake_is_current(out, 30, CONFIG) is False  # nothing baked yet

    b = bake_emosqp(CONFIG, out)
    assert b.n_vars == 30
    assert b.n_cons > 0
    assert (tmp_path / "emosqp" / "solver_meta.zig").exists()
    assert bake_is_current(out, 30, CONFIG) is True

    # A different n_vars does not match the bake.
    assert bake_is_current(out, 31, CONFIG) is False


def test_bake_is_current_false_without_meta(tmp_path):
    assert bake_is_current(str(tmp_path), 30, CONFIG) is False
