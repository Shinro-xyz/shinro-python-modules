"""E2E scenario compilation pipeline tests.

Covers the two-script workflow (``gen_scenario.py`` → ``build_scenario.py``)
unified by ``make compile``: the zig-free gen stage (graph pair + ``[compile]``
validation) and the zig-gated build stage (compile → oracle → stamp → verify).
The zig-gated tests skip cleanly when zig is unavailable, matching the
``test_zig_lowering.py`` convention.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scripts.gen_scenario import _COMPILE_KEYS, load_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN = REPO_ROOT / "scripts" / "gen_scenario.py"
BUILD = REPO_ROOT / "scripts" / "build_scenario.py"
SCENARIO = REPO_ROOT / "tests" / "integration" / "scenarios" / "base_tracking.toml"
TEMPLATE = REPO_ROOT / "src" / "shinro" / "configs" / "scenarios" / "_template.toml"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess:
    # Subprocesses must import shinro from the SOURCE tree (src/), matching
    # the pytest run's pythonpath — otherwise config-hash tests tamper the
    # repo copy while the subprocess resolves the installed wheel's copy
    # (untampered) and staleness goes undetected.
    env = {**os.environ, "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}"}
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, env=env)


# ─── [compile] validation (no zig) ─────────────────────────────────────────


def _scenario_toml(tmp_path, compile_section: str) -> Path:
    cfg = tmp_path / "s.toml"
    cfg.write_text(
        '[controller]\nconfig = "configs/controllers/lqr_base.toml"\n'
        '[estimator]\nconfig = "configs/estimators/kalman_base.toml"\n' + compile_section
    )
    return cfg


def test_load_scenario_requires_compile_section(tmp_path):
    with pytest.raises(ValueError, match=r"missing \[compile\]"):
        load_scenario(str(_scenario_toml(tmp_path, "")))


def test_load_scenario_rejects_unknown_compile_key(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        load_scenario(str(_scenario_toml(tmp_path, "[compile]\nn_x = 3\nn_u = 3\nbogus = 1\n")))


def test_load_scenario_rejects_invalid_optimize(tmp_path):
    with pytest.raises(ValueError, match="ReleaseSafe"):
        load_scenario(str(_scenario_toml(tmp_path, '[compile]\nn_x = 3\nn_u = 3\noptimize = "release-safe"\n')))


def test_gen_scenario_matches_shipped_graph(tmp_path):
    """The gen stage reproduces the shipped KF+LQR graph node-for-node."""
    from scripts.gen_base import build_base_graph
    from shinro.codegen import lower_zig

    out = tmp_path / "g"
    result = _run(GEN, str(SCENARIO), "--out", str(out))
    assert result.returncode == 0, result.stderr

    shipped_dir = tmp_path / "shipped"
    shipped_dir.mkdir()
    lower_zig(build_base_graph(), str(shipped_dir / "graph_data.zig"))

    a = json.loads((out / "graph_data_manifest.json").read_text())
    b = json.loads((shipped_dir / "graph_data_manifest.json").read_text())
    assert a["nodes"] == b["nodes"]
    assert a["inputs"] == b["inputs"]
    assert a["outputs"] == b["outputs"]
    assert a["state_outputs"] == b["state_outputs"]


def test_gen_scenario_exit_codes(tmp_path):
    assert _run(GEN, str(tmp_path / "missing.toml"), "--out", str(tmp_path)).returncode == 2
    bad = _scenario_toml(tmp_path, '[compile]\nn_x = 3\nn_u = 3\noptimize = "release-safe"\n')
    assert _run(GEN, str(bad), "--out", str(tmp_path)).returncode == 2


# ─── scenario template drift guard ──────────────────────────────────────────


def test_scenario_template_stays_in_sync_with_compile_schema():
    """The template must parse, use only known [compile] keys, and document
    every schema key (active or commented) — so a schema addition forces the
    template to teach it."""
    text = TEMPLATE.read_text()
    cfg = tomllib.loads(text)

    compile_cfg = cfg.get("compile", {})
    assert set(compile_cfg) <= _COMPILE_KEYS, f"template [compile] has keys outside the schema: {set(compile_cfg) - _COMPILE_KEYS}"
    for key in _COMPILE_KEYS:
        assert key in text, f"template does not document [compile] key '{key}'"


# ─── e2e build stage (zig-gated) ───────────────────────────────────────────


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig not on PATH")
def test_e2e_compile_verified(tmp_path):
    """gen → build → oracle → stamp → verify, all green."""
    out = tmp_path / "scenario"
    gen = _run(GEN, str(SCENARIO), "--out", str(out))
    assert gen.returncode == 0, gen.stderr

    build = _run(BUILD, str(out), "--scenario", str(SCENARIO))
    assert build.returncode == 0, build.stderr
    assert "oracle B" in build.stdout

    so = out / "lib" / "libbase.so"
    assert so.exists()
    record = out / "lib" / "libbase.deployment.json"
    assert record.exists()
    rec = json.loads(record.read_text())
    assert rec["master_hash"]
    assert rec["slots"]["binary"] == hashlib.sha256(so.read_bytes()).hexdigest()


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig not on PATH")
def test_e2e_cross_compile_skips_oracle(tmp_path):
    """A non-native target skips the host oracle (can't dlopen a cross .so)
    but keeps the integrity check, stamp, and verify."""
    out = tmp_path / "scenario"
    assert _run(GEN, str(SCENARIO), "--out", str(out)).returncode == 0

    build = _run(BUILD, str(out), "--scenario", str(SCENARIO), "--target", "aarch64-linux-gnu")
    assert build.returncode == 0, build.stderr
    assert "skipping host oracle" in build.stderr
    assert "oracle B" not in build.stdout

    so = out / "lib" / "libbase.so"
    assert so.exists()
    record = out / "lib" / "libbase.deployment.json"
    assert record.exists()


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig not on PATH")
def test_e2e_stale_graph_rejected(tmp_path):
    """Building a graph that no longer matches the scenario fails loudly."""
    out = tmp_path / "scenario"
    assert _run(GEN, str(SCENARIO), "--out", str(out)).returncode == 0

    lqr = REPO_ROOT / "src" / "shinro" / "configs" / "controllers" / "lqr_base.toml"
    original = lqr.read_bytes()
    try:
        lqr.write_bytes(original + b"\n# tamper\n")
        build = _run(BUILD, str(out), "--scenario", str(SCENARIO))
        assert build.returncode == 2
        assert "stale" in build.stderr
    finally:
        lqr.write_bytes(original)


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig not on PATH")
def test_e2e_shared_graph_untouched(tmp_path):
    """Compiling a scenario never clobbers the shipped src/shinro/runtime/graph_data.zig."""
    shipped = REPO_ROOT / "src/shinro/runtime" / "graph_data.zig"
    before = shipped.read_bytes()
    out = tmp_path / "scenario"
    assert _run(GEN, str(SCENARIO), "--out", str(out)).returncode == 0
    assert _run(BUILD, str(out), "--scenario", str(SCENARIO)).returncode == 0
    assert shipped.read_bytes() == before


# ─── policy-only (onnx_rl) scenarios — committed toy fixture ─────────────────

# A checked-in 3 -> 4 -> 2 tanh MLP (scripts/gen_toy_onnx.py) plus its controller
# config and a policy-only compile scenario, so the whole ONNX -> .so path runs
# against a stable fixture with no model synthesis in the test.
TOY_SCENARIO = REPO_ROOT / "tests" / "fixtures" / "configs" / "scenarios" / "toy_mlp_policy.toml"
TOY_ONNX = REPO_ROOT / "tests" / "fixtures" / "models" / "toy_mlp.onnx"


def _policy_scenario(tmp_path, *, controller_config, compile_section):
    """A minimal policy-only scenario around a given controller config."""
    cfg = tmp_path / "policy_scenario.toml"
    cfg.write_text(f'[controller]\nconfig = "{controller_config}"\n' + compile_section)
    return cfg


def test_toy_onnx_fixture_matches_its_generator(tmp_path):
    """The committed .onnx matches scripts/gen_toy_onnx.py (regenerate-to-check)."""
    import numpy as np
    from onnx import numpy_helper

    onnx = pytest.importorskip("onnx")
    fresh = tmp_path / "toy_mlp.onnx"
    result = _run(REPO_ROOT / "scripts" / "gen_toy_onnx.py", "--out", str(fresh))
    assert result.returncode == 0, result.stderr

    committed = onnx.load(str(TOY_ONNX)).graph
    regenerated = onnx.load(str(fresh)).graph
    assert [n.op_type for n in committed.node] == [n.op_type for n in regenerated.node]
    assert [t.name for t in committed.initializer] == [t.name for t in regenerated.initializer]
    for a, b in zip(committed.initializer, regenerated.initializer, strict=True):
        np.testing.assert_array_equal(numpy_helper.to_array(a), numpy_helper.to_array(b))


def test_policy_scenario_omits_estimator(tmp_path):
    """A policy-only scenario (no [estimator]) loads and lowers standalone."""
    from shinro.codegen.scenario_gen import gen_scenario, load_scenario

    pytest.importorskip("onnx")
    spec = load_scenario(str(TOY_SCENARIO))
    assert spec["estimator_config"] is None

    out = tmp_path / "g"
    cg, graph_path = gen_scenario(str(TOY_SCENARIO), str(out))
    assert cg.inputs == ["state"]
    assert cg.outputs == ["u"]
    assert cg.state_outputs == []
    assert graph_path.exists()


def test_policy_scenario_provenance_pins_onnx_weights(tmp_path):
    """The deployment record's config slot commits to the exact .onnx file."""
    from shinro.codegen.scenario_gen import gen_scenario

    pytest.importorskip("onnx")
    out = tmp_path / "g"
    gen_scenario(str(TOY_SCENARIO), str(out))
    configs = json.loads((out / "graph_data_manifest.json").read_text())["provenance"]["configs"]
    key = next(k for k in configs if k.endswith("toy_mlp.onnx"))
    assert configs[key] == hashlib.sha256(TOY_ONNX.read_bytes()).hexdigest()


def test_non_policy_scenario_still_requires_estimator(tmp_path):
    """Only onnx_rl may drop [estimator]; a classical controller still needs one."""
    from shinro.codegen.scenario_gen import load_scenario

    scenario = _policy_scenario(
        tmp_path,
        controller_config="configs/controllers/lqr_base.toml",
        compile_section="[compile]\nn_x = 3\nn_u = 3\n",
    )
    with pytest.raises(ValueError, match=r"missing \[estimator\]"):
        load_scenario(str(scenario))


def test_compile_rejects_unsafe_artifact_name(tmp_path):
    """artifact_name is a file-name stem: separators must be rejected."""
    from shinro.codegen.scenario_gen import load_scenario

    bad = _scenario_toml(tmp_path, '[compile]\nn_x = 3\nn_u = 3\nartifact_name = "../evil"\n')
    with pytest.raises(ValueError, match="artifact_name"):
        load_scenario(str(bad))


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig not on PATH")
def test_e2e_policy_named_artifact(tmp_path):
    """The committed policy scenario builds lib_neural_network.so and verifies."""
    out = tmp_path / "scenario"
    gen = _run(GEN, str(TOY_SCENARIO), "--out", str(out))
    assert gen.returncode == 0, gen.stderr
    build = _run(BUILD, str(out), "--scenario", str(TOY_SCENARIO))
    assert build.returncode == 0, build.stderr
    assert "oracle B" in build.stdout

    so = out / "lib" / "lib_neural_network.so"
    record = out / "lib" / "lib_neural_network.deployment.json"
    assert so.exists()
    assert (out / "lib" / "lib_neural_network.manifest.json").exists()
    assert record.exists()
    assert not (out / "lib" / "libbase.so").exists()
    assert json.loads(record.read_text())["slots"]["binary"] == hashlib.sha256(so.read_bytes()).hexdigest()
