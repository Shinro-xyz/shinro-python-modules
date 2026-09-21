"""Tests for the unified ``shinro`` CLI (:mod:`shinro.cli`).

Covers the dispatcher verbs end to end where it is cheap (check / run / trace /
verify) and behind a ``zig`` guard for the full build → verify pipeline. The
plant-only cartpole scenario keeps the run/check tests off MuJoCo.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from shinro.cli import _resolve_out, main
from shinro.factories.registry import _CONTROLLER_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "samples"
LQR = str(CONFIG / "controllers" / "lqr_base.toml")
CARTPOLE = str(REPO_ROOT / "tests" / "integration" / "scenarios" / "cartpole_balance.toml")
BASE = str(REPO_ROOT / "tests" / "integration" / "scenarios" / "base_tracking.toml")


class TestResolveOut:
    @staticmethod
    def _spec(**over):
        return {"compile": {"optimize": "debug", "target": "native", "out": None, **over}}

    def test_default_is_per_scenario_and_mode(self):
        """The default keys on the scenario *and* the build mode, so builds never clobber."""
        assert (
            _resolve_out("tests/integration/scenarios/base_tracking.toml", self._spec(), None, None, None)
            == "build/base_tracking/debug-native"
        )
        assert (
            _resolve_out("a/cartpole_balance.toml", self._spec(), None, "release", "aarch64-linux-gnu")
            == "build/cartpole_balance/release-aarch64-linux-gnu"
        )

    def test_declared_out_wins_over_default(self):
        assert _resolve_out("a/b.toml", self._spec(out="build/custom"), None, None, None) == "build/custom"

    def test_cli_out_wins_over_declared(self):
        assert _resolve_out("a/b.toml", self._spec(out="build/custom"), "build/flag", None, None) == "build/flag"


class TestCheck:
    def test_component_constructs(self, capsys):
        assert main(["check", LQR]) == 0
        assert "OK: LQR (controller) constructs" in capsys.readouterr().out

    def test_scenario_constructs(self, capsys):
        assert main(["check", CARTPOLE]) == 0
        assert "scenario 'cartpole_balance' constructs" in capsys.readouterr().out

    def test_unrecognized_config_is_usage(self, tmp_path, capsys):
        bad = tmp_path / "neither.toml"
        bad.write_text("[foo]\nbar = 1\n")
        assert main(["check", str(bad)]) == 2
        assert "not a readable component config" in capsys.readouterr().err

    def test_missing_file_is_usage(self, capsys):
        assert main(["check", "does/not/exist.toml"]) == 2
        assert "not a readable component config" in capsys.readouterr().err


class TestRun:
    def test_plant_only_passes(self, capsys):
        assert main(["run", CARTPOLE]) == 0
        out = capsys.readouterr().out
        assert "steady_state:" in out
        assert "run OK: 1200 steps" in out

    def test_behavior_gate_fails_when_short(self, capsys):
        """Five steps is far from settled, so [scenario.tolerance] is exceeded."""
        assert main(["run", CARTPOLE, "--steps", "5"]) == 1
        assert "BEHAVIOR GATE FAILED" in capsys.readouterr().err

    def test_seed_is_reproducible(self):
        assert main(["run", CARTPOLE, "--seed", "7"]) == 0
        assert main(["run", CARTPOLE, "--seed", "7"]) == 0


class TestTrace:
    def test_inventory(self, capsys):
        assert main(["trace"]) == 0
        assert "controllers" in capsys.readouterr().out

    def test_list_contract(self, capsys):
        assert main(["trace", "--list", LQR]) == 0
        assert "current_state" in capsys.readouterr().out

    def test_trace_runs_oracle(self, capsys):
        assert main(["trace", LQR, "--shape", "current_state=3", "--shape", "target_state=3"]) == 0
        assert "oracle A" in capsys.readouterr().out


class TestVerify:
    def test_missing_record_is_usage(self, tmp_path, capsys):
        assert main(["verify", BASE, "--out", str(tmp_path / "nothing")]) == 2
        assert "no deployment record" in capsys.readouterr().err


class TestArgParse:
    def test_unknown_verb_exits_2(self):
        with pytest.raises(SystemExit) as e:
            main(["nope"])
        assert e.value.code == 2

    def test_module_entrypoint_help(self):
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
        proc = subprocess.run([sys.executable, "-m", "shinro.cli", "--help"], capture_output=True, text=True, env=env, cwd=REPO_ROOT)
        assert proc.returncode == 0
        assert "build" in proc.stdout


@pytest.mark.skipif(shutil.which("zig") is None, reason="zig not on PATH")
class TestBuildVerifyE2E:
    def test_build_then_verify(self, tmp_path, capsys):
        out = str(tmp_path / "out")
        assert main(["build", BASE, "--out", out]) == 0
        assert (Path(out) / "lib" / "libbase.so").exists()
        assert "oracle B" in capsys.readouterr().out

        assert main(["verify", BASE, "--out", out]) == 0
        assert "OK:" in capsys.readouterr().out

    def test_default_out_is_per_mode(self, tmp_path, monkeypatch, capsys):
        """With no --out, build and verify agree on build/<stem>/<optimize>-<target>."""
        # Config resolution is CWD-relative and shinro ships no configs, so a
        # scenario run from a different CWD needs its sample configs present there.
        shutil.copytree(REPO_ROOT / "samples", tmp_path / "samples")
        monkeypatch.chdir(tmp_path)
        assert main(["build", BASE]) == 0
        out = Path("build") / "base_tracking" / "debug-native"
        assert (out / "lib" / "libbase.so").exists()
        capsys.readouterr()
        assert main(["verify", BASE]) == 0
        assert "OK:" in capsys.readouterr().out

    def test_verify_detects_drift(self, tmp_path, capsys):
        out = str(tmp_path / "out")
        assert main(["build", BASE, "--out", out]) == 0
        capsys.readouterr()
        config = REPO_ROOT / "samples" / "controllers" / "lqr_base.toml"
        original = config.read_bytes()
        try:
            config.write_bytes(original + b"\n# tamper\n")
            assert main(["verify", BASE, "--out", out]) == 1
            assert "VERIFY FAILED" in capsys.readouterr().out
        finally:
            config.write_bytes(original)


class TestImportFlag:
    """``--import MODULE`` makes a third-party @register_* component visible.

    shinro imports only its built-ins, so without the flag the registry lookup
    fails; the flag is the CLI's equivalent of a driver script's ``import``.
    """

    MOD = "shinro_ext_component"
    TYPE = "ExtP"

    @pytest.fixture
    def ext_module(self, tmp_path, monkeypatch):
        """Create a real importable module that registers a controller."""
        (tmp_path / f"{self.MOD}.py").write_text(
            "from dataclasses import dataclass\n"
            "from shinro.components import Controller\n"
            "from shinro.factories.registry import register_controller\n"
            "from shinro.utils.array_backend import NumpyBackend, parse_matrix\n"
            "\n"
            "@dataclass(frozen=True)\n"
            "class ExtCfg:\n"
            "    gain: list[list[float]]\n"
            "    name: str = 'extp'\n"
            "\n"
            "@register_controller('ExtP')\n"
            "class ExtP(Controller):\n"
            "    Config = ExtCfg\n"
            "\n"
            "    def __init__(self, gain_matrix, backend=None):\n"
            "        self.bk = backend or NumpyBackend()\n"
            "        self.K = gain_matrix\n"
            "\n"
            "    def compute(self, current_state, target_state=None):\n"
            "        if target_state is None:\n"
            "            target_state = self.bk.zeros_like(current_state)\n"
            "        return self.K @ (target_state - current_state)\n"
            "\n"
            "    @classmethod\n"
            "    def from_config(cls, config, backend=None):\n"
            "        bk = backend or NumpyBackend()\n"
            "        return cls(parse_matrix(bk, cls.parse_config(config).gain), backend=bk)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        yield self.MOD
        # The decorator side-effect persists in the registry, so undo both.
        sys.modules.pop(self.MOD, None)
        _CONTROLLER_REGISTRY.pop(self.TYPE, None)

    @pytest.fixture
    def ext_config(self, tmp_path):
        cfg = tmp_path / "extp.toml"
        cfg.write_text("type = 'ExtP'\ngain = [[1.0, 0.0], [0.0, 1.0]]\n")
        return str(cfg)

    def test_flag_after_verb(self, ext_module, ext_config, capsys):
        assert main(["check", ext_config, "--import", ext_module]) == 0
        assert f"OK: {self.TYPE} (controller) constructs" in capsys.readouterr().out

    def test_flag_before_verb(self, ext_module, ext_config, capsys):
        assert main(["--import", ext_module, "check", ext_config]) == 0
        assert f"OK: {self.TYPE} (controller) constructs" in capsys.readouterr().out

    def test_invisible_without_flag(self, ext_module, ext_config, capsys):
        """The import is what enables the type — without it the lookup fails."""
        assert main(["check", ext_config]) == 2
        assert self.TYPE in capsys.readouterr().err

    def test_inventory_lists_imported_type(self, ext_module, capsys):
        assert main(["trace", "--import", ext_module]) == 0
        assert self.TYPE in capsys.readouterr().out

    def test_unimportable_module_is_clean_error(self, ext_config, capsys):
        assert main(["check", ext_config, "--import", "definitely_not_a_module_xyz"]) == 2
        err = capsys.readouterr().err
        assert "cannot import" in err
        assert "definitely_not_a_module_xyz" in err
