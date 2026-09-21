"""config_resolver: filesystem-only resolution, and missing paths fail loudly.

shinro ships no configs, so resolution is absolute-path-or-CWD only. The
regression these tests pin down is the old silent fallback: a relative path
that did not resolve from the CWD used to have its *basename* joined onto the
packaged ``shinro/configs`` directory, so a wrong-CWD path silently loaded
shinro's own config instead of failing.
"""

from __future__ import annotations

import pytest

from shinro.utils.config_resolver import resolve_config_path


class TestResolves:
    def test_absolute_path(self, tmp_path):
        cfg = tmp_path / "c.toml"
        cfg.write_text("type = 'x'\n")
        assert resolve_config_path(str(cfg)) == cfg

    def test_cwd_relative_path(self, tmp_path, monkeypatch):
        (tmp_path / "configs").mkdir()
        cfg = tmp_path / "configs" / "c.toml"
        cfg.write_text("type = 'x'\n")
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path("configs/c.toml").exists()

    def test_nested_cwd_relative_path(self, tmp_path, monkeypatch):
        (tmp_path / "myproj" / "configs").mkdir(parents=True)
        cfg = tmp_path / "myproj" / "configs" / "c.toml"
        cfg.write_text("type = 'x'\n")
        monkeypatch.chdir(tmp_path)
        assert resolve_config_path("myproj/configs/c.toml").exists()


class TestFailsLoudly:
    def test_missing_absolute_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config not found"):
            resolve_config_path(str(tmp_path / "nope.toml"))

    def test_missing_relative_path_names_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="shinro ships no configs"):
            resolve_config_path("mysub2/robot_config.toml")

    def test_bare_filename_does_not_reach_the_package(self, tmp_path, monkeypatch):
        """A bare name must not resolve into shinro's own (now nonexistent) configs."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            resolve_config_path("robot_config.toml")

    def test_packaged_config_dir_is_gone(self):
        """The packaged configs directory no longer exists — nothing to fall back to."""
        from pathlib import Path

        import shinro

        pkg_dir = Path(shinro.__file__).parent
        assert not (pkg_dir / "configs").exists()
