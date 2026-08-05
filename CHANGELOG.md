# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version numbers are derived from git tags via `setuptools-scm` — a release is
simply a `vX.Y.Z` tag pushed to `origin` (see `make release-patch`, `make
release-minor`, `make release-major` and the README "Releases" section).

## [Unreleased]

### Added

- Packaging: converted the repo to a `src/shinro/` installable package with
  `setuptools-scm` versioning, TOML configs shipped in the wheel, and the MCP
  server exposed as the `shinro-mcp` console command.
- Config resolution: `shinro.utils.config_resolver` lets factories load
  `configs/...` paths from an installed wheel or a source checkout.
- CI: installs the package and runs import/version smoke checks.

### Fixed

- MCP dependency bounded to `mcp>=1.28,<2` (mcp 2.x removed `mcp.server.fastmcp`).

## [0.1.0] - 2026-08-05

### Added

- Initial release as an installable package.

[Unreleased]: https://github.com/Shinro-xyz/shinro-python-modules/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Shinro-xyz/shinro-python-modules/releases/tag/v0.1.0
