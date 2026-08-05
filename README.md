# Shinro Python Modules

A clean, modular Python control framework built on five abstract base classes
— **Controller**, **Plant**, **StateEstimator**, **TrajectoryGenerator**, and
**PhysicsEngine** — with registry-based factories that compose them from TOML
config. The architecture is robot-agnostic: **LeKiwi** (holonomic base +
6-DOF arm) is the current reference robot used in the demos below, not a
framework constraint.

> **Naming note:** this repo was previously scoped and named for LeKiwi only
> (`lerobot-mpc-lekiwi`). It was renamed to `shinro-python-modules` because
> the registry/factory/ABC pattern generalized beyond one robot.

## Documentation

Conceptual and operational documentation lives in
[docs.shinro.xyz](https://docs.shinro.xyz):

- **[Control Architecture](https://docs.shinro.xyz/control-architecture/)** — why the five ABCs exist, how they compose, sim/hardware parity
- **[Python Modules](https://docs.shinro.xyz/python-modules/)** — the operational guide: install, run, extend, component catalog

This README stays limited to repo-local setup and contributor pointers.

## Install

From a source checkout (recommended for development):

```bash
pip install -e .            # core (numpy, scipy, osqp, mcp)
pip install -e ".[mujoco]"  # add MuJoCo physics backend
pip install -e ".[torch]"   # add torch backend
pip install -e ".[lerobot]" # add learned-policy adapter
```

Once published, consumers can install from a git ref or an index without a
checkout:

```bash
pip install "shinro[mujoco,torch] @ git+https://github.com/<org>/shinro-python-modules@v0.1.0"
```

The MCP server is installed as a console command: `shinro-mcp`.

## Releases

Versions are derived from git tags via `setuptools-scm`. A release is a
`vX.Y.Z` tag pushed to `origin`; the `.github/workflows/release.yml` workflow
builds the wheel + sdist, smoke-tests them, and attaches them to a GitHub
Release with auto-generated notes.

Semantic versioning:
- `patch` — backwards-compatible fix (`v0.1.0` → `v0.1.1`)
- `minor` — backwards-compatible feature (`v0.1.1` → `v0.2.0`)
- `major` — incompatible API change (`v0.2.0` → `v1.0.0`)

To cut a release:

```bash
# 1. (optional but recommended) draft release notes
make release-patch   # or release-minor / release-major

# 2. commit the changelog, then push the tag to trigger the release
git push origin v0.1.1
```

Development builds between tags are auto-numbered (`0.1.2.devN+g<sha>`).
Local build + install sanity: `make install` and `make build`.

## Run a demo

```bash
python -m demos.demo_simple                              # terminal-only, no viewer
python -m demos.demo_arm_trajectory                       # arm trajectory + live viewer
python -m demos.demo_base_tracking                        # base tracking, LQR + observer
python -m demos.demo_base_tracking --controller mpc       # base tracking, MPC
python -m demos.demo_pick_and_place                       # full pick-and-place sequence
```

Auto-generate a robot config from a MuJoCo model:

```bash
python scripts/generate_robot_config.py lekiwi-sim/mjcf_lcmm_robot.xml > robot_config.toml
```

## Repo structure

See [Python Modules](https://docs.shinro.xyz/python-modules/) for the annotated
architecture and component catalog. For the raw file tree, browse the repo on
GitHub rather than reading it out of this README — it drifts.

## Contributing / agent instructions

See [`AGENTS.md`](./AGENTS.md) for the codebase index (Hermes) and lab-notes
workflow used by agents working in this repo.

## Status

See the [Roadmap](https://docs.shinro.xyz/roadmap/) for what's shipped vs.
planned, including Shinro Studio integration status.
