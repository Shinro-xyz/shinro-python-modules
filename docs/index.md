---
title: Home
---

# shinro

Whole-body control framework: controllers, plants, estimators, trajectories, and
a MuJoCo-backed simulation factory.

`shinro` is built on **five abstract base classes** — `Controller`, `Plant`,
`StateEstimator`, `TrajectoryGenerator`, and `PhysicsEngine` — with concrete
implementations assembled from TOML config via registry-based factories, and a
swappable numpy/torch array backend.

## Start here

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)** — install and run your first simulation end to end.
- **[How it works](how-it-works.md)** — the ABC model, factories, and the compose/lower pipeline.
- **[Components](components.md)** — how config-driven components are declared and validated.
- **[API Reference](reference/components.md)** — every exported symbol, generated from source.

</div>

## Layout

| Package | Contents |
|---|---|
| [`shinro.components`](reference/components.md) | The five ABCs and the `ConfigDriven` mixin |
| [`shinro.trajectories`](reference/trajectories.md) | Reference path generators |
| [`shinro.controllers`](reference/controllers.md) | LQR, PID, MPC, MPPI, SMC, RL adapters |
| [`shinro.estimators`](reference/estimators.md) | Kalman filter, Luenberger observer |
| [`shinro.plants`](reference/plants.md) | Robot models |
| [`shinro.factories`](reference/factories.md) | Registry-based TOML factories, `Scenario` |
| [`shinro.utils`](reference/utils.md) | Array backend, linearization, controllability |
| [`shinro.simulation`](reference/simulation.md) | Robot simulation factory |

## Install

```bash
pip install -e ".[mujoco,media]"    # MuJoCo + plotting for the demos
```

## How this reference stays current

The **API Reference** section is generated, not written by hand.
`scripts/gen_api.py` walks each subpackage's `__all__` and emits one page per
subpackage, so **adding an export is all it takes** for it to appear here on the
next build. A subpackage without `__all__` falls back to an AST scan of its
source files.

```bash
python scripts/gen_api.py   # regenerate reference page + nav
mkdocs serve                # preview at http://127.0.0.1:8000
```

Prose pages are hand-written and listed in `docs/_nav_prose.yml`. The generated
nav is composed into `docs/SUMMARY.md`, which is a build artifact.
