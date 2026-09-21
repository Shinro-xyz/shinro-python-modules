# `samples/` — example configs (documentation, not installed)

These TOMLs are **examples**: they show what a config looks like and how to
write one. shinro ships **no** config files — nothing here is installed, and
nothing here is resolved automatically.

Your project owns its configs. Copy the example you need, edit it, and point
your component/scenario at it.

## Layout

```
samples/
├── controllers/    one example config per registered controller
├── estimators/     kalman_*.toml and luenberger_*.toml twins per plant
├── plants/         one example config per registered plant
├── trajectories/   example waypoint / schedule generators
├── scenarios/      complete closed-loop scenarios (+ _template.toml)
└── robot_config.toml   MuJoCo sim manifest (plant list, joint groups)
```

`samples/scenarios/_template.toml` is the starting point for a new scenario —
it is a fully commented scenario with every section annotated.

## How to use a sample

Copy it into your project and reference it by a path **relative to your project
root**:

```bash
cp -r /path/to/shinro/samples/configs controllers/   # or whatever layout you like
```

```python
from shinro import ControllerFactory

lqr = ControllerFactory("configs/controllers/lqr_base.toml").create()
```

Then run from your project root:

```bash
cd my_project && python my_script.py
```

## Path resolution — the contract

`resolve_config_path` resolves a config path **against your current working
directory**, or uses it as-is when absolute. There is deliberately:

- **no** fallback into the shinro package (shinro ships no configs), and
- **no** basename search.

A path that does not resolve is a loud `FileNotFoundError` naming your CWD:

```
Config not found: configs/controllers/lqr_base.toml
(resolved relative to CWD /home/me/my_project; shinro ships no configs —
copy the sample you need from the repo's samples/)
```

That loudness is the point: an earlier implementation fell back to a packaged
copy by basename, so a misspelled or wrong-CWD path silently loaded *shinro's*
config instead of failing. If a path is wrong, you now find out immediately.

Two consequences worth knowing:

- **Run from your project root**, or use absolute paths. A scenario TOML's
  inner `config = "..."` references are resolved from the CWD too — not
  relative to the scenario file.
- **Nested paths are fine** (`configs/controllers/mine.toml`) but never
  flattened to a basename, so they cannot silently collide with something else.

## Running these samples from the repo

The samples reference each other with `samples/...` paths, so the in-repo
examples work when run from the repository root:

```bash
python -m demos.demo_double_pendulum          # uses samples/scenarios/...
shinro check samples/scenarios/cartpole_balance.toml
make compile SCENARIO=tests/integration/scenarios/mppi_pendulum_compile.toml
```

## Not to be confused with `tests/fixtures/configs/`

`tests/fixtures/configs/` is a separate, **test-only** namespace (arm/pendulum
variants used by the test suite). Those are not examples and are not covered by
this directory.
