---
name: port-numpy-component
description: Port a raw numpy implementation of a controller, plant, estimator, trajectory generator, or physics engine into the shinro package — make it backend-agnostic, register it, config-drive it, test it, lint it, and document it in lab notes. Use when the user hands over a numpy script or asks to integrate a new component into the framework.
license: MIT
compatibility: opencode
metadata:
  author: shinro
  version: "1.0.0"
  domain: robotics
  triggers: port numpy, convert numpy to framework, add new controller, add new plant, add new estimator, add new trajectory, add new physics engine, register component, from_config
  role: specialist
  scope: implementation
  output-format: code
  related-skills: python-pro, test-repo, code-documenter
---

# Port Numpy Component

Turn a raw numpy implementation of any framework component (controller, plant,
state estimator, trajectory generator, physics engine) into a first-class
`shinro.*` package component. Follow the steps in order; do not skip the
verification or documentation steps.

## When to Use This Skill

- The user provides a numpy script and wants it integrated into the package
- Adding a new component (controller, plant, estimator, trajectory, physics engine) to the framework
- Converting an existing standalone/piloting implementation to a registered, config-driven component

## Repo Reference

- ABCs to inherit from: `src/shinro/components.py` (Controller, Plant, StateEstimator, TrajectoryGenerator, PhysicsEngine)
- Registration decorators: `src/shinro/factories/registry.py` — `register_controller`, `register_plant`, `register_estimator`, `register_trajectory`, `register_plant_detector`
- Backend abstraction: `src/shinro/utils/array_backend.py` (NumpyBackend / TorchBackend)
- Public API export points: `src/shinro/{controllers,plants,estimators,trajectories,physics_engine}/__init__.py`
- Component config: top-level `robot_config.toml` (`[[plants]]`, `[[controllers]]`, `[[estimators]]`, `[[trajectories]]` arrays)
- Test templates: `tests/test_controllers.py`, `tests/test_plants.py`, `tests/test_estimators.py`, `tests/test_trajectories.py`
- Lab notes: `lab-notes/daily/<date>.md`

## Workflow

1. **Understand** — Read the script. Identify the math: state dimensions, governing equations, parameters, inputs/outputs. Map it to the right ABC in `src/shinro/components.py`. Read one existing component of the same type to mirror its structure and conventions.
2. **Test** — Run the raw script to establish baseline behavior (expected outputs, convergence, edge cases) before any changes.
3. **Backend-agnostic** — Replace `np.xxx` with `self.bk.xxx()`. If the backend is missing a method, add it to `ArrayBackend`/`NumpyBackend`/`TorchBackend` in `src/shinro/utils/array_backend.py`. Do not hardcode numpy; the component must run under both backends.
4. **Register** — Add the appropriate `@register_*` decorator from `src/shinro/factories/registry.py` and implement a `from_config` classmethod (validate params, raise on invalid config).
5. **Config** — Register the component in `robot_config.toml` under the matching section array (`[[controllers]]`, `[[plants]]`, `[[estimators]]`, `[[trajectories]]`), with a `type` field matching the registered name.
6. **Export** — Add the class to the corresponding subpackage `__init__.py` so it is importable as `shinro.<subpackage>.<ClassName>`.
7. **Test** — Write tests matching the existing pattern in `tests/test_*.py`: construction validation, compute output shapes, mathematical accuracy (verify the governing equation analytically), convergence, error handling, and `from_config`.
8. **Lint** — Run `ruff check .` then `pyright` (see `make lint` for the pyright path list). Fix all findings.
9. **Full suite** — Run `make test`. All tests must pass.
10. **Docstrings** — Add module, class, method, and property docstrings matching the codebase convention: Sphinx-compatible `:math:` inline math, `Args:`, `Returns:`, and `Config fields:` blocks.
11. **Document** — Write a semantic summary in `lab-notes/daily/<date>.md`: what was built, why, key design decisions, and test results.

## Constraints

### MUST DO
- Import code as `shinro.*`, never by top-level module path
- Inherit from the correct ABC in `src/shinro/components.py`
- Use `self.bk` for all array ops; never call `np.` directly inside the component
- Add the `@register_*` decorator and `from_config` classmethod
- Run the full lint + test suite before finishing
- Write the lab note — this step is mandatory, not optional

### MUST NOT DO
- Query or re-index the stale `.codebase/` SQLite index
- Bypass the registry or `from_config`
- Skip docstrings or ship undocumented public methods
- Leave `np.` calls in component code
- Skip the lab note
