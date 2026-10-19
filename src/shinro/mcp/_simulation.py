"""Design-time simulation — close the loop, and search over closing it many ways.

`simulate_scenario` answers "what happens if I run this?". `sweep_scenarios` answers
"which of these should I keep?" — it runs several trials and ranks them by one metric,
which is the difference between a technician trying things and a researcher comparing
them.

Both reuse the library's own runner (:func:`shinro.simulation.runner.run_scenario`), so
a number here means the same thing as a number anywhere else in the project. Neither
builds or loads a kernel; the compiled path stays on the codegen server.

A **trial is a config file**, not a parameter dictionary: candidates name scenario and
component-config paths that the caller wrote. That keeps one source of truth, makes a
sweep reviewable as a diff, and needs no TOML writer.
"""

import json

from shinro.mcp._common import server

#: Exceptions a bad scenario path or an unsupported scenario shape can raise, plus the
#: runner's own validation. Returned as an error string so the server never dies.
_SCENARIO_ERRORS = (FileNotFoundError, ValueError, KeyError, TypeError, RuntimeError)

#: Component sections a candidate may point at a different config file.
_COMPONENT_KEYS = ("controller_config", "estimator_config", "plant_config")


def _numeric(value) -> list | None:
    """Return a JSON-able flat list for array-like data, or ``None`` for e.g. dicts.

    Feedforward phase schedules record dict setpoints, not vectors; those are reported
    as ``None`` rather than serialised as a dict the caller has to special-case.
    """
    import numpy as np

    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.dtype == object or arr.ndim == 0:
        return None
    return arr.reshape(-1).tolist()


def _build(scenario_path: str, overrides: dict):
    """Build a scenario, applying per-component config-path overrides in memory.

    `ScenarioFactory` loads the scenario TOML into `self.config` and `build()` reads
    that dict, so pointing a section at another config file before building is enough —
    nothing on disk is touched.
    """
    from shinro.factories import ScenarioFactory

    factory = ScenarioFactory(scenario_path)
    for component, key in (("controller", "controller_config"), ("estimator", "estimator_config"), ("plant", "plant_config")):
        path = overrides.get(key)
        if not path:
            continue
        if component not in factory.config:
            raise ValueError(f"{scenario_path}: candidate overrides {key} but the scenario has no [{component}] section")
        factory.config[component]["config"] = path
    return factory.build()


def _run(scenario_path: str, steps: int | None, seed: int | None, overrides: dict):
    """Build and run one scenario, returning ``(scenario, result)``."""
    from shinro.simulation.runner import run_scenario

    scenario = _build(scenario_path, overrides)
    return scenario, run_scenario(scenario, steps=steps, seed=seed)


def _summarise(scenario, result, scenario_path: str) -> dict:
    """The numeric summary both simulation tools report (no trajectory)."""
    import numpy as np

    n = len(result)
    cfg = getattr(scenario, "config", {}) or {}
    dt = cfg.get("scenario", {}).get("dt") or getattr(getattr(scenario, "plant", None), "dt", None)
    last = result.records[-1] if n else None

    # Tracking error is the L-infinity distance from the reference; only computable
    # when references are vectors (not feedforward dict setpoints).
    tracking = None
    per_step = []
    for record in result.records:
        ref, plant = _numeric(record.reference), _numeric(record.plant_state)
        if ref is None or plant is None or len(ref) != len(plant):
            per_step = []
            break
        per_step.append(float(np.max(np.abs(np.asarray(plant) - np.asarray(ref)))))
    if per_step:
        tracking = {"final": per_step[-1], "max": max(per_step), "mean": float(np.mean(per_step))}

    controls = [_numeric(r.control) for r in result.records]
    max_action = (
        float(max(np.max(np.abs(np.asarray(c))) for c in controls if c is not None))
        if any(c is not None for c in controls)
        else None
    )

    return {
        "scenario": cfg.get("scenario", {}).get("name", scenario_path),
        "steps": n,
        "dt": dt,
        "duration": (n * dt) if (n and dt) else None,
        "final": {
            "plant_state": _numeric(last.plant_state) if last else None,
            "estimated": _numeric(last.estimated) if last else None,
            "reference": _numeric(last.reference) if last else None,
        },
        "tracking": tracking,
        "max_abs_action": max_action,
        "all_finite": bool(np.all(np.isfinite(result.plant_state))) if n else True,
        # [scenario.tolerance] checking from the TOML, when it declares any.
        "check": result.check(),
    }


def _lookup(payload: dict, dotted: str):
    """Read a dotted path (``"tracking.final"``) out of a summary, or ``None``."""
    current = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


@server.tool()
def simulate_scenario(
    scenario_path: str,
    steps: int | None = None,
    seed: int | None = None,
    samples: int = 20,
) -> str:
    """Run a scenario closed-loop in numpy and report how well it tracked.

    The scenario needs a `[plant]` section (a `type` + `config`, or a `[sim]` robot);
    compile-only scenarios (dims declared in `[compile]`) cannot be simulated. Steps
    default to `[scenario].duration / dt`.

    Use this to **tune**: change a gain or a weight, re-run, compare
    `final_tracking_error` and `max_tracking_error`. When the numbers look right,
    compile the same scenario with the codegen server and the `.so` will behave the
    same way (the oracle proves it).

    Args:
        scenario_path: Scenario TOML, relative to the server's working directory.
        steps: Number of control steps; defaults to the scenario's duration / dt.
        seed: RNG seed for measurement noise and stochastic components.
        samples: How many evenly spaced trajectory points to include (default 20);
            pass 0 for summary metrics only.
    """
    import numpy as np

    try:
        scenario, result = _run(scenario_path, steps, seed, {})
    except _SCENARIO_ERRORS as e:
        return json.dumps({"error": f"{e.__class__.__name__}: {e}"})

    payload = _summarise(scenario, result, scenario_path)

    n = len(result)
    if samples and n:
        idx = np.unique(np.linspace(0, n - 1, min(samples, n)).astype(int))
        payload["trajectory"] = {
            "t": [result.records[i].t for i in idx],
            "plant_state": [(_numeric(result.records[i].plant_state) or []) for i in idx],
            "estimated": [(_numeric(result.records[i].estimated) or []) for i in idx],
            "reference": [(_numeric(result.records[i].reference) or []) for i in idx],
        }

    return json.dumps(payload)


@server.tool()
def sweep_scenarios(
    candidates: list,
    scenario_path: str | None = None,
    steps: int | None = None,
    seed: int | None = None,
    objective: str = "tracking.final",
    direction: str = "min",
    max_trials: int = 50,
) -> str:
    """Run several trials and rank them by one metric.

    This is the compare step of a design loop: the answer is a sorted table, so an
    agent can keep the best and discard the rest in one call instead of N.

    A **trial is a config file**. Each candidate is either

    - a scenario path string, or
    - an object ``{"label": ..., "scenario": <path?>, "controller_config": <path?>,
      "estimator_config": <path?>, "plant_config": <path?>}`` where the config paths
      replace the sections of the scenario (in memory — no file is modified).

    You author the candidate configs; the sweep does not invent parameter grids, which
    keeps every trial reviewable as a diff and means the winner is a file you can point
    at, not a value only this call knows about.

    Args:
        candidates: List of scenario paths and/or candidate objects (see above).
        scenario_path: Base scenario for candidates that omit ``"scenario"``.
        steps: Control steps per trial; defaults to each scenario's duration / dt.
        seed: RNG seed, applied to every trial so comparisons are like-for-like.
        objective: Dotted path into the summary — e.g. ``tracking.final`` (default),
            ``tracking.max``, ``max_abs_action``, ``duration``.
        direction: ``"min"`` (default) or ``"max"``.
        max_trials: Safety cap on the number of trials (default 50).
    """
    if direction not in ("min", "max"):
        return json.dumps({"error": f"direction must be 'min' or 'max', got '{direction}'"})
    if not candidates:
        return json.dumps({"error": "candidates is empty"})
    if len(candidates) > max_trials:
        return json.dumps({"error": f"{len(candidates)} candidates exceeds max_trials={max_trials}"})

    ranked: list[dict] = []
    failed: list[dict] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, str):
            spec: dict = {"scenario": candidate, "label": candidate.rsplit("/", 1)[-1].removesuffix(".toml")}
        elif isinstance(candidate, dict):
            spec = candidate
        else:
            failed.append({"label": f"candidate-{index}", "error": "must be a path string or an object"})
            continue

        trial_path = spec.get("scenario") or scenario_path
        label = str(spec.get("label") or (trial_path.rsplit("/", 1)[-1].removesuffix(".toml") if trial_path else f"candidate-{index}"))
        if not trial_path:
            failed.append({"label": label, "error": "no scenario path (set 'scenario' or pass scenario_path)"})
            continue

        overrides = {key: spec[key] for key in _COMPONENT_KEYS if spec.get(key)}
        try:
            scenario, result = _run(trial_path, steps, seed, overrides)
        except _SCENARIO_ERRORS as e:
            failed.append({"label": label, "error": f"{e.__class__.__name__}: {e}"})
            continue

        summary = _summarise(scenario, result, trial_path)
        value = _lookup(summary, objective)
        if value is None:
            failed.append({"label": label, "error": f"objective '{objective}' is not a number in this run's summary"})
            continue
        ranked.append(
            {
                "label": label,
                "value": value,
                "scenario": trial_path,
                **{key: spec[key] for key in _COMPONENT_KEYS if spec.get(key)},
                "tracking": summary["tracking"],
                "max_abs_action": summary["max_abs_action"],
                "all_finite": summary["all_finite"],
            }
        )

    ranked.sort(key=lambda row: row["value"], reverse=(direction == "max"))
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position

    best = ranked[0] if ranked else None
    return json.dumps(
        {
            "objective": objective,
            "direction": direction,
            "steps": steps,
            "trials": len(candidates),
            "ranked": ranked,
            "failed": failed,
            # What to keep: the winner's files, ready to wire into the scenario.
            "best": (
                {k: v for k, v in best.items() if k in ("label", "value", "scenario", *_COMPONENT_KEYS)}
                if best
                else None
            ),
        }
    )
