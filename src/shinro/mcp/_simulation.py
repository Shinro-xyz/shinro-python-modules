"""Design-time simulation — close the loop without writing a script.

Everything else on this server answers "what would one control action be?". This
answers "what happens if I run it?", which is what tuning actually needs: an agent can
change a gain, re-run, and compare final and worst-case tracking error.

It reuses the library's own runner (:func:`shinro.simulation.runner.run_scenario`), the
same path the tests and the physics-backed suites use, so a number seen here means the
same thing as a number seen anywhere else. It is *numpy* simulation — it does not build
or load a kernel; the compiled path stays on the codegen server.
"""

import json

from shinro.mcp._common import server

#: Exceptions a bad scenario path or an unsupported scenario shape can raise, plus the
#: runner's own validation. Returned as an error string so the server never dies.
_SCENARIO_ERRORS = (FileNotFoundError, ValueError, KeyError, TypeError, RuntimeError)


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

    from shinro.factories import ScenarioFactory
    from shinro.simulation.runner import run_scenario

    try:
        scenario = ScenarioFactory(scenario_path).build()
        result = run_scenario(scenario, steps=steps, seed=seed)
    except _SCENARIO_ERRORS as e:
        return json.dumps({"error": f"{e.__class__.__name__}: {e}"})

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
        tracking = {
            "final": per_step[-1],
            "max": max(per_step),
            "mean": float(np.mean(per_step)),
        }

    controls = [_numeric(r.control) for r in result.records]
    max_action = (
        float(max(np.max(np.abs(np.asarray(c))) for c in controls if c is not None))
        if any(c is not None for c in controls)
        else None
    )
    finite = bool(np.all(np.isfinite(result.plant_state))) if n else True

    payload: dict = {
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
        "all_finite": finite,
        # [scenario.tolerance] checking from the TOML, when it declares any.
        "check": result.check(),
    }

    if samples and n:
        idx = np.unique(np.linspace(0, n - 1, min(samples, n)).astype(int))
        payload["trajectory"] = {
            "t": [result.records[i].t for i in idx],
            "plant_state": [(_numeric(result.records[i].plant_state) or []) for i in idx],
            "estimated": [(_numeric(result.records[i].estimated) or []) for i in idx],
            "reference": [(_numeric(result.records[i].reference) or []) for i in idx],
        }

    return json.dumps(payload)
