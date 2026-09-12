"""Closed-loop and feedforward scenario runners — the public simulation API.

Moved from ``tests/integration/helpers/scenario_runner.py``: the loop spec
(duration, dt, noise, adversarial faults, input limits, tolerances) is declared
in the scenario TOML, so running a scenario is data, not code. Two drivers:

* :func:`iter_scenario` / :func:`run_scenario` — a closed-loop run for
  single-plant scenarios (base tracking, arm Cartesian, adversarial). The
  estimator feeds the controller, which drives the plant, and the per-step
  history is yielded as :class:`StepRecord` entries (collect with
  :func:`run_scenario`).
* :func:`iter_phase_schedule` / :func:`run_phase_schedule` — a feedforward run
  for ``phase_list`` schedules (pick-and-place) where the schedule itself is
  the control: each step carries per-signal setpoints routed to plants by
  name or to actuators via the scenario's ``[signals]`` table, applied to
  the composed ``RobotSim``.

Importing this module does not require mujoco: plant-only scenarios run on the
minimal install (the physics engine is only needed for sim-backed scenarios).
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shinro.controllers.smc import SlidingModeController

MAX_CONTROL = 1e6
DEFAULT_SEED = 42


@dataclass
class StepRecord:
    """One recorded step of an integration run.

    Args:
        t: Simulation time (s).
        reference: Reference setpoint at this step (feedforward runs carry a
            dict of per-signal setpoints).
        true_state: Plant state (noiseless).
        measurement: Noisy/adversarial measurement fed to the estimator.
        estimated: Estimator output.
        control: Control input applied to the plant (feedforward: the applied
            setpoints).
        plant_state: Plant state read back after the step.
    """

    t: float
    reference: Any
    true_state: np.ndarray
    measurement: np.ndarray
    estimated: np.ndarray
    control: Any
    plant_state: np.ndarray


def _stack(values: list) -> Any:
    """Stack per-step values into an (n, k) array when they are homogeneous.

    Feedforward runs record dict setpoints (``{"left_arm": ..., "torso": ...}``) as
    ``reference``/``control`` — those are returned as the raw list.
    """
    if not values:
        return np.empty((0,))
    try:
        rows = [np.asarray(v, dtype=np.float64).ravel() for v in values]
        if len({r.shape[0] for r in rows}) != 1:
            return values
        return np.asarray(rows)
    except (TypeError, ValueError):
        return values


@dataclass
class SimResult:
    """Collected run history plus tolerance checking against the scenario TOML.

    Iterable/indexable like the records list (``__iter__``/``__getitem__``
    delegate to ``records``), so existing test helpers operate on it unchanged.

    Args:
        records: One :class:`StepRecord` per step.
        config: The raw scenario TOML dict — ``check()`` reads
            ``[scenario.tolerance]`` from it.
    """

    records: list[StepRecord]
    config: dict = field(default_factory=dict)

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    @property
    def t(self) -> np.ndarray:
        return np.array([r.t for r in self.records])

    @property
    def true_state(self) -> np.ndarray:
        return np.asarray([r.true_state for r in self.records])

    @property
    def estimated(self) -> np.ndarray:
        return np.asarray([r.estimated for r in self.records])

    @property
    def plant_state(self) -> np.ndarray:
        return np.asarray([r.plant_state for r in self.records])

    @property
    def measurement(self) -> np.ndarray:
        return np.asarray([r.measurement for r in self.records])

    @property
    def reference(self) -> Any:
        return _stack([r.reference for r in self.records])

    @property
    def control(self) -> Any:
        return _stack([r.control for r in self.records])

    def steady_state_error(self, tail: int = 100) -> float:
        """Mean ``||reference - plant_state||`` over the trailing ``tail`` steps."""
        errs = [
            float(np.linalg.norm(np.asarray(r.reference) - np.asarray(r.plant_state)))
            for r in self.records
        ]
        return float(np.mean(errs[-tail:]))

    def estimator_error(self, tail: int = 100) -> float:
        """Mean ``||estimated - true_state||`` over the trailing ``tail`` steps."""
        errs = [
            float(np.linalg.norm(np.asarray(r.estimated) - np.asarray(r.true_state)))
            for r in self.records
        ]
        return float(np.mean(errs[-tail:]))

    def check(self, tail: int = 100) -> dict:
        """Apply ``[scenario.tolerance]`` from the scenario TOML.

        Returns a report — never raises (assertion helpers belong to tests):

            {"steady_state": {"error": 0.031, "tol": 0.05, "ok": True},
             "estimator":    {"error": 0.010, "tol": 0.03, "ok": True},
             "ok": True}

        Metrics that are not computable for this run kind (e.g. steady-state
        tracking for feedforward phase schedules, whose references are dicts)
        are reported with ``ok: None``.
        """
        tol = self.config.get("scenario", {}).get("tolerance", {})
        report: dict[str, Any] = {}
        if "steady_state" in tol:
            try:
                err = self.steady_state_error(tail)
                report["steady_state"] = {
                    "error": err, "tol": float(tol["steady_state"]), "ok": err <= float(tol["steady_state"]),
                }
            except (TypeError, ValueError):
                report["steady_state"] = {"ok": None, "reason": "not computable for this run kind"}
        if "estimator" in tol:
            try:
                err = self.estimator_error(tail)
                report["estimator"] = {
                    "error": err, "tol": float(tol["estimator"]), "ok": err <= float(tol["estimator"]),
                }
            except (TypeError, ValueError):
                report["estimator"] = {"ok": None, "reason": "not computable for this run kind"}
        oks = [v["ok"] for v in report.values()]
        report["ok"] = all(oks) if oks and all(o is not None for o in oks) else None
        return report


def _control_input_dim(scenario) -> int:
    """Infer the control input dimension from the plant's model."""
    _, B = scenario.plant.get_model()
    return int(B.shape[1])


def _inject_noise(state: np.ndarray, noise_cfg: dict | None, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian measurement noise from a ``[noise.measurement]`` config."""
    if not noise_cfg:
        return state.copy()
    std = np.asarray(noise_cfg.get("std", 0.0), dtype=np.float64)
    std = np.broadcast_to(std, state.shape)
    return state + rng.normal(0.0, std)


def _inject_adversarial(
    measurement: np.ndarray,
    adversarial_cfg: dict | None,
    step: int,
) -> tuple[np.ndarray, bool]:
    """Apply an adversarial fault at the configured step.

    Returns a tuple of (possibly-corrupted measurement, whether a fault fired).
    """
    if not adversarial_cfg or adversarial_cfg.get("inject_at") != step:
        return measurement, False
    value = adversarial_cfg.get("value", "nan")
    out = measurement.copy()
    if value == "nan":
        out[:] = np.nan
    elif value == "inf":
        out[:] = np.inf
    else:
        out[:] = float(value)
    return out, True


def _control_limits(scenario) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) clip limits from ``[scenario.input_limits]``.

    Defaults to a large symmetric bound so unconstrained scenarios stay stable.
    """
    limits = scenario.config.get("scenario", {}).get("input_limits")
    n_u = _control_input_dim(scenario)
    if limits is None:
        return np.full(n_u, -MAX_CONTROL), np.full(n_u, MAX_CONTROL)
    return (
        np.asarray(limits["min"], dtype=np.float64),
        np.asarray(limits["max"], dtype=np.float64),
    )


def _compute_control(ctrl, estimate: np.ndarray, reference: np.ndarray, u_prev: np.ndarray, takes_u_prev: bool) -> np.ndarray:
    """Compute the control input via the uniform controller signature.

    Every scenario-runnable controller implements
    ``compute(current_state, target_state=None)`` — LQR/PID/MPC/MPPI all
    regulate ``current_state`` toward ``target_state`` (MPC internally forms
    the tracking error). Controllers that declare ``u_prev`` (MPC_DeltaU) get
    the previous control; the flag is computed once per run, not per step.
    """
    if takes_u_prev:
        return ctrl.compute(estimate, reference, u_prev=u_prev)
    return ctrl.compute(estimate, reference)


def _total_steps(scenario, steps: int | None) -> int:
    """Resolve the step count from ``steps`` or ``[scenario].duration / dt``."""
    dt = float(scenario.config.get("scenario", {}).get("dt", scenario.sim.engine.dt if scenario.sim is not None else 0.01))
    duration = float(scenario.config.get("scenario", {}).get("duration", 5.0))
    total = steps if steps is not None else int(round(duration / dt))
    return total


def iter_scenario(scenario, steps: int | None = None, seed: int | None = None) -> Iterator[StepRecord]:
    """Run the closed loop for a single-plant scenario, yielding one record per step.

    The fixed ABC dataflow — ``y → estimator → x̂ → controller → u → [clip] →
    plant`` — with noise, adversarial faults, and input limits from the TOML.
    The same loop the codegen pipeline traces into a compiled graph.

    Args:
        scenario: Composed scenario from :class:`~shinro.factories.scenario_factory.ScenarioFactory`.
        steps: Number of steps. Defaults to ``[scenario].duration / dt``.
        seed: Noise RNG seed. Overrides ``[noise.measurement].seed`` when given;
            that TOML key overrides the ``DEFAULT_SEED`` default.

    Yields:
        One :class:`StepRecord` per step.

    Raises:
        ValueError: If the scenario has no estimator (feedforward scenarios
            need :func:`iter_phase_schedule`), or if the estimator propagates
            NaN/Inf from an adversarial fault.
        RuntimeError: If a NaN/Inf value enters the plant state.
    """
    plant = scenario.plant
    ctrl = scenario.controller
    est = scenario.estimator
    traj = scenario.trajectory

    if est is None:
        raise ValueError("closed-loop run requires an estimator — feedforward scenarios need a phase_list trajectory")

    total_steps = _total_steps(scenario, steps)
    if len(traj) < total_steps:  # type: ignore[arg-type]
        warnings.warn(
            f"trajectory has {len(traj)} steps but the scenario requests {total_steps} — "
            "truncating the run (extend the trajectory or lower [scenario].duration)"
        )
    total_steps = min(total_steps, len(traj))  # type: ignore[arg-type]

    noise_cfg = scenario.config.get("noise", {}).get("measurement")
    seed = seed if seed is not None else (noise_cfg or {}).get("seed", DEFAULT_SEED)
    rng = np.random.default_rng(seed)

    lo, hi = _control_limits(scenario)
    n_u = _control_input_dim(scenario)
    u_prev = np.zeros(n_u)

    # One-time controller contract checks (not per step): SMC needs f_x/g_x
    # dynamics a scenario does not wire; u_prev is passed only to controllers
    # that declare it (MPC_DeltaU).
    if isinstance(ctrl, SlidingModeController):
        raise NotImplementedError("SMC requires f_x/g_x dynamics — not scenario-runnable")
    ctrl_takes_u_prev = "u_prev" in inspect.signature(ctrl.compute).parameters

    dt = float(scenario.config.get("scenario", {}).get("dt", scenario.sim.engine.dt if scenario.sim is not None else 0.01))
    for step in range(total_steps):
        true_state = np.asarray(plant.get_state(), dtype=np.float64).flatten()
        reference = np.asarray(traj[step], dtype=np.float64).flatten()  # type: ignore[index]

        measurement = _inject_noise(true_state, noise_cfg, rng)
        measurement, faulted = _inject_adversarial(measurement, scenario.config.get("adversarial"), step)

        estimate = np.asarray(
            est.estimate(measurement.reshape(-1, 1), u_prev.reshape(-1, 1)), dtype=np.float64
        ).flatten()
        if np.any(np.isnan(estimate)) or np.any(np.isinf(estimate)):
            raise ValueError(
                f"Estimator returned non-finite estimate at step {step} (faulted={faulted}). "
                "NaN/Inf measurements must not corrupt the estimate."
            )

        control = np.clip(_compute_control(ctrl, estimate, reference, u_prev, ctrl_takes_u_prev), lo, hi)

        plant.step(control)
        if scenario.sim is not None:
            scenario.sim.step()

        plant_state = np.asarray(plant.get_state(), dtype=np.float64).flatten()
        if not np.all(np.isfinite(plant_state)):
            raise RuntimeError(f"Plant state non-finite at step {step}.")

        yield StepRecord(step * dt, reference, true_state, measurement, estimate, control, plant_state)
        u_prev = control


def run_scenario(scenario, steps: int | None = None, seed: int | None = None) -> SimResult:
    """Run the closed loop to completion and collect the history.

    Args:
        scenario: Composed scenario.
        steps: Number of steps. Defaults to ``[scenario].duration / dt``.
        seed: Noise RNG seed. Overrides ``[noise.measurement].seed``.

    Returns:
        A :class:`SimResult` — one :class:`StepRecord` per step plus TOML
        tolerance checking.
    """
    return SimResult(list(iter_scenario(scenario, steps=steps, seed=seed)), config=scenario.config)


def _resolve_schedule_targets(scenario, schedule: dict):
    """Map each schedule signal name to its setter — plant ``step`` or actuator passthrough.

    Resolved once per run. A key matching a declared plant name drives that
    plant; any other key must be declared in the scenario's ``[signals]``
    table as ``name = { actuator = "<actuator>" }`` and routed to
    ``engine.set_joint_ctrl``. Unknown keys are a loud error naming the
    declared plants and signals.

    Args:
        scenario: Composed scenario (sim-backed).
        schedule: Dict of signal name → per-step setpoints.

    Returns:
        Dict of signal name → callable(setpoint array).
    """
    sim = scenario.sim
    declared_signals = scenario.config.get("signals", {})
    targets = {}
    for key in schedule:
        if key in sim.plants:
            targets[key] = sim.plants[key].step
            continue
        actuator = declared_signals.get(key, {}).get("actuator")
        if actuator is None:
            raise ValueError(
                f"schedule key '{key}' is neither a declared plant "
                f"({sorted(sim.plants)}) nor a declared signal "
                f"({sorted(declared_signals)})"
            )
        if actuator not in sim.engine.actuator_names:
            raise ValueError(
                f"signal '{key}' targets actuator '{actuator}', but the engine "
                f"has no such actuator (declared: {sorted(sim.engine.actuator_names)})"
            )
        targets[key] = lambda v, a=actuator: sim.engine.set_joint_ctrl(a, float(np.asarray(v).ravel()[0]))
    return targets


def iter_phase_schedule(scenario, steps: int | None = None) -> Iterator[StepRecord]:
    """Run a ``phase_list`` schedule feedforward through the composed sim.

    The schedule is a dict of signal name → per-step setpoints. Plant-named
    signals are routed to ``plant.step(u)``; other signals must be declared in
    the scenario's ``[signals]`` table (``name = { actuator = ... }``) and go
    straight to engine actuators. Nothing is robot-specific.

    Args:
        scenario: Composed scenario (its trajectory must be a phase dict).
        steps: Number of steps. Defaults to the schedule length.

    Yields:
        One :class:`StepRecord` per step (``estimated == true_state``, control
        == the applied setpoints — no feedback estimator). Reference/control
        carry the full applied signal dict.

    Raises:
        ValueError: If the trajectory is not a phase dict, the scenario is not
            sim-backed, a schedule key matches no plant and no ``[signals]``
            entry, or the signal lengths disagree.
    """
    schedule = scenario.trajectory
    if not isinstance(schedule, dict) or not schedule:
        raise ValueError("feedforward run requires a phase schedule dict.")
    if scenario.sim is None:
        raise ValueError("feedforward runs drive the sim — sim-backed scenarios only.")

    targets = _resolve_schedule_targets(scenario, schedule)
    lengths = {k: len(v) for k, v in schedule.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"schedule signals disagree in length: {lengths}")
    n = steps if steps is not None else next(iter(lengths.values()))
    dt = float(scenario.config.get("scenario", {}).get("dt", scenario.sim.engine.dt))
    primary = scenario.plant

    for step in range(n):
        applied = {k: np.asarray(v[step], dtype=np.float64).flatten() for k, v in schedule.items()}
        for key, set_value in targets.items():
            set_value(applied[key].copy())
            scenario.sim.step()

        state = np.asarray(primary.get_state(), dtype=np.float64).flatten()
        if not np.all(np.isfinite(state)):
            raise RuntimeError(f"Non-finite state at phase step {step}.")

        yield StepRecord(
            t=step * dt,
            reference=applied,
            true_state=state,
            measurement=state,
            estimated=state,
            control=applied,
            plant_state=state,
        )


def run_phase_schedule(scenario, steps: int | None = None) -> SimResult:
    """Run the feedforward schedule to completion and collect the history."""
    return SimResult(list(iter_phase_schedule(scenario, steps=steps)), config=scenario.config)
