"""Drivers for full Trajectory → Controller → Estimator → Plant → Engine loops.

Two drivers are provided:

* :func:`run_scenario` — a closed-loop run for single-plant scenarios (base
  tracking, arm Cartesian, adversarial). The estimator feeds the controller,
  which drives the plant, and the history is returned as
  :class:`StepRecord` entries.

* :func:`run_phase_schedule` — a feedforward run for ``phase_list`` schedules
  (pick-and-place) where the schedule itself is the control: each step carries
  per-signal arm/base/jaw setpoints that are applied directly to the composed
  ``RobotSim``. This validates the multi-plant wiring without double-integrating
  the scripted setpoints through a feedback controller.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from shinro.controllers.lqr import LQR
from shinro.controllers.mpc_lti import MPC_LTI, MPC_LTI_DeltaU
from shinro.controllers.pid import PIDController
from shinro.factories.scenario_factory import Scenario

MAX_CONTROL = 1e6


@dataclass
class StepRecord:
    """One recorded step of an integration run.

    Args:
        t: Simulation time (s).
        reference: Reference setpoint at this step.
        true_state: Plant state (noiseless).
        measurement: Noisy/adversarial measurement fed to the estimator.
        estimated: Estimator output.
        control: Control input applied to the plant.
        plant_state: Plant state read back after the step.
    """

    t: float
    reference: Any
    true_state: np.ndarray
    measurement: np.ndarray
    estimated: np.ndarray
    control: Any
    plant_state: np.ndarray


def _control_input_dim(scenario: Scenario) -> int:
    """Infer the control input dimension from the plant's model."""
    _, B = scenario.plant.get_model()
    return int(B.shape[1])


def _inject_noise(state: np.ndarray, noise_cfg: dict | None, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian measurement noise from a ``[noise.measurement]`` config.

    Args:
        state: True state.
        noise_cfg: Optional dict with ``std`` and ``seed``.
        rng: Shared RNG.

    Returns:
        Noisy measurement of the same shape as ``state``.
    """
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

    Args:
        measurement: Noisy measurement (may already be noisy).
        adversarial_cfg: Optional dict with ``inject_at`` and ``value``.
        step: Current step index.

    Returns:
        Tuple of (possibly-corrupted measurement, whether a fault fired).
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


def _control_limits(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """Return (low, high) clip limits for the control input.

    Reads ``[scenario.input_limits]`` from the config; defaults to a large
    symmetric bound so the loop stays stable for unconstrained scenarios.
    """
    limits = scenario.config.get("scenario", {}).get("input_limits")
    n_u = _control_input_dim(scenario)
    if limits is None:
        lo = np.full(n_u, -MAX_CONTROL)
        hi = np.full(n_u, MAX_CONTROL)
    else:
        lo = np.asarray(limits["min"], dtype=np.float64)
        hi = np.asarray(limits["max"], dtype=np.float64)
    return lo, hi


def run_scenario(scenario: Scenario, steps: int | None = None, seed: int = 42) -> list[StepRecord]:
    """Run the full closed-loop for a single-plant scenario.

    Args:
        scenario: Composed scenario from :class:`ScenarioFactory`.
        steps: Number of steps to run. Defaults to ``duration / dt``.
        seed: Seed for the measurement-noise RNG.

    Returns:
        List of :class:`StepRecord`, one per step.

    Raises:
        ValueError: If the estimator propagates NaN/Inf from an adversarial
            fault (NaN/Inf measurements must not corrupt the estimate).
        RuntimeError: If a NaN/Inf value enters the plant state.
    """
    plant = scenario.plant
    ctrl = scenario.controller
    est = scenario.estimator
    traj = scenario.trajectory

    dt = float(scenario.config.get("scenario", {}).get("dt", scenario.sim.engine.dt))
    duration = float(scenario.config.get("scenario", {}).get("duration", 5.0))
    total_steps = steps if steps is not None else int(round(duration / dt))
    total_steps = min(total_steps, len(traj))  # type: ignore[arg-type]

    rng = np.random.default_rng(seed)
    noise_cfg = scenario.config.get("noise", {}).get("measurement")
    adversarial_cfg = scenario.config.get("adversarial")
    lo, hi = _control_limits(scenario)

    n_u = _control_input_dim(scenario)
    u_prev = np.zeros(n_u)

    assert est is not None, "run_scenario requires an estimator"
    records = []
    for step in range(total_steps):
        t = step * dt
        true_state = np.asarray(plant.get_state(), dtype=np.float64).flatten()

        reference = np.asarray(traj[step], dtype=np.float64).flatten()  # type: ignore[index]

        measurement = _inject_noise(true_state, noise_cfg, rng)
        measurement, faulted = _inject_adversarial(measurement, adversarial_cfg, step)

        estimate = np.asarray(est.estimate(measurement.reshape(-1, 1), u_prev.reshape(-1, 1)), dtype=np.float64).flatten()

        if np.any(np.isnan(estimate)) or np.any(np.isinf(estimate)):
            raise ValueError(
                f"Estimator returned non-finite estimate at step {step} (faulted={faulted}). "
                "NaN/Inf measurements must not corrupt the estimate."
            )

        control = _compute_control(ctrl, estimate, reference, u_prev)
        control = np.clip(control, lo, hi)

        plant.step(control)
        scenario.sim.step()

        plant_state = np.asarray(plant.get_state(), dtype=np.float64).flatten()
        if not np.all(np.isfinite(plant_state)):
            raise RuntimeError(f"Plant state non-finite at step {step}.")

        records.append(StepRecord(t, reference, true_state, measurement, estimate, control, plant_state))
        u_prev = control

    return records


def run_phase_schedule(scenario: Scenario, steps: int | None = None) -> list[StepRecord]:
    """Run a ``phase_list`` schedule feedforward through the composed RobotSim.

    The schedule is a dict of ``{"arm", "base", "jaw"}`` per-step setpoints.
    The arm setpoint is the 6D twist passed to ``sim.arm.step()`` (with the jaw
    injected as the last channel), and the base setpoint is the 3D velocity
    passed to ``sim.base.step()``.

    Args:
        scenario: Composed scenario (its ``trajectory`` must be a phase dict).
        steps: Number of steps to run. Defaults to the schedule length.

    Returns:
        List of :class:`StepRecord` (estimated == true_state, control ==
        the applied setpoints, since there is no feedback estimator).

    Raises:
        ValueError: If the trajectory is not a ``phase_list`` dict.
    """
    schedule = scenario.trajectory
    if not isinstance(schedule, dict) or "arm" not in schedule:
        raise ValueError("run_phase_schedule requires a phase_list trajectory dict.")

    n = steps if steps is not None else len(schedule["arm"])
    dt = float(scenario.config.get("scenario", {}).get("dt", scenario.sim.engine.dt))

    records = []
    for step in range(n):
        t = step * dt
        arm_twist = np.asarray(schedule["arm"][step], dtype=np.float64).flatten().copy()
        arm_twist[5] = float(schedule["jaw"][step])
        base_vel = np.asarray(schedule["base"][step], dtype=np.float64).flatten()

        arm_plant = scenario.sim.get_plant("arm")
        base_plant = scenario.sim.get_plant("base")

        arm_plant.step(arm_twist)
        base_plant.step(base_vel)
        scenario.sim.step()

        arm_state = np.asarray(arm_plant.get_state(), dtype=np.float64).flatten()
        base_state = np.asarray(base_plant.get_state(), dtype=np.float64).flatten()

        if not np.all(np.isfinite(arm_state)) or not np.all(np.isfinite(base_state)):
            raise RuntimeError(f"Non-finite state at phase step {step}.")

        records.append(
            StepRecord(
                t=t,
                reference={"arm": arm_twist, "base": base_vel},
                true_state=arm_state,
                measurement=arm_state,
                estimated=arm_state,
                control={"arm": arm_twist, "base": base_vel},
                plant_state=arm_state,
            )
        )

    return records


def _compute_control(ctrl, estimate: np.ndarray, reference: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
    """Compute the control input, dispatching on controller type.

    LQR/PID take ``(current, target)``; MPC_LTI variants take ``(error)`` or
    ``(error, u_prev)``.

    Args:
        ctrl: Controller.
        estimate: Flat state estimate.
        reference: Reference setpoint array.
        u_prev: Previous control.

    Returns:
        Control input (n_u,).
    """
    if isinstance(ctrl, (LQR, PIDController)):
        return ctrl.compute(estimate, reference)
    if isinstance(ctrl, MPC_LTI_DeltaU):
        return ctrl.compute(estimate - reference, u_prev=u_prev)  # type: ignore[call-arg]
    if isinstance(ctrl, MPC_LTI):
        return ctrl.compute(estimate - reference)
    return ctrl.compute(estimate, reference)
