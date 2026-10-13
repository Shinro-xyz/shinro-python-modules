"""The live runner drives controllers through the same role-based dispatch as
``compose`` and gate A — including plant-model terms (SMC's ``f_x``/``g_x``).

SMC used to be rejected by ``iter_scenario`` because a scenario does not wire
``f_x``/``g_x`` ports. The runner now maps every controller input by role
(``controller_input_kwargs``), evaluating the plant pair live at its operating
point, so SMC runs in the closed loop like any other controller.
"""

from __future__ import annotations

import numpy as np
import pytest

from shinro.codegen.compose import controller_input_kwargs
from shinro.factories import ScenarioFactory
from shinro.plants.inverted_pendulum import InvertedPendulum
from shinro.simulation.runner import run_scenario
from shinro.utils.array_backend import NumpyBackend

SMC_SCENARIO = "tests/fixtures/scenarios/smc_pendulum_run.toml"


# --- the live closed loop -------------------------------------------------


def test_smc_scenario_is_runnable_and_regulates():
    """SMC (a model-input controller) runs live and drives the tilt upright."""
    recs = run_scenario(ScenarioFactory(SMC_SCENARIO).build())
    assert len(recs) == 800
    assert abs(recs[0].plant_state[0]) > 0.1
    assert abs(recs[-1].plant_state[0]) < 0.02


def test_smc_scenario_stays_finite_and_bounded():
    recs = run_scenario(ScenarioFactory(SMC_SCENARIO).build())
    states = np.array([r.plant_state for r in recs])
    controls = np.array([r.control for r in recs])
    assert np.all(np.isfinite(states))
    assert np.all(np.abs(controls) <= 5.0 + 1e-9)


# --- the role dispatcher --------------------------------------------------


def test_state_role_is_the_tracking_error_without_a_reference():
    """A regulator that declares no reference input gets the error x_hat - x_ref."""
    x_hat = np.array([1.0, 2.0])
    x_ref = np.array([0.5, -0.5])
    kwargs = controller_input_kwargs(["x"], x_hat, x_ref, np.zeros(1))
    assert np.allclose(kwargs["x"], x_hat - x_ref)


def test_state_and_reference_are_separate_when_a_reference_is_declared():
    x_hat = np.array([1.0, 2.0])
    x_ref = np.array([0.5, -0.5])
    kwargs = controller_input_kwargs(["current_state", "target_state"], x_hat, x_ref, np.zeros(1))
    assert np.allclose(kwargs["current_state"], x_hat)
    assert np.allclose(kwargs["target_state"], x_ref)


def test_model_role_uses_the_plant_operating_point():
    plant = InvertedPendulum(backend=NumpyBackend())
    x_hat = np.array([0.3, -0.2])
    kwargs = controller_input_kwargs(["x", "f_x", "g_x"], x_hat, np.zeros(2), np.zeros(1), plant=plant)
    f_x, g_x = plant.model_terms(x_hat, n_u=1)
    assert np.allclose(kwargs["f_x"], np.asarray(f_x).ravel())
    assert np.allclose(kwargs["g_x"], np.asarray(g_x))


def test_model_role_without_a_plant_is_loud():
    with pytest.raises(ValueError, match="plant-model term"):
        controller_input_kwargs(["x", "f_x", "g_x"], np.zeros(2), np.zeros(2), np.zeros(1), plant=None)


def test_free_host_input_defaults_to_none():
    """MPPI's epsilon stays ``None`` so the eager controller samples its own."""
    kwargs = controller_input_kwargs(
        ["current_state", "target_state", "epsilon"], np.zeros(2), np.zeros(2), np.zeros(1)
    )
    assert kwargs["epsilon"] is None
    supplied = controller_input_kwargs(
        ["epsilon"], np.zeros(2), np.zeros(2), np.zeros(1), host_values={"epsilon": np.ones(3)}
    )
    assert np.allclose(supplied["epsilon"], np.ones(3))
