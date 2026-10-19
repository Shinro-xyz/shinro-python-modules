"""Tests for the plant seam — nonlinear controllers and estimators via the control server.

Linear components carry their own model (LQR's A/B, PID's gains) and need nothing.
SMC, MPPI, EKF and UKF take their model from a plant, so before the plant tools they
constructed fine and then failed on the first call. These pin the seam:

* a plant can be created from a TOML or inline params and reports its dims/model,
* ``attach_plant`` is what makes the general form runnable,
* a nonlinear controller with no plant fails with a clear, actionable message,
* linear components still work untouched (the role dispatcher reproduces the old
  call shapes — the component suite is the regression net for that).
"""

import json

import numpy as np
import pytest

from shinro.mcp._common import _attached, _plant_store, _store
from shinro.mcp.server import (
    attach_plant,
    controller_compute,
    create_controller,
    create_estimator,
    create_plant,
    estimator_estimate,
    list_plant_types,
    list_plants,
    plant_model,
)

PENDULUM = "samples/plants/inverted_pendulum.toml"
CARTPOLE = "samples/plants/cartpole.toml"


@pytest.fixture(autouse=True)
def clear_stores():
    """Each test starts with empty component, plant, and attachment registries."""
    for store in (_store, _plant_store, _attached):
        store.clear()
    yield
    for store in (_store, _plant_store, _attached):
        store.clear()


class TestCreatePlant:
    """Verify plant creation, dims, and the listing tools."""

    def test_create_from_config_reports_dims(self):
        """A plant TOML yields a stored instance with its state/input dims."""
        data = json.loads(create_plant("pend", config_path=PENDULUM))
        assert data["stored"] == "pend"
        assert data["type"] == "InvertedPendulum"
        assert data["n_x"] == 2
        assert data["n_u"] == 1
        assert data["attached_to"] == []

    def test_create_from_inline_params(self):
        """Inline params create the same plant without a TOML."""
        data = json.loads(create_plant("pole", type="InvertedPendulum", params={"dt": 0.02}))
        assert data["n_x"] == 2

    def test_unknown_type_returns_error(self):
        """An unregistered plant type reports the available ones."""
        data = json.loads(create_plant("x", type="Nope", params={}))
        assert "error" in data
        assert "Available" in data["error"]

    def test_missing_config_file_returns_error(self):
        """A nonexistent config path is reported, not raised."""
        data = json.loads(create_plant("x", config_path="nope.toml"))
        assert "error" in data

    def test_no_source_returns_usage(self):
        """Neither config nor type+params is a usage error."""
        data = json.loads(create_plant("x"))
        assert "Provide either" in data["error"]

    def test_list_plant_types_includes_the_registry(self):
        """Every registered plant type is listed."""
        types = json.loads(list_plant_types())["plant_types"]
        assert "CartPole" in types
        assert "InvertedPendulum" in types
        assert types == sorted(types)

    def test_list_plants_reports_stored_instances(self):
        """The listing reports each stored plant with its dims and attachments."""
        create_plant("pend", config_path=PENDULUM)
        data = json.loads(list_plants())
        assert data["pend"]["type"] == "InvertedPendulum"

    def test_plant_model_returns_discrete_ab(self):
        """plant_model reports the linearization used by LQR/MPC."""
        create_plant("cart", config_path=CARTPOLE)
        data = json.loads(plant_model("cart"))
        assert np.asarray(data["A"]).shape == (4, 4)
        assert np.asarray(data["B"]).shape == (4, 1)
        assert data["n_x"] == 4
        assert data["dt"] is not None
        # Without an operating point there are no nonlinear terms to report.
        assert "dynamics" not in data
        assert "model_terms" not in data

    def test_plant_model_evaluates_the_nonlinear_terms_at_a_point(self):
        """Passing state also returns f(x, u) and the (f_x, g_x) pair SMC consumes.

        `get_model()` is only the *linearization*; these are the exact nonlinear
        values at the requested operating point (there is no get_dynamics hook —
        the vector field is `dynamics(x, u)`).
        """
        create_plant("pend", config_path=PENDULUM)
        data = json.loads(plant_model("pend", state=[0.1, 0.0]))
        assert data["operating_point"]["x"] == [0.1, 0.0]
        assert len(data["dynamics"]) == 2
        assert set(data["model_terms"]) == {"f_x", "g_x"}
        assert np.asarray(data["model_terms"]["g_x"]).shape == (2, 1)
        # The linearization stays available alongside the nonlinear evaluation.
        assert np.asarray(data["A"]).shape == (2, 2)

    def test_plant_model_defaults_to_the_operating_control(self):
        """A non-affine plant's g is zero at u=0, so the default u is its operating point.

        Rotor thrust is ~w^2, so control_matrix(x, 0) is identically zero. Defaulting
        to zeros would report a g_x that looks broken; the tool uses
        operating_control() (hover for a quadrotor), matching model_terms.
        """
        create_plant("qp", type="Quadrotor", params={"dt": 0.01})
        data = json.loads(plant_model("qp", state=[0.0] * 12))
        u = np.asarray(data["operating_point"]["u"])
        assert np.all(u > 0), f"expected hover speeds, got {u}"
        g = np.asarray(data["model_terms"]["g_x"])
        assert np.count_nonzero(g) > 0, "g_x should not be all zeros at the operating point"
        # The control-affine reconstruction must reproduce the vector field.
        recon = np.asarray(data["model_terms"]["f_x"]) + g @ u
        assert np.allclose(recon, data["dynamics"], atol=1e-9)

    def test_plant_model_accepts_an_operating_control(self):
        """An explicit u is used for f(x, u); otherwise the operating control applies."""
        create_plant("pend", config_path=PENDULUM)
        moved = json.loads(plant_model("pend", state=[0.1, 0.0], control=[0.5]))
        zeroed = json.loads(plant_model("pend", state=[0.1, 0.0]))
        assert moved["operating_point"]["u"] == [0.5]
        assert moved["dynamics"] != zeroed["dynamics"]

    def test_plant_model_unknown_name_returns_error(self):
        """An unknown plant name reports the miss."""
        assert "error" in json.loads(plant_model("nope"))


class TestAttach:
    """Verify the attachment tool's bookkeeping and failure modes."""

    def test_attach_records_on_both_sides(self):
        """Attachment shows up in the component→plant map and in list_plants."""
        create_plant("pend", config_path=PENDULUM)
        create_controller("smc", config_path="samples/controllers/smc.toml")
        data = json.loads(attach_plant("smc", "pend"))
        assert data["attached"] == "smc"
        assert _attached["smc"] == "pend"
        assert json.loads(list_plants())["pend"]["attached_to"] == ["smc"]

    def test_attach_unknown_component_returns_error(self):
        """Attaching to a name that was never created is reported."""
        create_plant("pend", config_path=PENDULUM)
        data = json.loads(attach_plant("nope", "pend"))
        assert "error" in data

    def test_attach_unknown_plant_returns_error(self):
        """Attaching a plant that was never created is reported."""
        create_controller("smc", config_path="samples/controllers/smc.toml")
        data = json.loads(attach_plant("smc", "nope"))
        assert "error" in data

    def test_attach_to_self_contained_component_is_rejected(self):
        """LQR carries its own A/B, so a plant attachment is meaningless and refused."""
        create_plant("pend", config_path=PENDULUM)
        create_controller("lqr", config_path="samples/controllers/lqr_base.toml")
        data = json.loads(attach_plant("lqr", "pend"))
        assert "error" in data
        assert "does not accept a plant" in data["error"]


class TestNonlinearControllers:
    """Verify SMC and MPPI run once their plant is attached."""

    def test_smc_computes_with_a_plant(self):
        """SMC's compute(x, f_x, g_x) is satisfied by the role dispatcher + plant."""
        create_plant("pend", config_path=PENDULUM)
        create_controller("smc", config_path="samples/controllers/smc.toml")
        attach_plant("smc", "pend")
        data = json.loads(controller_compute("smc", state=[0.1, 0.0], reference=[0.0, 0.0]))
        assert "action" in data
        assert len(data["action"]) == 1

    def test_smc_without_a_plant_fails_loudly(self):
        """The missing-model error names the plant, instead of a TypeError about g_x."""
        create_controller("smc", config_path="samples/controllers/smc.toml")
        result = controller_compute("smc", state=[0.1, 0.0], reference=[0.0, 0.0])
        assert "Error in compute" in result
        assert "no plant" in result.lower()

    def test_mppi_computes_with_a_plant(self):
        """MPPI rolls out plant.dynamics and samples its own epsilon when none is fed."""
        create_plant("pend", config_path=PENDULUM)
        create_controller(
            "mppi",
            type="MPPI",
            params={
                "dt": 0.05,
                "horizon": 5,
                "num_samples": 20,
                "temperature": 0.5,
                "noise_sigma": [0.3],
                "state_cost": [1.0, 1.0],
                "control_cost": [0.1],
                "u_min": [-1.0],
                "u_max": [1.0],
                "seed": 0,
            },
        )
        attach_plant("mppi", "pend")
        data = json.loads(controller_compute("mppi", state=[0.1, 0.0], reference=[0.0, 0.0]))
        assert "action" in data
        assert len(data["action"]) == 1


class TestNonlinearEstimators:
    """Verify the EKF and UKF run once their plant is attached."""

    def test_ekf_estimates_with_a_plant(self):
        """The EKF differentiates the plant's dynamics and measurement model."""
        create_plant("pend", config_path=PENDULUM)
        create_estimator(
            "ekf",
            type="ExtendedKalmanFilter",
            params={"dt": 0.02, "process_noise": [0.01, 0.01], "measurement_noise": [0.1, 0.1]},
        )
        attach_plant("ekf", "pend")
        data = json.loads(estimator_estimate("ekf", measurement=[0.0, 1.0], control_input=[0.0]))
        assert len(data["state_estimate"]) == 2

    def test_ekf_without_a_plant_fails_loudly(self):
        """Without a process model the filter says so rather than returning garbage."""
        create_estimator(
            "ekf",
            type="ExtendedKalmanFilter",
            params={"dt": 0.02, "process_noise": [0.01, 0.01], "measurement_noise": [0.1, 0.1]},
        )
        result = estimator_estimate("ekf", measurement=[0.0, 1.0], control_input=[0.0])
        assert "Error in estimate" in result
        assert "dynamics_fn" in result

    def test_ukf_estimates_with_a_plant(self):
        """The UKF propagates sigma points through the plant's dynamics."""
        create_plant("pend", config_path=PENDULUM)
        create_estimator(
            "ukf",
            type="UnscentedKF",
            params={
                "dt": 0.02,
                "process_noise": [0.01, 0.01],
                "measurement_noise": [0.1, 0.1],
                "alpha": 1.0,
                "beta": 2.0,
                "kappa": 0.0,
                "measurement_matrix": [[1.0, 0.0], [0.0, 1.0]],
            },
        )
        attach_plant("ukf", "pend")
        data = json.loads(estimator_estimate("ukf", measurement=[0.0, 1.0], control_input=[0.0]))
        assert len(data["state_estimate"]) == 2


class TestLinearRegression:
    """The role dispatcher must not change the linear components' behaviour."""

    def test_pid_still_computes_regulation(self):
        """PID at a zero reference still yields -kp * error."""
        create_controller("pid", type="PID", params={"dt": 0.02, "kp": [2.0], "ki": [0.0], "kd": [0.0]})
        data = json.loads(controller_compute("pid", state=[5.0]))
        assert data["action"][0] == pytest.approx(-10.0)

    def test_mpc_deltau_still_accumulates_from_u_prev(self):
        """MPC_DeltaU keeps receiving u_prev through the role dispatcher."""
        create_controller(
            "mpc",
            type="MPC_DeltaU",
            params={
                "dt": 0.02,
                "horizon": 5,
                "state_cost": [1.0, 1.0],
                "control_cost": [0.1, 0.1],
                "delta_u_penalty": [0.5, 0.5],
                "A_dynamics": [[1.0, 0.0], [0.0, 1.0]],
                "B_dynamics": [[0.02, 0.0], [0.0, 0.02]],
            },
        )
        data = json.loads(controller_compute("mpc", state=[0.5, -0.3], u_prev=[0.2, 0.1]))
        assert len(data["action"]) == 2
