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

import inspect
import json

import numpy as np
import pytest

from shinro.codegen.compose import controller_input_kwargs
from shinro.codegen.infer_contract import infer_contract
from shinro.mcp._common import _attached, _plant_store, _store
from shinro.mcp.server import (
    attach_plant,
    controller_compute,
    create_controller,
    create_estimator,
    create_plant,
    estimator_estimate,
    list_controller_types,
    list_plant_types,
    list_plants,
    plant_model,
)

PENDULUM = "samples/plants/inverted_pendulum.toml"
CARTPOLE = "samples/plants/cartpole.toml"

#: Inline configs for every controller whose compute() the dispatcher must satisfy.
#: Module-level (read-only) rather than a class attribute so it is not shared mutable
#: state between test instances.
CONTROLLER_CONFIGS = {
    "PID": {"dt": 0.02, "kp": [1.0], "ki": [0.0], "kd": [0.0]},
    "LQR": {"dt": 0.1, "state_cost": [1.0], "control_cost": [1.0], "A_dynamics": [[1.0]], "B_dynamics": [[0.1]]},
    "MPC_LTI": {"dt": 0.02, "horizon": 5, "state_cost": [1.0], "control_cost": [0.1],
                "A_dynamics": [[1.0]], "B_dynamics": [[0.02]]},
    "MPC_DeltaU": {"dt": 0.02, "horizon": 5, "state_cost": [1.0], "control_cost": [0.1],
                   "delta_u_penalty": [0.5], "A_dynamics": [[1.0]], "B_dynamics": [[0.02]]},
    "MPPI": {"dt": 0.05, "horizon": 5, "num_samples": 10, "temperature": 0.5, "noise_sigma": [0.3],
             "state_cost": [1.0], "control_cost": [0.1], "u_min": [-1.0], "u_max": [1.0], "seed": 0},
    "SMC": {"dt": 0.02, "c": [1.0, 2.0], "k1": 1.0, "phi": 0.1, "k2": 0.0, "smoother": "sat", "alpha": 0.0},
}


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

    def test_plant_model_returns_one_stable_schema(self):
        """The response always has both views and an explicit evaluation point.

        No mode-switching: calling without state still reports f / f_x / g_x, at the
        plant's current state and operating control, and says so in u_source.
        """
        create_plant("cart", config_path=CARTPOLE)
        data = json.loads(plant_model("cart"))
        assert np.asarray(data["linearization"]["A"]).shape == (4, 4)
        assert np.asarray(data["linearization"]["B"]).shape == (4, 1)
        assert data["n_x"] == 4
        assert data["dt"] is not None

        evaluation = data["evaluation"]
        assert evaluation["x"] == data["state"]  # defaults to the current state
        assert evaluation["u_source"] == "zero"  # CartPole has no operating control
        assert len(evaluation["dynamics"]) == 4
        assert set(evaluation["model_terms"]) == {"f_x", "g_x"}

    def test_plant_model_reports_the_evaluation_point_it_used(self):
        """An explicit operating point is echoed, with u_source=c caller."""
        create_plant("pend", config_path=PENDULUM)
        data = json.loads(plant_model("pend", state=[0.1, 0.0], control=[0.5]))
        evaluation = data["evaluation"]
        assert evaluation["x"] == [0.1, 0.0]
        assert evaluation["u"] == [0.5]
        assert evaluation["u_source"] == "caller"
        assert len(evaluation["dynamics"]) == 2
        assert np.asarray(evaluation["model_terms"]["g_x"]).shape == (2, 1)
        # The linearization stays available alongside the nonlinear evaluation.
        assert np.asarray(data["linearization"]["A"]).shape == (2, 2)

    def test_plant_model_defaults_to_the_operating_control(self):
        """A non-affine plant's g is zero at u=0, so the default is its operating control.

        Rotor thrust is ~w^2, so control_matrix(x, 0) is identically zero. The tool uses
        operating_control() (hover speeds) and labels it, matching model_terms.
        """
        create_plant("qp", type="Quadrotor", params={"dt": 0.01})
        evaluation = json.loads(plant_model("qp", state=[0.0] * 12))["evaluation"]
        assert evaluation["u_source"] == "operating_control"
        u = np.asarray(evaluation["u"])
        assert np.all(u > 0), f"expected hover speeds, got {u}"
        g = np.asarray(evaluation["model_terms"]["g_x"])
        assert np.count_nonzero(g) > 0, "g_x should not be all zeros at the operating point"
        # The reconstruction is exact AT the operating point, by construction.
        recon = np.asarray(evaluation["model_terms"]["f_x"]) + g @ u
        assert np.allclose(recon, evaluation["dynamics"], atol=1e-9)
        assert evaluation["affine_residual"] < 1e-9

    def test_plant_model_measures_affine_validity_instead_of_asserting_it(self):
        """affine_exact distinguishes a control-affine plant from a local tangent.

        The question "is f = f_x + g_x*u exact, or only near u_op?" used to live in
        prose; it is now measured by probing whether g depends on u.
        """
        create_plant("pend", config_path=PENDULUM)
        create_plant("qp", type="Quadrotor", params={"dt": 0.01})
        affine = json.loads(plant_model("pend", state=[0.1, 0.0]))["evaluation"]
        tangent = json.loads(plant_model("qp", state=[0.0] * 12))["evaluation"]
        assert affine["affine_exact"] is True
        assert tangent["affine_exact"] is False, "rotor thrust ~ w^2 is not control-affine"

    def test_plant_model_accepts_an_operating_control(self):
        """An explicit u changes f(x, u) while the operating control is the default."""
        create_plant("pend", config_path=PENDULUM)
        moved = json.loads(plant_model("pend", state=[0.1, 0.0], control=[0.5]))
        defaulted = json.loads(plant_model("pend", state=[0.1, 0.0]))
        assert moved["evaluation"]["u"] == [0.5]
        assert moved["evaluation"]["dynamics"] != defaulted["evaluation"]["dynamics"]

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


class TestDispatcherMatchesSignatures:
    """`ctrl.compute(**kwargs)` must receive exactly the controller's own parameters.

    The kwargs dict is built from `infer_contract(ctrl).input_names` and filled by
    role, so its keys must equal the real `compute` signature — no missing argument
    (TypeError) and no extra one. The same mapping drives the simulation runner and
    gate A, so this also pins live/compiled parity.
    """

    @pytest.mark.parametrize("ctype", sorted(CONTROLLER_CONFIGS))
    def test_kwargs_equal_the_compute_signature(self, ctype):
        """The dispatcher's keys are exactly the parameters compute() declares."""
        create_plant("pend", config_path=PENDULUM)
        create_controller("c", type=ctype, params=CONTROLLER_CONFIGS[ctype])
        ctrl = _store["c"]
        params = [p for p in inspect.signature(ctrl.compute).parameters if p != "self"]
        kwargs = controller_input_kwargs(
            infer_contract(ctrl).input_names,
            np.array([0.1, 0.0]),
            np.zeros(2),
            np.zeros(1),
            plant=_plant_store["pend"],
        )
        assert list(kwargs) == params

    def test_unknown_parameter_becomes_a_free_host_input(self):
        """A name outside the role vocabulary is a host input (None), like MPPI's epsilon.

        This is the deliberate fallback, and it matches what compose() does, so a
        third-party controller parameter is a host input in both the live and the
        compiled path rather than a silent error.
        """
        create_controller("mppi", type="MPPI", params=CONTROLLER_CONFIGS["MPPI"])
        ctrl = _store["mppi"]
        kwargs = controller_input_kwargs(infer_contract(ctrl).input_names, np.zeros(2), np.zeros(2), np.zeros(1))
        assert kwargs["epsilon"] is None


class TestThirdPartyController:
    """A controller registered outside the package must work through the tools.

    Nothing in the tool layer enumerates known controller types: create_* dispatches
    through the registry, and compute() is driven by role from the inferred contract.
    This pins that with a controller defined in the test, including one that returns a
    plain list rather than an ndarray.
    """

    def test_custom_controller_is_created_and_computed(self):
        from dataclasses import dataclass

        from shinro.components import Controller
        from shinro.factories.registry import _CONTROLLER_REGISTRY, register_controller
        from shinro.utils.array_backend import NumpyBackend

        @dataclass(frozen=True)
        class BangBangConfig:
            dt: float = 0.01
            effort: float = 1.0
            name: str = "bangbang_test"

        @register_controller("BangBangTest")
        class BangBangTest(Controller):
            Config = BangBangConfig

            def __init__(self, effort: float = 1.0, dt: float = 0.01, backend=None):
                self.bk = backend or NumpyBackend()
                self.effort = effort
                self.dt = dt

            @classmethod
            def from_config(cls, config, backend=None):
                cfg = config if isinstance(config, BangBangConfig) else cls.load_config(config)
                return cls(effort=cfg.effort, dt=cfg.dt, backend=backend or NumpyBackend())

            def compute(self, current_state, target_state=None):
                error = np.asarray(current_state, float) - (
                    0 if target_state is None else np.asarray(target_state, float)
                )
                return (np.sign(error) * self.effort).tolist()  # plain list on purpose

        try:
            assert "BangBangTest" in json.loads(list_controller_types())["controller_types"]
            assert "Created" in create_controller("bb", type="BangBangTest", params={"effort": 2.0})
            data = json.loads(controller_compute("bb", state=[1.0, -0.0], reference=[0.0, 0.0]))
            assert data["action"] == [2.0, 0.0]
        finally:
            _CONTROLLER_REGISTRY.pop("BangBangTest", None)
