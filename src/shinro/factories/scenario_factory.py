"""Compose a full control loop (Trajectory → Controller → Estimator → Plant → Engine) from one TOML.

A **scenario** declares every ABC role plus the scenario parameters (duration,
tolerances, noise, physics) in a single file. :class:`ScenarioFactory` builds the
whole loop and validates that the dimensions of each component agree with the
plant.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shinro.components import Controller, Plant, StateEstimator, TrajectoryGenerator
from shinro.controllers.mppi import MPPIController
from shinro.factories.controller_factory import ControllerFactory
from shinro.factories.estimator_factory import EstimatorFactory
from shinro.factories.registry import _PLANT_REGISTRY
from shinro.factories.trajectory_factory import TrajectoryFactory
from shinro.utils.array_backend import ArrayBackend, NumpyBackend
from shinro.utils.config_resolver import resolve_config_path
from shinro.utils.linearization import discretize_euler, linearize_plant

if TYPE_CHECKING:
    from shinro.simulation.robotsim import RobotSim


@dataclass
class Scenario:
    """Fully composed control loop plus scenario parameters.

    ``sim`` is the :class:`RobotSim` instance (engine + plants) for MuJoCo-
    backed scenarios, or ``None`` for plant-only scenarios where the plant
    self-integrates its analytical dynamics. ``plant`` is the primary plant
    driven by the controller; ``controller``, ``estimator`` and ``trajectory``
    are the other loop roles. For feedforward scenarios (e.g. ``phase_list``
    pick-and-place) ``controller`` and ``estimator`` are ``None`` and the
    schedule itself is the control. ``config`` is the raw TOML dict (used by
    the runner and the tests for tolerances, noise, etc.).
    """

    sim: RobotSim | None
    plant: Plant
    controller: Controller | None
    estimator: StateEstimator | None
    trajectory: TrajectoryGenerator | Any
    config: dict[str, Any] = field(default_factory=dict)


class ScenarioFactory:
    """Build a :class:`Scenario` from a single TOML config.

    Config sections:
        [scenario]    name, description, duration, dt, tolerance, input_limits
        [physics]     free_joint, model_path
        [plant]       name (sim-backed) OR type + config + initial_state (plant-only)
        [controller]  type, config (optional)
        [estimator]   type, config (optional)
        [trajectory]  type, config
        [sim]         config (path to the RobotSim TOML; optional)
        [noise]       measurement (optional)
        [adversarial] inject_at, value (optional)

    Two modes:

    1. **Sim-backed** (default): the plant is looked up by ``[plant].name`` on
       the ``RobotSim`` built from ``[sim]``. When ``[physics].free_joint`` is
       set, the LeKiwi MJCF is rewritten so the arm hangs off a mobile
       free-jointed base.

    2. **Plant-only**: when ``[sim]`` is absent, the plant is built directly
       from the registry via ``[plant].type`` (registered name) and
       ``[plant].config`` (path to a plant TOML). ``[plant].initial_state``
       optionally seeds the state. The plant self-integrates its analytical
       dynamics — no MuJoCo engine. ``[scenario].dt`` must equal the plant's
       own ``dt`` (the plant integrates at its own time step).

    When a controller/estimator config omits ``A_dynamics``/``B_dynamics`` in
    plant-only mode, the discrete-time model is derived from the plant via
    :func:`shinro.utils.linearization.linearize_plant` (upright equilibrium)
    and first-order Euler discretization, so the plant TOML stays the single
    source of physics truth.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(resolve_config_path(config_path), "rb") as f:
            self.config = tomllib.load(f)

    def build(self, backend: ArrayBackend | None = None) -> Scenario:
        """Build and validate the full scenario.

        Args:
            backend: Array backend for controller/estimator/trajectory.
                Defaults to numpy (the physics engine is always numpy-backed).

        Returns:
            A composed :class:`Scenario`.

        Raises:
            KeyError: If a required section or registry type is missing.
            ValueError: If component dimensions disagree with the plant.
        """
        plant_cfg = self.config["plant"]
        sim_cfg = self.config.get("sim", {"config": "robot_config.toml"})
        physics_cfg = self.config.get("physics", {})

        if "sim" in self.config:
            sim, plant = self._build_sim_plant(plant_cfg, sim_cfg, physics_cfg)
            derive_model = False
        else:
            sim, plant = self._build_plant_only(plant_cfg)
            derive_model = True
            scenario_dt = self.config.get("scenario", {}).get("dt")
            if scenario_dt is not None and abs(float(scenario_dt) - float(plant.dt)) > 1e-12:
                raise ValueError(
                    f"[scenario].dt ({scenario_dt}) must equal the plant's dt ({plant.dt}) "
                    "in plant-only scenarios: the plant self-integrates at its own time step."
                )

        def _create(factory_cls, path: str):
            if backend is not None:
                return factory_cls(path).create(backend=backend)
            return factory_cls(path).create()

        trajectory = _create(TrajectoryFactory, self.config["trajectory"]["config"])

        controller = None
        estimator = None
        if "controller" in self.config:
            controller = self._create_loop_role(
                ControllerFactory, self.config["controller"]["config"], plant, backend, derive_model
            )
        if "estimator" in self.config:
            estimator = self._create_loop_role(
                EstimatorFactory, self.config["estimator"]["config"], plant, backend, derive_model
            )

        # MPPI is function-based: its dynamics/cost are produced from the
        # plant's model via a batched adapter, so it must be wired after the
        # plant is built (this also sets its D_u for validation). Q/R come
        # from the controller's own config.
        if isinstance(controller, MPPIController):
            controller.attach_plant(plant)

        if controller is not None and estimator is not None:
            n_x = plant.get_state().shape[0]
            _, B = plant.get_model()
            n_u = B.shape[1]
            self._validate_dimensions(n_x, n_u, controller, estimator)

        return Scenario(
            sim=sim,
            plant=plant,
            controller=controller,
            estimator=estimator,
            trajectory=trajectory,
            config=self.config,
        )

    @staticmethod
    def _build_sim_plant(plant_cfg: dict, sim_cfg: dict, physics_cfg: dict) -> tuple[RobotSim, Plant]:
        """Build the RobotSim and look up the plant by name."""
        from shinro.simulation.robotsim import RobotSim

        xml_string, assets = ScenarioFactory._physics_xml(physics_cfg)
        sim = RobotSim(resolve_config_path(sim_cfg["config"]), xml_string=xml_string, assets=assets)  # type: ignore[arg-type]
        plant = sim.get_plant(plant_cfg["name"])
        if plant is None:
            raise KeyError(f"Plant name '{plant_cfg['name']}' not found in RobotSim. Available plants: {sorted(sim._plants.keys())}")
        return sim, plant

    @staticmethod
    def _build_plant_only(plant_cfg: dict) -> tuple[None, Plant]:
        """Build a standalone analytical plant from the registry (no MuJoCo sim).

        Requires ``[plant].type`` (registered name) and ``[plant].config``
        (path to a plant TOML). Applies ``[plant].initial_state`` if given and
        enforces that ``[scenario].dt`` matches the plant's own ``dt``.
        """
        ptype = plant_cfg.get("type")
        if ptype is None:
            raise KeyError(
                "Plant-only scenarios (no [sim] section) require [plant].type "
                "(registered plant name) and [plant].config (path to a plant TOML)."
            )
        pconfig = plant_cfg.get("config")
        if pconfig is None:
            raise KeyError("Plant-only scenarios require [plant].config (path to a plant TOML).")
        if ptype not in _PLANT_REGISTRY:
            raise KeyError(f"Unknown plant type '{ptype}'. Registered: {sorted(_PLANT_REGISTRY)}")

        with open(resolve_config_path(pconfig), "rb") as f:
            plant_dict = tomllib.load(f)
        plant = _PLANT_REGISTRY[ptype].from_config(plant_dict, backend=NumpyBackend())

        if "initial_state" in plant_cfg:
            init = plant_cfg["initial_state"]
            n_x = plant.get_state().shape[0]
            if len(init) != n_x:
                raise ValueError(
                    f"[plant].initial_state has length {len(init)} but plant '{ptype}' has state dimension {n_x}."
                )
            plant.state = plant.bk.array(init)

        return None, plant

    @staticmethod
    def _create_loop_role(factory_cls, config_path: str, plant: Plant, backend: ArrayBackend | None, derive_model: bool):
        """Build a controller/estimator, deriving A/B from the plant when omitted.

        In plant-only mode (``derive_model``), a config that omits
        ``A_dynamics``/``B_dynamics`` gets the plant's linearized model
        (upright equilibrium) discretized with first-order Euler at the
        plant's ``dt``. Sim-backed scenarios are untouched (their
        ``A = I, B = dt * I`` defaults are correct for the velocity-commanded
        base).
        """
        with open(resolve_config_path(config_path), "rb") as f:
            cfg = tomllib.load(f)
        if derive_model and "A_dynamics" not in cfg and "B_dynamics" not in cfg:
            A_c, B_c = linearize_plant(plant)
            A_d, B_d = discretize_euler(A_c, B_c, plant.dt, backend=plant.bk)
            cfg = {**cfg, "A_dynamics": A_d, "B_dynamics": B_d}
        if backend is not None:
            return factory_cls(config=cfg).create(backend=backend)
        return factory_cls(config=cfg).create()

    @staticmethod
    def _physics_xml(physics_cfg: dict) -> tuple[str, dict] | tuple[None, None]:
        """Build the (xml_string, assets) pair for RobotSim from a physics config.

        When ``free_joint`` is truthy, loads the stock LeKiwi MJCF, rewrites it
        with :func:`demos.helpers.inject_free_joint`, and loads the mesh assets
        so the model can be built from the string.

        Args:
            physics_cfg: The ``[physics]`` section of the scenario config.

        Returns:
            Tuple of (xml_string, assets). Both are None when the stock MJCF is
            loaded from the model path.
        """
        if not physics_cfg.get("free_joint"):
            return None, None

        from pathlib import Path

        from demos.helpers import inject_free_joint, load_model_assets
        from lekiwi_sim import HERE, MJCF_PATH

        xml = inject_free_joint(Path(MJCF_PATH).read_text())
        assets = load_model_assets(HERE / "lekiwi-sim" / "meshes")
        return xml, assets

    @staticmethod
    def _validate_dimensions(n_x: int, n_u: int, controller: Controller, estimator: StateEstimator) -> None:
        """Sanity-check that controller/estimator agree with the plant dims.

        Estimators expose their internal ``A``; controllers built from config
        default to ``A = I, B = dt I`` with dimension equal to their Q matrix.
        Mismatched dims are a scenario-authoring error.
        """
        est_x = getattr(estimator, "A", None)
        if est_x is not None and est_x.shape[0] != n_x:
            raise ValueError(f"Estimator state dimension {est_x.shape[0]} does not match plant state dimension {n_x}.")
        ctrl_B = getattr(controller, "B", None)
        if ctrl_B is not None and ctrl_B.shape[1] != n_u:
            raise ValueError(f"Controller input dimension {ctrl_B.shape[1]} does not match plant input dimension {n_u}.")
        ctrl_nu = getattr(controller, "D_u", None)
        if ctrl_nu is not None and ctrl_nu != n_u:
            raise ValueError(f"Controller input dimension {ctrl_nu} does not match plant input dimension {n_u}.")
