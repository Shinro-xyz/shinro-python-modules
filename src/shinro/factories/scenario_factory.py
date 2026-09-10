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

    A scenario is a runnable simulation: :meth:`run` executes the whole loop
    (closed-loop or feedforward, chosen by whether ``controller`` is set),
    driven entirely by the TOML — duration, dt, noise, adversarial faults,
    input limits and tolerances. :meth:`iter_run` yields one
    :class:`~shinro.simulation.runner.StepRecord` per step for live rendering
    or early stopping; :meth:`reset` restores the TOML initial state and
    clears controller/estimator state.
    """

    sim: RobotSim | None
    plant: Plant
    controller: Controller | None
    estimator: StateEstimator | None
    trajectory: TrajectoryGenerator | Any
    config: dict[str, Any] = field(default_factory=dict)

    def run(self, steps: int | None = None, seed: int | None = None):
        """Run the scenario to completion.

        Args:
            steps: Number of steps. Defaults to ``[scenario].duration / dt``.
            seed: Noise RNG seed. Overrides ``[noise.measurement].seed``.

        Returns:
            A :class:`~shinro.simulation.runner.SimResult` — one
            :class:`~shinro.simulation.runner.StepRecord` per step, plus
            ``check()`` tolerance reporting against the scenario TOML.
        """
        from shinro.simulation.runner import run_phase_schedule, run_scenario

        if self.controller is None:
            return run_phase_schedule(self, steps=steps)
        return run_scenario(self, steps=steps, seed=seed)

    def iter_run(self, steps: int | None = None, seed: int | None = None):
        """Iterate over the run, yielding one StepRecord per step.

        For live rendering or early stopping; :meth:`run` collects the same
        records into a :class:`~shinro.simulation.runner.SimResult`.
        """
        from shinro.simulation.runner import iter_phase_schedule, iter_scenario

        if self.controller is None:
            return iter_phase_schedule(self, steps=steps)
        return iter_scenario(self, steps=steps, seed=seed)

    def reset(self) -> None:
        """Restore the TOML initial state and clear controller/estimator state.

        Deterministic reproduction is ``scenario.reset(); scenario.run(seed=0)``.
        """
        if self.sim is not None:
            self.sim.reset()
        else:
            init = self.config.get("plant", {}).get("initial_state")
            if init is not None:
                self.plant.state = self.plant.bk.array(init)
            else:
                self.plant.state = self.plant.bk.zeros_like(self.plant.get_state())
        if self.controller is not None:
            self.controller.reset()
        if self.estimator is not None:
            self.estimator.reset()


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

    The plant TOML is the single source of physics truth. Controller/estimator
    configs never need to restate it: ``dt`` is filled from ``plant.dt`` when
    omitted (a declared ``dt`` that disagrees is a loud error), and in
    plant-only mode an omitted ``A_dynamics``/``B_dynamics`` model is derived
    from the plant via :func:`shinro.utils.linearization.linearize_plant`
    (upright equilibrium) and first-order Euler discretization.
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
        """Build a controller/estimator with plant-derived injection.

        The plant is the single source of physics truth. In **both** modes the
        config's ``dt`` is filled from ``plant.dt`` when omitted, and a
        declared ``dt`` that disagrees with the plant's is a loud error (a
        mismatched PID/MPPI dt silently mis-scales integration). In plant-only
        mode (``derive_model``), a config that omits ``A_dynamics``/
        ``B_dynamics`` additionally gets the plant's linearized model (upright
        equilibrium) discretized with first-order Euler at the plant's ``dt`` —
        the same derivation the compile path uses. Sim-backed scenarios keep
        their model defaults untouched (``A = I, B = dt * I`` is correct for
        the velocity-commanded base). See
        :func:`shinro.utils.linearization.inject_plant_derived`.
        """
        return factory_cls(config_path).create(backend=backend, plant=plant, derive_model=derive_model)

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
