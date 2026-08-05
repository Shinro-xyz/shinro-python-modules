"""Compose a full control loop (Trajectory → Controller → Estimator → Plant → Engine) from one TOML.

A **scenario** declares every ABC role plus the scenario parameters (duration,
tolerances, noise, physics) in a single file. :class:`ScenarioFactory` builds the
whole loop and validates that the dimensions of each component agree with the
plant.
"""

import tomllib
from dataclasses import dataclass, field
from typing import Any

from components import Controller, Plant, StateEstimator, TrajectoryGenerator
from factories.controller_factory import ControllerFactory
from factories.estimator_factory import EstimatorFactory
from factories.trajectory_factory import TrajectoryFactory
from simulation.robotsim import RobotSim
from utils.array_backend import ArrayBackend


@dataclass
class Scenario:
    """Fully composed control loop plus scenario parameters.

    ``sim`` is the :class:`RobotSim` instance (engine + plants); ``plant`` is
    the primary plant driven by the controller; ``controller``, ``estimator``
    and ``trajectory`` are the other loop roles. For feedforward scenarios
    (e.g. ``phase_list`` pick-and-place) ``controller`` and ``estimator`` are
    ``None`` and the schedule itself is the control. ``config`` is the raw TOML
    dict (used by the runner and the tests for tolerances, noise, etc.).
    """

    sim: RobotSim
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
        [plant]       name
        [controller]  type, config (optional)
        [estimator]   type, config (optional)
        [trajectory]  type, config
        [sim]         config (path to the RobotSim TOML)
        [noise]       measurement (optional)
        [adversarial] inject_at, value (optional)

    The plant is looked up by name on the ``RobotSim`` built from ``[sim]``.
    Controller/estimator/trajectory are built from their own config files via
    the existing factories. When ``[physics].free_joint`` is set, the LeKiwi
    MJCF is rewritten so the arm hangs off a mobile free-jointed base.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "rb") as f:
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

        xml_string, assets = self._physics_xml(physics_cfg)
        sim = RobotSim(sim_cfg["config"], xml_string=xml_string, assets=assets)  # type: ignore[arg-type]
        plant = sim.get_plant(plant_cfg["name"])
        if plant is None:
            raise KeyError(
                f"Plant name '{plant_cfg['name']}' not found in RobotSim. "
                f"Available plants: {sorted(sim._plants.keys())}"
            )

        def _create(factory_cls, path: str):
            if backend is not None:
                return factory_cls(path).create(backend=backend)
            return factory_cls(path).create()

        trajectory = _create(TrajectoryFactory, self.config["trajectory"]["config"])

        controller = None
        estimator = None
        if "controller" in self.config:
            controller = _create(ControllerFactory, self.config["controller"]["config"])
        if "estimator" in self.config:
            estimator = _create(EstimatorFactory, self.config["estimator"]["config"])

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
            raise ValueError(
                f"Estimator state dimension {est_x.shape[0]} does not match plant state dimension {n_x}."
            )
        ctrl_B = getattr(controller, "B", None)
        if ctrl_B is not None and ctrl_B.shape[1] != n_u:
            raise ValueError(
                f"Controller input dimension {ctrl_B.shape[1]} does not match plant input dimension {n_u}."
            )
