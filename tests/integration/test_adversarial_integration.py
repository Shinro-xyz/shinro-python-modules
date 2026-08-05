"""Adversarial integration: faults injected into the full closed loop must not corrupt control.

The contract (agreed in design): an estimator that receives a NaN/Inf
measurement must NOT silently propagate it into the control signal. The runner
raises :class:`ValueError` if a non-finite estimate ever reaches the
controller, and the tests below pin that behavior end-to-end.
"""

import numpy as np
import pytest

from shinro.factories import ScenarioFactory

from .helpers.scenario_runner import run_scenario

pytestmark = [pytest.mark.integration]

SCENARIO = "tests/integration/scenarios/adversarial_nan.toml"


class TestAdversarialNaN:
    """A NaN measurement surfaces as an error rather than corrupting control."""

    def test_nan_measurement_raises(self, mujoco_available):
        """The loop raises once the estimator sees a NaN measurement."""
        scenario = ScenarioFactory(SCENARIO).build()
        with pytest.raises(ValueError, match="non-finite estimate"):
            run_scenario(scenario)

    def test_run_without_fault_stays_clean(self, mujoco_available, tmp_path):
        """The same loop without the adversarial section runs to completion."""
        import textwrap

        clean = tmp_path / "clean.toml"
        clean.write_text(
            textwrap.dedent(
                """\
                [scenario]
                name = "clean"
                duration = 16.0
                dt = 0.02
                tolerance = { steady_state = 0.05, estimator = 0.03 }
                input_limits = { min = [-0.5, -0.5, -1.0], max = [0.5, 0.5, 1.0] }

                [physics]
                free_joint = true

                [plant]
                name = "base"

                [controller]
                type = "LQR"
                config = "configs/controllers/lqr_base.toml"

                [estimator]
                type = "KalmanFilter"
                config = "configs/estimators/kalman_base.toml"

                [trajectory]
                type = "waypoints"
                config = "configs/trajectories/base_straight.toml"

                [sim]
                config = "robot_config.toml"
                """
            )
        )
        scenario = ScenarioFactory(str(clean)).build()
        records = run_scenario(scenario)
        assert np.all(np.isfinite([r.plant_state[0] for r in records]))

    def test_measurement_fault_precedes_error(self, mujoco_available):
        """The fault fires exactly once, at the configured step."""
        scenario = ScenarioFactory(SCENARIO).build()
        with pytest.raises(ValueError):
            run_scenario(scenario, steps=60)
