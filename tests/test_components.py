import pytest
from components import Controller, Plant, StateEstimator, TrajectoryGenerator, PhysicsEngine
from utils.array_backend import NumpyBackend


class TestABCs:
    """Verify that abstract base classes cannot be instantiated directly."""

    def test_controller_abstract(self):
        """Controller ABC raises TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            Controller()

    def test_plant_abstract(self):
        """Plant ABC raises TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            Plant()

    def test_state_estimator_abstract(self):
        """StateEstimator ABC raises TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            StateEstimator()

    def test_trajectory_generator_abstract(self):
        """TrajectoryGenerator ABC raises TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            TrajectoryGenerator()

    def test_physics_engine_abstract(self):
        """PhysicsEngine ABC raises TypeError on direct instantiation."""
        with pytest.raises(TypeError):
            PhysicsEngine()

    def test_physics_engine_default_backend(self):
        """PhysicsEngine.backend property defaults to NumpyBackend."""
        assert PhysicsEngine.backend.fget(None) is not None
        bk = PhysicsEngine.backend.fget(None)
        assert isinstance(bk, NumpyBackend)

    def test_plant_dynamics_default_none(self):
        """Plant.dynamics() returns None by default."""
        class MinimalPlant(Plant):
            def get_state(self): return None
            def get_model(self): return None
            def step(self, u): return None
            def physics_engine(self, engine): pass
        p = MinimalPlant()
        assert p.dynamics(None, None) is None
