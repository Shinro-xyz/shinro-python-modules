import numpy as np
import pytest


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestBatchedDynamicsAdapter:
    """Verify the adapter's batched dynamics and cost against analytic results."""

    def _base_plant(self, bk):
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        return HolonomicMobileRobot(
            num_wheels=3, radius_robots=0.1, gamma=0.0, radius_wheels=0.03, dt=0.02, backend=bk
        )

    def _pendulum_plant(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        return InvertedPendulum(mass=0.1, length=0.5, damping=0.0, gravity=9.81, dt=0.01, backend=bk)

    def test_lti_dynamics_matches_matmul(self, bk):
        """The LTI dynamics equals x @ A.T + u @ B.T for a batch."""
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter
        plant = self._base_plant(bk)
        adapter = BatchedDynamicsAdapter(plant)
        A, B = plant.get_model()
        x = bk.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-2.0, 0.5, 1.5]])
        u = bk.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0], [0.5, -0.5, 0.1], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        out = adapter.dynamics_fn(x, u, 0.02)
        expected = x @ A.T + u @ B.T
        assert _to_np(out, bk).shape == (5, 3)
        assert np.allclose(_to_np(out, bk), _to_np(expected, bk))

    def test_lti_zero_input_keeps_state(self, bk):
        """With zero control and A = I, the state is unchanged."""
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter
        adapter = BatchedDynamicsAdapter(self._base_plant(bk))
        x = bk.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
        u = bk.zeros((2, 3))
        out = adapter.dynamics_fn(x, u, 0.02)
        assert np.allclose(_to_np(out, bk), _to_np(x, bk))

    def test_nonlinear_dynamics_matches_euler(self, bk):
        """The nonlinear dynamics equals a manual Euler step of plant.dynamics."""
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter
        plant = self._pendulum_plant(bk)
        adapter = BatchedDynamicsAdapter(plant)
        dt = 0.01
        x = bk.array([[0.1, 0.0], [0.5, 0.3], [-0.2, 1.0], [1.5, -0.5], [0.0, 0.0]])
        u = bk.array([[0.1], [0.2], [-0.3], [0.5], [0.0]])
        out = adapter.dynamics_fn(x, u, dt)
        expected = []
        for i in range(5):
            x_next = x[i] + dt * plant.dynamics(x[i], u[i])
            expected.append(_to_np(x_next, bk))
        assert _to_np(out, bk).shape == (5, 2)
        assert np.allclose(_to_np(out, bk), np.array(expected), atol=1e-10)

    def test_cost_matches_analytic(self, bk):
        """The batched cost equals (x-x_ref)ᵀQ(x-x_ref) + uᵀRu per sample."""
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter
        adapter = BatchedDynamicsAdapter(self._base_plant(bk))
        Q = bk.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
        R = bk.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        x_ref = bk.array([1.0, 1.0, 1.0])
        x = bk.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        u = bk.zeros((3, 3))
        out = adapter.cost_fn(x, u, Q, R, x_ref=x_ref)
        expected = np.array([
            (x[0] - x_ref) @ Q @ (x[0] - x_ref),
            (x[1] - x_ref) @ Q @ (x[1] - x_ref),
            (x[2] - x_ref) @ Q @ (x[2] - x_ref),
        ])
        assert _to_np(out, bk).shape == (3,)
        assert np.allclose(_to_np(out, bk), expected)

    def test_cost_diagonal_weights(self, bk):
        """Diagonal Q/R are treated as elementwise weights."""
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter
        adapter = BatchedDynamicsAdapter(self._base_plant(bk))
        Q = bk.array([1.0, 2.0, 3.0])
        R = bk.array([0.5, 0.5, 0.5])
        x = bk.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
        u = bk.array([[0.1, 0.1, 0.1], [0.0, 0.0, 0.0]])
        out = adapter.cost_fn(x, u, Q, R)
        expected = np.array([
            1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 0.5 * 0.01 * 3,
            1.0 * 4.0 + 2.0 * 9.0 + 3.0 * 16.0,
        ])
        assert np.allclose(_to_np(out, bk), expected)

    def test_cost_regulation_to_origin(self, bk):
        """Without x_ref, the cost is xᵀQx + uᵀRu (regulation to origin)."""
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter
        adapter = BatchedDynamicsAdapter(self._base_plant(bk))
        Q = bk.eye(3)
        R = bk.eye(3)
        x = bk.array([[1.0, 2.0, 3.0]])
        u = bk.array([[0.0, 0.0, 0.0]])
        out = adapter.cost_fn(x, u, Q, R)
        assert np.allclose(_to_np(out, bk), [14.0])

    def test_torch_backend_keeps_tensors(self, bk):
        """On a torch backend, batched dynamics and cost stay torch tensors."""
        pytest.importorskip("torch")
        if not hasattr(bk, "torch"):
            pytest.skip("requires TorchBackend")
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter
        adapter = BatchedDynamicsAdapter(self._base_plant(bk))
        x = bk.zeros((4, 3))
        u = bk.zeros((4, 3))
        out = adapter.dynamics_fn(x, u, 0.02)
        assert isinstance(out, bk.torch.Tensor)
        cost = adapter.cost_fn(x, u, None, None)
        assert isinstance(cost, bk.torch.Tensor)
        assert cost.shape == (4,)


class TestBatchedAdapterNonlinearVmap:
    """Verify torch.vmap vectorizes nonlinear plant dynamics over the batch."""

    def test_vmap_pendulum(self, bk):
        """vmap(pendulum.dynamics) produces (N, 2) from (N, 2), (N, 1)."""
        torch = pytest.importorskip("torch")
        if not hasattr(bk, "torch"):
            pytest.skip("requires TorchBackend")
        from shinro.plants.inverted_pendulum import InvertedPendulum
        plant = InvertedPendulum(backend=bk)
        x = bk.array([[0.1, 0.0], [0.5, 0.3], [1.0, -1.0], [0.0, 0.0], [-0.2, 0.7]])
        u = bk.array([[0.1], [0.2], [0.0], [0.5], [-0.3]])
        out = torch.vmap(plant.dynamics, in_dims=(0, 0))(x, u)
        assert out.shape == (5, 2)
        for i in range(5):
            single = plant.dynamics(x[i], u[i])
            assert torch.allclose(out[i], single)

    def test_vmap_cartpole(self, bk):
        """vmap(cartpole.dynamics) produces (N, 4) from (N, 4), (N, 1)."""
        torch = pytest.importorskip("torch")
        if not hasattr(bk, "torch"):
            pytest.skip("requires TorchBackend")
        from shinro.plants.cartpole import CartPole
        plant = CartPole(backend=bk)
        x = bk.array([[0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 1.0, 0.0]])
        u = bk.array([[1.0], [0.0], [-1.0]])
        out = torch.vmap(plant.dynamics, in_dims=(0, 0))(x, u)
        assert out.shape == (3, 4)
        for i in range(3):
            single = plant.dynamics(x[i], u[i])
            assert torch.allclose(out[i], single)
