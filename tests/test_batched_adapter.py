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

    def test_nonlinear_dynamics_matches_per_sample_calls(self, bk):
        """The rollout equals a manual Euler step of per-sample ``dynamics`` calls.

        The plant's ``dynamics`` is batch-capable, so this checks the rank
        handling: the ``(N, ·)`` call the adapter makes must agree with ``N``
        single-state calls.
        """
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
        assert np.allclose(_to_np(out, bk), np.array(expected), atol=1e-12)

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


class TestDynamicsDispatch:
    """Which dynamics path the adapter picks, and that the nonlinear one traces."""

    def _pendulum(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        return InvertedPendulum(mass=0.1, length=0.5, damping=0.0, gravity=9.81, dt=0.01, backend=bk)

    def test_nonlinear_vs_lti(self, bk):
        """A plant with ``dynamics`` is nonlinear; an (A, B)-only plant is LTI."""
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter
        assert BatchedDynamicsAdapter(self._pendulum(bk)).dynamics_path == "nonlinear"
        base = HolonomicMobileRobot(
            num_wheels=3, radius_robots=0.1, gamma=0.0, radius_wheels=0.03, dt=0.02, backend=bk
        )
        assert BatchedDynamicsAdapter(base).dynamics_path == "lti"

    def test_nonlinear_path_uses_caller_backend(self):
        """Tracers plus a caller-supplied backend: the plant emits graph nodes."""
        from shinro.codegen.trace_backend import TraceBackend
        from shinro.codegen.tracing import Graph, Tracer
        from shinro.utils.array_backend import NumpyBackend
        from shinro.utils.batched_adapter import BatchedDynamicsAdapter
        adapter = BatchedDynamicsAdapter(self._pendulum(NumpyBackend()))
        g = Graph()
        x = Tracer(g, (3, 2), g.input("x", (3, 2)))
        u = Tracer(g, (3, 1), g.input("u", (3, 1)))
        out = adapter.dynamics_fn(x, u, 0.01, bk=TraceBackend(g))
        assert isinstance(out, Tracer)
        assert out.shape == (3, 2)
        ops = [n.op for n in g.nodes]
        assert "sin" in ops
        assert "slice" in ops
        assert "stack" in ops

    def test_attach_plant_routes_controller_backend(self, bk):
        """attach_plant evaluates the nonlinear dynamics with the controller's bk."""
        from shinro.codegen.trace_backend import TraceBackend
        from shinro.codegen.tracing import Graph, Tracer
        from shinro.controllers.mppi import MPPIController
        ctrl = MPPIController(
            num_samples=3, temperature=1.0, dt=0.01, horizon=2,
            noise_sigma=[0.5], u_min=[-1.0], u_max=[1.0], seed=0, backend=bk,
        )
        ctrl.attach_plant(self._pendulum(bk))
        g = Graph()
        x = Tracer(g, (3, 2), g.input("x", (3, 2)))
        u = Tracer(g, (3, 1), g.input("u", (3, 1)))
        original = ctrl.bk
        ctrl.bk = TraceBackend(g)  # type: ignore[assignment]  # tracing swaps in a TraceBackend
        try:
            assert ctrl.dynamics_fn is not None
            out = ctrl.dynamics_fn(x, u, 0.01)
        finally:
            ctrl.bk = original
        assert isinstance(out, Tracer)
        assert out.shape == (3, 2)
        assert any(n.op == "sin" for n in g.nodes)

    def test_pendulum_dynamics_traces(self):
        """InvertedPendulum.dynamics emits graph nodes under a TraceBackend."""
        from shinro.codegen.trace_backend import TraceBackend
        from shinro.codegen.tracing import Graph, Tracer
        from shinro.plants.inverted_pendulum import InvertedPendulum
        from shinro.utils.array_backend import NumpyBackend
        plant = InvertedPendulum(backend=NumpyBackend())
        g = Graph()
        x = Tracer(g, (4, 2), g.input("x", (4, 2)))
        u = Tracer(g, (4, 1), g.input("u", (4, 1)))
        out = plant.dynamics(x, u, bk=TraceBackend(g))
        assert isinstance(out, Tracer)
        assert out.shape == (4, 2)
        ops = [n.op for n in g.nodes]
        assert "sin" in ops
        assert "slice" in ops
        assert "stack" in ops
