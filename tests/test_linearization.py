import pytest
import numpy as np
from utils.array_backend import NumpyBackend


def _to_np(x, bk):
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestNumpyBackendJacobian:
    """Verify NumpyBackend.jacobian via central finite differences."""

    def setup_method(self):
        self.bk = NumpyBackend()

    def test_linear_scalar(self):
        """Jacobian of f(x) = 2x is [[2]]."""
        J = self.bk.jacobian(lambda x: np.array([2.0 * x[0]]), np.array([3.0]))
        assert np.allclose(J, [[2.0]])

    def test_linear_vector(self):
        """Jacobian of f(x) = Ax where A is 2x3."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        x = np.array([1.0, 2.0, 3.0])
        J = self.bk.jacobian(lambda x: A @ x, x)
        assert np.allclose(J, A)

    def test_quadratic(self):
        """Jacobian of f(x) = [x0^2 + x1, x0 * x1] at (2, 3)."""
        def f(x):
            return np.array([x[0]**2 + x[1], x[0] * x[1]])
        J = self.bk.jacobian(f, np.array([2.0, 3.0]))
        expected = np.array([[4.0, 1.0], [3.0, 2.0]])
        assert np.allclose(J, expected, atol=1e-5)

    def test_trig(self):
        """Jacobian of f(x) = [sin(x0), cos(x1)] at (0, pi/2)."""
        def f(x):
            return np.array([np.sin(x[0]), np.cos(x[1])])
        J = self.bk.jacobian(f, np.array([0.0, np.pi / 2]))
        expected = np.array([[1.0, 0.0], [0.0, -1.0]])
        assert np.allclose(J, expected, atol=1e-5)

    def test_scalar_output_vector_input(self):
        """Jacobian of f(x) = x0^2 + x1^2 + x2^2 is 2*x^T."""
        def f(x):
            return np.array(x[0]**2 + x[1]**2 + x[2]**2)
        x = np.array([1.0, 2.0, 3.0])
        J = self.bk.jacobian(f, x)
        assert np.allclose(J, [[2.0, 4.0, 6.0]])

    def test_identity(self):
        """Jacobian of f(x) = x is I."""
        def f(x):
            return x
        J = self.bk.jacobian(f, np.array([1.0, 2.0, 3.0]))
        assert np.allclose(J, np.eye(3))

    def test_eps_parameter_affects_accuracy(self):
        """Larger eps reduces accuracy for nonlinear functions."""
        def f(x):
            return np.array([x[0]**3])
        J_fine = self.bk.jacobian(f, np.array([2.0]), eps=1e-6)
        J_coarse = self.bk.jacobian(f, np.array([2.0]), eps=1e-1)
        expected = np.array([[12.0]])
        assert np.allclose(J_fine, expected, atol=1e-4)
        coarse_error = abs(J_coarse[0, 0] - 12.0)
        fine_error = abs(J_fine[0, 0] - 12.0)
        assert coarse_error > fine_error


class TestTorchBackendJacobian:
    """Verify TorchBackend.jacobian via autograd (exact)."""

    def setup_method(self):
        torch = pytest.importorskip("torch")
        from utils.array_backend import TorchBackend
        self.bk = TorchBackend(device="cpu")
        self.torch = torch

    def _t(self, data):
        return self.torch.tensor(data, dtype=self.torch.float64)

    def test_linear_scalar(self):
        def f(x):
            return 2.0 * x
        J = self.bk.jacobian(f, self._t([3.0]))
        assert self.torch.allclose(J, self._t([[2.0]]))

    def test_linear_vector(self):
        A = self._t([[1, 2, 3], [4, 5, 6]])
        def f(x):
            return A @ x
        x = self._t([1.0, 2.0, 3.0])
        J = self.bk.jacobian(f, x)
        assert self.torch.allclose(J, A)

    def test_quadratic(self):
        def f(x):
            return self.torch.stack([x[0]**2 + x[1], x[0] * x[1]])
        x = self._t([2.0, 3.0])
        J = self.bk.jacobian(f, x)
        expected = self._t([[4.0, 1.0], [3.0, 2.0]])
        assert self.torch.allclose(J, expected, atol=1e-5)

    def test_trig(self):
        def f(x):
            return self.torch.stack([self.torch.sin(x[0]), self.torch.cos(x[1])])
        x = self._t([0.0, np.pi / 2])
        J = self.bk.jacobian(f, x)
        expected = self._t([[1.0, 0.0], [0.0, -1.0]])
        assert self.torch.allclose(J, expected, atol=1e-5)

    def test_scalar_output_vector_input(self):
        def f(x):
            return x[0]**2 + x[1]**2 + x[2]**2
        x = self._t([1.0, 2.0, 3.0])
        J = self.bk.jacobian(f, x)
        assert self.torch.allclose(J, self._t([[2.0, 4.0, 6.0]]))

    def test_identity(self):
        def f(x):
            return x
        x = self._t([1.0, 2.0, 3.0])
        J = self.bk.jacobian(f, x)
        assert self.torch.allclose(J, self.torch.eye(3, dtype=self.torch.float64))

    def test_eps_ignored(self):
        """eps parameter is accepted but ignored (autograd is exact)."""
        def f(x):
            return x[0]**3
        x = self._t([2.0])
        J1 = self.bk.jacobian(f, x, eps=1e-6)
        J2 = self.bk.jacobian(f, x, eps=1e-2)
        expected = self._t([[12.0]])
        assert self.torch.allclose(J1, expected)
        assert self.torch.allclose(J2, expected)

    def test_numpy_input_converted(self):
        """Passing a numpy array is converted to torch internally."""
        def f(x):
            return self.torch.stack([x[0]**2, x[1]**2])
        J = self.bk.jacobian(f, np.array([2.0, 3.0]))
        expected = self._t([[4.0, 0.0], [0.0, 6.0]])
        assert self.torch.allclose(J, expected, atol=1e-5)


class TestLinearize:
    """Verify the linearize() function produces correct (A, B) pairs."""

    def _f(self, x, u, bk):
        """Backend-agnostic helper: convert to numpy for computation, return numpy."""
        x_np = _to_np(x, bk)
        u_np = _to_np(u, bk)
        return bk.from_numpy(self._dynamics_np(x_np, u_np))

    def _dynamics_np(self, x, u):
        raise NotImplementedError

    def test_simple_integrator(self, bk):
        """f(x,u) = u → A=0, B=I."""
        from utils.linearization import linearize
        def f(x, u):
            return u
        x0 = bk.zeros(2)
        u0 = bk.zeros(2)
        A, B = linearize(f, x0, u0, bk)
        assert np.allclose(_to_np(A, bk), np.zeros((2, 2)))
        assert np.allclose(_to_np(B, bk), np.eye(2))

    def test_double_integrator(self, bk):
        """f([x,v], u) = [v, u] → A=[[0,1],[0,0]], B=[[0],[1]]."""
        from utils.linearization import linearize
        def f(x, u):
            return np.array([x[1], u[0]])
        x0 = bk.zeros(2)
        u0 = bk.zeros(1)
        A, B = linearize(f, x0, u0, bk)
        assert np.allclose(_to_np(A, bk), [[0, 1], [0, 0]], atol=1e-5)
        assert np.allclose(_to_np(B, bk), [[0], [1]], atol=1e-5)

    def test_pendulum_upright(self, bk):
        """Linearized pendulum at upright: A=[[0,1],[-g/l,0]], B=[[0],[1/(ml^2)]]."""
        from utils.linearization import linearize
        g, l, m = 9.81, 0.5, 0.1
        def f(x, u):
            theta, theta_dot = x
            torque = u[0]
            alpha = -(g / l) * np.sin(theta) + torque / (m * l**2)
            return np.array([theta_dot, alpha])
        x0 = bk.zeros(2)
        u0 = bk.zeros(1)
        A, B = linearize(f, x0, u0, bk)
        expected_A = np.array([[0, 1], [-g / l, 0]])
        expected_B = np.array([[0], [1.0 / (m * l**2)]])
        assert np.allclose(_to_np(A, bk), expected_A, atol=1e-5)
        assert np.allclose(_to_np(B, bk), expected_B, atol=1e-5)

    def test_pendulum_downward(self, bk):
        """Linearized pendulum at downward: A=[[0,1],[g/l,0]], B=[[0],[1/(ml^2)]]."""
        from utils.linearization import linearize
        g, l, m = 9.81, 0.5, 0.1
        def f(x, u):
            theta, theta_dot = x
            torque = u[0]
            alpha = -(g / l) * np.sin(theta) + torque / (m * l**2)
            return np.array([theta_dot, alpha])
        x0 = bk.array([np.pi, 0.0])
        u0 = bk.zeros(1)
        A, B = linearize(f, x0, u0, bk)
        expected_A = np.array([[0, 1], [g / l, 0]])
        expected_B = np.array([[0], [1.0 / (m * l**2)]])
        assert np.allclose(_to_np(A, bk), expected_A, atol=1e-5)
        assert np.allclose(_to_np(B, bk), expected_B, atol=1e-5)

    def test_cartpole_upright(self, bk):
        """Linearized cartpole at upright equilibrium."""
        from utils.linearization import linearize
        mc, mp, l, g = 1.0, 0.1, 0.5, 9.81
        def f(x, u):
            x_pos, theta, xd, thd = x
            F = u[0]
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            denom = mc + mp * sin_theta**2
            xdd = (F + mp * sin_theta * (l * thd**2 + g * cos_theta)) / denom
            thdd = (-F * cos_theta - mp * l * thd**2 * cos_theta * sin_theta
                    - (mc + mp) * g * sin_theta) / (l * denom)
            return np.array([xd, thd, xdd, thdd])
        x0 = bk.zeros(4)
        u0 = bk.zeros(1)
        A, B = linearize(f, x0, u0, bk)
        expected_A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, mp * g / mc, 0, 0],
            [0, -(mc + mp) * g / (mc * l), 0, 0],
        ])
        expected_B = np.array([[0], [0], [1.0 / mc], [-1.0 / (mc * l)]])
        assert np.allclose(_to_np(A, bk), expected_A, atol=1e-4)
        assert np.allclose(_to_np(B, bk), expected_B, atol=1e-4)

    def test_default_backend(self):
        """linearize() uses NumpyBackend when no backend is given."""
        from utils.linearization import linearize
        def f(x, u):
            return u
        x0 = np.zeros(2)
        u0 = np.zeros(2)
        A, B = linearize(f, x0, u0)
        assert np.allclose(A, np.zeros((2, 2)))
        assert np.allclose(B, np.eye(2))

    def test_eps_affects_result(self, bk):
        """Larger eps reduces accuracy for cubic function."""
        from utils.linearization import linearize
        def f(x, u):
            return np.array([x[0]**3 + u[0]])
        x0 = bk.array([2.0])
        u0 = bk.array([0.0])
        A_fine, B_fine = linearize(f, x0, u0, bk, eps=1e-6)
        A_coarse, B_coarse = linearize(f, x0, u0, bk, eps=1e-1)
        assert np.allclose(_to_np(A_fine, bk), [[12.0]], atol=1e-3)
        assert not np.allclose(_to_np(A_coarse, bk), [[12.0]], atol=1e-3)


class TestLinearizeEdgeCases:
    """Edge cases for linearize()."""

    def test_zero_operating_point(self, bk):
        """Linearization at zero works for a system with no drift."""
        from utils.linearization import linearize
        def f(x, u):
            return np.array([x[1], -x[0] + u[0]])
        x0 = bk.zeros(2)
        u0 = bk.zeros(1)
        A, B = linearize(f, x0, u0, bk)
        assert np.allclose(_to_np(A, bk), [[0, 1], [-1, 0]], atol=1e-5)
        assert np.allclose(_to_np(B, bk), [[0], [1]], atol=1e-5)

    def test_nonzero_operating_point(self, bk):
        """Linearization at a nonzero operating point captures local dynamics."""
        from utils.linearization import linearize
        def f(x, u):
            return np.array([x[1], -np.sin(x[0]) + u[0]])
        x0 = bk.array([0.5, 0.0])
        u0 = bk.array([np.sin(0.5)])
        A, B = linearize(f, x0, u0, bk)
        expected_A = np.array([[0, 1], [-np.cos(0.5), 0]])
        expected_B = np.array([[0], [1]])
        assert np.allclose(_to_np(A, bk), expected_A, atol=1e-5)
        assert np.allclose(_to_np(B, bk), expected_B, atol=1e-5)

    def test_scalar_state_and_input(self, bk):
        """Linearization works with scalar (1D) state and input."""
        from utils.linearization import linearize
        def f(x, u):
            return np.array([-x[0] + u[0]])
        x0 = bk.array([0.0])
        u0 = bk.array([0.0])
        A, B = linearize(f, x0, u0, bk)
        assert np.allclose(_to_np(A, bk), [[-1.0]])
        assert np.allclose(_to_np(B, bk), [[1.0]])

    def test_multi_input(self, bk):
        """Linearization with multi-dimensional input."""
        from utils.linearization import linearize
        def f(x, u):
            return np.array([u[0] + u[1], u[0] - u[1]])
        x0 = bk.zeros(2)
        u0 = bk.zeros(2)
        A, B = linearize(f, x0, u0, bk)
        assert np.allclose(_to_np(A, bk), np.zeros((2, 2)))
        assert np.allclose(_to_np(B, bk), [[1, 1], [1, -1]])

    def test_linear_system_is_exact(self, bk):
        """Linearization of a linear system recovers the exact matrices."""
        from utils.linearization import linearize
        A_true = np.array([[0.5, 1.0], [-0.2, 0.8]])
        B_true = np.array([[0.0], [1.0]])
        def f(x, u):
            return A_true @ x + B_true @ u
        x0 = bk.array([1.0, -0.5])
        u0 = bk.array([0.3])
        A, B = linearize(f, x0, u0, bk)
        assert np.allclose(_to_np(A, bk), A_true, atol=1e-10)
        assert np.allclose(_to_np(B, bk), B_true, atol=1e-10)

    def test_high_dimensional(self, bk):
        """Linearization works for 10-dimensional state and 5-dimensional input."""
        from utils.linearization import linearize
        n, m = 10, 5
        rng = np.random.default_rng(42)
        A_true = rng.standard_normal((n, n))
        B_true = rng.standard_normal((n, m))
        def f(x, u):
            return A_true @ x + B_true @ u
        x0 = bk.array(rng.standard_normal(n))
        u0 = bk.array(rng.standard_normal(m))
        A, B = linearize(f, x0, u0, bk)
        assert np.allclose(_to_np(A, bk), A_true, atol=1e-8)
        assert np.allclose(_to_np(B, bk), B_true, atol=1e-8)


class TestJacobianEdgeCases:
    """Edge cases for jacobian() — numpy-specific (finite differences)."""

    def setup_method(self):
        self.bk = NumpyBackend()

    def test_zero_input(self):
        """Jacobian at zero input for a smooth function."""
        def f(x):
            return np.array([np.sin(x[0]), np.cos(x[1])])
        x = self.bk.zeros(2)
        J = self.bk.jacobian(f, x)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]])
        assert np.allclose(J, expected, atol=1e-5)

    def test_large_input(self):
        """Jacobian at large input values."""
        def f(x):
            return np.array([x[0]**2, x[1]**3])
        x = self.bk.array([1e3, 1e2])
        J = self.bk.jacobian(f, x)
        expected = np.array([[2e3, 0], [0, 3e4]])
        assert np.allclose(J, expected, atol=1e-1)

    def test_negative_input(self):
        """Jacobian at negative input values."""
        def f(x):
            return np.array([x[0]**2, x[1]**3])
        x = self.bk.array([-2.0, -3.0])
        J = self.bk.jacobian(f, x)
        expected = np.array([[-4.0, 0], [0, 27.0]])
        assert np.allclose(J, expected, atol=1e-5)

    def test_single_element(self):
        """Jacobian of a 1D-to-1D function is a 1x1 matrix."""
        def f(x):
            return np.array([5.0 * x[0] + 3.0])
        J = self.bk.jacobian(f, self.bk.array([2.0]))
        assert np.allclose(J, [[5.0]])

    def test_constant_function(self):
        """Jacobian of a constant function is zero."""
        def f(x):
            return np.array([1.0, 2.0, 3.0])
        J = self.bk.jacobian(f, self.bk.array([1.0, 2.0]))
        assert np.allclose(J, np.zeros((3, 2)))

    def test_eps_very_small(self):
        """Very small eps can cause numerical issues for numpy backend."""
        def f(x):
            return np.array([x[0]**2])
        J = self.bk.jacobian(f, self.bk.array([1.0]), eps=1e-12)
        assert np.allclose(J, [[2.0]], atol=1e-3)
