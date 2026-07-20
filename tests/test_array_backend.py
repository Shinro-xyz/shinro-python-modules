import pytest
import numpy as np
from utils.array_backend import NumpyBackend


class TestNumpyBackend:
    """Verify every method of NumpyBackend produces correct outputs."""

    def setup_method(self):
        self.bk = NumpyBackend()

    def test_array(self):
        """Create an array from a Python list."""
        a = self.bk.array([1, 2, 3])
        assert isinstance(a, np.ndarray)
        assert a.dtype == np.float64
        assert np.allclose(a, [1, 2, 3])

    def test_zeros(self):
        """Create a zero-filled array with given shape."""
        z = self.bk.zeros(2, 3)
        assert z.shape == (2, 3)
        assert np.all(z == 0)

    def test_zeros_like(self):
        """Create a zero-filled array matching a reference shape."""
        x = np.array([[1, 2], [3, 4]], dtype=float)
        z = self.bk.zeros_like(x)
        assert z.shape == (2, 2)
        assert np.all(z == 0)

    def test_eye(self):
        """Create an identity matrix."""
        I = self.bk.eye(3)
        assert I.shape == (3, 3)
        assert np.allclose(I, np.eye(3))

    def test_diag(self):
        """Create a diagonal matrix from a 1D array."""
        d = self.bk.diag([1, 2, 3])
        assert d.shape == (3, 3)
        assert np.allclose(d, np.diag([1, 2, 3]))

    def test_inv(self):
        """Compute the matrix inverse: A @ A^{-1} = I."""
        A = np.array([[4, 7], [2, 6]], dtype=float)
        Ainv = self.bk.inv(A)
        assert np.allclose(A @ Ainv, np.eye(2))

    def test_pinv(self):
        """Compute the Moore-Penrose pseudoinverse: A pinv(A) A = A."""
        A = np.array([[1, 2], [2, 4]], dtype=float)
        Apinv = self.bk.pinv(A)
        assert np.allclose(A @ Apinv @ A, A)

    def test_solve(self):
        """Solve a linear system: A @ x = b."""
        A = np.array([[3, 1], [1, 2]], dtype=float)
        b = np.array([9, 8], dtype=float)
        x = self.bk.solve(A, b)
        assert np.allclose(A @ x, b)

    def test_norm(self):
        """Compute the 2-norm of a vector."""
        v = np.array([3, 4], dtype=float)
        assert np.allclose(self.bk.norm(v), 5.0)

    def test_cross(self):
        """Compute the cross product of two 3D vectors."""
        a = np.array([1, 0, 0], dtype=float)
        b = np.array([0, 1, 0], dtype=float)
        c = self.bk.cross(a, b)
        assert np.allclose(c, [0, 0, 1])

    def test_sin_cos_arccos(self):
        """Verify sin, cos, and arccos are consistent."""
        theta = np.pi / 4
        s = self.bk.sin(theta)
        c = self.bk.cos(theta)
        assert np.allclose(s, np.sqrt(2) / 2)
        assert np.allclose(c, np.sqrt(2) / 2)
        assert np.allclose(self.bk.arccos(c), theta)

    def test_trace(self):
        """Compute the trace of a matrix."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        assert self.bk.trace(A) == 5.0

    def test_clip(self):
        """Clip values to a specified range."""
        x = np.array([-1, 0.5, 2], dtype=float)
        clipped = self.bk.clip(x, 0.0, 1.0)
        assert np.allclose(clipped, [0, 0.5, 1])

    def test_where(self):
        """Select elements from a or b based on a condition."""
        cond = np.array([True, False, True])
        a = np.array([1, 2, 3], dtype=float)
        b = np.array([10, 20, 30], dtype=float)
        result = self.bk.where(cond, a, b)
        assert np.allclose(result, [1, 20, 3])

    def test_any(self):
        """Check if any element is True."""
        assert self.bk.any(np.array([False, True]))
        assert not self.bk.any(np.array([False, False]))

    def test_copy(self):
        """Create a deep copy that is independent of the original."""
        x = np.array([1, 2, 3], dtype=float)
        y = self.bk.copy(x)
        y[0] = 99
        assert x[0] == 1

    def test_kron(self):
        """Compute the Kronecker product of two matrices."""
        a = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([[0, 5], [6, 7]], dtype=float)
        k = self.bk.kron(a, b)
        assert k.shape == (4, 4)

    def test_eigvals(self):
        """Compute eigenvalues of a diagonal matrix."""
        A = np.array([[1, 0], [0, 2]], dtype=float)
        eigs = self.bk.eigvals(A)
        assert np.allclose(np.sort(eigs), [1, 2])

    def test_matrix_rank(self):
        """Compute the rank of singular and full-rank matrices."""
        A = np.array([[1, 2], [2, 4]], dtype=float)
        assert self.bk.matrix_rank(A) == 1
        B = np.eye(3)
        assert self.bk.matrix_rank(B) == 3

    def test_cond(self):
        """Compute the condition number of an identity matrix."""
        A = np.eye(3)
        assert np.allclose(self.bk.cond(A), 1.0)

    def test_svd(self):
        """Compute the SVD: U @ diag(s) @ Vh = A."""
        A = np.array([[1, 0], [0, 2], [0, 0]], dtype=float)
        U, s, Vh = self.bk.svd(A)
        assert U.shape[0] == 3
        assert len(s) == 2
        assert Vh.shape[1] == 2
        assert np.allclose(U[:, :2] @ np.diag(s) @ Vh, A)

    def test_real(self):
        """Extract the real part of a complex array."""
        x = np.array([1 + 2j, 3 + 4j])
        assert np.allclose(self.bk.real(x), [1, 3])

    def test_sort(self):
        """Sort an array in ascending order."""
        x = np.array([3, 1, 2], dtype=float)
        assert np.allclose(self.bk.sort(x), [1, 2, 3])

    def test_sqrt(self):
        """Compute the element-wise square root."""
        x = np.array([4, 9, 16], dtype=float)
        assert np.allclose(self.bk.sqrt(x), [2, 3, 4])

    def test_abs(self):
        """Compute the element-wise absolute value."""
        x = np.array([-1, -2, 3], dtype=float)
        assert np.allclose(self.bk.abs(x), [1, 2, 3])

    def test_sum(self):
        """Compute the sum of all elements."""
        x = np.array([1, 2, 3, 4], dtype=float)
        assert self.bk.sum(x) == 10.0

    def test_reshape(self):
        """Reshape an array without changing its data."""
        x = np.array([1, 2, 3, 4], dtype=float)
        y = self.bk.reshape(x, 2, 2)
        assert y.shape == (2, 2)

    def test_ravel(self):
        """Flatten a 2D array to 1D."""
        x = np.array([[1, 2], [3, 4]], dtype=float)
        y = self.bk.ravel(x)
        assert y.shape == (4,)

    def test_linspace(self):
        """Generate evenly spaced numbers over an interval."""
        x = self.bk.linspace(0, 1, 5)
        assert len(x) == 5
        assert np.allclose(x[0], 0)
        assert np.allclose(x[-1], 1)

    def test_matrix_power(self):
        """Raise a square matrix to an integer power."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        A2 = self.bk.matrix_power(A, 2)
        assert np.allclose(A2, A @ A)

    def test_cholesky(self):
        """Compute the Cholesky decomposition: L @ L^T = A."""
        A = np.array([[4, 2], [2, 3]], dtype=float)
        L = self.bk.cholesky(A)
        assert np.allclose(L @ L.T, A)

    def test_vstack(self):
        """Stack arrays vertically."""
        a = np.array([1, 2], dtype=float)
        b = np.array([3, 4], dtype=float)
        s = self.bk.vstack([a, b])
        assert s.shape == (2, 2)

    def test_hstack(self):
        """Stack arrays horizontally."""
        a = np.array([[1], [2]], dtype=float)
        b = np.array([[3], [4]], dtype=float)
        s = self.bk.hstack([a, b])
        assert s.shape == (2, 2)

    def test_block(self):
        """Assemble a block matrix from nested blocks."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5], [6]], dtype=float)
        C = np.array([[7, 8]], dtype=float)
        D = np.array([[9]], dtype=float)
        M = self.bk.block([[A, B], [C, D]])
        assert M.shape == (3, 3)

    def test_tile(self):
        """Tile an array by repeating along each axis."""
        x = np.array([1, 2], dtype=float)
        t = self.bk.tile(x, 3)
        assert np.allclose(t, [1, 2, 1, 2, 1, 2])

    def test_to_numpy(self):
        """Convert to numpy (no-op for NumpyBackend)."""
        x = np.array([1, 2, 3], dtype=float)
        assert self.bk.to_numpy(x) is x

    def test_from_numpy(self):
        """Convert from numpy (no-op for NumpyBackend)."""
        x = np.array([1, 2, 3], dtype=float)
        assert self.bk.from_numpy(x) is x


class TestTorchBackend:
    """Verify key TorchBackend methods produce correct outputs.

    Skipped if torch is not installed.
    """

    def setup_method(self):
        torch = pytest.importorskip("torch")
        from utils.array_backend import TorchBackend
        self.bk = TorchBackend(device="cpu")
        self.torch = torch

    def test_array(self):
        """Create a torch tensor from a Python list."""
        a = self.bk.array([1, 2, 3])
        assert isinstance(a, self.torch.Tensor)
        assert a.dtype == self.torch.float64

    def test_zeros(self):
        """Create a zero-filled torch tensor."""
        z = self.bk.zeros(2, 3)
        assert z.shape == (2, 3)
        assert self.bk.to_numpy(self.bk.sum(z)) == 0

    def test_eye(self):
        """Create an identity torch tensor."""
        I = self.bk.eye(3)
        assert I.shape == (3, 3)
        assert self.bk.allclose(I, self.torch.eye(3, dtype=self.torch.float64))

    def test_inv(self):
        """Compute the matrix inverse: A @ A^{-1} = I."""
        A = self.bk.array([[4, 7], [2, 6]])
        Ainv = self.bk.inv(A)
        assert self.bk.allclose(A @ Ainv, self.torch.eye(2, dtype=self.torch.float64))

    def test_svd(self):
        """Compute the SVD of a 3x2 matrix."""
        A = self.bk.array([[1, 0], [0, 2], [0, 0]])
        U, s, Vh = self.bk.svd(A)
        assert U.shape[0] == 3
        assert len(s) == 2
        assert Vh.shape[1] == 2

    def test_cholesky(self):
        """Compute the Cholesky decomposition: L @ L^T = A."""
        A = self.bk.array([[4, 2], [2, 3]])
        L = self.bk.cholesky(A)
        assert self.bk.allclose(L @ L.T, A)

    def test_to_numpy_roundtrip(self):
        """Verify to_numpy and from_numpy are inverses."""
        x = np.array([1, 2, 3], dtype=float)
        y = self.bk.from_numpy(x)
        z = self.bk.to_numpy(y)
        assert np.allclose(x, z)

    def allclose(self, a, b):
        return self.torch.allclose(a, b, atol=1e-8)

    def allclose(self, a, b):
        return self.torch.allclose(a, b, atol=1e-8)


class TestTorchBackendBatched:
    """Verify TorchBackend linalg methods work with batched (3D+) inputs.

    PyTorch's ``torch.linalg.*`` natively supports batching on the last 2
    dimensions. These tests verify that the backend wrapper passes through
    batched tensors correctly.
    """

    def setup_method(self):
        torch = pytest.importorskip("torch")
        from utils.array_backend import TorchBackend
        self.bk = TorchBackend(device="cpu")
        self.torch = torch

    def test_inv_batched(self):
        """Batched matrix inverse: (batch, 2, 2) -> (batch, 2, 2)."""
        A = self.bk.array([[[4, 7], [2, 6]], [[1, 0], [0, 1]]])
        Ainv = self.bk.inv(A)
        assert Ainv.shape == (2, 2, 2)
        I = self.torch.eye(2, dtype=self.torch.float64).unsqueeze(0)
        assert self.bk.allclose(A[0] @ Ainv[0], I[0])
        assert self.bk.allclose(A[1] @ Ainv[1], I[0])

    def test_solve_batched(self):
        """Batched linear solve: (batch, 2, 2) @ x = (batch, 2)."""
        A = self.bk.array([[[3, 1], [1, 2]], [[1, 0], [0, 1]]])
        b = self.bk.array([[9, 8], [5, 3]])
        x = self.bk.solve(A, b)
        assert x.shape == (2, 2)
        assert self.bk.allclose(A[0] @ x[0], b[0])
        assert self.bk.allclose(A[1] @ x[1], b[1])

    def test_cholesky_batched(self):
        """Batched Cholesky: (batch, 2, 2) -> (batch, 2, 2)."""
        A = self.bk.array([[[4, 2], [2, 3]], [[5, 1], [1, 4]]])
        L = self.bk.cholesky(A)
        assert L.shape == (2, 2, 2)
        assert self.bk.allclose(L[0] @ L[0].T, A[0])
        assert self.bk.allclose(L[1] @ L[1].T, A[1])

    def test_svd_batched(self):
        """Batched SVD: (batch, 2, 2) -> U(batch, 2, 2), s(batch, 2), Vh(batch, 2, 2)."""
        A = self.bk.array([[[1, 0], [0, 2]], [[2, 0], [0, 1]]])
        U, s, Vh = self.bk.svd(A)
        assert U.shape == (2, 2, 2)
        assert s.shape == (2, 2)
        assert Vh.shape == (2, 2, 2)

    def test_eigvals_batched(self):
        """Batched eigenvalues: (batch, 2, 2) -> (batch, 2)."""
        A = self.bk.array([[[1, 0], [0, 2]], [[3, 0], [0, 4]]])
        eigs = self.bk.eigvals(A)
        assert eigs.shape == (2, 2)

    def test_matrix_rank_batched(self):
        """Batched matrix rank: (batch, 2, 2) -> (batch,)."""
        A = self.bk.array([[[1, 2], [2, 4]], [[1, 0], [0, 1]]])
        ranks = self.bk.matrix_rank(A)
        assert ranks.shape == (2,)
        assert self.bk.to_numpy(ranks[0]) == 1
        assert self.bk.to_numpy(ranks[1]) == 2

    def test_cond_batched(self):
        """Batched condition number: (batch, 2, 2) -> (batch,)."""
        A = self.bk.array([[[1, 0], [0, 1]], [[2, 0], [0, 1]]])
        conds = self.bk.cond(A)
        assert conds.shape == (2,)
        assert self.bk.allclose(conds[0], self.torch.tensor(1.0, dtype=self.torch.float64))

    def test_norm_with_axis(self):
        """norm with axis computes per-row norms on batched input."""
        x = self.bk.array([[3, 4], [1, 0]])
        n = self.bk.norm(x, axis=1)
        assert n.shape == (2,)
        assert self.bk.allclose(n[0], self.torch.tensor(5.0, dtype=self.torch.float64))
        assert self.bk.allclose(n[1], self.torch.tensor(1.0, dtype=self.torch.float64))

    def test_sum_with_axis(self):
        """sum with axis reduces along the specified dimension."""
        x = self.bk.array([[1, 2, 3], [4, 5, 6]])
        s = self.bk.sum(x, axis=1)
        assert s.shape == (2,)
        assert self.bk.allclose(s[0], self.torch.tensor(6.0, dtype=self.torch.float64))
        assert self.bk.allclose(s[1], self.torch.tensor(15.0, dtype=self.torch.float64))

    def test_cross_with_axis(self):
        """cross with axis computes per-row cross products on batched input."""
        a = self.bk.array([[1, 0, 0], [0, 1, 0]])
        b = self.bk.array([[0, 1, 0], [1, 0, 0]])
        c = self.bk.cross(a, b, axis=1)
        assert c.shape == (2, 3)
        assert self.bk.allclose(c[0], self.torch.tensor([0, 0, 1], dtype=self.torch.float64))
        assert self.bk.allclose(c[1], self.torch.tensor([0, 0, -1], dtype=self.torch.float64))
