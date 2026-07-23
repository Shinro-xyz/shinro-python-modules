from abc import ABC, abstractmethod
from typing import Any
import numpy as np


def parse_matrix(bk, value):
    """Convert a TOML config value to a matrix.

    Flat list (e.g. ``[1.0, 2.0, 3.0]``) → diagonal matrix.
    Nested list (e.g. ``[[1, 0], [0, 2]]``) → full matrix.

    Args:
        bk: ArrayBackend instance.
        value: List or list-of-lists from a TOML config.

    Returns:
        Array in the backend's native type.
    """
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
        return bk.array(value)
    return bk.diag(value)


class ArrayBackend(ABC):
    """Abstract interface for array operations used by all components.

    Every component takes an optional ``backend`` parameter. If None,
    NumpyBackend is used. This lets the entire control stack switch
    between numpy and torch by changing one object.

    The ``@`` operator (matmul) is intentionally not wrapped — it works
    identically in numpy and torch for 2D arrays.
    """

    @abstractmethod
    def array(self, data) -> Any:
        ...

    @abstractmethod
    def zeros(self, *shape) -> Any:
        ...

    @abstractmethod
    def zeros_like(self, x) -> Any:
        ...

    @abstractmethod
    def eye(self, n) -> Any:
        ...

    @abstractmethod
    def diag(self, x) -> Any:
        ...

    @abstractmethod
    def inv(self, x) -> Any:
        ...

    @abstractmethod
    def pinv(self, x) -> Any:
        ...

    @abstractmethod
    def solve(self, A, b) -> Any:
        ...

    @abstractmethod
    def norm(self, x, axis=None) -> Any:
        ...

    @abstractmethod
    def cross(self, a, b, axis=-1) -> Any:
        ...

    @abstractmethod
    def sin(self, x) -> Any:
        ...

    @abstractmethod
    def cos(self, x) -> Any:
        ...

    @abstractmethod
    def arccos(self, x) -> Any:
        ...

    @abstractmethod
    def trace(self, x) -> Any:
        ...

    @abstractmethod
    def clip(self, x, lo, hi) -> Any:
        ...

    @abstractmethod
    def where(self, cond, a, b) -> Any:
        ...

    @abstractmethod
    def any(self, x) -> Any:
        ...

    @abstractmethod
    def copy(self, x) -> Any:
        ...

    @abstractmethod
    def kron(self, a, b) -> Any:
        ...

    @abstractmethod
    def eigvals(self, x) -> Any:
        ...

    @abstractmethod
    def matrix_rank(self, x) -> Any:
        ...

    @abstractmethod
    def cond(self, x) -> Any:
        ...

    @abstractmethod
    def svd(self, x) -> Any:
        ...

    @abstractmethod
    def real(self, x) -> Any:
        ...

    @abstractmethod
    def sort(self, x) -> Any:
        ...

    @abstractmethod
    def sqrt(self, x) -> Any:
        ...

    @abstractmethod
    def abs(self, x) -> Any:
        ...

    @abstractmethod
    def sum(self, x, axis=None) -> Any:
        ...

    @abstractmethod
    def reshape(self, x, *shape) -> Any:
        ...

    @abstractmethod
    def ravel(self, x) -> Any:
        ...

    @abstractmethod
    def linspace(self, start, stop, num) -> Any:
        ...

    @abstractmethod
    def cholesky(self, x) -> Any:
        ...

    @abstractmethod
    def matrix_power(self, A, n) -> Any:
        ...

    @abstractmethod
    def vstack(self, arrays) -> Any:
        ...

    @abstractmethod
    def hstack(self, arrays) -> Any:
        ...

    @abstractmethod
    def block(self, blocks) -> Any:
        ...

    @abstractmethod
    def tile(self, x, reps) -> Any:
        ...

    @abstractmethod
    def to_numpy(self, x) -> Any:
        ...

    @abstractmethod
    def allclose(self, a, b) -> Any:
        ...

    @abstractmethod
    def from_numpy(self, x) -> Any:
        ...

    @abstractmethod
    def jacobian(self, f, x, eps=1e-6) -> Any:
        """Compute the Jacobian of f at x using finite differences or autograd.

        Args:
            f: Callable ``f(x) -> y`` where x is shape (n,) and y is shape (m,).
            x: Point at which to evaluate the Jacobian, shape (n,).
            eps: Step size for finite differences (ignored by autograd backends).

        Returns:
            Jacobian matrix of shape (m, n).
        """
        ...


class NumpyBackend(ArrayBackend):
    """ArrayBackend implementation using numpy.

    All operations delegate directly to ``np.xxx``. The ``to_numpy`` and
    ``from_numpy`` methods are no-ops since the data is already numpy.
    """

    def array(self, data):
        return np.array(data, dtype=np.float64)

    def zeros(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return np.zeros(shape, dtype=np.float64)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def eye(self, n):
        return np.eye(n, dtype=np.float64)

    def diag(self, x):
        return np.diag(x)

    def inv(self, x):
        return np.linalg.inv(x)

    def pinv(self, x):
        return np.linalg.pinv(x)

    def solve(self, A, b):
        return np.linalg.solve(A, b)

    def norm(self, x, axis=None):
        return np.linalg.norm(x, axis=axis)

    def cross(self, a, b, axis=-1):
        return np.cross(a, b, axis=axis)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def arccos(self, x):
        return np.arccos(x)

    def trace(self, x):
        return np.trace(x)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def where(self, cond, a, b):
        return np.where(cond, a, b)

    def any(self, x):
        return np.any(x)

    def copy(self, x):
        return x.copy()

    def kron(self, a, b):
        return np.kron(a, b)

    def eigvals(self, x):
        return np.linalg.eigvals(x)

    def matrix_rank(self, x):
        if x.ndim >= 2 and any(s == 0 for s in x.shape[-2:]):
            return np.asarray(0)
        return np.linalg.matrix_rank(x)

    def cond(self, x):
        return np.linalg.cond(x)

    def svd(self, x):
        return np.linalg.svd(x)

    def real(self, x):
        return np.real(x)

    def sort(self, x):
        return np.sort(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def abs(self, x):
        return np.abs(x)

    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def reshape(self, x, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return x.reshape(shape)

    def ravel(self, x):
        return x.ravel()

    def linspace(self, start, stop, num):
        return np.linspace(start, stop, num)

    def cholesky(self, x):
        return np.linalg.cholesky(x)

    def matrix_power(self, A, n):
        return np.linalg.matrix_power(A, n)

    def vstack(self, arrays):
        return np.vstack(arrays)

    def hstack(self, arrays):
        return np.hstack(arrays)

    def block(self, blocks):
        return np.block(blocks)

    def tile(self, x, reps):
        return np.tile(x, reps)

    def to_numpy(self, x):
        return x

    def from_numpy(self, x):
        return x

    def allclose(self, a, b):
        return np.allclose(a, b, atol=1e-8)

    def jacobian(self, f, x, eps=1e-6):
        """Central finite-difference Jacobian.

        Args:
            f: Callable ``f(x) -> y`` where x is shape (n,) and y is shape (m,).
            x: Point at which to evaluate, shape (n,).
            eps: Step size for finite differences.

        Returns:
            Jacobian matrix of shape (m, n).
        """
        x = np.asarray(x, dtype=np.float64)
        fx = np.atleast_1d(np.asarray(f(x), dtype=np.float64))
        m = fx.shape[0]
        n = x.shape[0]
        J = np.zeros((m, n), dtype=np.float64)
        for i in range(n):
            h = np.zeros(n, dtype=np.float64)
            h[i] = eps
            J[:, i] = (np.atleast_1d(np.asarray(f(x + h), dtype=np.float64))
                       - np.atleast_1d(np.asarray(f(x - h), dtype=np.float64))) / (2.0 * eps)
        return J


class TorchBackend(ArrayBackend):
    """ArrayBackend implementation using PyTorch.

    All operations delegate to ``torch.xxx``. Data lives on the device
    specified at construction time. The ``to_numpy`` and ``from_numpy``
    methods handle device transfers.

    Args:
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    """

    def __init__(self, device="cpu"):
        import torch

        self.torch = torch
        self.device = device

    def array(self, data):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], self.torch.Tensor):
            return self.torch.stack(data)
        return self.torch.tensor(data, device=self.device, dtype=self.torch.float64)

    def zeros(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self.torch.zeros(*shape, device=self.device, dtype=self.torch.float64)

    def zeros_like(self, x):
        return self.torch.zeros_like(x)

    def eye(self, n):
        return self.torch.eye(n, device=self.device, dtype=self.torch.float64)

    def diag(self, x):
        if isinstance(x, list):
            x = self.torch.tensor(x, dtype=self.torch.float64)
        return self.torch.diag(x)

    def inv(self, x):
        return self.torch.linalg.inv(x)

    def pinv(self, x):
        return self.torch.linalg.pinv(x)

    def solve(self, A, b):
        if isinstance(b, list):
            b = self.torch.stack(b)
        return self.torch.linalg.solve(A, b)

    def norm(self, x, axis=None):
        return self.torch.linalg.norm(x, dim=axis)

    def cross(self, a, b, axis=-1):
        return self.torch.cross(a, b, dim=axis)

    def sin(self, x):
        return self.torch.sin(x)

    def cos(self, x):
        return self.torch.cos(x)

    def arccos(self, x):
        return self.torch.arccos(x)

    def trace(self, x):
        return self.torch.trace(x)

    def clip(self, x, lo, hi):
        if not isinstance(x, self.torch.Tensor):
            x = self.torch.tensor(x, dtype=self.torch.float64)
        return self.torch.clamp(x, lo, hi)

    def where(self, cond, a, b):
        return self.torch.where(cond, a, b)

    def any(self, x):
        return self.torch.any(x)

    def copy(self, x):
        return x.clone()

    def kron(self, a, b):
        return self.torch.kron(a, b)

    def eigvals(self, x):
        return self.torch.linalg.eigvals(x)

    def matrix_rank(self, x):
        if x.ndim >= 2 and any(s == 0 for s in x.shape[-2:]):
            return self.torch.tensor(0, device=x.device, dtype=x.dtype)
        return self.torch.linalg.matrix_rank(x)

    def cond(self, x):
        return self.torch.linalg.cond(x)

    def svd(self, x):
        return self.torch.linalg.svd(x)

    def real(self, x):
        return self.torch.real(x)

    def sort(self, x):
        return self.torch.sort(x).values

    def sqrt(self, x):
        return self.torch.sqrt(x)

    def abs(self, x):
        return self.torch.abs(x)

    def sum(self, x, axis=None):
        return self.torch.sum(x, dim=axis)

    def reshape(self, x, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return x.reshape(shape)

    def ravel(self, x):
        return x.ravel()

    def linspace(self, start, stop, num):
        return self.torch.linspace(start, stop, num, device=self.device, dtype=self.torch.float64)

    def cholesky(self, x):
        return self.torch.linalg.cholesky(x)

    def matrix_power(self, A, n):
        return self.torch.linalg.matrix_power(A, n)

    def vstack(self, arrays):
        return self.torch.vstack(arrays)

    def hstack(self, arrays):
        return self.torch.hstack(arrays)

    def block(self, blocks):
        rows = [self.torch.cat(row, dim=1) for row in blocks]
        return self.torch.cat(rows, dim=0)

    def tile(self, x, reps):
        if isinstance(reps, int):
            reps = (reps,)
        return self.torch.tile(x, reps)

    def to_numpy(self, x):
        return x.cpu().numpy()

    def from_numpy(self, x):
        if isinstance(x, list):
            x = np.array(x, dtype=np.float64)
        return self.torch.from_numpy(x).to(self.device)

    def allclose(self, a, b):
        return bool(self.torch.allclose(a, b, atol=1e-8))

    def jacobian(self, f, x, eps=1e-6):
        """Jacobian via torch.autograd.functional.jacobian.

        Args:
            f: Callable ``f(x) -> y`` where x is shape (n,) and y is shape (m,).
            x: Point at which to evaluate, shape (n,).
            eps: Ignored (used for API compatibility with NumpyBackend).

        Returns:
            Jacobian matrix of shape (m, n).
        """
        if not isinstance(x, self.torch.Tensor):
            x = self.torch.tensor(x, dtype=self.torch.float64)
        x_t = x.detach().clone().requires_grad_(True)
        y = f(x_t)
        if y.ndim == 0:
            y = y.unsqueeze(0)
        J = self.torch.autograd.functional.jacobian(f, x_t)
        return J.reshape(y.shape[0], x_t.shape[0])
