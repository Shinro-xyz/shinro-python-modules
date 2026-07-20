from abc import ABC, abstractmethod
from typing import Any
import numpy as np


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
    def norm(self, x) -> Any:
        ...

    @abstractmethod
    def cross(self, a, b) -> Any:
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
    def sum(self, x) -> Any:
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
    def from_numpy(self, x) -> Any:
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

    def norm(self, x):
        return np.linalg.norm(x)

    def cross(self, a, b):
        return np.cross(a, b)

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

    def sum(self, x):
        return np.sum(x)

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
        return self.torch.diag(x)

    def inv(self, x):
        return self.torch.linalg.inv(x)

    def pinv(self, x):
        return self.torch.linalg.pinv(x)

    def solve(self, A, b):
        return self.torch.linalg.solve(A, b)

    def norm(self, x):
        return self.torch.linalg.norm(x)

    def cross(self, a, b):
        return self.torch.cross(a, b)

    def sin(self, x):
        return self.torch.sin(x)

    def cos(self, x):
        return self.torch.cos(x)

    def arccos(self, x):
        return self.torch.arccos(x)

    def trace(self, x):
        return self.torch.trace(x)

    def clip(self, x, lo, hi):
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

    def sum(self, x):
        return self.torch.sum(x)

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
        return self.torch.block(blocks)

    def tile(self, x, reps):
        return self.torch.tile(x, reps)

    def to_numpy(self, x):
        return x.cpu().numpy()

    def from_numpy(self, x):
        return self.torch.from_numpy(x).to(self.device)
