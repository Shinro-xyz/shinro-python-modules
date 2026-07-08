from abc import ABC, abstractmethod
import numpy as np


class ArrayBackend(ABC):
    """Abstract interface for array operations used by all components.

    Every component takes an optional `backend` parameter. If None,
    NumpyBackend is used. This lets the entire control stack switch
    between numpy and torch by changing one object.
    """

    @abstractmethod
    def array(self, data): ...

    @abstractmethod
    def zeros(self, *shape): ...

    @abstractmethod
    def zeros_like(self, x): ...

    @abstractmethod
    def eye(self, n): ...

    @abstractmethod
    def diag(self, x): ...

    @abstractmethod
    def inv(self, x): ...

    @abstractmethod
    def pinv(self, x): ...

    @abstractmethod
    def solve(self, A, b): ...

    @abstractmethod
    def norm(self, x): ...

    @abstractmethod
    def cross(self, a, b): ...

    @abstractmethod
    def sin(self, x): ...

    @abstractmethod
    def cos(self, x): ...

    @abstractmethod
    def arccos(self, x): ...

    @abstractmethod
    def trace(self, x): ...

    @abstractmethod
    def clip(self, x, lo, hi): ...

    @abstractmethod
    def where(self, cond, a, b): ...

    @abstractmethod
    def any(self, x): ...

    @abstractmethod
    def copy(self, x): ...

    @abstractmethod
    def kron(self, a, b): ...

    @abstractmethod
    def matrix_power(self, A, n): ...

    @abstractmethod
    def vstack(self, arrays): ...

    @abstractmethod
    def hstack(self, arrays): ...

    @abstractmethod
    def block(self, blocks): ...

    @abstractmethod
    def tile(self, x, reps): ...

    @abstractmethod
    def to_numpy(self, x): ...

    @abstractmethod
    def from_numpy(self, x): ...


class NumpyBackend(ArrayBackend):
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
