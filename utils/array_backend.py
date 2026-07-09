from abc import ABC, abstractmethod
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
    def array(self, data):
        """Create an array from a Python list or nested lists.

        Args:
            data: Python list, tuple, or nested structure of scalars.

        Returns:
            A new array of the backend's native type.
        """

    @abstractmethod
    def zeros(self, *shape):
        """Create an array filled with zeros.

        Args:
            *shape: Either ``zeros(3, 4)`` or ``zeros((3, 4))``.

        Returns:
            Zero-filled array of the given shape.
        """

    @abstractmethod
    def zeros_like(self, x):
        """Create a zero-filled array with the same shape and type as x.

        Args:
            x: Reference array.

        Returns:
            Zero-filled array matching x's shape.
        """

    @abstractmethod
    def eye(self, n):
        """Create an n x n identity matrix.

        Args:
            n: Number of rows and columns.

        Returns:
            Identity matrix of shape (n, n).
        """

    @abstractmethod
    def diag(self, x):
        """Create a diagonal matrix from a 1D array, or extract the diagonal.

        Args:
            x: 1D array (creates diagonal matrix) or 2D array (extracts diagonal).

        Returns:
            Diagonal matrix or diagonal vector.
        """

    @abstractmethod
    def inv(self, x):
        """Compute the matrix inverse.

        Args:
            x: Square matrix of shape (n, n).

        Returns:
            Inverse matrix of shape (n, n).
        """

    @abstractmethod
    def pinv(self, x):
        """Compute the Moore-Penrose pseudoinverse.

        Args:
            x: Matrix of shape (m, n).

        Returns:
            Pseudoinverse of shape (n, m).
        """

    @abstractmethod
    def solve(self, A, b):
        """Solve the linear system Ax = b.

        Args:
            A: Coefficient matrix of shape (n, n).
            b: Right-hand side of shape (n,) or (n, k).

        Returns:
            Solution x of shape (n,) or (n, k).
        """

    @abstractmethod
    def norm(self, x):
        """Compute the 2-norm of a vector.

        Args:
            x: Vector of shape (n,).

        Returns:
            Scalar norm.
        """

    @abstractmethod
    def cross(self, a, b):
        """Compute the cross product of two 3D vectors.

        Args:
            a: First vector of shape (3,).
            b: Second vector of shape (3,).

        Returns:
            Cross product vector of shape (3,).
        """

    @abstractmethod
    def sin(self, x):
        """Compute the element-wise sine.

        Args:
            x: Input array.

        Returns:
            Array of sin(x).
        """

    @abstractmethod
    def cos(self, x):
        """Compute the element-wise cosine.

        Args:
            x: Input array.

        Returns:
            Array of cos(x).
        """

    @abstractmethod
    def arccos(self, x):
        """Compute the element-wise arccosine.

        Args:
            x: Input array with values in [-1, 1].

        Returns:
            Array of arccos(x) in [0, pi].
        """

    @abstractmethod
    def trace(self, x):
        """Compute the trace of a matrix.

        Args:
            x: Square matrix of shape (n, n).

        Returns:
            Scalar trace (sum of diagonal elements).
        """

    @abstractmethod
    def clip(self, x, lo, hi):
        """Clip values to a range.

        Args:
            x: Input array.
            lo: Lower bound (scalar or array).
            hi: Upper bound (scalar or array).

        Returns:
            Clipped array.
        """

    @abstractmethod
    def where(self, cond, a, b):
        """Select elements from a or b based on a condition.

        Args:
            cond: Boolean array.
            a: Values where cond is True.
            b: Values where cond is False.

        Returns:
            Array with elements chosen from a or b.
        """

    @abstractmethod
    def any(self, x):
        """Check if any element is True.

        Args:
            x: Boolean array.

        Returns:
            True if any element is non-zero / True.
        """

    @abstractmethod
    def copy(self, x):
        """Create a deep copy of an array.

        Args:
            x: Input array.

        Returns:
            New array with the same data.
        """

    @abstractmethod
    def kron(self, a, b):
        """Compute the Kronecker product of two matrices.

        Args:
            a: Matrix of shape (m, n).
            b: Matrix of shape (p, q).

        Returns:
            Kronecker product of shape (m*p, n*q).
        """

    @abstractmethod
    def matrix_power(self, A, n):
        """Raise a square matrix to an integer power.

        Args:
            A: Square matrix of shape (m, m).
            n: Non-negative integer exponent.

        Returns:
            Matrix A^n of shape (m, m).
        """

    @abstractmethod
    def vstack(self, arrays):
        """Stack arrays vertically (row-wise).

        Args:
            arrays: Sequence of arrays with the same number of columns.

        Returns:
            Vertically stacked array.
        """

    @abstractmethod
    def hstack(self, arrays):
        """Stack arrays horizontally (column-wise).

        Args:
            arrays: Sequence of arrays with the same number of rows.

        Returns:
            Horizontally stacked array.
        """

    @abstractmethod
    def block(self, blocks):
        """Assemble a block matrix from nested lists of blocks.

        Args:
            blocks: Nested list of arrays forming the blocks.

        Returns:
            Block matrix.
        """

    @abstractmethod
    def tile(self, x, reps):
        """Tile an array by repeating along each axis.

        Args:
            x: Input array.
            reps: Tuple of repetitions along each axis.

        Returns:
            Tiled array.
        """

    @abstractmethod
    def to_numpy(self, x):
        """Convert an array to a numpy array (no-op if already numpy).

        Args:
            x: Array in the backend's native type.

        Returns:
            Numpy array with the same data.
        """

    @abstractmethod
    def from_numpy(self, x):
        """Convert a numpy array to the backend's native type.

        Args:
            x: Numpy array.

        Returns:
            Array in the backend's native type with the same data.
        """


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
