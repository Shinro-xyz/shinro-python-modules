import numpy as np
import pytest

from shinro.utils.array_backend import NumpyBackend


@pytest.fixture
def numpy_backend():
    return NumpyBackend()


@pytest.fixture
def torch_backend():
    pytest.importorskip("torch")
    from shinro.utils.array_backend import TorchBackend
    return TorchBackend(device="cpu")


@pytest.fixture(params=["numpy", "torch"])
def bk(request):
    if request.param == "numpy":
        return NumpyBackend()
    pytest.importorskip("torch")
    from shinro.utils.array_backend import TorchBackend
    return TorchBackend(device="cpu")


@pytest.fixture
def rng():
    return np.random.default_rng(42)
