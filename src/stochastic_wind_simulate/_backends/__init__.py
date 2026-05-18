"""Backend abstraction layer.

Each backend module provides the same operations
(cholesky, IFFT, random, array conversion) using its native library.
"""

from ._jax import JaxBackend
from ._numpy import NumpyBackend
from ._torch import TorchBackend

__all__ = ["JaxBackend", "NumpyBackend", "TorchBackend"]
