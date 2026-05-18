"""NumPy backend operations."""

from __future__ import annotations

import numpy as np
from scipy import linalg


class NumpyBackend:
    """NumPy-based backend.

    Uses list comprehensions for frequency parallelism
    (no vmap equivalent).  Optional numba acceleration
    is used in the nonstationary path when available.
    """

    float_dtype = np.float64
    complex_dtype = np.complex128
    xp = np  # array module

    def __init__(self, seed: int = 0):
        self.seed = seed
        np.random.seed(seed)

    # -- array helpers --------------------------------------------------------

    def asarray(self, x, dtype=None):
        if dtype is None:
            dtype = self.float_dtype
        return np.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self.float_dtype
        return np.zeros(shape, dtype=dtype)

    def eye(self, n, dtype=None):
        if dtype is None:
            dtype = self.float_dtype
        return np.eye(n, dtype=dtype)

    def arange(self, *args, dtype=None, **kwargs):
        if dtype is None and len(args) < 2:
            dtype = self.float_dtype
        return np.arange(*args, dtype=dtype, **kwargs)

    def to_numpy(self, x):
        return np.asarray(x)

    # -- linear algebra -------------------------------------------------------

    def cholesky(self, matrix):
        return linalg.cholesky(matrix, lower=True)

    def matmul(self, a, b):
        return np.matmul(a, b)

    # -- FFT ------------------------------------------------------------------

    def ifft(self, x, axis=-1):
        return np.fft.ifft(x, axis=axis)

    # -- random ---------------------------------------------------------------

    def random_uniform(self, shape, minval=0.0, maxval=1.0):
        self.seed += 1
        np.random.seed(self.seed)
        return np.random.uniform(minval, maxval, shape)

    def random_phases(self, shape):
        """Generate random phases in [0, 2*pi)."""
        return self.random_uniform(shape, 0.0, 2.0 * np.pi)

    # -- mutable array helpers -------------------------------------------------

    @staticmethod
    def slice_set(arr, idx, values):
        """Mutable slice assignment (in-place, returns arr for chaining)."""
        arr[idx] = values
        return arr

    # -- vmap (list-comprehension fallback) ------------------------------------

    @staticmethod
    def vmap(fn):
        """Vectorise over the first dimension via list comprehension."""
        def _vmapped(*args):
            n = len(args[0])
            return np.array([fn(*[a[i] for a in args]) for i in range(n)])
        return _vmapped

    @property
    def name(self):
        return "numpy"
