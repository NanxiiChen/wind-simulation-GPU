"""JAX backend operations."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit, random, vmap
from jax.scipy import linalg


class JaxBackend:
    """JAX-based backend.

    Uses ``vmap`` + ``jit`` for frequency parallelism.
    JAX arrays are immutable, so accumulation uses ``.at[idx].set(val)``.
    """

    float_dtype = jnp.float32
    complex_dtype = jnp.complex64
    xp = jnp  # array module

    def __init__(self, seed: int = 0):
        self._key = random.PRNGKey(seed)

    @property
    def key(self):
        return self._key

    def split_key(self):
        self._key, subkey = random.split(self._key)
        return subkey

    # -- array helpers --------------------------------------------------------

    def asarray(self, x, dtype=None):
        if dtype is None:
            dtype = self.float_dtype
        return jnp.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self.float_dtype
        return jnp.zeros(shape, dtype=dtype)

    def eye(self, n, dtype=None):
        if dtype is None:
            dtype = self.float_dtype
        return jnp.eye(n, dtype=dtype)

    def arange(self, *args, dtype=None, **kwargs):
        if dtype is None:
            dtype = self.float_dtype
        return jnp.arange(*args, dtype=dtype, **kwargs)

    def to_numpy(self, x):
        return np.asarray(x)

    # -- linear algebra -------------------------------------------------------

    def cholesky(self, matrix):
        return linalg.cholesky(matrix, lower=True)

    def matmul(self, a, b):
        return jnp.matmul(a, b)

    # -- FFT ------------------------------------------------------------------

    def ifft(self, x, axis=-1):
        return jnp.fft.ifft(x, axis=axis)

    # -- random ---------------------------------------------------------------

    def random_uniform(self, shape, minval=0.0, maxval=1.0):
        key = self.split_key()
        return random.uniform(key, shape, dtype=self.float_dtype,
                              minval=minval, maxval=maxval)

    def random_phases(self, shape):
        """Generate random phases in [0, 2*pi)."""
        return self.random_uniform(shape, 0.0, 2.0 * jnp.pi)

    # -- immutable array helpers -----------------------------------------------

    @staticmethod
    def slice_set(arr, idx, values):
        """Functional slice assignment for immutable JAX arrays."""
        return arr.at[idx].set(values)

    # -- vmap / jit -----------------------------------------------------------

    @staticmethod
    def vmap(fn, in_axes=0):
        return vmap(fn, in_axes=in_axes)

    @staticmethod
    def jit(fn, static_argnums=()):
        return jit(fn, static_argnums=static_argnums)

    @property
    def name(self):
        return "jax"
