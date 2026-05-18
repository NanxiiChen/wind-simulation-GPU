"""PyTorch backend operations."""

from __future__ import annotations

import numpy as np
import torch
import torch.func as func


class TorchBackend:
    """PyTorch-based backend.

    Uses ``torch.func.vmap`` for frequency parallelism
    and supports GPU via explicit device placement.
    """

    float_dtype = torch.float32
    complex_dtype = torch.complex64
    xp = torch  # array module

    def __init__(self, seed: int = 0, device: str | None = None):
        self.seed = seed
        torch.manual_seed(seed)
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @property
    def device(self):
        return self._device

    # -- array helpers --------------------------------------------------------

    def asarray(self, x, dtype=None):
        if dtype is None:
            dtype = self.float_dtype
        return torch.as_tensor(x, dtype=dtype, device=self._device)

    def zeros(self, shape, dtype=None):
        if dtype is None:
            dtype = self.float_dtype
        return torch.zeros(shape, dtype=dtype, device=self._device)

    def eye(self, n, dtype=None):
        if dtype is None:
            dtype = self.float_dtype
        return torch.eye(n, dtype=dtype, device=self._device)

    def arange(self, *args, dtype=None, **kwargs):
        if dtype is None:
            dtype = self.float_dtype
        return torch.arange(*args, dtype=dtype, device=self._device, **kwargs)

    def to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    # -- linear algebra -------------------------------------------------------

    def cholesky(self, matrix):
        """Cholesky decomposition.

        ``torch.linalg.cholesky`` returns *upper* triangular by default,
        so we transpose to get the same lower-triangular result as
        scipy/jax.
        """
        return torch.linalg.cholesky(matrix, upper=False)

    def matmul(self, a, b):
        return torch.matmul(a, b)

    # -- FFT ------------------------------------------------------------------

    def ifft(self, x, axis=-1):
        return torch.fft.ifft(x, dim=axis)

    # -- random ---------------------------------------------------------------

    def random_uniform(self, shape, minval=0.0, maxval=1.0):
        result = torch.rand(shape, device=self._device, dtype=self.float_dtype)
        return result * (maxval - minval) + minval

    def random_phases(self, shape):
        """Generate random phases in [0, 2*pi)."""
        return self.random_uniform(shape, 0.0, 2.0 * torch.pi)

    # -- mutable array helpers -------------------------------------------------

    @staticmethod
    def slice_set(arr, idx, values):
        """Mutable slice assignment (in-place, returns arr for chaining)."""
        arr[idx] = values
        return arr

    # -- vmap ----------------------------------------------------------------

    def vmap(self, fn):
        """Vectorise over the first dimension via ``torch.func.vmap``."""
        return func.vmap(fn)

    @property
    def name(self):
        return "torch"
