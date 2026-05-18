"""Wind field simulators.

The core algorithm (spectrum → coherence → CSD → Cholesky → IFFT)
is defined **once** in ``_BaseSimulator``.  Each backend only provides
the handful of array primitives that differ:

- ``_xp``         — array module (``numpy`` / ``jax.numpy`` / ``torch``)
- ``_jit``        — JIT decorator (real for JAX, identity for others)
- ``_vmap``       — vectorising map (``vmap`` / ``func.vmap`` / list-comp)
- ``_cholesky``   — Cholesky decomposition
- ``_fft``        — inverse FFT

Everything else — batching, memory estimation, IFFT pipeline — is shared.
"""

from __future__ import annotations

from typing import Optional

from .coherence import davenport_coherence, cross_spectrum, simulation_frequencies
from .params import SimulationParams
from .spectrum import get_spectrum


# ═══════════════════════════════════════════════════════════════════════
# Shared base — algorithm, batching, memory
# ═══════════════════════════════════════════════════════════════════════

class _BaseSimulator:
    """Shared simulation logic.

    Subclasses set up backend-specific primitives in ``__init__``.
    """

    def __init__(self, spectrum_type, params):
        self.params = SimulationParams(**params) if params else SimulationParams()
        if isinstance(spectrum_type, str):
            spectrum_cls = get_spectrum(spectrum_type)
        elif isinstance(spectrum_type, type):
            spectrum_cls = spectrum_type
        else:
            raise TypeError(
                f"spectrum_type must be a string or class, got {type(spectrum_type)}"
            )
        self.spectrum = spectrum_cls(self._xp, self.params.to_dict())

    @property
    def backend_name(self):
        return self._backend_name

    # -- public API ------------------------------------------------------

    def simulate_wind(
        self, positions, wind_speeds, component: str = "u",
        max_memory_gb: float = 4.0, freq_batch_size: Optional[int] = None,
        auto_batch: bool = True, **kwargs,
    ):
        positions = self._asarray(positions)
        wind_speeds = self._asarray(wind_speeds)
        n, N = positions.shape[0], self.params.N
        est_mem = self.estimate_memory(n, N)

        use_batching = bool(auto_batch and est_mem > max_memory_gb)
        if freq_batch_size is not None:
            use_batching = True

        if use_batching:
            if freq_batch_size is None:
                _, freq_batch_size = self._optimal_batch_sizes(n, N, max_memory_gb)
            return self._simulate_batched(positions, wind_speeds, component, freq_batch_size)
        return self._simulate_direct(positions, wind_speeds, component)

    def update_params(self, **kwargs):
        self.params = self.params.update(**kwargs)
        self.spectrum.params = self.params.to_dict()

    # ------------------------------------------------------------------
    # Core algorithm  (shared — each backend only provides primitives)
    # ------------------------------------------------------------------

    def build_amplitude_matrix(self, positions, wind_speeds, frequencies, component: str):
        """Build complex amplitude matrix B of shape (n, N_batch).

        The inner ``_single_freq`` is defined locally so that spatial
        data (which is constant across frequencies) is captured via
        closure rather than passed through ``vmap``.  ``self._jit``
        wraps it for JAX; NumPy / Torch use an identity decorator.
        """
        xp = self._xp
        n = positions.shape[0]
        heights = positions[:, 2]

        # Spatial grids — same for every frequency
        x_i, x_j = positions[:, 0:1], positions[:, 0:1].T
        y_i, y_j = positions[:, 1:2], positions[:, 1:2].T
        z_i, z_j = positions[:, 2:3], positions[:, 2:3].T
        U_i = wind_speeds[:, None]
        U_j = wind_speeds[None, :]
        C_x, C_y, C_z = self.params.C_x, self.params.C_y, self.params.C_z
        eye_n = self._eye(n)

        phi = self._random_phases(len(frequencies), n)

        def _single_freq(freq, phi_l):
            s = self._spec_fn(freq, heights, component)
            s = self._clip_positive(s)
            coh = self._coh_fn(
                x_i, x_j, y_i, y_j, z_i, z_j,
                freq, U_i, U_j, C_x, C_y, C_z,
            )
            csd = self._csd_fn(s[:, None], s[None, :], coh)
            csd = 0.5 * (csd + csd.T)
            H = self._cholesky(csd + eye_n * 1e-12)
            return self._matmul(self._to_complex(H), xp.exp(1j * phi_l))

        B = self._vmap(self._jit(_single_freq))(frequencies, phi)
        return B.T

    def _process_amplitude_to_samples(self, B, N: int, M: int, dw: float):
        """IFFT pipeline — shared formula, backend-specific FFT."""
        n = B.shape[0]
        B_full = self._zeros_c((n, M))
        B_full = self._slice_set(B_full, (slice(None), slice(None, N)), B)
        G = self._fft(B_full, axis=1) * M
        p = self._arange(M)
        corr = self._xp.exp(1j * (p * self._xp.pi / M))
        scale = self._xp.sqrt(self._asarray(2.0 * dw))
        return self._asarray(scale * self._xp.real(G * corr[None, :]),
                             dtype=self._real_dtype)

    # -- simulation paths ------------------------------------------------

    def _simulate_direct(self, positions, wind_speeds, component):
        N, M, dw = self.params.N, self.params.M, self.params.dw
        freqs = self._asarray(simulation_frequencies(self._xp, N, dw))
        B = self.build_amplitude_matrix(positions, wind_speeds, freqs, component)
        samples = self._process_amplitude_to_samples(B, N, M, dw)
        return self._to_numpy(samples), self._to_numpy(freqs)

    def _simulate_batched(self, positions, wind_speeds, component, batch_size):
        N, M, dw = self.params.N, self.params.M, self.params.dw
        n = positions.shape[0]
        freqs = self._asarray(simulation_frequencies(self._xp, N, dw))
        n_batches = (N + batch_size - 1) // batch_size
        B_full = self._zeros_c((n, N))

        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, N)
            B_batch = self.build_amplitude_matrix(
                positions, wind_speeds, freqs[start:end], component)
            B_full = self._slice_set(B_full, (slice(None), slice(start, end)), B_batch)

        samples = self._process_amplitude_to_samples(B_full, N, M, dw)
        return self._to_numpy(samples), self._to_numpy(freqs)

    # -- memory ----------------------------------------------------------

    def estimate_memory(self, n_points: int, n_freqs: int) -> float:
        dtype_sz = 4
        S = n_freqs * n_points * n_points * dtype_sz
        H = S * 2
        B = n_points * (n_freqs * 2) * dtype_sz * 2
        return (S + H + B) * self._memory_factor / (1024**3)

    def _optimal_batch_sizes(self, n_points, n_freqs, max_memory_gb):
        budget = max_memory_gb * (1024**3)
        per_freq = (3 * n_points**2 + 4 * n_points) * 4 * self._memory_factor
        freq_batch = min(n_freqs, max(1, int(budget / per_freq)))
        return n_points, freq_batch


# ═══════════════════════════════════════════════════════════════════════
# JAX backend
# ═══════════════════════════════════════════════════════════════════════

class JaxWindSimulator(_BaseSimulator):
    """JAX — real ``jit`` + ``vmap``."""

    _memory_factor = 2.0

    def __init__(self, key=0, spectrum_type="kaimal", **params):
        import jax.numpy as jnp
        import jax.random as jr
        from jax import jit, vmap
        from jax.scipy.linalg import cholesky as _chol
        from functools import partial as _p

        self._xp = jnp
        self._real_dtype = jnp.float32
        self._key = jr.PRNGKey(key)
        self._asarray = jnp.asarray
        self._to_numpy = lambda x: __import__('numpy').asarray(x)
        self._backend_name = "jax"

        self._eye = _p(jnp.eye, dtype=jnp.float32)
        self._zeros_c = _p(jnp.zeros, dtype=jnp.complex64)
        self._arange = _p(jnp.arange, dtype=jnp.float32)
        self._matmul = jnp.matmul
        self._cholesky = _p(_chol, lower=True)
        self._fft = _p(jnp.fft.ifft)
        self._clip_positive = _p(jnp.maximum, 0.0)
        self._to_complex = lambda x: x
        self._jit = jit
        self._vmap = vmap
        self._slice_set = lambda arr, idx, val: arr.at[idx].set(val)

        # Pre-JIT helpers — compiled once, reduce outer JIT trace work
        # @jit
        def _coh(x_i, x_j, y_i, y_j, z_i, z_j, freq, U_i, U_j, C_x, C_y, C_z):
            return davenport_coherence(jnp, x_i, x_j, y_i, y_j, z_i, z_j,
                                       freq, U_i, U_j, C_x, C_y, C_z)

        # @jit
        def _csd(s_i, s_j, coh):
            return cross_spectrum(jnp, s_i, s_j, coh)

        self._coh_fn = _coh
        self._csd_fn = _csd

        def _random_phases(n_freq, n_pts):
            from jax.random import split, uniform
            self._key, sk = split(self._key)
            return uniform(sk, (n_freq, n_pts), minval=0, maxval=2 * jnp.pi)

        self._random_phases = _random_phases

        super().__init__(spectrum_type, params)

        # Pre-JIT spectrum (after super().__init__ so self.spectrum exists)
        _sp = self.spectrum

        @_p(jit, static_argnums=(2,))
        def _spec_fn(freq, heights, component):
            return _sp(freq, heights, component)

        self._spec_fn = _spec_fn


# ═══════════════════════════════════════════════════════════════════════
# NumPy backend
# ═══════════════════════════════════════════════════════════════════════

class NumpyWindSimulator(_BaseSimulator):
    """NumPy — identity ``jit``, list-comprehension ``vmap``."""

    _memory_factor = 2.0

    def __init__(self, seed=0, spectrum_type="kaimal", **params):
        import numpy as np
        from scipy.linalg import cholesky as _chol
        from functools import partial as _p

        self._xp = np
        self._real_dtype = np.float64
        self._seed = seed
        np.random.seed(seed)
        self._asarray = np.asarray
        self._to_numpy = np.asarray
        self._backend_name = "numpy"

        self._eye = np.eye
        self._zeros_c = _p(np.zeros, dtype=np.complex128)
        self._arange = _p(np.arange, dtype=np.float64)
        self._matmul = np.matmul
        self._cholesky = _p(_chol, lower=True)
        self._fft = np.fft.ifft
        self._clip_positive = _p(np.maximum, 0.0)
        self._to_complex = _p(np.ndarray.astype, dtype=np.complex128)
        self._jit = lambda fn: fn
        self._slice_set = lambda arr, idx, val: arr.__setitem__(idx, val) or arr

        self._coh_fn = lambda *a: davenport_coherence(np, *a)
        self._csd_fn = lambda *a: cross_spectrum(np, *a)

        def _vmap(fn):
            def _loop(*args):
                return np.array([fn(*[a[i] for a in args]) for i in range(len(args[0]))])
            return _loop
        self._vmap = _vmap

        def _random_phases(n_freq, n_pts):
            self._seed += 1
            np.random.seed(self._seed)
            return np.random.uniform(0, 2 * np.pi, (n_freq, n_pts))

        self._random_phases = _random_phases

        super().__init__(spectrum_type, params)
        self._spec_fn = lambda f, h, c: self.spectrum(f, h, c)


# ═══════════════════════════════════════════════════════════════════════
# Torch backend
# ═══════════════════════════════════════════════════════════════════════

class TorchWindSimulator(_BaseSimulator):
    """PyTorch — identity ``jit``, ``torch.func.vmap``."""

    _memory_factor = 5.0

    def __init__(self, seed=0, spectrum_type="kaimal", **params):
        import torch
        import torch.func as func

        self._xp = torch
        self._real_dtype = torch.float32
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._seed = seed
        torch.manual_seed(seed)
        self._asarray = lambda x, dtype=None: torch.as_tensor(
            x, dtype=dtype or torch.float32, device=self._device)
        self._to_numpy = lambda x: (x.detach().cpu().numpy() if hasattr(x, 'detach')
                                    else __import__('numpy').asarray(x))
        self._backend_name = "torch"

        self._eye = lambda n: torch.eye(n, device=self._device)
        self._zeros_c = lambda shape: torch.zeros(shape, dtype=torch.complex64, device=self._device)
        self._arange = lambda n: torch.arange(n, dtype=torch.float32, device=self._device)
        self._matmul = torch.matmul
        self._cholesky = lambda m: torch.linalg.cholesky(m, upper=False)
        self._fft = lambda x, axis: torch.fft.ifft(x, dim=axis)

        def _clip_positive(x):
            return torch.maximum(x, torch.tensor(0.0, device=self._device))

        self._clip_positive = _clip_positive
        self._to_complex = lambda x: x.to(torch.complex64)
        self._jit = lambda fn: fn   # identity
        self._vmap = lambda fn: func.vmap(fn)
        self._slice_set = lambda arr, idx, val: arr.__setitem__(idx, val) or arr

        self._coh_fn = lambda *a: davenport_coherence(torch, *a)
        self._csd_fn = lambda *a: cross_spectrum(torch, *a)

        def _random_phases(n_freq, n_pts):
            torch.manual_seed(self._seed)
            self._seed += 1
            return torch.rand(n_freq, n_pts, device=self._device) * 2 * torch.pi

        self._random_phases = _random_phases

        super().__init__(spectrum_type, params)
        self._spec_fn = lambda f, h, c: self.spectrum(f, h, c)


# ═══════════════════════════════════════════════════════════════════════
# Convenience constructors
# ═══════════════════════════════════════════════════════════════════════

def create_simulator(backend: str = "jax", spectrum: str = "kaimal",
                     seed: int = 0, **params) -> _BaseSimulator:
    if backend == "jax":
        return JaxWindSimulator(key=seed, spectrum_type=spectrum, **params)
    elif backend == "numpy":
        return NumpyWindSimulator(seed=seed, spectrum_type=spectrum, **params)
    elif backend == "torch":
        return TorchWindSimulator(seed=seed, spectrum_type=spectrum, **params)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")
