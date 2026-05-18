"""Nonstationary wind field simulation.

Wraps a backend-specific stationary simulator and adds time-varying
evolutionary power spectral density (EPSD) support.  All backend
operations (arrays, linear algebra, vmap, jit) are delegated to the
wrapped simulator.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .coherence import davenport_coherence, cross_spectrum, simulation_frequencies

logger = logging.getLogger(__name__)


class NonstationaryWindSimulator:
    """Nonstationary wind simulator — wraps a stationary simulator.

    Parameters
    ----------
    sim:
        A backend-specific stationary simulator
        (``JaxWindSimulator``, ``NumpyWindSimulator``, or ``TorchWindSimulator``).
    """

    def __init__(self, sim):
        self._sim = sim  # delegate all backend ops

    # -- convenience property forwarding --------------------------------

    @property
    def params(self):
        return self._sim.params

    @property
    def spectrum(self):
        return self._sim.spectrum

    @property
    def backend_name(self):
        return self._sim.backend_name

    def update_params(self, **kwargs):
        self._sim.update_params(**kwargs)

    # -- public API ----------------------------------------------------

    def simulate_nonstationary(
        self,
        positions,
        wind_speeds,
        component: str = "u",
        mode: str = "chunked-vmap",
        modulation_amplitude: float = 0.2,
        modulation_values=None,
        evolution_psd_generator=None,
        max_memory_gb: float = 4.0,
        freq_batch_size: Optional[int] = None,
        time_batch_size: Optional[int] = None,
        auto_batch: bool = True,
    ):
        """Simulate a nonstationary wind field.

        Parameters
        ----------
        positions:
            (n, 3) spatial coordinates.
        wind_speeds:
            (n,) reference mean wind speeds at 10 m (will be modulated).
        component:
            ``"u"`` or ``"w"``.
        mode:
            ``"chunked-vmap"`` (default), ``"freq-for"``, or ``"full-vmap"``.
        modulation_amplitude:
            Amplitude of sinusoidal modulation.
        modulation_values:
            Optional (M,) array of explicit modulation factors.
        evolution_psd_generator:
            Optional callable for custom EPSD.
        max_memory_gb:
            Memory budget for auto-chunking.
        freq_batch_size, time_batch_size:
            Manual chunk sizes (``None`` = auto).
        auto_batch:
            Enable automatic chunking when memory exceeds budget.

        Returns
        -------
        wind_samples:
            (n, M) numpy array of wind speed time series.
        frequencies:
            (N,) numpy array of simulation frequencies.
        """
        s = self._sim
        positions = s._asarray(positions)
        wind_speeds = s._asarray(wind_speeds)

        n = positions.shape[0]
        N = s.params.N
        M = s.params.M
        dw = s.params.dw
        dt = s.params.dt
        T_total = s.params.T

        xp = s._xp
        freqs = s._asarray(simulation_frequencies(xp, N, dw))
        times = s._asarray(s._arange(M)) * dt

        if modulation_values is not None:
            modulation_values = s._asarray(modulation_values)
            if modulation_values.shape != (M,):
                raise ValueError(
                    f"modulation_values must have shape ({M},), "
                    f"got {modulation_values.shape}"
                )

        # Spatial grids
        x_i, x_j = positions[:, 0:1], positions[:, 0:1].T
        y_i, y_j = positions[:, 1:2], positions[:, 1:2].T
        z_i, z_j = positions[:, 2:3], positions[:, 2:3].T
        heights = positions[:, 2]

        phi = s._random_phases(N, n)
        V_accum = s._zeros_c((M, n))
        eye_n = s._eye(n)

        # Chunking
        est_mem = self._estimate_memory(n, N, M)
        freq_batch, time_batch = self._choose_chunks(
            mode, n, N, M, est_mem, max_memory_gb,
            freq_batch_size, time_batch_size, auto_batch,
        )
        n_fb = (N + freq_batch - 1) // freq_batch
        n_tb = (M + time_batch - 1) // time_batch

        logger.info("Nonstationary memory estimate: %.2f GB", est_mem)
        logger.info("Chunks: %d freq x %d time (freq_batch=%d, time_batch=%d)",
                    n_fb, n_tb, freq_batch, time_batch)

        # -- shared single-time amplitude (captures freq_l, phi_l from _single_freq) --
        def _make_single_time(freq_l, phi_l):
            def _single_time(time_idx, time_m):
                s_ev = self._evolutional_psd(
                    freq_l, heights, wind_speeds, component,
                    time_m, time_idx, T_total,
                    modulation_amplitude, modulation_values,
                    evolution_psd_generator,
                )
                s_ev = s._clip_positive(s_ev)
                U_mod = self._modulated_wind(
                    wind_speeds, time_m, time_idx, T_total,
                    modulation_amplitude, modulation_values,
                )
                Ui, Uj = U_mod[:, None], U_mod[None, :]
                coh = davenport_coherence(
                    xp, x_i, x_j, y_i, y_j, z_i, z_j,
                    freq_l, Ui, Uj,
                    s.params.C_x, s.params.C_y, s.params.C_z,
                )
                csd = cross_spectrum(xp, s_ev[:, None], s_ev[None, :], coh)
                csd = 0.5 * (csd + csd.T)
                H = s._cholesky(csd + eye_n * 1e-12)
                phase_vec = xp.exp(1j * phi_l)
                return s._matmul(s._to_complex(H), phase_vec) * xp.exp(
                    1j * 2.0 * xp.pi * freq_l * time_m
                )
            return _single_time

        # -- backend-specific chunk functions (defined ONCE, reused) --
        if s.backend_name == "jax":
            from jax import jit

            @jit
            def _single_freq(freq_l, time_chunk, time_indices, phi_l):
                return s._vmap(_make_single_time(freq_l, phi_l))(time_indices, time_chunk)

            @jit
            def _compute_chunk(freq_chunk, time_chunk, time_indices, phi_chunk):
                return s._vmap(
                    lambda f, p: _single_freq(f, time_chunk, time_indices, p)
                )(freq_chunk, phi_chunk)

        elif s.backend_name == "torch":
            def _single_freq(freq_l, time_chunk, time_indices, phi_l):
                return s._vmap(_make_single_time(freq_l, phi_l))(time_indices, time_chunk)

            def _compute_chunk(freq_chunk, time_chunk, time_indices, phi_chunk):
                return s._vmap(
                    lambda f, p: _single_freq(f, time_chunk, time_indices, p)
                )(freq_chunk, phi_chunk)
        else:
            def _single_freq(freq_l, time_chunk, time_indices, phi_l):
                st = _make_single_time(freq_l, phi_l)
                return np.array([st(time_indices[j], time_chunk[j])
                                 for j in range(len(time_chunk))])

            def _compute_chunk(freq_chunk, time_chunk, time_indices, phi_chunk):
                return np.array([
                    _single_freq(freq_chunk[i], time_chunk, time_indices, phi_chunk[i])
                    for i in range(len(freq_chunk))
                ])

        # -- main loop over chunks ---------------------------------
        t_start = time.time()
        chunk_counter = 0

        for fb in range(n_fb):
            fs, fe = fb * freq_batch, min((fb + 1) * freq_batch, N)
            fc = freqs[fs:fe]
            pc = phi[fs:fe]

            for tb in range(n_tb):
                ts, te = tb * time_batch, min((tb + 1) * time_batch, M)
                tc = times[ts:te]
                ti = s._asarray(xp.arange(ts, te))

                if mode == "freq-for":
                    chunk_sum = s._zeros_c((te - ts, n))
                    for k in range(len(fc)):
                        contrib = _compute_chunk(fc[k:k+1], tc, ti, pc[k:k+1])
                        chunk_sum = chunk_sum + (contrib[0] if contrib.ndim == 3 else contrib)
                else:
                    contrib = _compute_chunk(fc, tc, ti, pc)
                    chunk_sum = contrib.sum(axis=0) if contrib.ndim == 3 else contrib

                V_accum = s._slice_set(
                    V_accum, (slice(ts, te), slice(None)), V_accum[ts:te] + chunk_sum
                )
                chunk_counter += 1
                if chunk_counter == 1:
                    self._sync(V_accum)
                    logger.info("First chunk done (incl. compile): %.3f s",
                                time.time() - t_start)

        wind = xp.sqrt(s._asarray(2.0 * dw)) * xp.real(V_accum)
        wind = s._asarray(wind.T, dtype=s._real_dtype)
        self._sync(wind)
        logger.info("Total nonstationary time: %.3f s", time.time() - t_start)

        return s._to_numpy(wind), s._to_numpy(freqs)

    # -- helpers -----------------------------------------------------

    def _evolutional_psd(self, freq, heights, wind_speeds, component,
                          time_val, time_idx, T_total,
                          mod_amp, mod_vals, psd_gen):
        if psd_gen is not None:
            result = psd_gen(
                freq=freq, heights=heights, wind_speeds=wind_speeds,
                component=component, time_value=time_val,
                time_index=time_idx, total_time=T_total, simulator=self,
            )
            return self._sim._asarray(result)
        U_mod = self._modulated_wind(
            wind_speeds, time_val, time_idx, T_total, mod_amp, mod_vals)
        return self._sim.spectrum(freq, heights, component, U_d=U_mod)

    def _modulated_wind(self, wind_speeds, time_val, time_idx, T_total,
                         mod_amp, mod_vals):
        s = self._sim
        if mod_vals is not None:
            return wind_speeds * mod_vals[time_idx]
        scale = s._asarray(1.0) + s._asarray(mod_amp) * s._xp.sin(
            2.0 * s._xp.pi * time_val / T_total)
        return wind_speeds * scale

    def _choose_chunks(self, mode, n, N, M, est_mem, max_mem,
                        freq_batch, time_batch, auto_batch):
        if mode == "full-vmap":
            return N, M
        if mode == "freq-for":
            return 1, M if time_batch is None else min(time_batch, M)
        if not auto_batch or est_mem <= max_mem:
            return (N if freq_batch is None else min(freq_batch, N),
                    M if time_batch is None else min(time_batch, M))
        factor = {"jax": 3.0, "numpy": 3.0, "torch": 5.0}[self.backend_name]
        budget = max_mem * (1024**3)
        per_pair = ((3 * n * n) + (3 * n)) * 4 * factor
        max_pairs = max(1, int(budget / per_pair))
        if max_pairs >= N * M:
            return N, M
        ratio = N / max(M, 1)
        fb = max(1, min(N, int((max_pairs * ratio) ** 0.5)))
        tb = max(1, min(M, max_pairs // max(fb, 1)))
        while fb * tb > max_pairs:
            if fb >= tb and fb > 1:
                fb -= 1
            elif tb > 1:
                tb -= 1
            else:
                break
        return max(1, fb), max(1, tb)

    def _estimate_memory(self, n_points, n_freqs, n_times,
                         freq_batch=None, time_batch=None):
        fc = n_freqs if freq_batch is None else min(freq_batch, n_freqs)
        tc = n_times if time_batch is None else min(time_batch, n_times)
        mat_bytes = fc * tc * n_points * n_points * 4
        vec_bytes = fc * tc * n_points * 4
        return (3 * mat_bytes + vec_bytes + vec_bytes * 2) * 3.0 / (1024**3)

    def _sync(self, arr):
        if self.backend_name == "jax":
            try:
                _ = arr.block_until_ready()
            except AttributeError:
                pass
