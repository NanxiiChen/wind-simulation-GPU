"""Wind field visualisation — backend-agnostic.

Works with any simulator produced by :func:`create_simulator`.
All theoretical computations are delegated to the simulator's
backend; plotting uses matplotlib (numpy).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .simulator import _BaseSimulator, create_simulator


class WindVisualizer:
    """Plot PSD and cross-correlation for simulated wind fields.

    Parameters
    ----------
    simulator:
        A simulator instance (``JaxWindSimulator``, ``NumpyWindSimulator``,
        or ``TorchWindSimulator``).
    seed:
        Seed for reproducible random index selection.
    """

    def __init__(self, simulator: _BaseSimulator, seed: int = 0):
        self.sim = simulator
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # PSD plot
    # ------------------------------------------------------------------

    def plot_psd(
        self,
        wind_samples: np.ndarray,
        heights: np.ndarray,
        show_num: int = 5,
        component: str = "u",
        indices=None,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ):
        """Compare empirical PSD (Welch) with theoretical spectrum.

        Parameters
        ----------
        wind_samples:
            (n, M) array of simulated wind speeds.
        heights:
            (n,) heights at each point.
        show_num:
            Number of random points to display (when ``indices`` is None).
        component:
            ``"u"`` or ``"w"``.
        indices:
            Specific point indices to plot (overrides ``show_num``).
        save_path:
            If given, save figure to this path.
        show:
            Whether to call ``plt.show()``.
        """
        n = wind_samples.shape[0]
        indices = self._resolve_indices(indices, show_num, n)

        # Theoretical PSD from simulator's spectrum
        N, dw = self.sim.params.N, self.sim.params.dw
        freqs_theory = self._freq_array(N, dw)
        S_theory = self._theoretical_psd(freqs_theory, heights, component)

        ncol = kwargs.get("ncol", 3)
        nrow = (len(indices) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
        axes = [axes] if nrow == 1 and ncol == 1 else np.atleast_1d(axes).flatten()

        dt = self.sim.params.dt
        fs = 1.0 / dt

        for idx, pt in enumerate(indices):
            ax = axes[idx]
            f_emp, psd_emp = signal.welch(
                wind_samples[pt], fs=fs, nperseg=1024,
                scaling="density", window="hann",
            )
            ax.loglog(f_emp, psd_emp, label=f"Point {pt + 1}")
            ax.loglog(
                freqs_theory, S_theory[:, idx],
                "--", color="black", linewidth=2, label="Theory",
            )
            ax.set(
                xlabel="Frequency (Hz)",
                ylabel="PSD (m²/s²)",
                title=f"PSD of {component.upper()} at Z={heights[pt]:.2f} m (Pt {pt + 1})",
            )
            ax.grid(True, which="both", ls="-", alpha=0.6)
            ax.legend()

        # Hide unused axes
        for ax in axes[len(indices):]:
            ax.set_visible(False)

        fig.tight_layout()
        self._finish(fig, save_path, show)

    # ------------------------------------------------------------------
    # Cross-correlation plot
    # ------------------------------------------------------------------

    def plot_cross_correlation(
        self,
        wind_samples: np.ndarray,
        positions: np.ndarray,
        wind_speeds: np.ndarray,
        component: str = "u",
        indices=None,
        downsample: int = 1,
        save_path: Optional[str] = None,
        show: bool = True,
        **kwargs,
    ):
        """Compare empirical cross-correlation with theoretical expectation.

        Parameters
        ----------
        wind_samples:
            (n, M) simulated wind speeds.
        positions:
            (n, 3) spatial coordinates.
        wind_speeds:
            (n,) mean wind speeds.
        component:
            ``"u"`` or ``"w"``.
        indices:
            Pair of point indices ``(i, j)`` (random if None).
        downsample:
            Downsampling factor for faster computation.
        save_path:
            If given, save figure to this path.
        show:
            Whether to call ``plt.show()``.
        """
        n = wind_samples.shape[0]

        if downsample > 1:
            wind_samples = wind_samples[:, ::downsample]
            dt = self.sim.params.dt * downsample
        else:
            dt = self.sim.params.dt

        i, j = self._resolve_pair_indices(indices, n)

        data_i = wind_samples[i]
        data_j = wind_samples[j]

        # Empirical correlation
        corr_i = data_i - np.mean(data_i)
        corr_j = data_j - np.mean(data_j)
        corr = signal.correlate(corr_i, corr_j, mode="full")
        corr /= np.max(np.abs(corr)) if np.max(np.abs(corr)) > 0 else 1.0
        lags = np.arange(-len(data_i) + 1, len(data_i))
        lag_times = lags * dt

        # Theoretical correlation
        N, M, dw = self.sim.params.N, self.sim.params.M, self.sim.params.dw
        freqs = self._freq_array(N, dw)

        S_i = self._theoretical_psd_point(freqs, positions[i, 2], component)
        S_j = self._theoretical_psd_point(freqs, positions[j, 2], component)
        U_i, U_j = wind_speeds[i], wind_speeds[j]

        coh_arr = self._coherence_array(
            freqs, positions[i], positions[j], U_i, U_j,
        )
        cross_spec = np.sqrt(S_i * S_j) * coh_arr

        # Build full-spectrum and IFFT for theoretical correlation
        full_spec = np.zeros(M, dtype=np.complex128)
        full_spec[1:N + 1] = cross_spec
        full_spec[M - N:] = np.flip(np.conj(cross_spec))
        theo_corr = np.real(np.fft.ifft(full_spec))
        theo_corr = np.fft.fftshift(theo_corr)
        theo_max = np.max(np.abs(theo_corr))
        if theo_max > 0:
            theo_corr /= theo_max
        theo_lags = np.arange(-M // 2, M // 2)
        theo_lag_times = theo_lags * self.sim.params.dt

        # Plot
        mid = len(corr) // 2
        rng = kwargs.get("range_points", len(theo_lag_times) // 2)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(
            lag_times[mid - rng:mid + rng + 1],
            corr[mid - rng:mid + rng + 1],
            label="Simulation",
        )
        t_mid = len(theo_corr) // 2
        ax.plot(
            theo_lag_times[t_mid - rng:t_mid + rng + 1],
            theo_corr[t_mid - rng:t_mid + rng + 1],
            "--", color="black", linewidth=2, label="Theory",
        )
        ax.set(
            title=f"Cross-correlation of {component.upper()} at Pts {i + 1} & {j + 1}",
            xlabel="Time lag (s)", ylabel="Cross-correlation",
        )
        ax.grid(True)
        ax.legend()
        self._finish(fig, save_path, show)

        if kwargs.get("return_data", False):
            return (lag_times[mid - rng:mid + rng + 1],
                    corr[mid - rng:mid + rng + 1],
                    theo_lag_times[t_mid - rng:t_mid + rng + 1],
                    theo_corr[t_mid - rng:t_mid + rng + 1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _freq_array(self, N, dw):
        """Frequency array as numpy (for plotting/theory)."""
        return np.arange(1, N + 1) * dw - dw / 2.0

    def _theoretical_psd(self, freqs, heights, component):
        """Theoretical PSD over all freqs and all heights.

        Uses the simulator's pre-JIT ``_spec_fn`` (JAX) or vmap (Torch)
        to avoid repeated tracing / compilation.
        """
        sim = self.sim
        f = sim._asarray(freqs)
        h = sim._asarray(heights)
        if sim.backend_name == "jax":
            from jax import vmap
            S = vmap(lambda freq: sim._spec_fn(freq, h, component))(f)
        elif sim.backend_name == "torch":
            from torch.func import vmap
            S = vmap(lambda freq: sim._spec_fn(freq, h, component))(f)
        else:
            S = sim._vmap(lambda freq: sim._spec_fn(freq, h, component))(f)
        return np.asarray(sim._to_numpy(S))

    def _theoretical_psd_point(self, freqs, height, component):
        """Theoretical PSD at a single height."""
        sim = self.sim
        f = sim._asarray(freqs)
        h = sim._asarray([height])
        if sim.backend_name == "jax":
            from jax import vmap
            S = vmap(lambda freq: sim._spec_fn(freq, h, component))(f)
        elif sim.backend_name == "torch":
            from torch.func import vmap
            S = vmap(lambda freq: sim._spec_fn(freq, h, component))(f)
        else:
            S = sim._vmap(lambda freq: sim._spec_fn(freq, h, component))(f)
        return np.asarray(sim._to_numpy(S)).flatten()

    def _coherence_array(self, freqs, pos_i, pos_j, U_i, U_j):
        """Coherence over frequency array, for a single point pair.

        Uses the simulator's pre-JIT ``_coh_fn`` with vmap (JAX/Torch)
        or batch call (NumPy) to avoid per-frequency overhead.
        """
        sim = self.sim
        f = sim._asarray(freqs)
        xp_i = sim._asarray([pos_i[0]]); xp_j = sim._asarray([pos_j[0]])
        yp_i = sim._asarray([pos_i[1]]); yp_j = sim._asarray([pos_j[1]])
        zp_i = sim._asarray([pos_i[2]]); zp_j = sim._asarray([pos_j[2]])
        ui = sim._asarray([U_i]); uj = sim._asarray([U_j])
        Cx, Cy, Cz = sim.params.C_x, sim.params.C_y, sim.params.C_z

        if sim.backend_name == "jax":
            from jax import vmap
            coh = vmap(
                lambda fi: sim._coh_fn(xp_i, xp_j, yp_i, yp_j, zp_i, zp_j,
                                       fi, ui, uj, Cx, Cy, Cz)
            )(f)
        elif sim.backend_name == "torch":
            from torch.func import vmap
            coh = vmap(
                lambda fi: sim._coh_fn(xp_i, xp_j, yp_i, yp_j, zp_i, zp_j,
                                       fi, ui, uj, Cx, Cy, Cz)
            )(f)
        else:
            coh = sim._vmap(
                lambda fi: sim._coh_fn(xp_i, xp_j, yp_i, yp_j, zp_i, zp_j,
                                       fi, ui, uj, Cx, Cy, Cz)
            )(f)
        return np.asarray(sim._to_numpy(coh)).flatten()

    def _resolve_indices(self, indices, show_num, n):
        if indices is not None:
            if isinstance(indices, (int, np.integer)):
                return (int(indices),)
            return tuple(indices)
        if show_num >= n:
            return tuple(range(n))
        return tuple(sorted(self._rng.choice(n, show_num, replace=False)))

    def _resolve_pair_indices(self, indices, n):
        if indices is None:
            idx = self._rng.randint(0, n)
            return idx, idx
        if isinstance(indices, (int, np.integer)):
            return int(indices), int(indices)
        if len(indices) == 2:
            return int(indices[0]), int(indices[1])
        raise ValueError("indices must be an int or a pair (i, j)")

    @staticmethod
    def _finish(fig, save_path, show):
        if save_path:
            fig.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
