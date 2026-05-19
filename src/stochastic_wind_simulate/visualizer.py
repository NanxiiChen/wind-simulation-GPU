"""Wind field visualisation — backend-agnostic.

Works with any simulator.  For stationary signals use ``plot_psd``
(Welch's method).  For nonstationary signals use
``plot_nonstationary_psd`` (short-time Fourier transform).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .simulator import _BaseSimulator


class WindVisualizer:
    """Plot PSD and cross-correlation for simulated wind fields.

    Parameters
    ----------
    simulator:
        A simulator instance (any backend).
    seed:
        Seed for reproducible random index selection.
    """

    def __init__(self, simulator: _BaseSimulator, seed: int = 0):
        self.sim = simulator
        self._rng = np.random.RandomState(seed)

    # ==================================================================
    # Stationary: Welch PSD
    # ==================================================================

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
        wind_samples: (n, M) array.
        heights: (n,) heights at each point.
        show_num: Number of random points to display.
        component: ``"u"`` or ``"w"``.
        indices: Specific point indices (overrides *show_num*).
        """
        n = wind_samples.shape[0]
        indices = self._resolve_indices(indices, show_num, n)

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
                xlabel="Frequency (Hz)", ylabel="PSD (m²/s²)",
                title=f"PSD of {component.upper()} at Z={heights[pt]:.2f} m (Pt {pt + 1})",
            )
            ax.grid(True, which="both", ls="-", alpha=0.6)
            ax.legend()

        for ax in axes[len(indices):]:
            ax.set_visible(False)
        fig.tight_layout()
        self._finish(fig, save_path, show)

    # ==================================================================
    # Stationary: cross-correlation
    # ==================================================================

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
        """Compare empirical cross-correlation with theory.

        Parameters
        ----------
        wind_samples: (n, M) array.
        positions: (n, 3) spatial coordinates.
        wind_speeds: (n,) mean wind speeds.
        component: ``"u"`` or ``"w"``.
        indices: Pair of point indices ``(i, j)``.
        downsample: Downsampling factor.
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

        corr_i = data_i - np.mean(data_i)
        corr_j = data_j - np.mean(data_j)
        corr = signal.correlate(corr_i, corr_j, mode="full")
        corr /= np.max(np.abs(corr)) if np.max(np.abs(corr)) > 0 else 1.0
        lags = np.arange(-len(data_i) + 1, len(data_i))
        lag_times = lags * dt

        N, M, dw = self.sim.params.N, self.sim.params.M, self.sim.params.dw
        freqs = self._freq_array(N, dw)

        S_i = self._theoretical_psd_point(freqs, positions[i, 2], component)
        S_j = self._theoretical_psd_point(freqs, positions[j, 2], component)
        U_i, U_j = wind_speeds[i], wind_speeds[j]

        coh_arr = self._coherence_array(freqs, positions[i], positions[j], U_i, U_j)
        cross_spec = np.sqrt(S_i * S_j) * coh_arr

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

        mid = len(corr) // 2
        rng = kwargs.get("range_points", len(theo_lag_times) // 2)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(lag_times[mid - rng:mid + rng + 1],
                corr[mid - rng:mid + rng + 1], label="Simulation")
        t_mid = len(theo_corr) // 2
        ax.plot(theo_lag_times[t_mid - rng:t_mid + rng + 1],
                theo_corr[t_mid - rng:t_mid + rng + 1],
                "--", color="black", linewidth=2, label="Theory")
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

    # ==================================================================
    # Nonstationary: short-time Fourier PSD
    # ==================================================================

    def plot_nonstationary_psd(
        self,
        wind_signal: np.ndarray,
        height: float,
        wind_speed: float = 30.0,
        component: str = "u",
        window_size: int = 256,
        overlap: int = 192,
        modulation_amplitude: float = 0.2,
        modulation_values=None,
        snapshot_count: int = 4,
        snapshot_times=None,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Plot nonstationary PSD via short-time Fourier transform.

        Computes a local spectrogram by sliding-window FFT and compares
        against the theoretical evolutionary PSD at selected snapshots.

        Parameters
        ----------
        wind_signal: (M,) time series for a **single** spatial point.
        height: Height of the point [m].
        wind_speed: Reference mean wind speed [m/s].
        component: ``"u"`` or ``"w"``.
        window_size: Window length in samples.
        overlap: Overlap between consecutive windows.
        modulation_amplitude: Sinusoidal modulation amplitude.
        modulation_values: Optional (M,) explicit modulation factors.
        snapshot_count: Number of PSD snapshots.
        snapshot_times: Explicit snapshot times [s].
        """
        if not hasattr(self.sim, '_evolutional_psd'):
            raise TypeError(
                "plot_nonstationary_psd requires a NonstationaryWindSimulator. "
                "Wrap: ns = NonstationaryWindSimulator(sim)"
            )

        sim = self.sim
        dt = sim.params.dt
        M = len(wind_signal)

        # -- local spectrogram ------------------------------------------
        est_freqs, est_times, spectrogram = self._local_spectrogram(
            wind_signal, dt, window_size, overlap)

        # -- modulation factors -----------------------------------------
        if modulation_values is not None:
            mod_f = np.asarray(modulation_values, dtype=np.float32)
        else:
            t_all = np.arange(M) * dt
            mod_f = 1.0 + modulation_amplitude * np.sin(
                2.0 * np.pi * t_all / sim.params.T)

        sim_freqs = self._freq_array(sim.params.N, sim.params.dw)
        h_arr = np.array([height], dtype=np.float32)
        ws_arr = np.array([wind_speed], dtype=np.float32)

        # -- full theoretical evolutionary PSD (all times x all freqs) --
        tgt_full = self._compute_evolutionary_psd_full(
            sim_freqs, h_arr, ws_arr, component, mod_f)

        # -- average over windows (same as validate.py) -----------------
        tgt_win = self._average_over_windows(
            tgt_full, np.arange(M) * dt, est_times, dt, window_size)

        # -- resample to spectrogram freq bins --------------------------
        theo_resampled = self._average_over_freq_bands(
            sim_freqs, tgt_win, est_freqs)

        # -- local variance & target variance ---------------------------
        est_var = self._local_variance(wind_signal, window_size, overlap)
        mean_resp = self._window_mean_response(sim_freqs, dt, window_size)
        tgt_var = np.array([
            np.trapezoid(tgt_win[ti] * (1.0 - mean_resp), sim_freqs)
            for ti in range(len(est_times))
        ])

        # -- snapshot PSDs ----------------------------------------------
        snap_idx = self._snapshot_indices(est_times, snapshot_count, snapshot_times)
        snap_t = est_times[snap_idx]
        snap_est = spectrogram[:, snap_idx]
        snap_tgt = self._average_over_freq_bands(
            sim_freqs,
            np.stack([tgt_full[int(np.argmin(np.abs(
                np.arange(M) * dt - tv)))] for tv in snap_t], axis=0),
            est_freqs,
        )

        # -- plot -------------------------------------------------------
        fig = self._plot_nonstationary_figure(
            est_times, est_var, tgt_var,
            np.arange(M) * dt, wind_signal,
            est_freqs, snap_est, snap_idx, snap_t, snap_tgt,
            component, height,
        )
        self._finish(fig, save_path, show)

    # ==================================================================
    # Nonstationary helpers
    # ==================================================================

    @staticmethod
    def _local_spectrogram(signal, dt, window_size, overlap):
        """Sliding-window PSD via FFT. Returns (freqs, times, spec)."""
        step = max(1, window_size - overlap)
        if len(signal) < window_size:
            raise ValueError(f"window_size ({window_size}) > signal ({len(signal)})")

        win = np.hanning(window_size).astype(np.float32)
        wp = np.sum(win ** 2)
        starts = np.arange(0, len(signal) - window_size + 1, step)
        centre_times = (starts + window_size / 2.0) * dt

        segs = np.stack([signal[s:s + window_size] for s in starts], axis=0)
        segs = segs - np.mean(segs, axis=1, keepdims=True)
        segs = segs * win[None, :]

        fv = np.fft.rfft(segs, axis=1)
        spec = (np.abs(fv) ** 2) / (wp / dt)
        if spec.shape[1] > 2:
            spec[:, 1:-1] *= 2.0
        return np.fft.rfftfreq(window_size, d=dt), centre_times, spec.T

    @staticmethod
    def _local_variance(signal, window_size, overlap):
        """Sliding-window variance."""
        step = max(1, window_size - overlap)
        starts = np.arange(0, len(signal) - window_size + 1, step)
        return np.var(np.stack(
            [signal[s:s + window_size] for s in starts], axis=0), axis=1)

    def _compute_evolutionary_psd_full(
        self, sim_freqs, heights, wind_speeds, component, mod_factors,
    ):
        """Theoretical evolutionary PSD at every time step via double vmap.

        Same approach as ``validate.py:theoretical_evolutional_psd``.
        Returns ``(n_times, n_freqs)``.
        """
        sim = self.sim
        f_arr = sim._asarray(sim_freqs)
        h_arr = sim._asarray(heights)
        ws_arr = sim._asarray(wind_speeds)
        mf_arr = sim._asarray(mod_factors)

        result = sim._vmap(lambda mf: sim._vmap(lambda f: sim._evolutional_psd(
            f, h_arr, ws_arr, component, mf, None))(f_arr))(mf_arr)
        return np.asarray(sim._to_numpy(result), dtype=np.float32).squeeze()

    @staticmethod
    def _average_over_windows(target_psd, target_times, window_centers,
                               dt, window_size):
        """Average target PSD over each time window (same as validate.py)."""
        half = 0.5 * window_size * dt
        out = []
        for center in window_centers:
            mask = (target_times >= center - half) & (target_times < center + half)
            if not np.any(mask):
                idx = int(np.argmin(np.abs(target_times - center)))
                out.append(target_psd[idx])
            else:
                out.append(np.mean(target_psd[mask], axis=0))
        return np.stack(out, axis=0)

    @staticmethod
    def _average_over_freq_bands(target_freqs, target_psd, est_freqs):
        """Bin-average target PSD onto estimated frequency bins."""
        n_times, _ = target_psd.shape
        n_est = len(est_freqs)

        edges = np.empty(n_est + 1, dtype=np.float32)
        edges[1:-1] = 0.5 * (est_freqs[:-1] + est_freqs[1:])
        edges[0] = max(0.0, est_freqs[0] - 0.5 * (est_freqs[1] - est_freqs[0]))
        edges[-1] = est_freqs[-1] + 0.5 * (est_freqs[-1] - est_freqs[-2])

        out = np.empty((n_times, n_est), dtype=np.float32)
        for ti in range(n_times):
            for fi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                mask = (target_freqs >= lo) & (target_freqs < hi)
                if np.any(mask):
                    bf, bv = target_freqs[mask], target_psd[ti, mask]
                    out[ti, fi] = (np.trapezoid(bv, bf) / (hi - lo)
                                   if len(bf) > 1 else bv[0])
                else:
                    out[ti, fi] = np.interp(
                        0.5 * (lo + hi), target_freqs, target_psd[ti])
        return out

    @staticmethod
    def _window_mean_response(freqs, dt, window_size):
        """Window mean-response for target variance (same as validate.py)."""
        nf = np.asarray(freqs, dtype=np.float64) * dt
        k = np.exp(-1j * np.pi * (window_size - 1) * nf)
        k = k * window_size * (np.sinc(window_size * nf) / np.sinc(nf))
        k[np.isclose(nf, 0.0)] = float(window_size) + 0.0j
        return np.abs(k / window_size) ** 2

    @staticmethod
    def _snapshot_indices(window_times, snapshot_count, snapshot_times):
        """Select snapshot indices from window centre times."""
        if snapshot_times is not None:
            return sorted(set(
                int(np.argmin(np.abs(window_times - tv))) for tv in snapshot_times))
        n = len(window_times)
        if snapshot_count >= n:
            return list(range(n))
        return sorted(set(int(round(i))
                          for i in np.linspace(0, n - 1, snapshot_count)))

    # ==================================================================
    # Nonstationary plot layout
    # ==================================================================

    def _plot_nonstationary_figure(
        self, est_times, est_var, tgt_var,
        sig_times, sig,
        est_freqs, snap_est, snap_idx, snap_t, snap_tgt,
        component, height,
    ):
        n_snap = len(snap_idx)
        ncol = min(2, n_snap)
        nrow = int(np.ceil(n_snap / ncol))

        fig = plt.figure(figsize=(15, 10 + 3 * max(0, nrow - 1)))
        gs = fig.add_gridspec(2 + nrow, 2)

        # -- top: variance envelope --
        ax_var = fig.add_subplot(gs[0, :])
        ax_var.plot(est_times, est_var, color="tab:blue", lw=1.8,
                    label="Estimated local variance")
        ax_var.plot(est_times, tgt_var, "--", color="tab:red", lw=2.0,
                    label="Target local variance")
        ax_var.set_title("Local Variance Envelope")
        ax_var.set_ylabel("Variance")
        ax_var.grid(True); ax_var.legend()
        for tv in snap_t:
            ax_var.axvline(tv, color="0.4", linestyle=":", lw=0.9)

        # -- middle: time history with std envelope --
        est_std = np.sqrt(np.maximum(est_var, 0.0))
        tgt_std = np.sqrt(np.maximum(tgt_var, 0.0))
        est_std_full = np.interp(sig_times, est_times, est_std)
        tgt_std_full = np.interp(sig_times, est_times, tgt_std)

        ax_time = fig.add_subplot(gs[1, :], sharex=ax_var)
        ax_time.plot(sig_times, sig, color="black", lw=0.9, alpha=0.8,
                     label="Sample wind speed")
        ax_time.fill_between(sig_times, -tgt_std_full, tgt_std_full,
                             color="tab:red", alpha=0.16,
                             label="Target std band")
        ax_time.plot(sig_times, tgt_std_full, "--", color="tab:red", lw=1.2)
        ax_time.plot(sig_times, -tgt_std_full, "--", color="tab:red", lw=1.2)
        ax_time.plot(sig_times, est_std_full, color="tab:blue", lw=1.2,
                     alpha=0.9, label="Est. std")
        ax_time.plot(sig_times, -est_std_full, color="tab:blue", lw=1.2, alpha=0.9)
        ax_time.set_title("Wind Time History with Local Std Envelope")
        ax_time.set_xlabel("Time (s)"); ax_time.set_ylabel("Wind speed")
        ax_time.grid(True); ax_time.legend(ncol=2, fontsize=9)
        for tv in snap_t:
            ax_time.axvline(tv, color="0.4", linestyle=":", lw=0.9)

        # -- bottom: PSD snapshots --
        comp = component.upper()
        for si in range(n_snap):
            ax = fig.add_subplot(gs[2 + si // ncol, si % ncol])
            ax.plot(est_freqs, snap_est[:, si],
                    color="tab:blue", lw=1.8, label="Simulated")
            ax.plot(est_freqs, snap_tgt[si, :],
                    "--", color="tab:red", lw=2.0, label="Theory")
            ax.set_title(f"PSD of {comp} at t = {snap_t[si]:.2f} s "
                         f"(Z = {height:.1f} m)")
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
            ax.set_yscale("log"); ax.grid(True); ax.legend(fontsize=9)

        for ax_idx in range(n_snap, ncol * nrow):
            ax = fig.add_subplot(gs[2 + ax_idx // ncol, ax_idx % ncol])
            ax.axis("off")

        fig.tight_layout()
        return fig

    # ==================================================================
    # Shared helpers
    # ==================================================================

    def _freq_array(self, N, dw):
        return np.arange(1, N + 1) * dw - dw / 2.0

    def _theoretical_psd(self, freqs, heights, component):
        sim = self.sim
        f = sim._asarray(freqs)
        h = sim._asarray(heights)
        S = sim._vmap(lambda freq: sim._spec_fn(freq, h, component))(f)
        return np.asarray(sim._to_numpy(S))

    def _theoretical_psd_point(self, freqs, height, component):
        sim = self.sim
        f = sim._asarray(freqs)
        h = sim._asarray([height])
        S = sim._vmap(lambda freq: sim._spec_fn(freq, h, component))(f)
        return np.asarray(sim._to_numpy(S)).flatten()

    def _coherence_array(self, freqs, pos_i, pos_j, U_i, U_j):
        sim = self.sim
        f = sim._asarray(freqs)
        xp_i = sim._asarray([pos_i[0]]); xp_j = sim._asarray([pos_j[0]])
        yp_i = sim._asarray([pos_i[1]]); yp_j = sim._asarray([pos_j[1]])
        zp_i = sim._asarray([pos_i[2]]); zp_j = sim._asarray([pos_j[2]])
        ui = sim._asarray([U_i]); uj = sim._asarray([U_j])
        Cx, Cy, Cz = sim.params.C_x, sim.params.C_y, sim.params.C_z

        coh = sim._vmap(
            lambda fi: sim._coh_fn(xp_i, xp_j, yp_i, yp_j, zp_i, zp_j,
                                   fi, ui, uj, Cx, Cy, Cz))(f)
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
