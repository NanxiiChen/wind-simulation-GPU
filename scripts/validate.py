#!/usr/bin/env python
"""Validate nonstationary wind simulation against theoretical evolutionary PSD.

Compares ensemble-averaged local spectrograms and variance envelopes
with the target evolutionary power spectral density.  Supports all
backends.

Examples
--------
.. code-block:: bash

    python scripts/validate.py --config configs/validate.py
    python scripts/validate.py --n-points 50 --n-freqs 512 --n-realizations 16
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stochastic_wind_simulate import (
    NonstationaryWindSimulator,
    create_simulator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spectrogram & window helpers
# ---------------------------------------------------------------------------

def local_spectrogram(signal, dt, window_size, overlap):
    step = max(1, window_size - overlap)
    window = np.hanning(window_size).astype(np.float32)
    window_power = np.sum(window**2)
    starts = np.arange(0, len(signal) - window_size + 1, step)
    center_times = (starts + window_size / 2.0) * dt

    segments = np.stack([signal[s:s + window_size] for s in starts], axis=0)
    segments -= np.mean(segments, axis=1, keepdims=True)
    segments *= window[None, :]

    fft_vals = np.fft.rfft(segments, axis=1)
    spec = (np.abs(fft_vals) ** 2) / (window_power / dt)
    if spec.shape[1] > 2:
        spec[:, 1:-1] *= 2.0
    freqs = np.fft.rfftfreq(window_size, d=dt)
    return freqs, center_times, spec.T


def windowed_variance(signal, window_size, overlap):
    step = max(1, window_size - overlap)
    starts = np.arange(0, len(signal) - window_size + 1, step)
    windows = np.stack([signal[s:s + window_size] for s in starts], axis=0)
    return np.var(windows, axis=1)


# ---------------------------------------------------------------------------
# Theoretical PSD helpers
# ---------------------------------------------------------------------------

def theoretical_evolutional_psd(sim, freqs, positions, wind_speeds,
                                component, times, modulation_amplitude,
                                modulation_values):
    heights = positions[:, 2]
    freqs_j = sim.backend.asarray(freqs)
    times_j = sim.backend.asarray(times)
    mod_vals_j = sim.backend.asarray(modulation_values) if modulation_values is not None else None
    T = sim.params.T

    # Use backend vmap for time-frequency scan
    if sim.backend.name == "jax":
        from jax import vmap
        def _one_time(ti, tv):
            return vmap(lambda f: sim._evolutional_psd(
                f, heights, wind_speeds, component, tv, ti, T,
                modulation_amplitude, mod_vals_j, None,
            ))(freqs_j)
        t_idx = sim.backend.xp.arange(len(times))
        result = vmap(_one_time)(t_idx, times_j)
    elif sim.backend.name == "torch":
        import torch.func as func
        def _one_time(ti, tv):
            return func.vmap(lambda f: sim._evolutional_psd(
                f, heights, wind_speeds, component, tv, ti, T,
                modulation_amplitude, mod_vals_j, None,
            ))(freqs_j)
        t_idx = sim.backend.arange(len(times))
        result = func.vmap(_one_time)(t_idx, times_j)
    else:
        result = []
        for ti, tv in enumerate(times):
            row = []
            for f in freqs:
                val = sim._evolutional_psd(
                    f, heights, wind_speeds, component, tv, ti, T,
                    modulation_amplitude, mod_vals_j, None,
                )
                row.append(np.asarray(sim.backend.to_numpy(val)))
            result.append(np.array(row))
        result = np.array(result)

    return np.asarray(sim.backend.to_numpy(result), dtype=np.float32)


def average_over_windows(target_psd, target_times, window_centers, dt, window_size):
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


def average_over_freq_bands(target_freqs, target_psd, est_freqs):
    """Bin-averaged target PSD onto estimated frequency bins."""
    edges = np.empty(len(est_freqs) + 1, dtype=np.float32)
    edges[1:-1] = 0.5 * (est_freqs[:-1] + est_freqs[1:])
    edges[0] = max(0.0, est_freqs[0] - 0.5 * (est_freqs[1] - est_freqs[0]))
    edges[-1] = est_freqs[-1] + 0.5 * (est_freqs[-1] - est_freqs[-2])

    out = np.empty((target_psd.shape[0], len(est_freqs)), dtype=np.float32)
    for ti in range(target_psd.shape[0]):
        for fi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (target_freqs >= lo) & (target_freqs < hi)
            if np.any(mask):
                bf = target_freqs[mask]
                bv = target_psd[ti, mask]
                if len(bf) == 1:
                    out[ti, fi] = bv[0]
                else:
                    out[ti, fi] = np.trapezoid(bv, bf) / (hi - lo)
            else:
                mid = 0.5 * (lo + hi)
                out[ti, fi] = np.interp(mid, target_freqs, target_psd[ti])
    return out


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def psd_error_metrics(est, tgt, skip_low=1):
    e = est[skip_low:, :].reshape(-1)
    t = tgt[skip_low:, :].reshape(-1)
    raw = float(np.linalg.norm(e - t) / np.linalg.norm(t))
    scale = float(np.dot(t, e) / np.dot(t, t))
    scaled = float(np.linalg.norm(e - scale * t) / np.linalg.norm(scale * t))
    return {"raw": raw, "scaled": scaled, "scale": scale}


def variance_error_metrics(est, tgt):
    raw = float(np.linalg.norm(est - tgt) / np.linalg.norm(tgt))
    corr = float(np.corrcoef(est, tgt)[0, 1])
    scale = float(np.dot(tgt, est) / np.dot(tgt, tgt))
    scaled = float(np.linalg.norm(est - scale * tgt) / np.linalg.norm(scale * tgt))
    return {"raw": raw, "scaled": scaled, "corr": corr, "scale": scale}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_validation(est_times, var_est, var_tgt,
                    sig_times, sig,
                    est_freqs, est_psd, tgt_psd_snap, snap_times,
                    skip_low, output_prefix, show):
    if not show:
        matplotlib.use("Agg")

    n_snap = len(snap_times)
    ncol = min(2, n_snap)
    nrow = int(np.ceil(n_snap / ncol))

    fig = plt.figure(figsize=(15, 10 + 3 * max(0, nrow - 1)))
    gs = fig.add_gridspec(2 + nrow, 2)

    # Variance envelope
    ax_var = fig.add_subplot(gs[0, :])
    ax_var.plot(est_times, var_est, color="tab:blue", lw=1.8, label="Estimated")
    ax_var.plot(est_times, var_tgt, "--", color="tab:red", lw=2.0, label="Target")
    ax_var.set_title("Local Variance Envelope")
    ax_var.set_ylabel("Variance")
    ax_var.grid(True)
    ax_var.legend()
    for tv in snap_times:
        ax_var.axvline(tv, color="0.4", linestyle=":", lw=0.9)

    # Time history with std envelope
    est_std = np.sqrt(np.maximum(var_est, 0.0))
    tgt_std = np.sqrt(np.maximum(var_tgt, 0.0))
    est_std_full = np.interp(sig_times, est_times, est_std)
    tgt_std_full = np.interp(sig_times, est_times, tgt_std)

    ax_time = fig.add_subplot(gs[1, :], sharex=ax_var)
    ax_time.plot(sig_times, sig, color="black", lw=0.9, alpha=0.8, label="Sample")
    ax_time.fill_between(sig_times, -tgt_std_full, tgt_std_full,
                         color="tab:red", alpha=0.16, label="Target std band")
    ax_time.plot(sig_times, tgt_std_full, "--", color="tab:red", lw=1.2)
    ax_time.plot(sig_times, -tgt_std_full, "--", color="tab:red", lw=1.2)
    ax_time.plot(sig_times, est_std_full, color="tab:blue", lw=1.2, alpha=0.9,
                 label="Est. std")
    ax_time.plot(sig_times, -est_std_full, color="tab:blue", lw=1.2, alpha=0.9)
    ax_time.set_title("Wind Time History with Local Std Envelope")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Wind speed")
    ax_time.grid(True)
    ax_time.legend(ncol=2, fontsize=9)
    for tv in snap_times:
        ax_time.axvline(tv, color="0.4", linestyle=":", lw=0.9)

    # PSD snapshots
    sl = slice(skip_low, None)
    for si in range(n_snap):
        ax = fig.add_subplot(gs[2 + si // ncol, si % ncol])
        ax.plot(est_freqs[sl], est_psd[sl, si], color="tab:blue", lw=1.8,
                label="Simulated")
        ax.plot(est_freqs[sl], tgt_psd_snap[si, sl], "--", color="tab:red", lw=2.0,
                label="Theory")
        ax.set_title(f"PSD at t = {snap_times[si]:.2f} s")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend(fontsize=9)

    fig.tight_layout()
    path = output_prefix.with_suffix(".png")
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Validate nonstationary simulation")
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--backend", type=str, default="jax",
                   choices=["jax", "numpy", "torch"])
    p.add_argument("--spectrum", type=str, default="kaimal")
    p.add_argument("--n-points", type=int, default=100)
    p.add_argument("--n-freqs", type=int, default=1024)
    p.add_argument("--n-realizations", type=int, default=32)
    p.add_argument("--point-index", type=int, default=0)
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--w-up", type=float, default=5.0)
    p.add_argument("--component", type=str, default="u", choices=["u", "w"])
    p.add_argument("--mode", type=str, default="chunked-vmap",
                   choices=["freq-for", "full-vmap", "chunked-vmap"])
    p.add_argument("--max-memory-gb", type=float, default=8.0)
    p.add_argument("--modulation-amplitude", type=float, default=0.2)
    p.add_argument("--skip-low-freq-bins", type=int, default=1)
    p.add_argument("--psd-snapshot-count", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="benchmark_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def main():
    args = parse_args()

    # Load config
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cfg", args.config)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.get_config().to_dict()
    else:
        cfg = {}

    backend = _first(args.backend, cfg.get("backend"), "jax")
    spectrum = _first(args.spectrum, cfg.get("spectrum"), "kaimal")
    params_cfg = cfg.get("params", {})
    spatial_cfg = cfg.get("spatial", {})
    wind_cfg = cfg.get("wind", {})
    ns_cfg = cfg.get("nonstationary", {})
    val_cfg = cfg.get("validation", {})

    n_points = _first(args.n_points, spatial_cfg.get("n_points"), 100)
    n_freqs = _first(args.n_freqs, params_cfg.get("N"), 1024)
    n_real = _first(args.n_realizations, val_cfg.get("n_realizations"), 32)
    pt_idx = _first(args.point_index, val_cfg.get("point_index"), 0)
    win_size = _first(args.window_size, val_cfg.get("window_size"), 64)
    overlap = _first(args.overlap, val_cfg.get("overlap"), 50)
    w_up = _first(args.w_up, params_cfg.get("w_up"), 5.0)
    component = _first(args.component, wind_cfg.get("component"), "u")
    mode = _first(args.mode, ns_cfg.get("mode"), "chunked-vmap")
    max_mem = _first(args.max_memory_gb, ns_cfg.get("max_memory_gb"), 8.0)
    mod_amp = _first(args.modulation_amplitude,
                     ns_cfg.get("modulation_amplitude"), 0.2)
    skip_low = _first(args.skip_low_freq_bins,
                      val_cfg.get("skip_low_freq_bins"), 1)
    snap_count = _first(args.psd_snapshot_count,
                        val_cfg.get("psd_snapshot_count"), 4)
    out_dir = Path(_first(args.output_dir, cfg.get("output_dir"), "benchmark_results"))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build simulator
    sim = NonstationaryWindSimulator(
        create_simulator(backend, spectrum, seed=args.seed,
                         N=n_freqs, w_up=w_up,
                         U_d=params_cfg.get("U_d", 20.0),
                         H_bar=params_cfg.get("H_bar", 20.0),
                         alpha_0=params_cfg.get("alpha_0", 0.12),
                         z_0=params_cfg.get("z_0", 0.01))
    )

    # Build positions
    positions = np.zeros((n_points, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1000.0, n_points)
    positions[:, 1] = 5.0
    positions[:, 2] = spatial_cfg.get("height", 35.0)
    wind_speeds = np.full(n_points, wind_cfg.get("speed", 30.0), dtype=np.float32)

    dt = sim.params.dt
    M = sim.params.M
    times = np.arange(M) * dt
    mod_vals = 1.0 + mod_amp * np.sin(2.0 * np.pi * times / sim.params.T)

    # Target theoretical PSD
    logger.info("Computing target evolutionary PSD...")
    target_freqs = np.arange(1, n_freqs + 1) * sim.params.dw - sim.params.dw / 2.0
    tgt_psd_full = theoretical_evolutional_psd(
        sim, target_freqs, positions, wind_speeds,
        component, times, mod_amp, mod_vals,
    )

    # Ensemble simulations
    spec_list, var_list = [], []
    repr_signal = None
    t0 = time.time()

    for ri in range(n_real):
        logger.info("Realization %d/%d", ri + 1, n_real)
        samples, _ = sim.simulate_nonstationary(
            positions, wind_speeds, component=component,
            mode=mode, modulation_amplitude=mod_amp,
            modulation_values=mod_vals,
            max_memory_gb=max_mem, auto_batch=True,
        )
        sig = samples[pt_idx]
        if repr_signal is None:
            repr_signal = sig.copy()

        ef, et, spec = local_spectrogram(sig, dt, win_size, overlap)
        spec_list.append(spec)
        var_list.append(windowed_variance(sig, win_size, overlap))

    est_psd = np.mean(np.stack(spec_list, axis=0), axis=0)
    est_var = np.mean(np.stack(var_list, axis=0), axis=0)
    ef_vals = ef  # from last realization

    # Target processed through same windowing
    pt_tgt = tgt_psd_full[:, :, pt_idx]
    tgt_win = average_over_windows(pt_tgt, times, et, dt, win_size)
    tgt_resampled = average_over_freq_bands(target_freqs, tgt_win, ef_vals)
    tgt_psd_comp = tgt_resampled.T

    # Snapshot times
    snap_idx = np.linspace(0, len(et) - 1, snap_count).astype(int)
    snap_times = et[snap_idx]
    snap_est = est_psd[:, snap_idx]
    snap_tgt = average_over_freq_bands(
        target_freqs,
        np.stack([pt_tgt[int(np.argmin(np.abs(times - tv)))] for tv in snap_times], axis=0),
        ef_vals,
    )

    # Target variance
    tgt_var = np.array([
        np.trapezoid(
            tgt_win[ti] * (1.0 - _window_mean_response(target_freqs, dt, win_size)),
            target_freqs,
        )
        for ti in range(len(et))
    ])

    # Error metrics
    psd_err = psd_error_metrics(est_psd, tgt_psd_comp, skip_low)
    var_err = variance_error_metrics(est_var, tgt_var)

    # Plot
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = out_dir / f"validation_{backend}_{mode}_{ts}"
    fig_path = plot_validation(
        et, est_var, tgt_var,
        times, repr_signal,
        ef_vals, snap_est, snap_tgt,
        snap_times, skip_low, prefix, args.show,
    )

    # Save summary
    csv_path = prefix.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "backend", "mode", "n_points", "n_freqs", "n_realizations",
            "psd_error_raw", "psd_error_scaled", "psd_scale",
            "var_error_raw", "var_error_scaled", "var_corr", "var_scale",
            "elapsed_s", "figure",
        ])
        w.writeheader()
        w.writerow({
            "backend": backend, "mode": mode,
            "n_points": n_points, "n_freqs": n_freqs,
            "n_realizations": n_real,
            "psd_error_raw": f"{psd_err['raw']:.6e}",
            "psd_error_scaled": f"{psd_err['scaled']:.6e}",
            "psd_scale": f"{psd_err['scale']:.6e}",
            "var_error_raw": f"{var_err['raw']:.6e}",
            "var_error_scaled": f"{var_err['scaled']:.6e}",
            "var_corr": f"{var_err['corr']:.6e}",
            "var_scale": f"{var_err['scale']:.6e}",
            "elapsed_s": f"{time.time() - t0:.3f}",
            "figure": str(fig_path),
        })

    logger.info("Validation done in %.1f s", time.time() - t0)
    logger.info("PSD error: raw=%.4e  scaled=%.4e (x%.2f)",
                psd_err["raw"], psd_err["scaled"], psd_err["scale"])
    logger.info("Var error: raw=%.4e  scaled=%.4e  corr=%.4f",
                var_err["raw"], var_err["scaled"], var_err["corr"])
    logger.info("Figure: %s", fig_path)
    logger.info("Summary: %s", csv_path)


def _window_mean_response(freqs, dt, win_size):
    """Window mean-response for target variance."""
    nf = np.asarray(freqs, dtype=np.float64) * dt
    k = np.exp(-1j * np.pi * (win_size - 1) * nf)
    k = k * win_size * (np.sinc(win_size * nf) / np.sinc(nf))
    k[np.isclose(nf, 0.0)] = float(win_size) + 0.0j
    return np.abs(k / win_size) ** 2


if __name__ == "__main__":
    main()
