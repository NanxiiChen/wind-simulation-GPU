#!/usr/bin/env python
"""Validate nonstationary simulation against theoretical evolutionary PSD.

Uses ``ml_collections`` + ``absl.flags``.  No argparse — everything
through ``--config.key=value``.

Examples
--------
    python scripts/validate.py --config=configs/validate.py
    python scripts/validate.py --config=configs/validate.py \
        --config.params.N=512 --config.validation.n_realizations=16
"""

import csv
import logging
import sys
import time
from pathlib import Path

from absl import app
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_collections import ConfigDict, config_flags
from stochastic_wind_simulate import (
    NonstationaryWindSimulator, WindVisualizer, create_simulator,
)

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "Path to config file", lock_config=False,
)
FLAGS = app.flags.FLAGS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = ConfigDict(dict(
    backend="jax", spectrum="kaimal", seed=42,
    params=dict(U_d=20.0, H_bar=20.0, z_0=0.01, alpha_0=0.12, w_up=5.0, N=1024),
    spatial=dict(n_points=100, height=35.0),
    wind=dict(speed=30.0, component="u"),
    nonstationary=dict(mode="chunked-vmap", modulation_amplitude=0.2, max_memory_gb=8.0),
    validation=dict(
        n_realizations=32, point_index=0,
        window_size=64, overlap=50, skip_low_freq_bins=1,
        psd_snapshot_count=4,
    ),
    output_dir="output", show=False,
))

# Aliases to visualizer methods
_local_spectrogram = WindVisualizer._local_spectrogram
_local_variance = WindVisualizer._local_variance
_average_over_windows = WindVisualizer._average_over_windows
_average_over_freq_bands = WindVisualizer._average_over_freq_bands
_window_mean_response = WindVisualizer._window_mean_response


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


def plot_validation(est_times, var_est, var_tgt, sig_times, sig,
                    est_freqs, est_psd, tgt_psd_snap, snap_times,
                    skip_low, output_prefix, show):
    if not show:
        matplotlib.use("Agg")

    n_snap = len(snap_times)
    ncol = min(2, n_snap)
    nrow = int(np.ceil(n_snap / ncol))

    fig = plt.figure(figsize=(15, 10 + 3 * max(0, nrow - 1)))
    gs = fig.add_gridspec(2 + nrow, 2)

    ax_var = fig.add_subplot(gs[0, :])
    ax_var.plot(est_times, var_est, color="tab:blue", lw=1.8, label="Estimated")
    ax_var.plot(est_times, var_tgt, "--", color="tab:red", lw=2.0, label="Target")
    ax_var.set_title("Local Variance Envelope")
    ax_var.set_ylabel("Variance"); ax_var.grid(True); ax_var.legend()
    for tv in snap_times:
        ax_var.axvline(tv, color="0.4", linestyle=":", lw=0.9)

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
    ax_time.plot(sig_times, est_std_full, color="tab:blue", lw=1.2, alpha=0.9, label="Est. std")
    ax_time.plot(sig_times, -est_std_full, color="tab:blue", lw=1.2, alpha=0.9)
    ax_time.set_title("Wind Time History with Local Std Envelope")
    ax_time.set_xlabel("Time (s)"); ax_time.set_ylabel("Wind speed")
    ax_time.grid(True); ax_time.legend(ncol=2, fontsize=9)
    for tv in snap_times:
        ax_time.axvline(tv, color="0.4", linestyle=":", lw=0.9)

    sl = slice(skip_low, None)
    for si in range(n_snap):
        ax = fig.add_subplot(gs[2 + si // ncol, si % ncol])
        ax.plot(est_freqs[sl], est_psd[sl, si], color="tab:blue", lw=1.8, label="Simulated")
        ax.plot(est_freqs[sl], tgt_psd_snap[si, sl], "--", color="tab:red", lw=2.0, label="Theory")
        ax.set_title(f"PSD at t = {snap_times[si]:.2f} s")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
        ax.set_yscale("log"); ax.grid(True); ax.legend(fontsize=9)

    fig.tight_layout()
    path = output_prefix.with_suffix(".png")
    fig.savefig(path, dpi=150)
    if show: plt.show()
    else: plt.close(fig)
    return path


def main(_):
    cfg = _CONFIG.value or _DEFAULT_CONFIG
    backend = cfg.backend
    spectrum = cfg.spectrum
    params = cfg.params
    spatial = cfg.spatial
    wind = cfg.wind
    ns_cfg = cfg.nonstationary
    val = cfg.validation

    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sim = NonstationaryWindSimulator(
        create_simulator(backend, spectrum, seed=cfg.seed,
                         N=params.N, w_up=params.w_up,
                         U_d=params.U_d, H_bar=params.H_bar,
                         z_0=params.z_0, alpha_0=params.alpha_0))

    positions = np.zeros((spatial.n_points, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1000.0, spatial.n_points)
    positions[:, 1] = 5.0; positions[:, 2] = spatial.height
    wind_speeds = np.full(spatial.n_points, wind.speed, dtype=np.float32)

    dt = sim.params.dt; M = sim.params.M
    times = np.arange(M) * dt
    mod_vals = 1.0 + ns_cfg.modulation_amplitude * np.sin(
        2.0 * np.pi * times / sim.params.T)

    logger.info("Computing target evolutionary PSD...")
    target_freqs = np.arange(1, params.N + 1) * sim.params.dw - sim.params.dw / 2.0
    viz = WindVisualizer(sim)
    tgt_psd_full = viz._compute_evolutionary_psd_full(
        target_freqs, positions[:, 2], wind_speeds,
        wind.component, mod_vals)

    spec_list, var_list = [], []
    repr_signal = None
    t0 = time.time()

    for ri in range(val.n_realizations):
        logger.info("Realization %d/%d", ri + 1, val.n_realizations)
        samples, _ = sim.simulate_nonstationary(
            positions, wind_speeds, component=wind.component,
            mode=ns_cfg.mode, modulation_amplitude=ns_cfg.modulation_amplitude,
            modulation_values=mod_vals,
            max_memory_gb=ns_cfg.max_memory_gb, auto_batch=True,
        )
        sig = samples[val.point_index]
        if repr_signal is None:
            repr_signal = sig.copy()

        ef, et, spec = _local_spectrogram(sig, dt, val.window_size, val.overlap)
        spec_list.append(spec)
        var_list.append(_local_variance(sig, val.window_size, val.overlap))

    est_psd = np.mean(np.stack(spec_list, axis=0), axis=0)
    est_var = np.mean(np.stack(var_list, axis=0), axis=0)

    pt_tgt = tgt_psd_full[:, :, val.point_index]
    tgt_win = _average_over_windows(pt_tgt, times, et, dt, val.window_size)
    tgt_resampled = _average_over_freq_bands(target_freqs, tgt_win, ef)
    tgt_psd_comp = tgt_resampled.T

    snap_idx = np.linspace(0, len(et) - 1, val.psd_snapshot_count).astype(int)
    snap_times = et[snap_idx]
    snap_est = est_psd[:, snap_idx]
    snap_tgt = _average_over_freq_bands(
        target_freqs,
        np.stack([pt_tgt[int(np.argmin(np.abs(times - tv)))]
                  for tv in snap_times], axis=0), ef)

    mean_resp = _window_mean_response(target_freqs, dt, val.window_size)
    tgt_var = np.array([np.trapezoid(
        tgt_win[ti] * (1.0 - mean_resp), target_freqs)
        for ti in range(len(et))])

    psd_err = psd_error_metrics(est_psd, tgt_psd_comp, val.skip_low_freq_bins)
    var_err = variance_error_metrics(est_var, tgt_var)

    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = out_dir / f"validation_{backend}_{ns_cfg.mode}_{ts}"
    fig_path = plot_validation(
        et, est_var, tgt_var, times, repr_signal,
        ef, snap_est, snap_tgt, snap_times,
        val.skip_low_freq_bins, prefix, cfg.get("show", False))

    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "backend", "mode", "n_points", "n_freqs", "n_realizations",
            "psd_error_raw", "psd_error_scaled", "psd_scale",
            "var_error_raw", "var_error_scaled", "var_corr", "var_scale",
            "elapsed_s", "figure",
        ])
        w.writeheader()
        w.writerow(dict(
            backend=backend, mode=ns_cfg.mode,
            n_points=spatial.n_points, n_freqs=params.N,
            n_realizations=val.n_realizations,
            psd_error_raw=f"{psd_err['raw']:.6e}",
            psd_error_scaled=f"{psd_err['scaled']:.6e}",
            psd_scale=f"{psd_err['scale']:.6e}",
            var_error_raw=f"{var_err['raw']:.6e}",
            var_error_scaled=f"{var_err['scaled']:.6e}",
            var_corr=f"{var_err['corr']:.6e}",
            var_scale=f"{var_err['scale']:.6e}",
            elapsed_s=f"{time.time() - t0:.3f}", figure=str(fig_path),
        ))

    logger.info("Validation done in %.1f s", time.time() - t0)
    logger.info("PSD error: raw=%.4e  scaled=%.4e (x%.2f)",
                psd_err["raw"], psd_err["scaled"], psd_err["scale"])
    logger.info("Var error: raw=%.4e  scaled=%.4e  corr=%.4f",
                var_err["raw"], var_err["scaled"], var_err["corr"])
    logger.info("Figure: %s", fig_path)
    logger.info("Summary: %s", csv_path)


if __name__ == "__main__":
    app.run(main)
