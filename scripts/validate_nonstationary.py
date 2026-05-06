import argparse
import csv
import logging
import time
from pathlib import Path

import matplotlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap

from stochastic_wind_simulate.jax_backend.simulator_nonstationary import JaxNonstationaryWindSimulator


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def build_positions(n_points, height=10.0):
    positions = np.zeros((n_points, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1000.0, n_points)
    positions[:, 1] = 5.0
    positions[:, 2] = height
    return jnp.asarray(positions)


def compute_local_spectrogram(signal, dt, window_size, overlap):
    step = max(1, window_size - overlap)
    if signal.shape[0] < window_size:
        raise ValueError("window_size must not exceed the time series length")

    window = np.hanning(window_size).astype(np.float32)
    window_power = np.sum(window ** 2)
    starts = np.arange(0, signal.shape[0] - window_size + 1, step)
    center_times = (starts + window_size / 2.0) * dt

    segments = np.stack([signal[start : start + window_size] for start in starts], axis=0)
    detrended = segments - np.mean(segments, axis=1, keepdims=True)
    tapered = detrended * window[None, :]

    fft_values = np.fft.rfft(tapered, axis=1)
    spectrogram = (np.abs(fft_values) ** 2) / (window_power / dt)
    if spectrogram.shape[1] > 2:
        spectrogram[:, 1:-1] *= 2.0
    freqs = np.fft.rfftfreq(window_size, d=dt)
    return freqs, center_times, spectrogram.T


def compute_windowed_variance(signal, window_size, overlap):
    step = max(1, window_size - overlap)
    starts = np.arange(0, signal.shape[0] - window_size + 1, step)
    windows = np.stack([signal[start : start + window_size] for start in starts], axis=0)
    return np.var(windows, axis=1)


def compute_theoretical_evolutional_psd(
    simulator,
    frequencies,
    positions,
    wind_speeds,
    component,
    times,
    modulation_amplitude=0.2,
    modulation_values=None,
    evolution_psd_generator=None,
):
    heights = positions[:, 2]
    frequencies = jnp.asarray(frequencies, dtype=jnp.float32)
    times = jnp.asarray(times, dtype=jnp.float32)
    if modulation_values is not None:
        modulation_values = jnp.asarray(modulation_values, dtype=jnp.float32)

    def _single_time(time_index, time_value):
        return vmap(
            lambda freq: simulator._calculate_evolutional_power_spectrum(
                freq,
                heights,
                wind_speeds,
                component,
                time_value=time_value,
                total_time=simulator.params["T"],
                modulation_amplitude=modulation_amplitude,
                modulation_values=modulation_values,
                time_index=time_index,
                evolution_psd_generator=evolution_psd_generator,
            )
        )(frequencies)

    time_indices = jnp.arange(times.shape[0])
    return np.asarray(vmap(_single_time)(time_indices, times), dtype=np.float32)


def average_target_over_windows(target_psd, target_times, window_centers, dt, window_size):
    half_window = 0.5 * window_size * dt
    averaged = []
    for center in window_centers:
        mask = (target_times >= center - half_window) & (target_times < center + half_window)
        if not np.any(mask):
            nearest_idx = int(np.argmin(np.abs(target_times - center)))
            averaged.append(target_psd[nearest_idx])
        else:
            averaged.append(np.mean(target_psd[mask], axis=0))
    return np.stack(averaged, axis=0)


def average_target_over_frequency_bands(target_freqs, target_psd, estimate_freqs):
    if estimate_freqs.shape[0] < 2:
        raise ValueError("estimate_freqs must contain at least two frequencies")

    band_edges = np.empty(estimate_freqs.shape[0] + 1, dtype=np.float32)
    band_edges[1:-1] = 0.5 * (estimate_freqs[:-1] + estimate_freqs[1:])
    band_edges[0] = max(0.0, estimate_freqs[0] - 0.5 * (estimate_freqs[1] - estimate_freqs[0]))
    band_edges[-1] = estimate_freqs[-1] + 0.5 * (estimate_freqs[-1] - estimate_freqs[-2])

    averaged = np.empty((target_psd.shape[0], estimate_freqs.shape[0]), dtype=np.float32)
    for time_idx in range(target_psd.shape[0]):
        for freq_idx, (left_edge, right_edge) in enumerate(zip(band_edges[:-1], band_edges[1:])):
            mask = (target_freqs >= left_edge) & (target_freqs < right_edge)
            if np.any(mask):
                band_freqs = target_freqs[mask]
                band_values = target_psd[time_idx, mask]
                if band_freqs.shape[0] == 1:
                    averaged[time_idx, freq_idx] = band_values[0]
                else:
                    averaged[time_idx, freq_idx] = np.trapezoid(band_values, band_freqs) / (right_edge - left_edge)
            else:
                averaged[time_idx, freq_idx] = np.interp(
                    0.5 * (left_edge + right_edge),
                    target_freqs,
                    target_psd[time_idx],
                )
    return averaged


def compute_window_mean_response(frequencies, dt, window_size):
    normalized_frequency = np.asarray(frequencies, dtype=np.float64) * dt
    kernel = np.exp(-1j * np.pi * (window_size - 1) * normalized_frequency)
    kernel = kernel * window_size * (np.sinc(window_size * normalized_frequency) / np.sinc(normalized_frequency))
    kernel[np.isclose(normalized_frequency, 0.0)] = float(window_size) + 0.0j
    return np.abs(kernel / window_size) ** 2


def compute_target_windowed_variance(target_freqs, target_psd, dt, window_size):
    mean_response = compute_window_mean_response(target_freqs, dt, window_size)
    variance_response = 1.0 - mean_response
    return np.trapezoid(target_psd * variance_response[None, :], target_freqs, axis=1)


def compute_psd_error_metrics(estimated_psd, target_psd, skip_low_freq_bins=1):
    if skip_low_freq_bins < 0 or skip_low_freq_bins >= estimated_psd.shape[0]:
        raise ValueError("skip_low_freq_bins must be in [0, n_freq_bins - 1)")

    estimated_slice = estimated_psd[skip_low_freq_bins:, :]
    target_slice = target_psd[skip_low_freq_bins:, :]
    estimated_flat = estimated_slice.reshape(-1)
    target_flat = target_slice.reshape(-1)

    raw_error = float(
        np.linalg.norm(estimated_flat - target_flat) / np.linalg.norm(target_flat)
    )

    scale = float(np.dot(target_flat, estimated_flat) / np.dot(target_flat, target_flat))
    scaled_target = scale * target_slice
    scaled_error = float(
        np.linalg.norm(estimated_slice - scaled_target) / np.linalg.norm(scaled_target)
    )

    design_matrix = np.stack([target_flat, np.ones_like(target_flat)], axis=1)
    affine_scale, affine_bias = np.linalg.lstsq(design_matrix, estimated_flat, rcond=None)[0]
    affine_target = affine_scale * target_slice + affine_bias
    affine_error = float(
        np.linalg.norm(estimated_slice - affine_target) / np.linalg.norm(affine_target)
    )

    return {
        "raw_error": raw_error,
        "scale_aligned_error": scaled_error,
        "affine_aligned_error": affine_error,
        "scale_factor": scale,
        "affine_scale_factor": float(affine_scale),
        "affine_bias": float(affine_bias),
    }


def compute_variance_error_metrics(estimated_variance, target_variance):
    relative_error = float(
        np.linalg.norm(estimated_variance - target_variance) / np.linalg.norm(target_variance)
    )
    correlation = float(np.corrcoef(estimated_variance, target_variance)[0, 1])
    scale = float(np.dot(target_variance, estimated_variance) / np.dot(target_variance, target_variance))
    scaled_target = scale * target_variance
    scaled_error = float(
        np.linalg.norm(estimated_variance - scaled_target) / np.linalg.norm(scaled_target)
    )
    return {
        "raw_error": relative_error,
        "scale_aligned_error": scaled_error,
        "correlation": correlation,
        "scale_factor": scale,
    }


def interpolate_envelope_to_signal_times(signal_times, envelope_times, envelope_values):
    return np.interp(signal_times, envelope_times, envelope_values)


def select_snapshot_indices(window_times, snapshot_count, snapshot_times=None):
    if window_times.size == 0:
        raise ValueError("window_times must not be empty")

    if snapshot_times:
        selected = []
        for time_value in snapshot_times:
            nearest_index = int(np.argmin(np.abs(window_times - time_value)))
            selected.append(nearest_index)
        return sorted(set(selected))

    snapshot_count = max(1, min(int(snapshot_count), int(window_times.size)))
    if snapshot_count == 1:
        return [int(window_times.size // 2)]

    raw_indices = np.linspace(0, window_times.size - 1, snapshot_count)
    return sorted(set(int(round(index)) for index in raw_indices))


def sample_theoretical_psd_at_times(target_psd, target_times, snapshot_times):
    selected_spectra = []
    for time_value in snapshot_times:
        nearest_index = int(np.argmin(np.abs(target_times - time_value)))
        selected_spectra.append(target_psd[nearest_index])
    return np.stack(selected_spectra, axis=0)


def plot_validation(
    estimate_times,
    variance_estimate,
    variance_target,
    sample_times,
    sample_signal,
    estimate_freqs,
    estimated_psd,
    theoretical_psd_snapshots,
    snapshot_times,
    skip_low_freq_bins,
    output_prefix,
    show,
):
    if not show and plt.get_backend().lower() != "agg":
        plt.switch_backend("Agg")

    snapshot_count = len(snapshot_times)
    snapshot_cols = min(2, snapshot_count)
    snapshot_rows = int(np.ceil(snapshot_count / snapshot_cols))
    fig = plt.figure(figsize=(15, 10 + 3 * max(0, snapshot_rows - 1)))
    grid = fig.add_gridspec(2 + snapshot_rows, 2)
    variance_axis = fig.add_subplot(grid[0, :])
    time_axis = fig.add_subplot(grid[1, :], sharex=variance_axis)

    snapshot_axes = []
    for snapshot_idx in range(snapshot_count):
        row_offset = 2 + snapshot_idx // snapshot_cols
        col_index = snapshot_idx % snapshot_cols
        snapshot_axes.append(fig.add_subplot(grid[row_offset, col_index]))

    variance_axis.plot(estimate_times, variance_estimate, color="tab:blue", linewidth=1.8, label="Estimated local variance")
    variance_axis.plot(estimate_times, variance_target, "--", color="tab:red", linewidth=2.0, label="Target local variance")
    variance_axis.set_title("Local Variance Envelope")
    variance_axis.set_ylabel("Variance")
    variance_axis.grid(True)
    variance_axis.legend()
    for time_value in snapshot_times:
        variance_axis.axvline(time_value, color="0.4", linestyle=":", linewidth=0.9, alpha=0.8)

    estimated_std = np.sqrt(np.maximum(variance_estimate, 0.0))
    target_std = np.sqrt(np.maximum(variance_target, 0.0))
    estimated_std_full = interpolate_envelope_to_signal_times(sample_times, estimate_times, estimated_std)
    target_std_full = interpolate_envelope_to_signal_times(sample_times, estimate_times, target_std)

    time_axis.plot(sample_times, sample_signal, color="black", linewidth=0.9, alpha=0.8, label="Sample wind speed")
    time_axis.fill_between(sample_times, -target_std_full, target_std_full, color="tab:red", alpha=0.16, label="Target local std band")
    time_axis.plot(sample_times, target_std_full, "--", color="tab:red", linewidth=1.2)
    time_axis.plot(sample_times, -target_std_full, "--", color="tab:red", linewidth=1.2)
    time_axis.plot(sample_times, estimated_std_full, color="tab:blue", linewidth=1.2, alpha=0.9, label="+ estimated local std")
    time_axis.plot(sample_times, -estimated_std_full, color="tab:blue", linewidth=1.2, alpha=0.9, label="- estimated local std")
    time_axis.set_title("Wind Time History With Local Std Envelope")
    time_axis.set_xlabel("Time (s)")
    time_axis.set_ylabel("Wind speed")
    time_axis.grid(True)
    time_axis.legend(ncol=2, fontsize=9)
    for time_value in snapshot_times:
        time_axis.axvline(time_value, color="0.4", linestyle=":", linewidth=0.9, alpha=0.8)

    for snapshot_idx, axis in enumerate(snapshot_axes):
        plot_slice = slice(skip_low_freq_bins, None)
        axis.plot(
            estimate_freqs[plot_slice],
            estimated_psd[plot_slice, snapshot_idx],
            color="tab:blue",
            linewidth=1.8,
            label="Simulated window PSD",
        )
        axis.plot(
            estimate_freqs[plot_slice],
            theoretical_psd_snapshots[snapshot_idx, plot_slice],
            "--",
            color="tab:red",
            linewidth=2.0,
            label="Theoretical current PSD",
        )
        axis.set_title(f"PSD at t = {snapshot_times[snapshot_idx]:.2f} s")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("PSD")
        axis.set_yscale("log")
        axis.grid(True)
        axis.legend(fontsize=9)

    for axis in snapshot_axes[snapshot_count:]:
        axis.axis("off")

    fig.tight_layout()
    figure_path = output_prefix.with_suffix(".png")
    fig.savefig(figure_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return figure_path


def main():
    parser = argparse.ArgumentParser(description="Validate JAX nonstationary wind simulation against target evolutional PSD")
    parser.add_argument("--n-points", type=int, default=100, help="Number of spatial points")
    parser.add_argument("--n-freqs", type=int, default=1024, help="Number of frequency components")
    parser.add_argument("--n-realizations", type=int, default=32, help="Number of realizations for ensemble averaging")
    parser.add_argument("--point-index", type=int, default=0, help="Spatial point index to validate")
    parser.add_argument("--window-size", type=int, default=64, help="Window size for local spectrum estimation")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap for local spectrum estimation")
    parser.add_argument("--w-up", type=float, default=5.0, help="Cutoff frequency")
    parser.add_argument("--component", type=str, default="u", choices=["u", "w"], help="Wind component")
    parser.add_argument("--mode", type=str, default="chunked-vmap", choices=["freq-for", "full-vmap", "chunked-vmap"], help="Execution mode")
    parser.add_argument("--max-memory-gb", type=float, default=8.0, help="Memory budget for nonstationary chunking")
    parser.add_argument("--freq-batch-size", type=int, default=None, help="Manual frequency chunk size")
    parser.add_argument("--time-batch-size", type=int, default=None, help="Manual time chunk size")
    parser.add_argument("--modulation-amplitude", type=float, default=0.2, help="Sinusoidal modulation amplitude")
    parser.add_argument("--skip-low-freq-bins", type=int, default=1, help="Number of lowest frequency bins to exclude from PSD error metrics")
    parser.add_argument("--variance-low-freq-bins", type=int, default=2, help="Number of lowest frequency bins to exclude when deriving target local variance")
    parser.add_argument("--psd-snapshot-count", type=int, default=4, help="Number of representative times to plot for PSD comparison")
    parser.add_argument("--psd-snapshot-times", type=float, nargs="*", default=None, help="Explicit times in seconds for PSD comparison snapshots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    simulator = JaxNonstationaryWindSimulator(key=42, spectrum_type="kaimal-nd")
    simulator.update_parameters(
        U_d=20.0,
        H_bar=20.0,
        alpha_0=0.12,
        z_0=0.01,
        w_up=args.w_up,
        N=args.n_freqs,
        M=args.n_freqs * 2,
    )

    positions = build_positions(args.n_points)
    wind_speeds = jnp.full((args.n_points,), 30.0, dtype=jnp.float32)
    dt = simulator.params["dt"]
    times = np.asarray(jnp.arange(simulator.params["M"]) * dt)
    modulation_values = 1.0 + args.modulation_amplitude * np.sin(2 * np.pi * times / simulator.params["T"])
    target_freqs = np.asarray(simulator.calculate_simulation_frequency(simulator.params["N"], simulator.params["dw"]))
    logging.info("Building target PSD from evolutional PSD generator")
    target_psd_full = compute_theoretical_evolutional_psd(
        simulator,
        target_freqs,
        positions,
        wind_speeds,
        args.component,
        times,
        modulation_amplitude=args.modulation_amplitude,
        modulation_values=modulation_values,
    )

    realization_spectra = []
    realization_variances = []
    representative_signal = None
    run_start = time.time()
    for realization_idx in range(args.n_realizations):
        logging.info("Simulating realization %d/%d", realization_idx + 1, args.n_realizations)
        samples, _ = simulator.simulate_wind_nonstationary(
            positions,
            wind_speeds,
            component=args.component,
            mode=args.mode,
            modulation_amplitude=args.modulation_amplitude,
            modulation_values=modulation_values,
            max_memory_gb=args.max_memory_gb,
            freq_batch_size=args.freq_batch_size,
            time_batch_size=args.time_batch_size,
            auto_batch=True,
        )
        signal = np.asarray(samples[args.point_index])
        if representative_signal is None:
            representative_signal = signal.copy()
        estimate_freqs, estimate_times, spectrogram = compute_local_spectrogram(
            signal,
            dt,
            args.window_size,
            args.overlap,
        )
        realization_spectra.append(spectrogram)
        realization_variances.append(compute_windowed_variance(signal, args.window_size, args.overlap))

    estimated_psd = np.mean(np.stack(realization_spectra, axis=0), axis=0)
    estimated_variance = np.mean(np.stack(realization_variances, axis=0), axis=0)

    point_target_psd = target_psd_full[:, :, args.point_index]
    target_psd_windowed = average_target_over_windows(point_target_psd, times, estimate_times, dt, args.window_size)
    target_psd_resampled = average_target_over_frequency_bands(target_freqs, target_psd_windowed, estimate_freqs)
    target_psd_for_comparison = target_psd_resampled.T
    snapshot_indices = select_snapshot_indices(estimate_times, args.psd_snapshot_count, args.psd_snapshot_times)
    snapshot_times = estimate_times[snapshot_indices]
    snapshot_estimated_psd = estimated_psd[:, snapshot_indices]
    snapshot_target_psd = sample_theoretical_psd_at_times(point_target_psd, times, snapshot_times)
    snapshot_target_resampled = average_target_over_frequency_bands(target_freqs, snapshot_target_psd, estimate_freqs)
    target_variance = compute_target_windowed_variance(
        target_freqs,
        target_psd_windowed,
        dt,
        args.window_size,
    )

    psd_error_metrics = compute_psd_error_metrics(
        estimated_psd,
        target_psd_for_comparison,
        skip_low_freq_bins=args.skip_low_freq_bins,
    )
    variance_error_metrics = compute_variance_error_metrics(estimated_variance, target_variance)

    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_prefix = output_dir / f"nonstationary_validation_{args.mode}_{timestamp}"
    figure_path = plot_validation(
        estimate_times,
        estimated_variance,
        target_variance,
        times,
        representative_signal,
        estimate_freqs,
        snapshot_estimated_psd,
        snapshot_target_resampled,
        snapshot_times,
        args.skip_low_freq_bins,
        output_prefix,
        args.show,
    )

    summary_path = output_prefix.with_suffix(".csv")
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "mode",
                "n_points",
                "n_freqs",
                "n_realizations",
                "point_index",
                "window_size",
                "overlap",
                "skip_low_freq_bins",
                "variance_low_freq_bins",
                "psd_snapshot_count",
                "psd_snapshot_times_s",
                "relative_psd_error_raw",
                "relative_psd_error_scaled",
                "relative_psd_error_affine",
                "psd_scale_factor",
                "psd_affine_scale_factor",
                "psd_affine_bias",
                "relative_variance_error_raw",
                "relative_variance_error_scaled",
                "variance_correlation",
                "variance_scale_factor",
                "elapsed_s",
                "figure_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "mode": args.mode,
                "n_points": args.n_points,
                "n_freqs": args.n_freqs,
                "n_realizations": args.n_realizations,
                "point_index": args.point_index,
                "window_size": args.window_size,
                "overlap": args.overlap,
                "skip_low_freq_bins": args.skip_low_freq_bins,
                "variance_low_freq_bins": args.variance_low_freq_bins,
                "psd_snapshot_count": len(snapshot_indices),
                "psd_snapshot_times_s": ";".join(f"{value:.6f}" for value in snapshot_times),
                "relative_psd_error_raw": f"{psd_error_metrics['raw_error']:.6e}",
                "relative_psd_error_scaled": f"{psd_error_metrics['scale_aligned_error']:.6e}",
                "relative_psd_error_affine": f"{psd_error_metrics['affine_aligned_error']:.6e}",
                "psd_scale_factor": f"{psd_error_metrics['scale_factor']:.6e}",
                "psd_affine_scale_factor": f"{psd_error_metrics['affine_scale_factor']:.6e}",
                "psd_affine_bias": f"{psd_error_metrics['affine_bias']:.6e}",
                "relative_variance_error_raw": f"{variance_error_metrics['raw_error']:.6e}",
                "relative_variance_error_scaled": f"{variance_error_metrics['scale_aligned_error']:.6e}",
                "variance_correlation": f"{variance_error_metrics['correlation']:.6e}",
                "variance_scale_factor": f"{variance_error_metrics['scale_factor']:.6e}",
                "elapsed_s": f"{time.time() - run_start:.6f}",
                "figure_path": str(figure_path),
            }
        )

    logging.info("Validation completed in %.2f seconds", time.time() - run_start)
    logging.info("Relative PSD error (raw): %.4e", psd_error_metrics["raw_error"])
    logging.info("Relative PSD error (scaled): %.4e with alpha=%.4f", psd_error_metrics["scale_aligned_error"], psd_error_metrics["scale_factor"])
    logging.info(
        "Relative PSD error (affine): %.4e with alpha=%.4f, beta=%.4e",
        psd_error_metrics["affine_aligned_error"],
        psd_error_metrics["affine_scale_factor"],
        psd_error_metrics["affine_bias"],
    )
    logging.info(
        "Relative variance error (raw): %.4e",
        variance_error_metrics["raw_error"],
    )
    logging.info(
        "Relative variance error (scaled): %.4e with alpha=%.4f",
        variance_error_metrics["scale_aligned_error"],
        variance_error_metrics["scale_factor"],
    )
    logging.info("Variance envelope correlation: %.4f", variance_error_metrics["correlation"])
    logging.info("Saved figure to %s", figure_path)
    logging.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
