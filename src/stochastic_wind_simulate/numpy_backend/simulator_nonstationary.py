import time
from importlib import import_module

import numpy as np
from scipy.linalg import cholesky

from .simulator import NumpyWindSimulator

try:
    njit = import_module("numba").njit

    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    NUMBA_AVAILABLE = False


@njit(cache=True)
def _numba_calculate_power_spectrum(freq, heights, mean_wind_speeds, z_0, z_d, K, alpha_0, component_code, spectrum_code):
    n_points = heights.shape[0]
    spectrum_values = np.empty(n_points, dtype=np.float64)

    for point_idx in range(n_points):
        height = heights[point_idx]
        mean_speed = max(mean_wind_speeds[point_idx], 1e-8)
        u_star = K * mean_speed / np.log((height - z_d) / z_0)
        reduced_frequency = freq * height / mean_speed

        if component_code == 0:
            if spectrum_code == 0:
                spectrum_values[point_idx] = (u_star ** 2 / freq) * (
                    200.0 * reduced_frequency / ((1.0 + 50.0 * reduced_frequency) ** (5.0 / 3.0))
                )
            elif spectrum_code == 2:
                spectrum_values[point_idx] = (u_star ** 2 / freq) * (
                    105.0 * reduced_frequency / ((0.44 + 33.0 * reduced_frequency) ** (5.0 / 3.0))
                )
            else:
                spectrum_values[point_idx] = 0.0
        else:
            if spectrum_code == 1:
                spectrum_values[point_idx] = (u_star ** 2 / freq) * (
                    6.0 * reduced_frequency / ((1.0 + 4.0 * reduced_frequency) ** 2.0)
                )
            elif spectrum_code == 2:
                spectrum_values[point_idx] = (u_star ** 2 / freq) * (
                    2.0 * reduced_frequency / ((1.0 + 5.3 * reduced_frequency) ** (5.0 / 3.0))
                )
            else:
                spectrum_values[point_idx] = 0.0

    return spectrum_values


@njit(cache=True)
def _numba_compute_single_frequency(
    freq_l,
    time_chunk,
    time_indices,
    phi_l,
    heights,
    wind_speeds,
    distance_term,
    total_time,
    modulation_amplitude,
    modulation_values,
    use_modulation_values,
    z_0,
    z_d,
    K,
    alpha_0,
    component_code,
    spectrum_code,
):
    n_points = heights.shape[0]
    phase_vector = np.exp(1j * phi_l)
    identity = np.eye(n_points, dtype=np.float64)
    frequency_result = np.empty((time_chunk.shape[0], n_points), dtype=np.complex128)

    for local_idx in range(time_chunk.shape[0]):
        time_value = time_chunk[local_idx]
        time_index = time_indices[local_idx]
        if use_modulation_values:
            modulation = modulation_values[time_index]
        else:
            modulation = 1.0 + modulation_amplitude * np.sin(2.0 * np.pi * time_value / total_time)

        mean_wind_speeds = wind_speeds * modulation
        spectrum_values = _numba_calculate_power_spectrum(
            freq_l,
            heights,
            mean_wind_speeds,
            z_0,
            z_d,
            K,
            alpha_0,
            component_code,
            spectrum_code,
        )

        csd_matrix = np.empty((n_points, n_points), dtype=np.float64)
        for row_idx in range(n_points):
            s_i = max(spectrum_values[row_idx], 0.0)
            for col_idx in range(n_points):
                s_j = max(spectrum_values[col_idx], 0.0)
                denominator = mean_wind_speeds[row_idx] + mean_wind_speeds[col_idx]
                safe_denominator = max(denominator, 1e-8)
                coherence = np.exp(-2.0 * freq_l * distance_term[row_idx, col_idx] / safe_denominator)
                csd_matrix[row_idx, col_idx] = np.sqrt(s_i * s_j) * coherence

        csd_matrix = 0.5 * (csd_matrix + csd_matrix.T)
        h_matrix = np.linalg.cholesky(csd_matrix + identity * 1e-12)
        complex_h_matrix = h_matrix.astype(np.complex128)
        phase_scale = np.exp(1j * 2.0 * np.pi * freq_l * time_value)
        frequency_result[local_idx] = (complex_h_matrix @ phase_vector) * phase_scale

    return frequency_result


class NumpyNonstationaryWindSimulator(NumpyWindSimulator):
    """NumPy nonstationary wind simulator with evolutionary PSD support."""

    def _get_numba_spectrum_code(self):
        """Return an integer code for supported built-in spectra or None."""
        spectrum_name = self.spectrum.__class__.__name__
        if spectrum_name == "KaimalWindSpectrumNonDimensional":
            return 0
        if spectrum_name == "PanofskyWindSpectrumNonDimensional":
            return 1
        if spectrum_name == "TeunissenWindSpectrumNonDimensional":
            return 2
        return None

    def _calculate_evolutional_mean_wind_speeds(
        self,
        wind_speeds,
        time_value,
        total_time,
        modulation_amplitude=0.2,
        modulation_values=None,
        time_index=None,
    ):
        """Build the instantaneous mean wind speeds for the current time."""
        if modulation_values is None:
            modulation = 1.0 + modulation_amplitude * np.sin(2 * np.pi * time_value / total_time)
        else:
            if time_index is None:
                raise ValueError("time_index is required when modulation_values is provided")
            modulation = modulation_values[time_index]
        return wind_speeds * modulation

    def _calculate_evolutional_modulation_factors(
        self,
        time_values,
        total_time,
        modulation_amplitude=0.2,
        modulation_values=None,
        time_indices=None,
    ):
        """Build modulation factors for a time chunk."""
        if modulation_values is None:
            return 1.0 + modulation_amplitude * np.sin(2.0 * np.pi * time_values / total_time)
        if time_indices is None:
            raise ValueError("time_indices is required when modulation_values is provided")
        return modulation_values[time_indices]

    def _calculate_evolutional_power_spectrum(
        self,
        freq,
        heights,
        wind_speeds,
        component,
        time_value,
        total_time,
        modulation_amplitude=0.2,
        modulation_values=None,
        time_index=None,
        evolution_psd_generator=None,
    ):
        """Evaluate the evolutional PSD at one time-frequency slice."""
        if evolution_psd_generator is not None:
            psd_values = evolution_psd_generator(
                freq=freq,
                heights=heights,
                wind_speeds=wind_speeds,
                component=component,
                time_value=time_value,
                time_index=time_index,
                total_time=total_time,
                simulator=self,
            )
            return np.asarray(psd_values, dtype=np.float64)

        spectrum_function = getattr(self.spectrum, f"calculate_power_spectrum_{component}", None)
        if spectrum_function is None:
            raise ValueError(f"Unsupported component: {component}.")

        mean_wind_speeds = self._calculate_evolutional_mean_wind_speeds(
            wind_speeds,
            time_value,
            total_time,
            modulation_amplitude=modulation_amplitude,
            modulation_values=modulation_values,
            time_index=time_index,
        )

        u_stars = self.spectrum.calculate_friction_velocity(
            heights,
            mean_wind_speeds,
            self.params["z_0"],
            self.params["z_d"],
            self.params["K"],
            self.params["alpha_0"],
        )
        reduced_frequencies = self.spectrum.calculate_f(
            freq,
            heights,
            mean_wind_speeds,
            self.params["alpha_0"],
        )
        return np.asarray(spectrum_function(freq, u_stars, reduced_frequencies), dtype=np.float64)

    def _calculate_evolutional_power_spectrum_batch(
        self,
        freq_values,
        heights,
        wind_speeds,
        component,
        time_values,
        total_time,
        modulation_amplitude=0.2,
        modulation_values=None,
        time_indices=None,
    ):
        """Evaluate the evolutional PSD on a frequency-time chunk."""
        spectrum_function = getattr(self.spectrum, f"calculate_power_spectrum_{component}", None)
        if spectrum_function is None:
            raise ValueError(f"Unsupported component: {component}.")

        modulation = self._calculate_evolutional_modulation_factors(
            time_values,
            total_time,
            modulation_amplitude=modulation_amplitude,
            modulation_values=modulation_values,
            time_indices=time_indices,
        )
        mean_wind_speeds = wind_speeds[None, :] * modulation[:, None]
        broadcast_heights = np.broadcast_to(heights[None, :], mean_wind_speeds.shape)
        u_stars = self.spectrum.calculate_friction_velocity(
            broadcast_heights,
            mean_wind_speeds,
            self.params["z_0"],
            self.params["z_d"],
            self.params["K"],
            self.params["alpha_0"],
        )
        reduced_frequencies = self.spectrum.calculate_f(
            np.asarray(freq_values, dtype=np.float64)[:, None, None],
            heights[None, None, :],
            mean_wind_speeds[None, :, :],
            self.params["alpha_0"],
        )
        return np.asarray(
            spectrum_function(
                np.asarray(freq_values, dtype=np.float64)[:, None, None],
                u_stars[None, :, :],
                reduced_frequencies,
            ),
            dtype=np.float64,
        ), mean_wind_speeds

    def _compute_frequency_chunk_high_memory(
        self,
        freq_chunk,
        time_chunk,
        time_indices,
        phi_chunk,
        heights,
        wind_speeds,
        distance_term,
        component,
        total_time,
        modulation_amplitude=0.2,
        modulation_values=None,
    ):
        """Materialize a larger frequency-time chunk to trade memory for speed."""
        phase_vectors = np.exp(1j * phi_chunk)
        spectrum_values, mean_wind_speeds = self._calculate_evolutional_power_spectrum_batch(
            freq_chunk,
            heights,
            wind_speeds,
            component,
            time_chunk,
            total_time,
            modulation_amplitude=modulation_amplitude,
            modulation_values=modulation_values,
            time_indices=time_indices,
        )
        spectrum_values = np.maximum(spectrum_values, 0.0)
        safe_denominator = np.maximum(
            mean_wind_speeds[:, :, None] + mean_wind_speeds[:, None, :],
            1e-8,
        )
        coherence = np.exp(
            -2.0
            * freq_chunk[:, None, None, None]
            * distance_term[None, None, :, :]
            / safe_denominator[None, :, :, :]
        )
        csd_matrix = (
            np.sqrt(spectrum_values[:, :, :, None] * spectrum_values[:, :, None, :])
            * coherence
        )
        csd_matrix = 0.5 * (csd_matrix + np.swapaxes(csd_matrix, -1, -2))
        h_matrix = np.linalg.cholesky(csd_matrix + np.eye(heights.shape[0], dtype=np.float64)[None, None, :, :] * 1e-12)
        chunk_result = np.matmul(
            h_matrix.astype(np.complex128),
            phase_vectors[:, None, :, None],
        )[..., 0]
        chunk_result *= np.exp(1j * 2.0 * np.pi * freq_chunk[:, None] * time_chunk[None, :])[:, :, None]
        return chunk_result

    def estimate_nonstationary_memory_requirement(
        self,
        n_points,
        n_frequencies,
        n_times,
        freq_batch_size=None,
        time_batch_size=None,
    ):
        """Estimate memory usage for the NumPy nonstationary kernel in GB."""
        freq_chunk = n_frequencies if freq_batch_size is None else min(freq_batch_size, n_frequencies)
        time_chunk = n_times if time_batch_size is None else min(time_batch_size, n_times)

        dtype_size = 8
        real_matrix_bytes = freq_chunk * time_chunk * n_points * n_points * dtype_size
        real_vector_bytes = freq_chunk * time_chunk * n_points * dtype_size
        complex_vector_bytes = freq_chunk * time_chunk * n_points * dtype_size * 2

        total_bytes = (
            3 * real_matrix_bytes
            + real_vector_bytes
            + complex_vector_bytes
        ) * 3.0
        return total_bytes / (1024 ** 3)

    def get_optimal_nonstationary_batch_sizes(self, n_points, n_frequencies, n_times, max_memory_gb=4.0):
        """Choose frequency and time chunk sizes for the NumPy nonstationary kernel."""
        bytes_budget = max_memory_gb * (1024 ** 3)
        per_pair_bytes = ((3 * n_points * n_points) + (3 * n_points)) * 8 * 3.0
        max_pairs = max(1, int(bytes_budget / per_pair_bytes))

        if max_pairs >= n_frequencies * n_times:
            return n_frequencies, n_times

        ratio = n_frequencies / max(n_times, 1)
        freq_batch = max(1, min(n_frequencies, int((max_pairs * ratio) ** 0.5)))
        time_batch = max(1, min(n_times, max_pairs // max(freq_batch, 1)))

        while freq_batch * time_batch > max_pairs:
            if freq_batch >= time_batch and freq_batch > 1:
                freq_batch -= 1
            elif time_batch > 1:
                time_batch -= 1
            else:
                break

        return max(1, freq_batch), max(1, time_batch)

    def simulate_wind_nonstationary(
        self,
        positions,
        wind_speeds,
        component="u",
        mode="chunked-vmap",
        modulation_amplitude=0.2,
        max_memory_gb=4.0,
        freq_batch_size=None,
        time_batch_size=None,
        auto_batch=True,
        **kwargs,
    ):
        """Simulate nonstationary wind using the same public interface as the JAX backend."""
        modulation_values = kwargs.pop("modulation_values", None)
        evolution_psd_generator = kwargs.pop("evolution_psd_generator", None)
        memory_mode = kwargs.pop("memory_mode", "auto")
        valid_modes = {"freq-for", "full-vmap", "chunked-vmap"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported nonstationary mode: {mode}. Expected one of {sorted(valid_modes)}")
        valid_memory_modes = {"auto", "balanced", "high"}
        if memory_mode not in valid_memory_modes:
            raise ValueError(
                f"Unsupported memory_mode: {memory_mode}. Expected one of {sorted(valid_memory_modes)}"
            )

        np.random.seed(self.seed)
        self.seed += 1

        positions = np.asarray(positions, dtype=np.float64)
        wind_speeds = np.asarray(wind_speeds, dtype=np.float64)

        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]
        dt = self.params["dt"]
        total_time = self.params["T"]

        frequencies = self.calculate_simulation_frequency(N, dw)
        times = np.arange(M, dtype=np.float64) * dt
        if modulation_values is not None:
            modulation_values = np.asarray(modulation_values, dtype=np.float64)
            if modulation_values.shape != (M,):
                raise ValueError(f"modulation_values must have shape ({M},), got {modulation_values.shape}")

        x_i, x_j = positions[:, 0:1], positions[:, 0:1].T
        y_i, y_j = positions[:, 1:2], positions[:, 1:2].T
        z_i, z_j = positions[:, 2:3], positions[:, 2:3].T
        distance_term = np.sqrt(
            self.params["C_x"] ** 2 * (x_i - x_j) ** 2
            + self.params["C_y"] ** 2 * (y_i - y_j) ** 2
            + self.params["C_z"] ** 2 * (z_i - z_j) ** 2
        )

        phi = np.random.uniform(0.0, 2.0 * np.pi, (N, n))
        V_accum = np.zeros((M, n), dtype=np.complex128)
        identity = np.eye(n, dtype=np.float64)
        component_code = 0 if component == "u" else 1 if component == "w" else None
        spectrum_code = self._get_numba_spectrum_code()
        use_high_memory_kernel = (
            memory_mode in {"auto", "high"}
            and evolution_psd_generator is None
            and component_code is not None
            and spectrum_code is not None
        )
        use_numba_kernel = (
            not use_high_memory_kernel
            and
            NUMBA_AVAILABLE
            and evolution_psd_generator is None
            and component_code is not None
            and spectrum_code is not None
        )

        def _compute_single_frequency(freq_l, time_chunk, time_indices, phi_l):
            if use_numba_kernel:
                return _numba_compute_single_frequency(
                    freq_l,
                    time_chunk,
                    time_indices,
                    phi_l,
                    positions[:, 2],
                    wind_speeds,
                    distance_term,
                    total_time,
                    modulation_amplitude,
                    modulation_values if modulation_values is not None else np.empty(0, dtype=np.float64),
                    modulation_values is not None,
                    self.params["z_0"],
                    self.params["z_d"],
                    self.params["K"],
                    self.params["alpha_0"],
                    component_code,
                    spectrum_code,
                )

            phase_vector = np.exp(1j * phi_l)
            frequency_result = np.empty((time_chunk.shape[0], n), dtype=np.complex128)

            for local_idx, (time_idx, time_m) in enumerate(zip(time_indices, time_chunk)):
                s_values = self._calculate_evolutional_power_spectrum(
                    freq_l,
                    positions[:, 2],
                    wind_speeds,
                    component,
                    time_value=time_m,
                    total_time=total_time,
                    modulation_amplitude=modulation_amplitude,
                    modulation_values=modulation_values,
                    time_index=int(time_idx),
                    evolution_psd_generator=evolution_psd_generator,
                )
                s_values = np.maximum(s_values, 0.0)
                s_i, s_j = s_values[:, None], s_values[None, :]

                mean_wind_speeds = self._calculate_evolutional_mean_wind_speeds(
                    wind_speeds,
                    time_m,
                    total_time,
                    modulation_amplitude=modulation_amplitude,
                    modulation_values=modulation_values,
                    time_index=int(time_idx),
                )
                U_i = mean_wind_speeds[:, None]
                U_j = mean_wind_speeds[None, :]

                coherence = NumpyNonstationaryWindSimulator.calculate_coherence(
                    x_i,
                    x_j,
                    y_i,
                    y_j,
                    z_i,
                    z_j,
                    freq_l,
                    U_i,
                    U_j,
                    self.params["C_x"],
                    self.params["C_y"],
                    self.params["C_z"],
                )

                csd_matrix = NumpyNonstationaryWindSimulator.calculate_cross_spectrum(s_i, s_j, coherence)
                csd_matrix = 0.5 * (csd_matrix + csd_matrix.T)
                H_matrix = cholesky(csd_matrix + identity * 1e-12, lower=True)
                frequency_result[local_idx] = (
                    H_matrix @ phase_vector
                ) * np.exp(1j * 2.0 * np.pi * freq_l * time_m)

            return frequency_result

        estimated_memory = self.estimate_nonstationary_memory_requirement(n, N, M)
        use_batching = False
        if mode == "full-vmap":
            freq_batch_size = N
            time_batch_size = M
        elif mode == "freq-for":
            freq_batch_size = 1
            if time_batch_size is None:
                time_batch_size = M
        elif auto_batch and estimated_memory > max_memory_gb:
            use_batching = True
        elif freq_batch_size is not None or time_batch_size is not None:
            use_batching = True

        if mode == "chunked-vmap" and use_batching:
            auto_freq_batch, auto_time_batch = self.get_optimal_nonstationary_batch_sizes(
                n, N, M, max_memory_gb
            )
            freq_batch_size = auto_freq_batch if freq_batch_size is None else min(freq_batch_size, N)
            time_batch_size = auto_time_batch if time_batch_size is None else min(time_batch_size, M)
        elif mode == "chunked-vmap":
            freq_batch_size = N
            time_batch_size = M
        else:
            freq_batch_size = N if freq_batch_size is None else min(freq_batch_size, N)
            time_batch_size = M if time_batch_size is None else min(time_batch_size, M)

        chunk_memory = self.estimate_nonstationary_memory_requirement(
            n, N, M, freq_batch_size, time_batch_size
        )
        n_freq_batches = self._get_batch_info(N, freq_batch_size)
        n_time_batches = self._get_batch_info(M, time_batch_size)
        self.last_nonstationary_run_info = {
            "mode": mode,
            "memory_mode": memory_mode,
            "n_points": n,
            "n_frequencies": N,
            "n_times": M,
            "estimated_full_memory_gb": estimated_memory,
            "freq_batch_size": int(freq_batch_size),
            "time_batch_size": int(time_batch_size),
            "n_freq_batches": int(n_freq_batches),
            "n_time_batches": int(n_time_batches),
            "chunk_memory_gb": float(chunk_memory),
            "auto_batch": bool(auto_batch),
        }

        print(f"Estimated nonstationary full-vmap memory: {estimated_memory:.2f} GB")
        if use_high_memory_kernel:
            print("NumPy nonstationary kernel: high-memory vectorized")
        elif use_numba_kernel:
            print("NumPy nonstationary kernel: numba-accelerated")
        else:
            print("NumPy nonstationary kernel: pure NumPy fallback")
        if mode == "freq-for":
            print(
                "Using nonstationary frequency-for mode with "
                f"time_batch_size={time_batch_size} ({n_time_batches} time chunks per frequency)"
            )
        elif mode == "full-vmap":
            print("Using nonstationary full-vmap mode")
        elif use_batching:
            print(
                "Using nonstationary batching with "
                f"freq_batch_size={freq_batch_size}, time_batch_size={time_batch_size} "
                f"({n_freq_batches} x {n_time_batches} chunks, ~{chunk_memory:.2f} GB/chunk)"
            )
        else:
            print("Nonstationary full-vmap fits in memory; running without chunking")

        overall_start = time.time()
        first_chunk_elapsed = None
        tenth_chunk_elapsed = None
        chunk_counter = 0

        for freq_batch_idx in range(n_freq_batches):
            start_freq, end_freq = self._get_batch_range(freq_batch_idx, freq_batch_size, N)
            freq_chunk = frequencies[start_freq:end_freq]
            phi_chunk = phi[start_freq:end_freq]

            for time_batch_idx in range(n_time_batches):
                start_time_idx, end_time_idx = self._get_batch_range(time_batch_idx, time_batch_size, M)
                time_chunk = times[start_time_idx:end_time_idx]
                time_indices = np.arange(start_time_idx, end_time_idx)

                if use_high_memory_kernel:
                    chunk_result = self._compute_frequency_chunk_high_memory(
                        freq_chunk,
                        time_chunk,
                        time_indices,
                        phi_chunk,
                        positions[:, 2],
                        wind_speeds,
                        distance_term,
                        component,
                        total_time,
                        modulation_amplitude=modulation_amplitude,
                        modulation_values=modulation_values,
                    )
                    chunk_sum = np.sum(chunk_result, axis=0)
                elif mode == "freq-for":
                    chunk_sum = _compute_single_frequency(
                        freq_chunk[0],
                        time_chunk,
                        time_indices,
                        phi_chunk[0],
                    )
                else:
                    chunk_sum = np.zeros((time_chunk.shape[0], n), dtype=np.complex128)
                    for freq_l, phi_l in zip(freq_chunk, phi_chunk):
                        chunk_sum += _compute_single_frequency(
                            freq_l,
                            time_chunk,
                            time_indices,
                            phi_l,
                        )

                V_accum[start_time_idx:end_time_idx] += chunk_sum

                if chunk_counter == 0:
                    first_chunk_elapsed = time.time() - overall_start
                    print(
                        "First nonstationary chunk completed "
                        f"(includes setup overhead): {first_chunk_elapsed:.3f} s"
                    )
                elif chunk_counter == 9:
                    tenth_chunk_elapsed = time.time() - overall_start
                    print(
                        "Tenth nonstationary chunk completed "
                        f"(steady-state compute): {tenth_chunk_elapsed:.3f} s"
                    )

                chunk_counter += 1

        wind_samples = np.sqrt(2.0 * dw) * np.real(V_accum)
        wind_samples = np.asarray(wind_samples.T, dtype=np.float64)

        total_elapsed = time.time() - overall_start
        print(f"Total nonstationary simulation time: {total_elapsed:.3f} s")
        if tenth_chunk_elapsed is None and chunk_counter < 10:
            print("Tenth nonstationary chunk timing skipped because fewer than 10 chunks were executed.")

        return wind_samples, frequencies