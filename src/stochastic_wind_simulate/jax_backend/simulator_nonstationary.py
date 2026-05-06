import time

import jax.numpy as jnp
from jax import jit, random, vmap
from jax.scipy.linalg import cholesky

from .simulator import JaxWindSimulator


class JaxNonstationaryWindSimulator(JaxWindSimulator):
    """JAX nonstationary wind simulator with evolutionary PSD support."""

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
            modulation = 1.0 + modulation_amplitude * jnp.sin(2 * jnp.pi * time_value / total_time)
        else:
            if time_index is None:
                raise ValueError("time_index is required when modulation_values is provided")
            modulation = modulation_values[time_index]
        return wind_speeds * modulation

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
            return jnp.asarray(psd_values, dtype=jnp.float32)

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
        return spectrum_function(freq, u_stars, reduced_frequencies)

    def estimate_nonstationary_memory_requirement(
        self,
        n_points,
        n_frequencies,
        n_times,
        freq_batch_size=None,
        time_batch_size=None,
    ):
        """Estimate memory usage for the vmapped nonstationary kernel in GB."""
        freq_chunk = n_frequencies if freq_batch_size is None else min(freq_batch_size, n_frequencies)
        time_chunk = n_times if time_batch_size is None else min(time_batch_size, n_times)

        dtype_size = 4
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
        """Choose frequency/time chunk sizes for the vmapped nonstationary kernel."""
        bytes_budget = max_memory_gb * (1024 ** 3)
        per_pair_bytes = ((3 * n_points * n_points) + (3 * n_points)) * 4 * 3.0
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
        """Simulate nonstationary wind using selectable execution modes."""
        modulation_values = kwargs.pop("modulation_values", None)
        evolution_psd_generator = kwargs.pop("evolution_psd_generator", None)
        valid_modes = {"freq-for", "full-vmap", "chunked-vmap"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported nonstationary mode: {mode}. Expected one of {sorted(valid_modes)}")

        if not isinstance(positions, jnp.ndarray):
            positions = jnp.array(positions)
        if not isinstance(wind_speeds, jnp.ndarray):
            wind_speeds = jnp.array(wind_speeds)

        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]
        dt = self.params["dt"]
        total_time = self.params["T"]

        frequencies = self.calculate_simulation_frequency(N, dw)
        times = jnp.arange(M) * dt
        if modulation_values is not None:
            modulation_values = jnp.asarray(modulation_values, dtype=jnp.float32)
            if modulation_values.shape != (M,):
                raise ValueError(f"modulation_values must have shape ({M},), got {modulation_values.shape}")

        x_i, x_j = positions[:, 0:1], positions[:, 0:1].T
        y_i, y_j = positions[:, 1:2], positions[:, 1:2].T
        z_i, z_j = positions[:, 2:3], positions[:, 2:3].T

        self.key, subkey = random.split(self.key)
        phi = random.uniform(subkey, (N, n), minval=0, maxval=2 * jnp.pi)
        V_accum = jnp.zeros((M, n), dtype=jnp.complex64)
        identity = jnp.eye(n, dtype=jnp.float32)

        @jit
        def _compute_single_frequency(freq_l, time_chunk, time_indices, phi_l):
            phase_vector = jnp.exp(1j * phi_l)

            def _single_time(time_idx, time_m):
                s_values = self._calculate_evolutional_power_spectrum(
                    freq_l,
                    positions[:, 2],
                    wind_speeds,
                    component,
                    time_value=time_m,
                    total_time=total_time,
                    modulation_amplitude=modulation_amplitude,
                    modulation_values=modulation_values,
                    time_index=time_idx,
                    evolution_psd_generator=evolution_psd_generator,
                )
                s_values = jnp.maximum(s_values, 0.0)
                s_i, s_j = s_values[:, None], s_values[None, :]

                mean_wind_speeds = self._calculate_evolutional_mean_wind_speeds(
                    wind_speeds,
                    time_m,
                    total_time,
                    modulation_amplitude=modulation_amplitude,
                    modulation_values=modulation_values,
                    time_index=time_idx,
                )
                U_i = mean_wind_speeds[:, None]
                U_j = mean_wind_speeds[None, :]

                coherence = JaxNonstationaryWindSimulator.calculate_coherence(
                    x_i, x_j, y_i, y_j, z_i, z_j, freq_l, U_i, U_j,
                    self.params["C_x"], self.params["C_y"], self.params["C_z"]
                )

                csd_matrix = JaxNonstationaryWindSimulator.calculate_cross_spectrum(s_i, s_j, coherence)
                csd_matrix = 0.5 * (csd_matrix + csd_matrix.T)
                H_matrix = cholesky(csd_matrix + identity * 1e-12, lower=True)
                return jnp.matmul(H_matrix, phase_vector) * jnp.exp(1j * 2.0 * jnp.pi * freq_l * time_m)

            return vmap(_single_time, in_axes=(0, 0))(time_indices, time_chunk)

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
            freq_batch_size = min(freq_batch_size, N)
            time_batch_size = min(time_batch_size, M)

        chunk_memory = self.estimate_nonstationary_memory_requirement(
            n, N, M, freq_batch_size, time_batch_size
        )
        n_freq_batches = self._get_batch_info(N, freq_batch_size)
        n_time_batches = self._get_batch_info(M, time_batch_size)
        self.last_nonstationary_run_info = {
            "mode": mode,
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

        @jit
        def _compute_frequency_time_chunk(freq_chunk, time_chunk, time_indices, phi_chunk):
            return vmap(
                lambda freq_l, phi_l: _compute_single_frequency(
                    freq_l,
                    time_chunk,
                    time_indices,
                    phi_l,
                )
            )(freq_chunk, phi_chunk)

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
                time_indices = jnp.arange(start_time_idx, end_time_idx)

                if mode == "freq-for":
                    chunk_sum = jnp.zeros((time_chunk.shape[0], n), dtype=jnp.complex64)
                    for freq_idx in range(freq_chunk.shape[0]):
                        chunk_sum = chunk_sum + _compute_single_frequency(
                            freq_chunk[freq_idx],
                            time_chunk,
                            time_indices,
                            phi_chunk[freq_idx],
                        )
                else:
                    chunk_result = _compute_frequency_time_chunk(
                        freq_chunk,
                        time_chunk,
                        time_indices,
                        phi_chunk,
                    )
                    chunk_sum = jnp.sum(chunk_result, axis=0)
                V_accum = V_accum.at[start_time_idx:end_time_idx].add(chunk_sum)

                if chunk_counter == 0:
                    V_accum.block_until_ready()
                    first_chunk_elapsed = time.time() - overall_start
                    print(
                        "First nonstationary chunk completed "
                        f"(includes JIT compilation): {first_chunk_elapsed:.3f} s"
                    )
                elif chunk_counter == 9:
                    V_accum.block_until_ready()
                    tenth_chunk_elapsed = time.time() - overall_start
                    print(
                        "Tenth nonstationary chunk completed "
                        f"(steady-state compute): {tenth_chunk_elapsed:.3f} s"
                    )

                chunk_counter += 1

        wind_samples = jnp.sqrt(2.0 * dw) * jnp.real(V_accum)
        wind_samples = jnp.asarray(wind_samples.T, dtype=jnp.float32)
        wind_samples.block_until_ready()

        total_elapsed = time.time() - overall_start
        print(f"Total nonstationary simulation time: {total_elapsed:.3f} s")
        if tenth_chunk_elapsed is None and chunk_counter < 10:
            print("Tenth nonstationary chunk timing skipped because fewer than 10 chunks were executed.")

        return wind_samples, frequencies