from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit, random, vmap
from jax.scipy.linalg import cholesky
from .psd import get_spectrum_class
from ..base_simulator import BaseWindSimulator


class JaxWindSimulator(BaseWindSimulator):
    """
    Stochastic wind field simulator class implemented using JAX.
    
    This class provides functionality for simulating fluctuating wind fields using
    the spectral representation method with automatic batching for memory management.
    JAX backend offers GPU acceleration and efficient vectorized computations.
    """

    def __init__(self, key=0, spectrum_type="kaimal-nd"):
        """
        Initialize the wind field simulator.

        Args:
            key (int): JAX random number seed for reproducible results
            spectrum_type (str): Type of wind spectrum to use (default: "kaimal-nd")
        """
        super().__init__()  # Initialize base class
        self.key = random.PRNGKey(key)
        self.spectrum = get_spectrum_class(spectrum_type)(**self.params)

    def _set_default_parameters(self) -> Dict:
        """
        Set default wind field simulation parameters.
        
        Returns:
            Dict: Dictionary containing default simulation parameters including
                 physical constants, grid specifications, and numerical settings
        """
        params = {
            "K": 0.4,  # Dimensionless constant
            "H_bar": 10.0,  # Average height of surrounding buildings (m)
            "z_0": 0.05,  # Surface roughness height
            "alpha_0": 0.16,  # Surface roughness exponent
            "C_x": 16.0,  # Decay coefficient in x direction
            "C_y": 6.0,  # Decay coefficient in y direction
            "C_z": 10.0,  # Decay coefficient in z direction
            "w_up": 5.0,  # Cutoff frequency (Hz)
            "N": 3000,  # Number of frequency segments
            "M": 6000,  # Number of time points (M=2N)
            "T": 600,  # Simulation duration (s)
            "dt": 0.1,  # Time step (s)
            "U_d": 25.0,  # Design basic wind speed (m/s)
        }
        params["dw"] = params["w_up"] / params["N"]  # Frequency increment
        params["z_d"] = params["H_bar"] - params["z_0"] / params["K"]  # Calculate zero plane displacement

        return params

    def simulate_wind(self, positions, wind_speeds, component="u", 
                     max_memory_gb=4.0, point_batch_size=None, 
                     freq_batch_size=None, auto_batch=True, **kwargs):
        """
        Simulate fluctuating wind field with automatic batching for memory management.
        
        This method now uses batching by default to handle large-scale simulations
        efficiently and avoid memory issues.

        Args:
            positions: Array of shape (n, 3), each row represents (x, y, z) coordinates
            wind_speeds: Array of shape (n,), represents mean wind speed at each point
            component: Wind component, 'u' for along-wind, 'w' for vertical
            max_memory_gb: Maximum memory limit in GB (default: 4.0)
            point_batch_size: Manual point batch size (auto-calculate if None)
            freq_batch_size: Manual frequency batch size (auto-calculate if None)
            auto_batch: If True, automatically determine if batching is needed

        Returns:
            wind_samples: Array of shape (n, M), fluctuating wind time series at each point
            frequencies: Frequency array
        """
        if not isinstance(positions, jnp.ndarray):
            positions = jnp.array(positions)
        
        n = positions.shape[0]
        N = self.params["N"]
        
        # Use base class method to determine batching strategy
        use_batching, point_batch_size, freq_batch_size = self._should_use_batching(
            n, N, max_memory_gb, point_batch_size, freq_batch_size, auto_batch
        )
        
        # Print information about memory and batching decisions
        estimated_memory = self.estimate_memory_requirement(n, N)
        if use_batching:
            n_point_batches = self._get_batch_info(n, point_batch_size)
            n_freq_batches = self._get_batch_info(N, freq_batch_size)
            self.print_batch_info(
                estimated_memory, max_memory_gb, use_batching, 
                point_batch_size, freq_batch_size, n_point_batches, n_freq_batches
            )
        else:
            self.print_batch_info(estimated_memory, max_memory_gb, use_batching)
        
        if use_batching:
            return self._simulate_wind_with_batching(
                positions, wind_speeds, component, 
                point_batch_size, freq_batch_size, **kwargs
            )
        else:
            # Use the direct method for small problems
            return self._simulate_fluctuating_wind(
                positions, wind_speeds, component, **kwargs
            )

    @staticmethod
    @jit
    def calculate_coherence(x_i, x_j, y_i, y_j, z_i, z_j, freq, U_zi, U_zj, C_x, C_y, C_z):
        """
        Calculate spatial coherence function for wind field correlation.
        
        Args:
            x_i, x_j: X-coordinates of points i and j
            y_i, y_j: Y-coordinates of points i and j  
            z_i, z_j: Z-coordinates of points i and j
            freq: Frequency (1/s)
            U_zi, U_zj: Wind speeds at points i and j
            C_x, C_y, C_z: Decay coefficients in x, y, z directions
            
        Returns:
            Coherence value between 0 and 1
        """
        distance_term = jnp.sqrt(
            C_x**2 * (x_i - x_j) ** 2
            + C_y**2 * (y_i - y_j) ** 2
            + C_z**2 * (z_i - z_j) ** 2
        )
        # Add numerical stability protection to avoid division by near-zero values
        # denominator = 2 * jnp.pi * (U_zi + U_zj)
        denominator = U_zi + U_zj
        safe_denominator = jnp.maximum(denominator, 1e-8)  # Set safe minimum value

        return jnp.exp(-2 * freq * distance_term / safe_denominator)

    @staticmethod
    @jit
    def calculate_cross_spectrum(S_ii, S_jj, coherence):
        """
        Calculate cross-spectral density function S_ij.
        
        Args:
            S_ii: Auto-spectral density at point i
            S_jj: Auto-spectral density at point j
            coherence: Coherence function between points i and j
            
        Returns:
            Cross-spectral density S_ij
        """
        return jnp.sqrt(S_ii * S_jj) * coherence

    @staticmethod
    def calculate_simulation_frequency(N, dw):
        """
        Calculate simulation frequency array.
        
        Args:
            N (int): Number of frequency segments
            dw (float): Frequency increment
            
        Returns:
            Array of simulation frequencies
        """
        return jnp.arange(1, N + 1) * dw - dw / 2
    

    def build_spectrum_matrix(self, positions, wind_speeds, frequencies, component, **kwargs):
        """
        Build cross-spectral density matrix S(w) for given frequencies.
        
        This method constructs the cross-spectral density matrices for all frequency
        points, which describe the correlation between wind fluctuations at different
        spatial locations.
        
        Args:
            positions: Array of shape (n, 3) with spatial coordinates
            wind_speeds: Array of shape (n,) with wind speeds at each point
            frequencies: Array of frequencies to compute spectra for
            component: Wind component ('u' for along-wind, 'w' for vertical)
            **kwargs: Additional parameters for spectrum calculation
            
        Returns:
            Array of shape (n_freq, n, n) containing cross-spectral density matrices
        """
        n = positions.shape[0]


        x_i, x_j = positions[:, 0:1], positions[:, 0:1].T  # [n, 1], [1, n]
        y_i, y_j = positions[:, 1:2], positions[:, 1:2].T  # [n, 1], [1, n]
        z_i, z_j = positions[:, 2:3], positions[:, 2:3].T  # [n, 1], [1, n]
        U_i, U_j = wind_speeds[:, None], wind_speeds[None, :]  # [n, 1], [1, n]

        @partial(jit, static_argnums=(2,))
        def _build_spectrum_for_position(freq, positions, component, **kwargs):
            s_values = self.spectrum.calculate_power_spectrum(
                freq, positions[:, 2], component, **kwargs
            )
            s_i, s_j = s_values[:, None], s_values[None, :]  # [n, 1], [1, n]
            coherence = self.calculate_coherence(
                x_i, x_j, y_i, y_j, z_i, z_j, freq, U_i, U_j,
                self.params["C_x"], self.params["C_y"], self.params["C_z"]
            )
            cross_spectrum = self.calculate_cross_spectrum(s_i, s_j, coherence)
            return cross_spectrum
        
        # Parallel computation of cross-spectral density matrix for each frequency point
        S_matrices = vmap(
            _build_spectrum_for_position,
            in_axes=(0, None, None),
        )(frequencies, positions, component)
        
        return S_matrices

    def simulate_wind(self, positions, wind_speeds, component="u", 
                     max_memory_gb=4.0, point_batch_size=None, 
                     freq_batch_size=None, auto_batch=True, **kwargs):
        """
        Simulate fluctuating wind field with automatic batching for memory management.
        
        This method now uses batching by default to handle large-scale simulations
        efficiently and avoid memory issues.

        Args:
            positions: Array of shape (n, 3), each row represents (x, y, z) coordinates
            wind_speeds: Array of shape (n,), represents mean wind speed at each point
            component: Wind component, 'u' for along-wind, 'w' for vertical
            max_memory_gb: Maximum memory limit in GB (default: 4.0)
            point_batch_size: Manual point batch size (auto-calculate if None)
            freq_batch_size: Manual frequency batch size (auto-calculate if None)
            auto_batch: If True, automatically determine if batching is needed

        Returns:
            wind_samples: Array of shape (n, M), fluctuating wind time series at each point
            frequencies: Frequency array
        """
        if not isinstance(positions, jnp.ndarray):
            positions = jnp.array(positions)
        
        n = positions.shape[0]
        N = self.params["N"]
        
        # Estimate memory requirement
        estimated_memory = self.estimate_memory_requirement(n, N)
        
        # Decide whether to use batching
        use_batching = False
        if auto_batch:
            if estimated_memory > max_memory_gb:
                use_batching = True
        else:
            # Force batching if batch sizes are specified
            if point_batch_size is not None or freq_batch_size is not None:
                use_batching = True
        
        if use_batching:
            # Use provided batch sizes or calculate optimal ones
            if point_batch_size is None or freq_batch_size is None:
                optimal_point_batch, optimal_freq_batch = self.get_optimal_batch_sizes(n, N, max_memory_gb)
                if point_batch_size is None:
                    point_batch_size = optimal_point_batch
                if freq_batch_size is None:
                    freq_batch_size = optimal_freq_batch
            
            return self._simulate_wind_with_batching(
                positions, wind_speeds, component, 
                point_batch_size, freq_batch_size, **kwargs
            )
        else:
            # Use the direct method for small problems
            return self._simulate_fluctuating_wind(
                positions, wind_speeds, component, **kwargs
            )

    def _simulate_fluctuating_wind(self, positions, wind_speeds, component, **kwargs):
        """
        Internal implementation of wind field simulation for small-scale problems.
        
        This method performs direct simulation without batching, suitable for
        problems that fit within memory constraints.
        
        Args:
            positions: Array of shape (n, 3) with spatial coordinates
            wind_speeds: Array of shape (n,) with wind speeds at each point
            component: Wind component ('u' or 'w')
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (wind_samples, frequencies) where wind_samples has shape (n, M)
        """
        self.key, subkey = random.split(self.key)
        
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]

        frequencies = self.calculate_simulation_frequency(N, dw)

        # Build cross-spectral density matrix
        S_matrices = self.build_spectrum_matrix(
            positions, wind_speeds, frequencies, component, **kwargs
        )

        # Process spectrum matrices to samples
        wind_samples = self._process_spectrum_to_samples(S_matrices, subkey, n, N, M, dw)

        return wind_samples, frequencies

    def simulate_wind_batched(self, positions, wind_speeds, component="u", 
                            max_memory_gb=4.0, point_batch_size=None, 
                            freq_batch_size=None, **kwargs):
        """
        Simulate fluctuating wind field with automatic batching for memory management.

        Args:
            positions: Array of shape (n, 3), each row represents (x, y, z) coordinates
            wind_speeds: Array of shape (n,), represents mean wind speed at each point
            component: Wind component, 'u' for along-wind, 'w' for vertical
            max_memory_gb: Maximum memory limit in GB
            point_batch_size: Manual point batch size (auto-calculate if None)
            freq_batch_size: Manual frequency batch size (auto-calculate if None)

        Returns:
            wind_samples: Array of shape (n, M), fluctuating wind time series at each point
            frequencies: Frequency array
        """
        if not isinstance(positions, jnp.ndarray):
            positions = jnp.array(positions)
        
        n = positions.shape[0]
        N = self.params["N"]
        
        # Check if batching is needed
        estimated_memory = self.estimate_memory_requirement(n, N)
        print(f"Estimated memory requirement: {estimated_memory:.2f} GB")
        
        if estimated_memory <= max_memory_gb and point_batch_size is None and freq_batch_size is None:
            # No batching needed, use regular simulation
            print("Memory requirement within limit, using regular simulation")
            return self.simulate_wind(positions, wind_speeds, component, **kwargs)
        
        # Use provided batch sizes or calculate optimal ones
        if point_batch_size is None or freq_batch_size is None:
            optimal_point_batch, optimal_freq_batch = self.get_optimal_batch_sizes(n, N, max_memory_gb)
            if point_batch_size is None:
                point_batch_size = optimal_point_batch
            if freq_batch_size is None:
                freq_batch_size = optimal_freq_batch
        
        print(f"Using batched simulation with point_batch_size={point_batch_size}, freq_batch_size={freq_batch_size}")
        
        return self._simulate_wind_with_batching(
            positions, wind_speeds, component, 
            point_batch_size, freq_batch_size, **kwargs
        )

    def _simulate_wind_with_batching(self, positions, wind_speeds, component,
                                   point_batch_size, freq_batch_size, **kwargs):
        """Internal implementation of batched wind field simulation."""
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]
        
        # Use base class methods for batch calculations
        n_point_batches = self._get_batch_info(n, point_batch_size)
        n_freq_batches = self._get_batch_info(N, freq_batch_size)
        
        frequencies = self.calculate_simulation_frequency(N, dw)
        
        # Initialize result array
        wind_samples = jnp.zeros((n, M))
        
        # Process in batches
        for point_batch_idx in range(n_point_batches):
            start_point, end_point = self._get_batch_range(point_batch_idx, point_batch_size, n)
            
            batch_positions = positions[start_point:end_point]
            batch_wind_speeds = wind_speeds[start_point:end_point]
            
            # Use base class method for progress reporting
            self.print_batch_progress(point_batch_idx, n_point_batches, "point", start_point, end_point)
            
            # Process this point batch
            batch_samples = self._simulate_point_batch(
                batch_positions, batch_wind_speeds, component, 
                freq_batch_size, frequencies, **kwargs
            )
            
            wind_samples = wind_samples.at[start_point:end_point].set(batch_samples)
        
        return wind_samples, frequencies
    
    def _simulate_point_batch(self, positions, wind_speeds, component, 
                            freq_batch_size, frequencies, **kwargs):
        """Simulate a batch of points, potentially with frequency batching."""        
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]
        
        if freq_batch_size >= N:
            # No frequency batching needed, use direct simulation
            return self._simulate_fluctuating_wind(positions, wind_speeds, component, **kwargs)[0]
        
        # Build spectrum matrices in frequency batches
        self.key, subkey = random.split(self.key)
        n_freq_batches = self._get_batch_info(N, freq_batch_size)
        S_matrices_full = jnp.zeros((N, n, n))
        
        for freq_batch_idx in range(n_freq_batches):
            start_freq, end_freq = self._get_batch_range(freq_batch_idx, freq_batch_size, N)
            
            batch_frequencies = frequencies[start_freq:end_freq]
            
            # Build spectrum matrix for this frequency batch
            S_batch = self.build_spectrum_matrix(
                positions, wind_speeds, batch_frequencies, component, **kwargs
            )
            
            S_matrices_full = S_matrices_full.at[start_freq:end_freq].set(S_batch)
        
        # Process the full spectrum for this point batch
        return self._process_spectrum_to_samples(S_matrices_full, subkey, n, N, M, dw)
    
    def _process_spectrum_to_samples(self, S_matrices, key, n, N, M, dw):
        """Process spectrum matrices to wind samples (extracted from main simulation)."""
        # Perform Cholesky decomposition for each frequency point
        @jit
        def cholesky_with_reg(S):
            return cholesky(S + jnp.eye(n) * 1e-12, lower=True)

        H_matrices = vmap(cholesky_with_reg)(S_matrices)

        # Generate random phases
        key, subkey = random.split(key)
        phi = random.uniform(subkey, (n, N), minval=0, maxval=2 * jnp.pi)

        @partial(jit, static_argnums=(1, 2, 3))
        def compute_B_for_point(j, N, M, n, H_matrices, phi):
            """Compute B values for point j, fully vectorized implementation"""
            m_indices = jnp.arange(n)
            mask = m_indices <= j
            
            H_jm_all = H_matrices[:, j, :]
            phi_transposed = phi.T
            exp_terms = jnp.exp(1j * phi_transposed)
            
            masked_terms = jnp.where(mask, H_jm_all * exp_terms, 0.0)
            B_values = jnp.sum(masked_terms, axis=1)
            
            return jnp.pad(B_values, (0, M - N), mode="constant")

        # Parallel computation of B for each point
        B = vmap(compute_B_for_point, in_axes=(0, None, None, None, None, None))(
            jnp.arange(n), N, M, n, H_matrices, phi
        )
        
        # Compute FFT to get G
        G = vmap(jit(jnp.fft.ifft))(B) * M

        # Calculate wind field samples
        @jit
        def compute_samples_for_point(j):
            p_indices = jnp.arange(M)
            exponent = jnp.exp(1j * (p_indices * jnp.pi / M))
            terms = G[j] * exponent
            return jnp.sqrt(2 * dw) * jnp.real(terms)

        wind_samples = vmap(compute_samples_for_point)(jnp.arange(n))
        
        return wind_samples
