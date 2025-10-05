from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit, random, vmap
from jax.scipy.linalg import cholesky

from .psd import get_spectrum_class
from ..base_simulator import BaseWindSimulator


class JaxWindSimulator(BaseWindSimulator):
    """
    Stochastic wind field simulator implemented using JAX.
    
    This class efficiently simulates fluctuating wind fields using the spectral 
    representation method. It supports only frequency batching for memory management,
    using vmap for parallel computation across frequencies to avoid storing large
    cross-spectral density matrices.
    """

    def __init__(self, key=0, spectrum_type="kaimal-nd"):
        """
        Initialize the JAX wind field simulator.
        
        Args:
            key (int): Random number seed for reproducible results
            spectrum_type (str): Type of wind spectrum to use (default: "kaimal-nd")
        """
        super().__init__()
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
            # Physical constants
            "K": 0.4,           # von Karman constant
            "H_bar": 10.0,      # Average height of surrounding buildings (m)
            "z_0": 0.05,        # Surface roughness height (m)
            "alpha_0": 0.16,    # Surface roughness exponent
            
            # Coherence coefficients
            "C_x": 16.0,        # Decay coefficient in x direction
            "C_y": 6.0,         # Decay coefficient in y direction
            "C_z": 10.0,        # Decay coefficient in z direction
            
            # Frequency domain parameters
            "w_up": 5.0,        # Cutoff frequency (Hz)
            "N": 3000,          # Number of frequency segments
            "M": 6000,          # Number of time points (M = 2*N)
            
            # Time domain parameters
            "T": 600,           # Simulation duration (s)
            "dt": 0.1,          # Time step (s)
            "U_d": 25.0,        # Design basic wind speed (m/s)
        }
        
        # Calculate dependent parameters
        params["dw"] = params["w_up"] / params["N"]  # Frequency increment
        params["z_d"] = params["H_bar"] - params["z_0"] / params["K"]  # Zero plane displacement
        
        return params

    @staticmethod
    @jit
    def calculate_coherence(x_i, x_j, y_i, y_j, z_i, z_j, freq, U_zi, U_zj, C_x, C_y, C_z):
        """
        Calculate spatial coherence function for wind field correlation.
        
        Args:
            x_i, x_j: X-coordinates of points i and j (shapes: (n, 1), (1, n))
            y_i, y_j: Y-coordinates of points i and j (shapes: (n, 1), (1, n))
            z_i, z_j: Z-coordinates of points i and j (shapes: (n, 1), (1, n))
            freq: Frequency (scalar or array)
            U_zi, U_zj: Wind speeds at points i and j (shapes: (n, 1), (1, n))
            C_x, C_y, C_z: Decay coefficients in x, y, z directions
            
        Returns:
            Coherence matrix of shape (n, n) with values between 0 and 1
        """
        # Calculate spatial separation term
        distance_term = jnp.sqrt(
            C_x**2 * (x_i - x_j) ** 2 +
            C_y**2 * (y_i - y_j) ** 2 +
            C_z**2 * (z_i - z_j) ** 2
        )
        
        # Add numerical stability protection
        denominator = U_zi + U_zj
        safe_denominator = jnp.maximum(denominator, 1e-8)
        
        # Davenport coherence function
        return jnp.exp(-2 * freq * distance_term / safe_denominator)

    @staticmethod
    @jit
    def calculate_cross_spectrum(S_ii, S_jj, coherence):
        """
        Calculate cross-spectral density function S_ij.
        
        Args:
            S_ii: Auto-spectral density at point i (shape: (n, 1))
            S_jj: Auto-spectral density at point j (shape: (1, n))
            coherence: Coherence function between points i and j (shape: (n, n))
            
        Returns:
            Cross-spectral density matrix S_ij of shape (n, n)
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
            Array of simulation frequencies of shape (N,)
        """
        return jnp.arange(1, N + 1) * dw - dw / 2

    def build_amplitude_matrix(self, positions, wind_speeds, frequencies, component, key, **kwargs):
        """
        Build amplitude matrix B for all frequencies using vmap parallel computation.
        
        This method avoids storing large cross-spectral density matrices by computing
        amplitude coefficients for each frequency independently and in parallel.
        
        Args:
            positions: Spatial coordinates (n, 3)
            wind_speeds: Mean wind speeds (n,)
            frequencies: Frequency array (N_batch,)
            component: Wind component ('u' for along-wind, 'w' for vertical)
            key: JAX random key for generating random phases
            **kwargs: Additional parameters for spectrum calculation
            
        Returns:
            B_matrix: Amplitude matrix (n, N_batch)
        """
        n = positions.shape[0]
        N_batch = len(frequencies)
        
        # Create spatial coordinate grids for coherence calculation
        x_i, x_j = positions[:, 0:1], positions[:, 0:1].T  # (n, 1), (1, n)
        y_i, y_j = positions[:, 1:2], positions[:, 1:2].T  # (n, 1), (1, n) 
        z_i, z_j = positions[:, 2:3], positions[:, 2:3].T  # (n, 1), (1, n)
        U_i, U_j = wind_speeds[:, None], wind_speeds[None, :]  # (n, 1), (1, n)
        
        # Generate random phases for each frequency and spatial point
        key, subkey = random.split(key)
        phi = random.uniform(subkey, (N_batch, n), minval=0, maxval=2 * jnp.pi)  # (N_batch, n)

        @partial(jit, static_argnums=(2,))
        def _single_freq_amplitude(freq_l, positions, component, phi_l):
            """
            Compute amplitude coefficients for a single frequency.
            
            Args:
                freq_l: Single frequency value (scalar)
                positions: Spatial coordinates (n, 3)
                component: Wind component string
                phi_l: Random phases for this frequency (n,)
                
            Returns:
                Complex amplitude coefficients (n,)
            """
            # Calculate auto-spectral densities at all points
            s_values = self.spectrum.calculate_power_spectrum(freq_l, positions[:, 2], component)  # (n,)
            s_i, s_j = s_values[:, None], s_values[None, :]  # (n, 1), (1, n)
            
            # Calculate spatial coherence matrix
            coherence = self.calculate_coherence(
                x_i, x_j, y_i, y_j, z_i, z_j, freq_l, U_i, U_j,
                self.params["C_x"], self.params["C_y"], self.params["C_z"]
            )  # (n, n)
            
            # Build cross-spectral density matrix
            csd_matrix = self.calculate_cross_spectrum(s_i, s_j, coherence)  # (n, n)
            
            # Cholesky decomposition for amplitude matrix
            H_matrix = cholesky(csd_matrix + jnp.eye(n) * 1e-12, lower=True)  # (n, n)
            H_matrix = jnp.tril(H_matrix)
            
            # Apply random phases and compute amplitude coefficients
            E = jnp.exp(1j * phi_l)  # (n,)
            return jnp.matmul(H_matrix, E)  # (n,)

        # Parallel computation across frequencies using vmap
        # phi has shape (N_batch, n), vmap takes phi[i, :] as phi_l for each frequency
        B_non_zero = vmap(_single_freq_amplitude, in_axes=(0, None, None, 0))(
            frequencies, positions, component, phi
        )  # (N_batch, n)
        
        return B_non_zero.T  # (n, N_batch)

    def simulate_wind(self, positions, wind_speeds, component="u", 
                     max_memory_gb=4.0, freq_batch_size=None, auto_batch=True, **kwargs):
        """
        Simulate fluctuating wind field time series.
        
        Only frequency batching is supported for memory management. The method
        automatically determines whether batching is needed based on memory estimation.
        
        Args:
            positions: Spatial coordinates (n, 3)
            wind_speeds: Mean wind speeds at each point (n,)
            component: Wind component ('u' for along-wind, 'w' for vertical)
            max_memory_gb: Maximum memory limit in GB (default: 4.0)
            freq_batch_size: Manual frequency batch size (None for auto-calculation)
            auto_batch: If True, automatically determine if batching is needed
            **kwargs: Additional parameters for spectrum calculation
            
        Returns:
            wind_samples: Wind time series (n, M)
            frequencies: Frequency array (N,)
        """
        # Ensure positions is a JAX array
        if not isinstance(positions, jnp.ndarray):
            positions = jnp.array(positions)
            
        n = positions.shape[0]
        N = self.params["N"]
        
        # Estimate memory requirements and determine batching strategy
        estimated_memory = self.estimate_memory_requirement(n, N)
        use_batching = False
        
        if auto_batch and estimated_memory > max_memory_gb:
            use_batching = True
        elif freq_batch_size is not None:
            use_batching = True
            
        # Execute appropriate simulation method
        if use_batching:
            if freq_batch_size is None:
                _, freq_batch_size = self.get_optimal_batch_sizes(n, N, max_memory_gb)
                
            return self._simulate_wind_with_freq_batching(
                positions, wind_speeds, component, freq_batch_size, **kwargs
            )
        else:
            return self._simulate_fluctuating_wind(
                positions, wind_speeds, component, **kwargs
            )

    def _simulate_fluctuating_wind(self, positions, wind_speeds, component, **kwargs):
        """
        Direct simulation without frequency batching for small problems.
        
        This method is used when the estimated memory requirement is within
        the specified limit and no manual batching is requested.
        
        Args:
            positions: Spatial coordinates (n, 3)
            wind_speeds: Mean wind speeds (n,)
            component: Wind component ('u' or 'w')
            **kwargs: Additional parameters for spectrum calculation
            
        Returns:
            wind_samples: Wind time series (n, M)
            frequencies: Frequency array (N,)
        """
        # Split random key for amplitude matrix generation
        self.key, subkey = random.split(self.key)
        
        # Extract simulation parameters
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"] 
        dw = self.params["dw"]
        
        # Generate frequency array
        frequencies = self.calculate_simulation_frequency(N, dw)  # (N,)
        
        # Build amplitude matrix for all frequencies at once
        B_matrices = self.build_amplitude_matrix(
            positions, wind_speeds, frequencies, component, subkey, **kwargs
        )  # (n, N)
        
        # Convert amplitude matrix to wind time series using FFT
        wind_samples = self._process_amplitude_to_samples(B_matrices, N, M, dw)  # (n, M)
        
        return wind_samples, frequencies

    def _simulate_wind_with_freq_batching(self, positions, wind_speeds, component, freq_batch_size, **kwargs):
        """
        Simulate wind field with frequency batching for memory management.
        
        This method processes frequencies in batches to reduce memory usage when
        the full amplitude matrix would exceed memory limits.
        
        Args:
            positions: Spatial coordinates (n, 3)
            wind_speeds: Mean wind speeds (n,)
            component: Wind component ('u' or 'w')
            freq_batch_size: Number of frequencies to process in each batch
            **kwargs: Additional parameters for spectrum calculation
            
        Returns:
            wind_samples: Wind time series (n, M)
            frequencies: Frequency array (N,)
        """
        # Extract simulation parameters
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]
        
        # Generate frequency array and calculate batch information
        frequencies = self.calculate_simulation_frequency(N, dw)  # (N,)
        n_freq_batches = self._get_batch_info(N, freq_batch_size)
        
        # Initialize amplitude matrix with complex dtype
        B_matrix = jnp.zeros((n, N), dtype=jnp.complex64)  # (n, N)
        
        # Process frequencies in batches
        for freq_batch_idx in range(n_freq_batches):
            # Get frequency range for current batch
            start_freq, end_freq = self._get_batch_range(freq_batch_idx, freq_batch_size, N)
            batch_frequencies = frequencies[start_freq:end_freq]  # (batch_size,)
            
            # Generate new random key for this batch
            self.key, subkey = random.split(self.key)
            
            # Compute amplitude matrix for current frequency batch
            B_batch = self.build_amplitude_matrix(
                positions, wind_speeds, batch_frequencies, component, subkey, **kwargs
            )  # (n, batch_size)
            
            # Store batch results in full amplitude matrix
            batch_size = B_batch.shape[1]
            B_matrix = B_matrix.at[:, start_freq:start_freq+batch_size].set(B_batch)
        
        # Convert amplitude matrix to wind time series
        wind_samples = self._process_amplitude_to_samples(B_matrix, N, M, dw)  # (n, M)
        
        return wind_samples, frequencies

    @partial(jit, static_argnums=(0, 2, 3, 4))
    def _process_amplitude_to_samples(self, B_matrices, N, M, dw):
        """
        Convert amplitude matrix to wind time series using inverse FFT.
        
        This method implements the spectral representation method by transforming
        frequency-domain amplitude coefficients to time-domain wind samples.
        
        Args:
            B_matrices: Complex amplitude matrix (n, N)
            N: Number of frequency points
            M: Number of time points (M = 2*N typically)
            dw: Frequency increment (Hz)
            
        Returns:
            wind_samples: Real-valued wind time series (n, M)
        """
        # Ensure complex data type for amplitude matrix
        B_matrices = jnp.asarray(B_matrices, dtype=jnp.complex64)  # (n, N)
        
        # Pad amplitude matrix with zeros to match time series length
        B_padded = jnp.pad(B_matrices, ((0, 0), (0, M - N)), mode='constant')  # (n, M)
        
        # Apply inverse FFT to transform to time domain
        G = jnp.fft.ifft(B_padded, axis=1) * M  # (n, M)
        
        # Generate phase correction factors
        p_indices = jnp.arange(M, dtype=jnp.float32)  # (M,)
        exponent = jnp.exp(1j * (p_indices * jnp.pi / M))  # (M,)
        
        # Apply spectral representation formula and extract real part
        wind_samples = jnp.sqrt(2 * dw) * jnp.real(G * exponent[None, :])  # (n, M)
        
        return jnp.asarray(wind_samples, dtype=jnp.float32)

