import numpy as np
from typing import Dict, Tuple
from scipy.linalg import cholesky
from .psd import get_spectrum_class
from ..base_simulator import BaseWindSimulator


class NumpyWindSimulator(BaseWindSimulator):
    """
    Stochastic wind field simulator class implemented using NumPy.
    
    This class provides functionality for simulating fluctuating wind fields using
    the spectral representation method with automatic batching for memory management.
    NumPy backend uses CPU computations and is suitable for moderate-scale simulations.
    """

    def __init__(self, key=0, spectrum_type="kaimal-nd"):
        """
        Initialize the wind field simulator.
        
        Args:
            key (int): Random number seed for reproducible results
            spectrum_type (str): Type of wind spectrum to use (default: "kaimal-nd")
        """
        super().__init__()  # Initialize base class
        self.seed = key
        np.random.seed(key)
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

    def update_parameters(self, **kwargs):
        """Update simulation parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        # Update dependent parameters
        self.params["dw"] = self.params["w_up"] / self.params["N"]
        self.params["z_d"] = (
            self.params["H_bar"] - self.params["z_0"] / self.params["K"]
        )
        self.spectrum.params = self.params  # Update spectrum parameters

    @staticmethod
    def calculate_coherence(x_i, x_j, y_i, y_j, z_i, z_j, freq, U_zi, U_zj, C_x, C_y, C_z):
        """Calculate spatial correlation function Coh."""
        distance_term = np.sqrt(
            C_x**2 * (x_i - x_j) ** 2
            + C_y**2 * (y_i - y_j) ** 2
            + C_z**2 * (z_i - z_j) ** 2
        )
        # Add numerical stability protection to avoid division by near-zero values
        denominator = U_zi + U_zj
        safe_denominator = np.maximum(denominator, 1e-8)  # Set safe minimum value

        return np.exp(-2 * freq * distance_term / safe_denominator)

    @staticmethod
    def calculate_cross_spectrum(S_ii, S_jj, coherence):
        """Calculate cross-spectral density function S_ij."""
        return np.sqrt(S_ii * S_jj) * coherence

    @staticmethod
    def calculate_simulation_frequency(N, dw):
        """Calculate simulation frequency array."""
        return np.arange(1, N + 1) * dw - dw / 2

    def build_amplitude_matrix(self, positions, wind_speeds, frequencies, component, **kwargs):
        """
        Build amplitude matrix B for all frequencies using parallel computation.
        
        This method avoids storing large cross-spectral density matrices by computing
        amplitude coefficients for each frequency independently and in parallel.
        
        Args:
            positions: Spatial coordinates (n, 3)
            wind_speeds: Mean wind speeds (n,)
            frequencies: Frequency array (N_batch,)
            component: Wind component ('u' for along-wind, 'w' for vertical)
            **kwargs: Additional parameters for spectrum calculation
            
        Returns:
            B_matrix: Amplitude matrix (n, N_batch)
        """
        positions = np.asarray(positions, dtype=np.float64)
        wind_speeds = np.asarray(wind_speeds, dtype=np.float64)
        frequencies = np.asarray(frequencies, dtype=np.float64)
        
        n = positions.shape[0]
        N_batch = len(frequencies)
        
        # Create spatial coordinate grids for coherence calculation
        x_i, x_j = positions[:, 0:1], positions[:, 0:1].T  # (n, 1), (1, n)
        y_i, y_j = positions[:, 1:2], positions[:, 1:2].T  # (n, 1), (1, n)
        z_i, z_j = positions[:, 2:3], positions[:, 2:3].T  # (n, 1), (1, n)
        U_i, U_j = wind_speeds[:, None], wind_speeds[None, :]  # (n, 1), (1, n)
        
        # Generate random phases for each frequency and spatial point
        np.random.seed(self.seed)
        phi = np.random.uniform(0, 2 * np.pi, (N_batch, n))  # (N_batch, n)

        def _single_freq_amplitude(freq_l, phi_l):
            """
            Compute amplitude coefficients for a single frequency.
            
            Args:
                freq_l: Single frequency value (scalar)
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
            # Ensure matrix is real and positive definite for Cholesky
            csd_real = csd_matrix.real + np.eye(n) * 1e-12
            H_matrix = cholesky(csd_real, lower=True)  # (n, n)
            
            # Apply random phases and compute amplitude coefficients
            E = np.exp(1j * phi_l)  # (n,)
            return np.matmul(H_matrix.astype(np.complex128), E)  # (n,)

        # Parallel computation across frequencies using list comprehension
        # NumPy doesn't have vmap, so we use list comprehension like current PyTorch
        B_non_zero = np.array([
            _single_freq_amplitude(frequencies[i], phi[i, :])
            for i in range(N_batch)
        ])  # (N_batch, n)
        
        return B_non_zero.T  # (n, N_batch)

    def simulate_wind(self, positions, wind_speeds, component="u", 
                     max_memory_gb=8.0, freq_batch_size=None, auto_batch=True, **kwargs):
        """
        Simulate fluctuating wind field time series.
        
        Only frequency batching is supported for memory management. The method
        automatically determines whether batching is needed based on memory estimation.
        
        Args:
            positions: Array of shape (n, 3), each row represents (x, y, z) coordinates
            wind_speeds: Array of shape (n,), represents mean wind speed at each point
            component: Wind component ('u' for along-wind, 'w' for vertical)
            max_memory_gb: Maximum memory limit in GB (default: 8.0, higher for CPU)
            freq_batch_size: Manual frequency batch size (None for auto-calculation)
            auto_batch: If True, automatically determine if batching is needed
            **kwargs: Additional parameters for spectrum calculation
            
        Returns:
            wind_samples: Array of shape (n, M), fluctuating wind time series at each point
            frequencies: Frequency array (N,)
        """
        np.random.seed(self.seed)
        self.seed += 1
        
        # Convert inputs to NumPy arrays
        positions = np.asarray(positions, dtype=np.float64)
        wind_speeds = np.asarray(wind_speeds, dtype=np.float64)
        
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
        positions = np.asarray(positions, dtype=np.float64)
        wind_speeds = np.asarray(wind_speeds, dtype=np.float64)
        
        # Extract simulation parameters
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"] 
        dw = self.params["dw"]
        
        # Generate frequency array
        frequencies = self.calculate_simulation_frequency(N, dw)  # (N,)
        
        # Build amplitude matrix for all frequencies at once
        B_matrices = self.build_amplitude_matrix(
            positions, wind_speeds, frequencies, component, **kwargs
        )  # (n, N)
        
        # Convert amplitude matrix to wind time series using FFT
        wind_samples = self._process_amplitude_to_samples(B_matrices, N, M, dw)  # (n, M)
        return wind_samples, frequencies

    def _simulate_wind_with_freq_batching(self, positions, wind_speeds, component, freq_batch_size, **kwargs):
        """
        Simulate wind with frequency batching to manage memory usage.
        
        This method processes frequencies in smaller batches to reduce memory requirements.
        Unlike the old approach, this directly computes amplitude coefficients for each
        batch without storing large S matrices.
        
        Args:
            positions: Spatial coordinates (n, 3)
            wind_speeds: Mean wind speeds (n,)
            component: Wind component ('u' or 'w')
            freq_batch_size: Number of frequencies to process simultaneously
            **kwargs: Additional parameters for spectrum calculation
            
        Returns:
            wind_samples: Wind time series (n, M)
            frequencies: Frequency array (N,)
        """
        positions = np.asarray(positions, dtype=np.float64)
        wind_speeds = np.asarray(wind_speeds, dtype=np.float64)
        
        # Extract simulation parameters
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"] 
        dw = self.params["dw"]
        
        # Generate full frequency array
        frequencies = self.calculate_simulation_frequency(N, dw)  # (N,)
        
        # Initialize amplitude matrix to accumulate results
        B_total = np.zeros((n, N), dtype=np.complex128)
        
        # Process frequencies in batches
        num_batches = (N + freq_batch_size - 1) // freq_batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * freq_batch_size
            end_idx = min(start_idx + freq_batch_size, N)
            
            # Extract frequency batch
            freq_batch = frequencies[start_idx:end_idx]  # (batch_size,)
            
            # Build amplitude matrix for this frequency batch
            B_batch = self.build_amplitude_matrix(
                positions, wind_speeds, freq_batch, component, **kwargs
            )  # (n, batch_size)
            
            # Store in appropriate position of full amplitude matrix
            B_total[:, start_idx:end_idx] = B_batch
        
        # Convert amplitude matrix to wind time series using FFT
        wind_samples = self._process_amplitude_to_samples(B_total, N, M, dw)  # (n, M)
        
        return wind_samples, frequencies

    def _process_amplitude_to_samples(self, B_matrices, N, M, dw):
        """
        Convert amplitude matrix to wind time series using FFT.
        
        This method implements the same FFT-based conversion as JAX and PyTorch versions.
        
        Args:
            B_matrices: Complex amplitude matrix (n, N)
            N: Number of frequency points
            M: Number of time points
            dw: Frequency step
            
        Returns:
            wind_samples: Wind time series (n, M)
        """
        n = B_matrices.shape[0]
        
        # Build full amplitude matrix B (n, M) with proper zero padding
        B_full = np.zeros((n, M), dtype=np.complex128)
        B_full[:, :N] = B_matrices  # Copy amplitude coefficients
        
        # Apply IFFT to convert to time domain
        # Scale by M to match JAX/PyTorch implementation
        G = np.fft.ifft(B_full, axis=1) * M  # (n, M)
        
        # Apply phase correction and scaling to get real wind samples
        p_indices = np.arange(M, dtype=np.float64)
        phase_correction = np.exp(1j * (p_indices * np.pi / M))
        
        # Final wind samples (real part only)
        wind_samples = np.sqrt(2 * dw) * np.real(G * phase_correction[None, :])
        
        return wind_samples

    def estimate_memory_requirement(self, n_points, n_frequencies):
        """
        Estimate memory requirement for NumPy backend in GB.
        
        With the new vmap-style architecture, we no longer store large S matrices.
        Memory usage is now dominated by the amplitude matrix and FFT operations.
        
        Args:
            n_points: Number of simulation points
            n_frequencies: Number of frequency points
            
        Returns:
            Estimated memory requirement in GB
        """
        # NumPy typically uses 64-bit floats (8 bytes) and 128-bit complex (16 bytes)
        float_size = 8
        complex_size = 16
        
        # Main memory components in new architecture:
        # 1. Amplitude matrix B: (n_points, n_frequencies) complex
        B_memory = n_points * n_frequencies * complex_size
        
        # 2. Full amplitude matrix for FFT: (n_points, M) where M = 2*N
        M = 2 * n_frequencies
        B_full_memory = n_points * M * complex_size
        
        # 3. Intermediate arrays and working memory (coherence, CSD matrix per frequency)
        # These are computed one frequency at a time, so memory is O(n^2) not O(N*n^2)
        temp_memory = n_points * n_points * complex_size * 3  # CSD, H, coherence matrices
        
        # 4. Random phase array: (n_frequencies, n_points)
        phi_memory = n_frequencies * n_points * float_size
        
        # Total with safety factor for NumPy operations
        total_bytes = (B_memory + B_full_memory + temp_memory + phi_memory) * 1.3
        
        return total_bytes / (1024**3)  # Convert to GB