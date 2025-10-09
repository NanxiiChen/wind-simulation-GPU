from typing import Dict

import torch
import torch.func as func
from torch import Tensor

from .psd import get_spectrum_class
from ..base_simulator import BaseWindSimulator


class TorchWindSimulator(BaseWindSimulator):
    """
    Stochastic wind field simulator implemented using PyTorch.
    
    This class efficiently simulates fluctuating wind fields using the spectral 
    representation method. It supports only frequency batching for memory management,
    using vmap for parallel computation across frequencies to avoid storing large
    cross-spectral density matrices.
    """

    def __init__(self, key=0, spectrum_type="kaimal-nd"):
        """
        Initialize the PyTorch wind field simulator.
        
        Args:
            key (int): Random number seed for reproducible results
            spectrum_type (str): Type of wind spectrum to use (default: "kaimal-nd")
        """
        super().__init__()
        self.seed = key
        torch.manual_seed(key)
        
        # Set computing device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        params["backend"] = "torch"
        
        return params

    def _to_tensor(self, value, device=None):
        """
        Safely convert input to tensor on device.

        Args:
            value: Value to convert (tensor or other type)
            device: Target device, defaults to class device

        Returns:
            Tensor on specified device
        """
        if device is None:
            device = self.device
            
        if isinstance(value, torch.Tensor):
            return value.to(device)
        else:
            return torch.tensor(value, device=device)

    @staticmethod
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
        # Get device from input tensor
        device = x_i.device
        
        # Convert coefficients to tensors
        C_x_t = torch.tensor(C_x, device=device)
        C_y_t = torch.tensor(C_y, device=device)
        C_z_t = torch.tensor(C_z, device=device)
        
        # Calculate spatial separation term
        distance_term = torch.sqrt(
            C_x_t**2 * (x_i - x_j) ** 2 +
            C_y_t**2 * (y_i - y_j) ** 2 +
            C_z_t**2 * (z_i - z_j) ** 2
        )
        
        # Add numerical stability protection
        denominator = U_zi + U_zj
        safe_denominator = torch.maximum(
            denominator, 
            torch.tensor(1e-8, device=device)
        )
        
        # Davenport coherence function
        return torch.exp(-2 * freq * distance_term / safe_denominator)

    @staticmethod
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
        return torch.sqrt(S_ii * S_jj) * coherence

    @staticmethod
    def calculate_simulation_frequency(N, dw, device=None):
        """
        Calculate simulation frequency array.
        
        Args:
            N (int): Number of frequency segments
            dw (float): Frequency increment
            device: Target device for tensor creation
            
        Returns:
            Array of simulation frequencies of shape (N,)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.arange(1, N + 1, device=device, dtype=torch.float32) * dw - dw / 2

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
        positions = torch.as_tensor(positions, device=self.device)
        wind_speeds = torch.as_tensor(wind_speeds, device=self.device)
        frequencies = torch.as_tensor(frequencies, device=self.device)
        
        n = positions.shape[0]
        N_batch = len(frequencies)
        
        # Create spatial coordinate grids for coherence calculation
        x_i, x_j = positions[:, 0:1], positions[:, 0:1].T  # (n, 1), (1, n)
        y_i, y_j = positions[:, 1:2], positions[:, 1:2].T  # (n, 1), (1, n) 
        z_i, z_j = positions[:, 2:3], positions[:, 2:3].T  # (n, 1), (1, n)
        U_i, U_j = wind_speeds[:, None], wind_speeds[None, :]  # (n, 1), (1, n)
        
        # Generate random phases for each frequency and spatial point
        torch.manual_seed(self.seed)
        phi = torch.rand((N_batch, n), device=self.device, dtype=torch.float32) * 2 * torch.pi  # (N_batch, n)

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
            csd_real = csd_matrix.real + torch.eye(n, device=self.device, dtype=torch.float32) * 1e-12
            H_matrix = torch.linalg.cholesky(csd_real)  # (n, n)
            
            # Apply random phases and compute amplitude coefficients
            E = torch.exp(1j * phi_l)  # (n,)
            return torch.matmul(H_matrix.to(torch.complex64), E)  # (n,)

        # Parallel computation across frequencies
        # B_non_zero = torch.stack([
        #     _single_freq_amplitude(frequencies[i], phi[i, :])
        #     for i in range(N_batch)
        # ])  # (N_batch, n)
        B_non_zero = func.vmap(_single_freq_amplitude)(frequencies, phi)  # (N_batch, n)
        
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
        # Ensure positions is a PyTorch tensor
        if not isinstance(positions, torch.Tensor):
            positions = torch.from_numpy(positions)
            
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
        positions = torch.as_tensor(positions, device=self.device)
        wind_speeds = torch.as_tensor(wind_speeds, device=self.device)
        
        # Extract simulation parameters
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"] 
        dw = self.params["dw"]
        
        # Generate frequency array
        frequencies = self.calculate_simulation_frequency(N, dw, device=self.device)  # (N,)
        
        # Build amplitude matrix for all frequencies at once
        B_matrices = self.build_amplitude_matrix(
            positions, wind_speeds, frequencies, component, **kwargs
        )  # (n, N)
        
        # Convert amplitude matrix to wind time series using FFT
        wind_samples = self._process_amplitude_to_samples(B_matrices, N, M, dw)  # (n, M)
        
        return wind_samples.cpu().numpy(), frequencies.cpu().numpy()

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
        positions = torch.as_tensor(positions, device=self.device)
        wind_speeds = torch.as_tensor(wind_speeds, device=self.device)
        
        # Extract simulation parameters
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"] 
        dw = self.params["dw"]
        
        # Generate full frequency array
        frequencies = self.calculate_simulation_frequency(N, dw, device=self.device)  # (N,)
        
        # Initialize amplitude matrix to accumulate results
        B_total = torch.zeros((n, N), dtype=torch.complex64, device=self.device)
        
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
        
        return wind_samples.cpu().numpy(), frequencies.cpu().numpy()



    def _process_amplitude_to_samples(self, B_matrices, N, M, dw):
        """
        Convert amplitude matrix to wind time series using FFT.
        
        This method implements the same FFT-based conversion as JAX version.
        
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
        B_full = torch.zeros((n, M), dtype=torch.complex64, device=self.device)
        B_full[:, :N] = B_matrices  # Copy amplitude coefficients
        
        # Apply IFFT to convert to time domain
        # Scale by M to match JAX implementation
        G = torch.fft.ifft(B_full, dim=1) * M  # (n, M)
        
        # Apply phase correction and scaling to get real wind samples
        p_indices = torch.arange(M, device=self.device, dtype=torch.float32)
        phase_correction = torch.exp(1j * (p_indices * torch.pi / M))
        
        # Convert dw to tensor if needed
        dw_tensor = torch.tensor(dw, device=self.device, dtype=torch.float32) if not isinstance(dw, torch.Tensor) else dw
        
        # Final wind samples (real part only)
        wind_samples = torch.sqrt(2 * dw_tensor) * (G * phase_correction.unsqueeze(0)).real
        
        return wind_samples
