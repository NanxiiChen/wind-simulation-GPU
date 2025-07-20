import numpy as np
from typing import Dict, Tuple
from scipy.linalg import cholesky
from .psd import get_spectrum_class


class NumpyWindSimulator:
    """Stochastic wind field simulator class implemented using NumPy - consistent logic with JAX version."""

    def __init__(self, key=0, spectrum_type="kaimal-nd"):
        """
        Initialize the wind field simulator.
        
        Args:
            key: Random number seed
            spectrum_type: Type of wind spectrum to use
        """
        self.seed = key
        np.random.seed(key)
        self.params = self._set_default_parameters()
        self.spectrum = get_spectrum_class(spectrum_type)(**self.params)

    def _set_default_parameters(self) -> Dict:
        """Set default wind field simulation parameters."""
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
    def calculate_coherence(x_i, x_j, y_i, y_j, z_i, z_j, w, U_zi, U_zj, C_x, C_y, C_z):
        """Calculate spatial correlation function Coh."""
        distance_term = np.sqrt(
            C_x**2 * (x_i - x_j) ** 2
            + C_y**2 * (y_i - y_j) ** 2
            + C_z**2 * (z_i - z_j) ** 2
        )
        # Add numerical stability protection to avoid division by near-zero values
        denominator = 2 * np.pi * (U_zi + U_zj)
        safe_denominator = np.maximum(denominator, 1e-8)  # Set safe minimum value

        return np.exp(-2 * w * distance_term / safe_denominator)

    @staticmethod
    def calculate_cross_spectrum(S_ii, S_jj, coherence):
        """Calculate cross-spectral density function S_ij."""
        return np.sqrt(S_ii * S_jj) * coherence

    @staticmethod
    def calculate_simulation_frequency(N, dw):
        """Calculate simulation frequency array."""
        return np.arange(1, N + 1) * dw - dw / 2

    def build_spectrum_matrix(self, positions, wind_speeds, frequencies, component, **kwargs):
        """Build cross-spectral density matrix S(w) using the new spectrum interface."""
        n = positions.shape[0]
        num_freqs = len(frequencies)

        def _build_spectrum_for_position(freq, positions, component, **kwargs):
            s_values = self.spectrum.calculate_power_spectrum(
                freq, positions[:, 2], component, **kwargs
            )
            s_i = s_values.reshape(n, 1)  # [n, 1]
            s_j = s_values.reshape(1, n)  # [1, n]
            
            # Create grid coordinates
            x_i = positions[:, 0][:, np.newaxis].repeat(n, axis=1)  # [n, n]
            x_j = positions[:, 0][np.newaxis, :].repeat(n, axis=0)  # [n, n]
            y_i = positions[:, 1][:, np.newaxis].repeat(n, axis=1)  # [n, n]
            y_j = positions[:, 1][np.newaxis, :].repeat(n, axis=0)  # [n, n]
            z_i = positions[:, 2][:, np.newaxis].repeat(n, axis=1)  # [n, n]
            z_j = positions[:, 2][np.newaxis, :].repeat(n, axis=0)  # [n, n]
            U_i = wind_speeds[:, np.newaxis].repeat(n, axis=1)  # [n, n]
            U_j = wind_speeds[np.newaxis, :].repeat(n, axis=0)  # [n, n]
            
            coherence = self.calculate_coherence(
                x_i, x_j, y_i, y_j, z_i, z_j, freq, U_i, U_j,
                self.params["C_x"], self.params["C_y"], self.params["C_z"]
            )
            cross_spectrum = self.calculate_cross_spectrum(s_i, s_j, coherence)
            return cross_spectrum
        
        # Compute cross-spectral density matrix for each frequency point
        S_matrices = np.array([
            _build_spectrum_for_position(freq, positions, component, **kwargs)
            for freq in frequencies
        ])
        
        return S_matrices

    def simulate_wind(self, positions, wind_speeds, component="u", **kwargs):
        """
        Simulate fluctuating wind field.
        
        Args:
            positions: Array of shape (n, 3), each row represents (x, y, z) coordinates
            wind_speeds: Array of shape (n,), represents mean wind speed at each point
            component: Wind component, 'u' for along-wind, 'w' for vertical
            
        Returns:
            wind_samples: Array of shape (n, M), fluctuating wind time series at each point
            frequencies: Frequency array
        """
        np.random.seed(self.seed)
        self.seed += 1
        
        # Convert inputs to NumPy arrays
        positions = np.asarray(positions, dtype=np.float64)
        wind_speeds = np.asarray(wind_speeds, dtype=np.float64)
        
        return self._simulate_fluctuating_wind(
            positions, wind_speeds, component, **kwargs
        )

    def _simulate_fluctuating_wind(self, positions, wind_speeds, component, **kwargs):
        """Internal implementation of wind field simulation - corresponding to JAX version."""
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]

        # Calculate frequency
        frequencies = self.calculate_simulation_frequency(N, dw)

        # Build cross-spectral density matrix
        S_matrices = self.build_spectrum_matrix(
            positions, wind_speeds, frequencies, component, **kwargs
        )

        # Perform Cholesky decomposition for each frequency point
        H_matrices = np.zeros((N, n, n), dtype=np.complex128)
        for i in range(N):
            # Add small diagonal terms to improve numerical stability
            S_reg = S_matrices[i] + np.eye(n) * 1e-12
            H_matrices[i] = cholesky(S_reg, lower=True)

        # Generate random phases - same as JAX version
        phi = np.random.uniform(0, 2*np.pi, (n, N))

        # Calculate B matrix - 修正版本，与JAX版本保持一致
        B = np.zeros((n, M), dtype=np.complex128)

        for j in range(n):
            # 创建掩码矩阵，其中 mask[m] = True if m <= j
            m_indices = np.arange(n)  # [n,]
            mask = m_indices <= j  # [n,] 布尔掩码
            
            # H_matrices[l, j, m] 对所有频率l的H_{jm}
            H_jm_all = H_matrices[:, j, :]  # [N, n]
            
            # phi[m, l] -> phi.T 得到 [N, n]
            phi_transposed = phi.T  # [N, n]
            
            # 计算 exp(i * phi_{ml})
            exp_terms = np.exp(1j * phi_transposed)  # [N, n]
            
            # 应用掩码并求和
            # 将mask广播到[N, n]的形状
            mask_expanded = np.broadcast_to(mask, (N, n))  # [N, n]
            masked_terms = np.where(mask_expanded, H_jm_all * exp_terms, 0.0)  # [N, n]
            B_values = np.sum(masked_terms, axis=1)  # [N,]
            
            # 将B_values放入B矩阵的前N个位置，其余位置保持为0
            B[j, :N] = B_values
        
        # FFT transform
        G = np.fft.ifft(B) * M
        
        # Calculate wind field samples
        wind_samples = np.zeros((n, M))
        p_indices = np.arange(M)
        exp_factor = np.exp(1j * (p_indices * np.pi / M))
        
        for j in range(n):
            wind_samples[j] = np.sqrt(2 * dw) * np.real(G[j] * exp_factor)
        
        return wind_samples, frequencies