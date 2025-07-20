from typing import Dict

import torch
import torch.func as func
from torch import Tensor
from .psd import get_spectrum_class


class TorchWindSimulator:
    """Stochastic wind field simulator class implemented using PyTorch."""

    def __init__(self, key=0, spectrum_type="kaimal-nd"):
        """
        Initialize the wind field simulator.

        Args:
            key: Random number seed
            spectrum_type: Type of wind spectrum to use
        """
        self.seed = key
        torch.manual_seed(key)

        # Set computing device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def calculate_coherence(
        x_i: Tensor,
        x_j: Tensor,
        y_i: Tensor,
        y_j: Tensor,
        z_i: Tensor,
        z_j: Tensor,
        w: Tensor,
        U_zi: Tensor,
        U_zj: Tensor,
        C_x: float,
        C_y: float,
        C_z: float,
    ) -> Tensor:
        """Calculate spatial correlation function Coh."""
        # Get device from input tensor
        device = x_i.device
        
        # Convert to tensor
        C_x_t = torch.tensor(C_x, device=device)
        C_y_t = torch.tensor(C_y, device=device)
        C_z_t = torch.tensor(C_z, device=device)
        
        # Compute using PyTorch entirely
        distance_term = torch.sqrt(
            C_x_t**2 * (x_i - x_j) ** 2
            + C_y_t**2 * (y_i - y_j) ** 2
            + C_z_t**2 * (z_i - z_j) ** 2
        )

        # Use PyTorch π constant
        denominator = 2 * torch.pi * (U_zi + U_zj)
        safe_denominator = torch.maximum(
            denominator, 
            torch.tensor(1e-8, device=device)
        )

        return torch.exp(-2 * w * distance_term / safe_denominator)

    @staticmethod
    def calculate_cross_spectrum(
        S_ii: Tensor, S_jj: Tensor, coherence: Tensor
    ) -> Tensor:
        """Calculate cross-spectral density function S_ij."""
        return torch.sqrt(S_ii * S_jj) * coherence

    @staticmethod
    def calculate_simulation_frequency(N: int, dw: float, device=None) -> Tensor:
        """Calculate simulation frequency array."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.arange(1, N + 1, device=device) * dw - dw / 2

    def build_spectrum_matrix(
        self, positions: Tensor, wind_speeds: Tensor, frequencies: Tensor, component, **kwargs
    ) -> Tensor:
        """
        Build cross-spectral density matrix S(w) using the new spectrum interface.
        """
        positions = torch.as_tensor(positions, device=self.device)
        wind_speeds = torch.as_tensor(wind_speeds, device=self.device)
        frequencies = torch.as_tensor(frequencies, device=self.device)

        n = positions.shape[0]
        num_freqs = len(frequencies)

        def _build_spectrum_for_position(freq, positions, component, **kwargs):
            s_values = self.spectrum.calculate_power_spectrum(
                freq, positions[:, 2], component, **kwargs
            )
            s_i = s_values.reshape(n, 1)  # [n, 1]
            s_j = s_values.reshape(1, n)  # [1, n]
            
            # Create grid for spatial coordinates
            x_i = positions[:, 0].unsqueeze(1).expand(n, n)  # [n, n]
            x_j = positions[:, 0].unsqueeze(0).expand(n, n)  # [n, n]
            y_i = positions[:, 1].unsqueeze(1).expand(n, n)  # [n, n]
            y_j = positions[:, 1].unsqueeze(0).expand(n, n)  # [n, n]
            z_i = positions[:, 2].unsqueeze(1).expand(n, n)  # [n, n]
            z_j = positions[:, 2].unsqueeze(0).expand(n, n)  # [n, n]
            U_i = wind_speeds.unsqueeze(1).expand(n, n)  # [n, n]
            U_j = wind_speeds.unsqueeze(0).expand(n, n)  # [n, n]
            
            coherence = self.calculate_coherence(
                x_i, x_j, y_i, y_j, z_i, z_j, freq, U_i, U_j,
                self.params["C_x"], self.params["C_y"], self.params["C_z"]
            )
            cross_spectrum = self.calculate_cross_spectrum(s_i, s_j, coherence)
            return cross_spectrum
        
        # Compute cross-spectral density matrix for each frequency point
        S_matrices = torch.stack([
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
        if not isinstance(positions, Tensor):
            positions = torch.from_numpy(positions)
        return self._simulate_fluctuating_wind(positions, wind_speeds, component, **kwargs)

    def _simulate_fluctuating_wind(self, positions, wind_speeds, component, **kwargs):
        """Internal implementation of wind field simulation."""
        positions = torch.as_tensor(positions, device=self.device)
        wind_speeds = torch.as_tensor(wind_speeds, device=self.device)

        n = positions.shape[0]
        N = self._to_tensor(self.params["N"], device=self.device)
        M = self._to_tensor(self.params["M"], device=self.device)
        dw = self._to_tensor(self.params["dw"], device=self.device)

        frequencies = self.calculate_simulation_frequency(int(N.item()), float(dw.item()), device=self.device)

        # Build cross-spectral density matrix
        S_matrices = self.build_spectrum_matrix(
            positions, wind_speeds, frequencies, component, **kwargs
        )

        # Perform Cholesky decomposition for each frequency point - use vmap instead of loop
        def cholesky_with_reg(S):
            return torch.linalg.cholesky(
                S + torch.eye(n, device=self.device) * 1e-12
            )


        H_matrices = func.vmap(cholesky_with_reg)(S_matrices)

        # Modify B matrix computation in _simulate_fluctuating_wind method
        N_int = int(N.item()) if isinstance(N, torch.Tensor) else int(N)
        M_int = int(M.item()) if isinstance(M, torch.Tensor) else int(M)

        # Generate random phases
        torch.manual_seed(self.seed)  # Ensure reproducibility
        phi = torch.rand((n, N_int), device=self.device) * 2 * torch.pi

        # Initialize B matrix
        B = torch.zeros((n, M_int), dtype=torch.complex64, device=self.device)

        # 计算B矩阵 - 修正版本，与JAX版本保持一致
        for j in range(n):
            # 创建掩码矩阵，其中 mask[m] = True if m <= j
            m_indices = torch.arange(n, device=self.device)  # [n,]
            mask = m_indices <= j  # [n,] 布尔掩码
            
            # H_matrices[l, j, m] 对所有频率l的H_{jm}
            H_jm_all = H_matrices[:, j, :]  # [N, n]
            
            # phi[m, l] -> phi.T 得到 [N, n]
            phi_transposed = phi.t()  # [N, n]
            
            # 计算 exp(i * phi_{ml})
            exp_terms = torch.exp(1j * phi_transposed)  # [N, n]
            
            # 应用掩码并求和
            # 将mask广播到[N, n]的形状
            mask_expanded = mask.unsqueeze(0).expand(N_int, -1)  # [N, n]
            masked_terms = torch.where(mask_expanded, H_jm_all * exp_terms, 0.0)  # [N, n]
            B_values = torch.sum(masked_terms, dim=1)  # [N,]
            
            # 将B_values放入B矩阵的前N个位置，其余位置保持为0
            B[j, :N_int] = B_values

        # Compute FFT
        G = torch.fft.ifft(B, dim=1) * M_int

        # Compute wind field samples - can keep vectorized
        p_indices = torch.arange(M, device=self.device)
        exponent = torch.exp(1j * (p_indices * torch.pi / M))
        wind_samples = torch.sqrt(2 * dw) * (G * exponent.unsqueeze(0)).real

        # Convert to NumPy array to ensure consistent output with JAX version
        return wind_samples.cpu().numpy(), frequencies.cpu().numpy()
