from typing import Dict

import torch
import torch.func as func
from torch import Tensor


class TorchWindSimulator:
    """Stochastic wind field simulator class implemented using PyTorch."""

    def __init__(self, key=0):
        """
        Initialize the wind field simulator.

        Args:
            key: Random number seed
        """
        self.seed = key
        torch.manual_seed(key)

        # Set computing device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = self._set_default_parameters()

    def _set_default_parameters(self) -> Dict:
        """Set default wind field simulation parameters."""
        params = {
            "K": 0.4,  # Dimensionless constant
            "H_bar": 10.0,  # Average height of surrounding buildings (m)
            "z_0": 0.05,  # Surface roughness height
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

    def calculate_friction_velocity(
        self, Z: Tensor, U_d: float, z_0: float, z_d: float, K: float
    ) -> Tensor:
        """Calculate wind friction velocity u_*."""
        return K * U_d / torch.log((Z - z_d) / z_0)

    def calculate_f(self, n: Tensor, Z: Tensor, U_d: float) -> Tensor:
        """Calculate dimensionless frequency f."""
        return n * Z / U_d

    def calculate_power_spectrum_u(
        self, n: Tensor, u_star: Tensor, f: Tensor
    ) -> Tensor:
        """Calculate along-wind fluctuating wind power spectral density S_u(n)."""
        return (u_star**2 / n) * (200 * f / ((1 + 50 * f) ** (5 / 3)))

    def calculate_power_spectrum_w(
        self, n: Tensor, u_star: Tensor, f: Tensor
    ) -> Tensor:
        """Calculate vertical fluctuating wind power spectral density S_w(n)."""
        return (u_star**2 / n) * (6 * f / ((1 + 4 * f) ** 2))

    def calculate_coherence(
        self,
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
        # Convert to tensor
        C_x_t = self._to_tensor(C_x, device=self.device)
        C_y_t = self._to_tensor(C_y, device=self.device)
        C_z_t = self._to_tensor(C_z, device=self.device)
        
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
            torch.tensor(1e-8, device=self.device)
        )

        return torch.exp(-2 * w * distance_term / safe_denominator)

    def calculate_cross_spectrum(
        self, S_ii: Tensor, S_jj: Tensor, coherence: Tensor
    ) -> Tensor:
        """Calculate cross-spectral density function S_ij."""
        return torch.sqrt(S_ii * S_jj) * coherence

    def calculate_simulation_frequency(self, N: int, dw: float) -> Tensor:
        """Calculate simulation frequency array."""
        return torch.arange(1, N + 1, device=self.device) * dw - dw / 2

    def build_spectrum_matrix(
        self, positions: Tensor, wind_speeds: Tensor, frequencies: Tensor, spectrum_func
    ) -> Tensor:
        """
        Build cross-spectral density matrix S(w) - vectorized implementation.
        """
        positions = torch.as_tensor(positions, device=self.device)
        wind_speeds = torch.as_tensor(wind_speeds, device=self.device)
        frequencies = torch.as_tensor(frequencies, device=self.device)

        n = positions.shape[0]
        num_freqs = len(frequencies)

        # Calculate friction velocity at each point
        u_stars = func.vmap(
            lambda z: self.calculate_friction_velocity(
                z,
                self.params["U_d"],
                self.params["z_0"],
                self.params["z_d"],
                self.params["K"],
            )
        )(positions[:, 2])

        # Calculate dimensionless frequency for all frequency points - avoid nested vmap
        f_values_all = torch.zeros((num_freqs, n), device=self.device)
        for freq_idx in range(num_freqs):
            freq = frequencies[freq_idx]
            f_values_all[freq_idx] = func.vmap(
                lambda z: self.calculate_f(freq, z, self.params["U_d"])
            )(positions[:, 2])

        # Calculate power spectral density for all frequency points - avoid nested vmap
        S_values_all = torch.zeros((num_freqs, n), device=self.device)
        for freq_idx in range(num_freqs):
            freq = frequencies[freq_idx]
            S_values_all[freq_idx] = func.vmap(
                lambda u_star, f_val: spectrum_func(freq, u_star, f_val)
            )(u_stars, f_values_all[freq_idx])

        # Create cross-spectral density matrix - fully vectorized implementation
        S_matrices = torch.zeros(
            (num_freqs, n, n), device=self.device, dtype=torch.float32
        )

        # Put auto-spectral densities on diagonal
        for freq_idx in range(num_freqs):
            S_matrices[freq_idx].diagonal().copy_(S_values_all[freq_idx])

        # Create grid to compute all point pairs
        x_i = positions[:, 0].unsqueeze(1).expand(n, n)  # [n, n]
        x_j = positions[:, 0].unsqueeze(0).expand(n, n)  # [n, n]
        y_i = positions[:, 1].unsqueeze(1).expand(n, n)  # [n, n]
        y_j = positions[:, 1].unsqueeze(0).expand(n, n)  # [n, n]
        z_i = positions[:, 2].unsqueeze(1).expand(n, n)  # [n, n]
        z_j = positions[:, 2].unsqueeze(0).expand(n, n)  # [n, n]
        U_i = wind_speeds.unsqueeze(1).expand(n, n)  # [n, n]
        U_j = wind_speeds.unsqueeze(0).expand(n, n)  # [n, n]

        # Batch compute cross-spectra for each frequency point
        for freq_idx in range(num_freqs):
            freq = frequencies[freq_idx]

            coh = self.calculate_coherence(
                x_i, x_j, y_i, y_j, z_i, z_j, freq,
                U_i, U_j,
                self.params["C_x"], self.params["C_y"], self.params["C_z"]
            )
            S_matrices[freq_idx] = self.calculate_cross_spectrum(
                S_values_all[freq_idx].unsqueeze(1).expand(n, n),  # S_ii
                S_values_all[freq_idx].unsqueeze(0).expand(n, n),  # S_jj
                coh
            )

        return S_matrices

    def simulate_wind(self, positions, wind_speeds, direction="u"):
        """
        Simulate fluctuating wind field.

        Args:
            positions: Array of shape (n, 3), each row represents (x, y, z) coordinates
            wind_speeds: Array of shape (n,), represents mean wind speed at each point
            direction: Wind direction, 'u' for along-wind, 'w' for vertical

        Returns:
            wind_samples: Array of shape (n, M), fluctuating wind time series at each point
            frequencies: Frequency array
        """
        if not isinstance(positions, Tensor):
            positions = torch.from_numpy(positions)
        return self._simulate_fluctuating_wind(positions, wind_speeds, direction)

    def _simulate_fluctuating_wind(self, positions, wind_speeds, direction):
        """Internal implementation of wind field simulation."""
        positions = torch.as_tensor(positions, device=self.device)
        wind_speeds = torch.as_tensor(wind_speeds, device=self.device)

        n = positions.shape[0]
        N = self._to_tensor(self.params["N"], device=self.device)
        M = self._to_tensor(self.params["M"], device=self.device)
        dw = self._to_tensor(self.params["dw"], device=self.device)

        frequencies = self.calculate_simulation_frequency(N, dw)
        spectrum_func = (
            self.calculate_power_spectrum_u
            if direction == "u"
            else self.calculate_power_spectrum_w
        )

        # Build cross-spectral density matrix
        S_matrices = self.build_spectrum_matrix(
            positions, wind_speeds, frequencies, spectrum_func
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
        phi = torch.rand((n, n, N_int), device=self.device) * 2 * torch.pi

        # Initialize B matrix
        B = torch.zeros((n, M_int), dtype=torch.complex64, device=self.device)

        for j in range(n):
            mask = torch.arange(n, device=self.device) <= j
            H_terms = H_matrices[:, j, :]
            H_masked = H_terms * mask
            H_masked = H_masked.to(torch.complex64) 
            phi_masked = phi[j, :, :] * mask.reshape(n, 1)
            exp_terms = torch.exp(1j * phi_masked.transpose(1, 0))
            exp_masked = exp_terms * mask

            B[j, :N_int] = torch.einsum("li,li->l", H_masked, exp_masked)

        # Compute FFT
        G = torch.fft.ifft(B, dim=1) * M_int

        # Compute wind field samples - can keep vectorized
        p_indices = torch.arange(M, device=self.device)
        exponent = torch.exp(1j * (p_indices * torch.pi / M))
        wind_samples = torch.sqrt(2 * dw) * (G * exponent.unsqueeze(0)).real

        # Convert to NumPy array to ensure consistent output with JAX version
        return wind_samples.cpu().numpy(), frequencies.cpu().numpy()
