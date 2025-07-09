from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit, random, vmap
from jax.scipy.linalg import cholesky


class JaxWindSimulator:
    """Stochastic wind field simulator class."""

    def __init__(self, key=0):
        """
        Initialize the wind field simulator.

        Args:
            key: JAX random number seed
        """
        self.key = random.PRNGKey(key)
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

    @staticmethod
    @jit
    def calculate_friction_velocity(Z, U_d, z_0, z_d, K):
        """Calculate wind friction velocity u_*."""
        return K * U_d / jnp.log((Z - z_d) / z_0)

    @staticmethod
    @jit
    def calculate_f(n, Z, U_d):
        """Calculate dimensionless frequency f."""
        return n * Z / U_d

    @staticmethod
    @jit
    def calculate_power_spectrum_u(n, u_star, f):
        """Calculate along-wind fluctuating wind power spectral density S_u(n)."""
        return (u_star**2 / n) * (200 * f / ((1 + 50 * f) ** (5 / 3)))

    @staticmethod
    @jit
    def calculate_power_spectrum_w(n, u_star, f):
        """Calculate vertical fluctuating wind power spectral density S_w(n)."""
        return (u_star**2 / n) * (6 * f / ((1 + 4 * f) ** 2))

    @staticmethod
    @jit
    def calculate_coherence(x_i, x_j, y_i, y_j, z_i, z_j, w, U_zi, U_zj, C_x, C_y, C_z):
        """Calculate spatial correlation function Coh."""
        distance_term = jnp.sqrt(
            C_x**2 * (x_i - x_j) ** 2
            + C_y**2 * (y_i - y_j) ** 2
            + C_z**2 * (z_i - z_j) ** 2
        )
        # Add numerical stability protection to avoid division by near-zero values
        denominator = 2 * jnp.pi * (U_zi + U_zj)
        safe_denominator = jnp.maximum(denominator, 1e-8)  # Set safe minimum value

        return jnp.exp(-2 * w * distance_term / safe_denominator)

    @staticmethod
    @jit
    def calculate_cross_spectrum(S_ii, S_jj, coherence):
        """Calculate cross-spectral density function S_ij."""
        return jnp.sqrt(S_ii * S_jj) * coherence

    @staticmethod
    def calculate_simulation_frequency(N, dw):
        """Calculate simulation frequency array."""
        # return jnp.array([(l - 0.5) * dw for l in range(1, N + 1)])
        return jnp.arange(1, N + 1) * dw - dw / 2
    
    @partial(jit, static_argnums=(0,3))
    def calculate_spectrum_for_position(self, freq, positions, spectrum_func):
        u_stars = self.calculate_friction_velocity(
            positions[:, 2],
            self.params["U_d"],
            self.params["z_0"],
            self.params["z_d"],
            self.params["K"],
        )
        f_values = self.calculate_f(freq, positions[:, 2], self.params["U_d"])
        S_values = spectrum_func(freq, u_stars, f_values)
        return S_values


    def build_spectrum_matrix(self, positions, wind_speeds, frequencies, spectrum_func):
        """Build cross-spectral density matrix S(w)."""
        n = positions.shape[0]

        x_i = jnp.expand_dims(positions[:, 0], 1).repeat(n, axis=1)  # [n, n]
        x_j = jnp.expand_dims(positions[:, 0], 0).repeat(n, axis=0)  # [n, n]
        y_i = jnp.expand_dims(positions[:, 1], 1).repeat(n, axis=1)  # [n, n]
        y_j = jnp.expand_dims(positions[:, 1], 0).repeat(n, axis=0)  # [n, n]
        z_i = jnp.expand_dims(positions[:, 2], 1).repeat(n, axis=1)  # [n, n]
        z_j = jnp.expand_dims(positions[:, 2], 0).repeat(n, axis=0)  # [n, n]

        U_i = jnp.expand_dims(wind_speeds, 1).repeat(n, axis=1)  # [n, n]
        U_j = jnp.expand_dims(wind_speeds, 0).repeat(n, axis=0)  # [n, n]

        @partial(jit, static_argnums=(2,))
        def _build_spectrum_for_position(freq, positions, spectrum_func):
            s_values = self.calculate_spectrum_for_position(
                freq, positions, spectrum_func
            )
            s_i = s_values.reshape(n, 1)  # [n, 1]
            s_j = s_values.reshape(1, n)  # [1, n]
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
        )(frequencies, positions, spectrum_func)
        
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
        self.key, subkey = random.split(self.key)
        if not isinstance(positions, jnp.ndarray):
            positions = jnp.array(positions)

        return self._simulate_fluctuating_wind(
            positions, wind_speeds, subkey, direction
        )

    def _simulate_fluctuating_wind(self, positions, wind_speeds, key, direction):
        """Internal implementation of wind field simulation."""
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]

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
            """
            计算第j个点的B值，完全向量化的实现
            B_j(w_l) = sum_{m=1}^{j} H_{jm}(w_l) * exp(i * phi_{ml})
            """
            # 创建掩码矩阵 [N, n]，其中 mask[l, m] = True if m <= j
            m_indices = jnp.arange(n)  # [n,]
            mask = m_indices <= j  # [n,] 布尔掩码
            
            # H_matrices[l, j, m] 对所有频率l的H_{jm}
            H_jm_all = H_matrices[:, j, :]  # [N, n]
            
            # phi[m, l] -> phi[:, :] -> phi.T 得到 [N, n]
            phi_transposed = phi.T  # [N, n]
            
            # 计算 exp(i * phi_{ml})
            exp_terms = jnp.exp(1j * phi_transposed)  # [N, n]
            
            # 应用掩码并求和
            masked_terms = jnp.where(mask, H_jm_all * exp_terms, 0.0)  # [N, n]
            B_values = jnp.sum(masked_terms, axis=1)  # [N,]
            
            # 零填充到M长度
            return jnp.pad(B_values, (0, M - N), mode="constant")

        # Parallel computation of B for each point
        B = vmap(compute_B_for_point, in_axes=(0, None, None, None, None, None))(
            jnp.arange(n), N, M, n, H_matrices, phi
        )
        
        # FFT计算G
        G = vmap(jit(jnp.fft.ifft))(B) * M

        # Calculate wind field samples
        @jit
        def compute_samples_for_point(j):
            p_indices = jnp.arange(M)
            exponent = jnp.exp(1j * (p_indices * jnp.pi / M))
            terms = G[j] * exponent
            return jnp.sqrt(2 * dw) * jnp.real(terms)

        wind_samples = vmap(compute_samples_for_point)(jnp.arange(n))

        return wind_samples, frequencies
