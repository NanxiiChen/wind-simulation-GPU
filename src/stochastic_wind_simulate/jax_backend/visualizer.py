from typing import Dict, Tuple, Optional, Union, List
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap

from .simulator import JaxWindSimulator
from .psd import get_spectrum_class


class JaxWindVisualizer:
    """
    Wind field visualization class implemented using JAX.
    
    This class provides functionality for visualizing wind field simulation results,
    including power spectral density plots and cross-correlation analysis.
    JAX backend offers JIT compilation and automatic differentiation capabilities.
    """

    def __init__(self, key=0, 
                 simulator: JaxWindSimulator = None, 
                 spectrum_type: str = "kaimal",
                 **kwargs):
        """
        Initialize the wind field visualizer.

        Args:
            key (int): Random number seed for reproducible results
            simulator (Optional[JaxWindSimulator]): Wind field simulator instance, 
                                                   creates new instance if None
            spectrum_type (str): Type of spectrum model to use (default: "kaimal")
            **kwargs: Additional parameters passed to the simulator
        """
        self.key = random.PRNGKey(key)
        self.simulator = simulator if simulator else JaxWindSimulator(key)


    def plot_psd(
        self,
        wind_samples: jnp.ndarray,
        Zs: jnp.ndarray,
        show_num=5,
        save_path: str = None,
        show=True,
        component="u",
        indices: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """
        Plot power spectral density comparison between simulation and theory.

        Args:
            wind_samples (jnp.ndarray): Simulated wind field samples
            Zs (jnp.ndarray): Height coordinates
            show_num (int): Number of random samples to display
            save_path (Optional[str]): Path to save the plot
            show (bool): Whether to display the plot
            component (str): Wind component to plot ("u", "v", or "w")
            indices (Optional[Union[int, Tuple[int, int]]]): Specific indices to plot,
                if None, random indices are selected
            **kwargs: Additional plotting parameters

        Returns:
            None
        """
        n = wind_samples.shape[0]

        if indices is None:
            self.key, subkey = random.split(self.key)
            indices = random.randint(subkey, (show_num,), 0, n)
        elif isinstance(indices, int):
            indices = (indices,)
        elif isinstance(indices, tuple) and len(indices) == 2:
            pass
        else:
            raise ValueError("indices must be an integer or sequence of integers")

        ncol = kwargs.get("ncol", 3)
        nrow = (len(indices) + ncol - 1) // ncol
        frequencies_theory = self.simulator.calculate_simulation_frequency(
            self.simulator.params["N"], self.simulator.params["dw"]
        )
        S_theory = vmap(
            self.simulator.spectrum.calculate_power_spectrum,
            in_axes=(0, None, None),
        )(frequencies_theory, Zs, component)

        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 5 * nrow))
        axes = axes.flatten() if nrow > 1 else [axes]
        for idx, i in enumerate(indices):
            ax = axes[idx]
            data = wind_samples[i]
            frequencies, psd = jax.scipy.signal.welch(
                data,
                fs=1 / self.simulator.params["dt"],
                nperseg=1024,
                scaling="density",
                window="hann",
            )
            ax.loglog(frequencies, psd, label=f"Point {i+1}")

            ax.loglog(
                frequencies_theory,
                S_theory[:, i],
                "--",
                color="black",
                linewidth=2,
                label="Theoretical Reference",
            )
            ax.set(
                xlabel="Frequency (Hz)",
                ylabel="Power Spectral Density (m²/s²)",
                title=f"PSD of {component.upper()} Wind at Point {i+1} (Z={Zs[i]:.2f} m)",
            )

            ax.grid(True, which="both", ls="-", alpha=0.6)
            ax.legend()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

    def _compute_correlation(self, data_i, data_j):
        """
        Compute sample cross-correlation function with maximum value normalization.

        Args:
            data_i: Time series data for first point
            data_j: Time series data for second point

        Returns:
            Normalized cross-correlation array
        """
        data_i_centered = data_i - jnp.mean(data_i)
        data_j_centered = data_j - jnp.mean(data_j)
        
        correlation = jax.scipy.signal.correlate(
            data_i_centered, data_j_centered, mode="full"
        )
        
        # Use maximum value normalization (consistent with theoretical values)
        corr_max = jnp.max(jnp.abs(correlation))
        return correlation / (corr_max if corr_max > 0 else 1.0)

    def _calculate_theoretical_correlation(self, S_ii, S_jj, coherence, M):
        """
        Calculate theoretical cross-correlation function with maximum value normalization.

        Args:
            S_ii: Power spectral density at point i
            S_jj: Power spectral density at point j
            coherence: Coherence function between points
            M: Number of frequency points

        Returns:
            Normalized theoretical cross-correlation array
        """
        # Original calculation logic preserved
        cross_spectrum = jnp.sqrt(S_ii * S_jj) * coherence

        full_spectrum = jnp.zeros(M, dtype=jnp.complex64)
        N = len(coherence)
        full_spectrum = full_spectrum.at[1 : N + 1].set(cross_spectrum)
        full_spectrum = full_spectrum.at[M - N :].set(
            jnp.flip(jnp.conj(cross_spectrum))
        )

        theo_correlation = jnp.real(jnp.fft.ifft(full_spectrum))
        theo_correlation = jnp.fft.fftshift(theo_correlation)

        theo_max = jnp.max(jnp.abs(theo_correlation))
        return theo_correlation / (theo_max if theo_max > 0 else 1.0)


    def plot_cross_correlation(
        self,
        wind_samples,
        positions,
        wind_speeds,
        save_path=None,
        show=True,
        component="u",
        indices=None,
        downsample=1,
        **kwargs,
    ):
        """
        Plot cross-correlation function and compare with theoretical values.

        Args:
            wind_samples: Wind field samples array
            positions: Spatial positions of measurement points  
            wind_speeds: Wind speed values
            save_path (Optional[str]): Path to save the plot
            show (bool): Whether to display the plot
            component (str): Wind component to analyze ("u", "v", or "w")
            indices (Optional[Union[int, Tuple[int, int]]]): Point indices for correlation,
                if None, random indices are selected
            downsample (int): Downsampling factor for data
            **kwargs: Additional plotting parameters

        Returns:
            None
        """
        n = wind_samples.shape[0]

        if downsample > 1:
            wind_samples = wind_samples[:, ::downsample]
            dt = self.simulator.params["dt"] * downsample
        else:
            dt = self.simulator.params["dt"]

        self.key, subkey = random.split(self.key)
        if indices is None:
            idx = random.randint(subkey, (1,), 0, n).item()
            indices = idx, idx
        elif isinstance(indices, int):
            indices = (indices, indices)
        elif isinstance(indices, tuple) and len(indices) == 2:
            pass
        else:
            raise ValueError("indices must be an integer or tuple of two integers")

        i, j = indices
        data_i = wind_samples[i]
        data_j = wind_samples[j]

        # Compute actual cross-correlation function
        correlation = self._compute_correlation(data_i, data_j)
        lags = jnp.arange(-len(data_i) + 1, len(data_i))
        lag_times = lags * dt

        # Extract position information
        x_i, y_i, z_i = positions[i]
        x_j, y_j, z_j = positions[j]
        U_zi, U_zj = wind_speeds[i], wind_speeds[j]

        N = self.simulator.params["N"]
        M = self.simulator.params["M"]
        dw = self.simulator.params["dw"]

        frequencies = self.simulator.calculate_simulation_frequency(N, dw)
        S_ii = self.simulator.spectrum.calculate_power_spectrum(
            frequencies, z_i, component
        )
        S_jj = self.simulator.spectrum.calculate_power_spectrum(
            frequencies, z_j, component
        )

        coherence = vmap(
            lambda freq: self.simulator.calculate_coherence(
                x_i,
                x_j,
                y_i,
                y_j,
                z_i,
                z_j,
                freq,
                U_zi,
                U_zj,
                self.simulator.params["C_x"],
                self.simulator.params["C_y"],
                self.simulator.params["C_z"],
            )
        )(frequencies)

        theo_correlation = self._calculate_theoretical_correlation(
            S_ii, S_jj, coherence, M
        )
        theo_lags = jnp.arange(-M // 2, M // 2)
        theo_lag_times = theo_lags * dt

        fig, ax = plt.subplots(figsize=(12, 8))
        mid = len(correlation) // 2
        range_points = kwargs.get("range_points", len(theo_lag_times) // 2)
        plot_corr = correlation[mid - range_points : mid + range_points + 1]
        plot_lag_times = lag_times[mid - range_points : mid + range_points + 1]

        # Cut theoretical correlation function to same range
        theo_mid = len(theo_correlation) // 2
        theo_plot = theo_correlation[
            theo_mid - range_points : theo_mid + range_points + 1
        ]
        theo_plot_times = theo_lag_times[
            theo_mid - range_points : theo_mid + range_points + 1
        ]

        ax.plot(
            plot_lag_times,
            plot_corr,
            label="simulation",
        )
        ax.plot(
            theo_plot_times,
            theo_plot,
            "--",
            color="black",
            linewidth=2,
            label="theoretical",
        )

        plt.title(
            f"Cross Correlation of {component.upper()} Wind at Points {i+1} and {j+1}\n"
        )
        plt.xlabel("Time Lag (s)")
        plt.ylabel("Cross Correlation")
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

        if kwargs.get("return_data", False):
            return plot_lag_times, plot_corr, theo_plot_times, theo_plot
        
    # def plot_cross_coherence(
    #     self,
    #     wind_samples,
    #     positions,
    #     wind_speeds,
    #     save_path=None,
    #     show=True,
    #     component="u",
    #     indices=None,
    #     downsample=1,
    #     **kwargs,
    # ):
    #     """
    #     绘制互相干函数图
    #     """
    #     n = positions.shape[0]
    #     # 处理降采样
    #     if downsample > 1:
    #         wind_samples = wind_samples[:, ::downsample]
    #         dt = self.simulator.params["dt"] * downsample
    #     else:
    #         dt = self.simulator.params["dt"]
    
    #     # 选择要分析的点对
    #     self.key, subkey = random.split(self.key)
    #     if indices is None:
    #         # 如果未指定，随机选择一对点
    #         idx1 = random.randint(subkey, (1,), 0, n).item()
    #         self.key, subkey = random.split(self.key)
    #         idx2 = random.randint(subkey, (1,), 0, n).item()
    #         indices = idx1, idx2
    #     elif isinstance(indices, int):
    #         # 如果是单个索引，选择该点与自身
    #         indices = (indices, indices)
    #     elif isinstance(indices, tuple) and len(indices) == 2:
    #         pass
    #     else:
    #         raise ValueError("indices must be an integer or tuple of two integers")
        
    #     i, j = indices
    #     data_i = wind_samples[i]
    #     data_j = wind_samples[j]
    #     # 计算实测相干函数 - 使用Welch方法
    #     nperseg = kwargs.get("nperseg", min(1024, len(data_i) // 4))
        
    #     # 计算自谱和互谱
    #     fxx, Pxx = jax.scipy.signal.welch(
    #         data_i, fs=1/dt, nperseg=nperseg, 
    #         scaling="density", window="hann",
    #     )
    #     fyy, Pyy = jax.scipy.signal.welch(
    #         data_j, fs=1/dt, nperseg=nperseg, 
    #         scaling="density", window="hann",
    #     )
    #     fxy, Pxy = jax.scipy.signal.csd(
    #         data_i, data_j, fs=1/dt, nperseg=nperseg,
    #         scaling="density", window="hann",
    #     )

    #     measured_coherence = jnp.abs(Pxy)**2 / (Pxx * Pyy +1e-10)

    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     ax.loglog(fxx, Pxx, label=f"PSD of Point {i+1}")
    #     ax.loglog(fyy, Pyy, label=f"PSD of Point {j+1}")
    #     ax.loglog(fxy, Pxy, label=f"CSD of Points {i+1} and {j+1}")
    #     ax.set(
    #         title=f"Power Spectral Density and Cross Spectral Density of {component.upper()} Wind at Points {i+1} and {j+1}",
    #         xlabel="Frequency (Hz)",
    #         ylabel="Power Spectral Density (m²/s²)",
    #     )
    #     plt.show(block=False)


    #     x_i, y_i, z_i = positions[i]
    #     x_j, y_j, z_j = positions[j]
    #     U_zi, U_zj = wind_speeds[i], wind_speeds[j]
    #     theoretical_coherence = vmap(
    #         lambda freq: self.simulator.calculate_coherence(
    #             x_i, x_j, y_i, y_j, z_i, z_j,
    #             2 * jnp.pi * freq, U_zi, U_zj,
    #             self.simulator.params["C_x"], self.simulator.params["C_y"], self.simulator.params["C_z"]
    #         )
    #     )(fxx)

    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     ax.semilogy(fxy, measured_coherence, label="Measured Coherence")
    #     ax.semilogy(fxx, theoretical_coherence, '--', color='black', linewidth=2, label="Theoretical Coherence")
    #     ax.set(
    #         title=f"Cross Coherence of {component.upper()} Wind at Points {i+1} and {j+1}",
    #         xlabel="Frequency (Hz)",
    #         ylabel="Coherence",
    #     )
    #     ax.grid(True, which="both", ls="-", alpha=0.6)
    #     ax.legend()
    #     if save_path:
    #         plt.savefig(save_path)
    #     if show:
    #         plt.show()
    #     else:
    #         plt.close()

    #     return indices
