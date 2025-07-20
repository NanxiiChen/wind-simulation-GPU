from typing import Dict, Tuple, Optional, Union, List
import torch
import torch.func as func
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from .simulator import TorchWindSimulator


class TorchWindVisualizer:
    """
    Wind field visualization class implemented using PyTorch.
    
    This class provides functionality for visualizing wind field simulation results,
    including power spectral density plots and cross-correlation analysis.
    PyTorch backend offers GPU acceleration for computational operations.
    """

    def __init__(self, key=0, simulator: Optional[TorchWindSimulator] = None, **kwargs):
        """
        Initialize the wind field visualizer.

        Args:
            key (int): Random number seed for reproducible results
            simulator (Optional[TorchWindSimulator]): Wind field simulator instance, 
                                                     creates new instance if None
            **kwargs: Additional parameters passed to the simulator
        """
        self.seed = key
        torch.manual_seed(key)
        self.simulator = simulator if simulator else TorchWindSimulator(key)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 设置参数
        self.params = {}
        for key, value in kwargs.items():
            self.params[key] = value

        # 如果没有提供参数，使用模拟器的参数
        if not self.params and hasattr(self.simulator, "params"):
            self.params = self.simulator.params

    def _to_tensor(self, value, device=None):
        """
        Safely convert input to tensor on device.

        Args:
            value: Value to convert (tensor or other type)
            device: Target device, defaults to class device

        Returns:
            torch.Tensor: Tensor on specified device
        """
        if device is None:
            device = self.device

        if isinstance(value, torch.Tensor):
            return value.to(device)
        else:
            return torch.tensor(value, device=device)


    def plot_psd(
        self,
        wind_samples: np.ndarray,
        Zs: torch.Tensor,
        show_num: int = 5,
        save_path: Optional[str] = None,
        show: bool = True,
        component: str = "u",
        indices: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """
        Plot power spectral density of wind field simulation results.

        Args:
            wind_samples (np.ndarray): Wind speed time series, shape (n_points, n_timesteps)
            Zs (torch.Tensor): Heights at each point, shape (n_points,)
            show_num (int): Number of sampling points to display (default: 5)
            save_path (Optional[str]): Path to save the image, None if not saving
            show (bool): Whether to display the plot (default: True)
            component (str): Wind component, 'u' for along-wind, 'w' for vertical
            indices (Optional[Union[int, Tuple[int, int]]]): Point indices to plot
            **kwargs: Additional plotting parameters
        """
        n = wind_samples.shape[0]

        # 随机选择点进行显示
        torch.manual_seed(self.seed)
        if indices is None:
            if show_num >= n:
                indices = torch.arange(n)
            else:
                indices = torch.randperm(n)[:show_num]
            indices = indices.tolist()
        elif isinstance(indices, int):
            indices = (indices,)
        elif isinstance(indices, tuple) or isinstance(indices, list):
            pass
        else:
            raise ValueError("indices must be an integer or integer tuple")

        ncol = kwargs.get("ncol", 3)
        nrow = (len(indices) + ncol - 1) // ncol

        # frequencies_theory, S_theory = func.vmap(
        #     self._calculate_theoretical_spectrum, in_dims=(0, None)
        # )(Zs, component)

        frequencies_theory = self.simulator.calculate_simulation_frequency(
            self.params["N"], self.params["dw"]
        ).to(self.device)
        S_theory = func.vmap(
            self.simulator.spectrum.calculate_power_spectrum,
            in_dims=(0, None, None),
        )(
            frequencies_theory,
            Zs.to(self.device),
            component,
        )



        frequencies_theory = frequencies_theory.cpu().numpy()
        S_theory = S_theory.cpu().numpy()

        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 5 * nrow))
        axes = axes.flatten() if nrow > 1 else [axes]
        # Calculate and plot power spectral density for each sampling point
        for idx, i in enumerate(indices):
            ax = axes[idx]
            data = wind_samples[i]
            frequencies, psd = signal.welch(
                data,
                fs=1 / self.params["dt"],
                nperseg=1024,
                scaling="density",
                window="hann",
            )
            ax.loglog(frequencies, psd, label=f"Point {i+1}")

            # Plot theoretical spectrum line
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
                title=f"PSD of {component.upper()} Wind at Z={Zs[i].item():.2f} m (Point {i+1})",
            )

            ax.grid(True, which="both", ls="-", alpha=0.6)
            ax.legend()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

    def _compute_correlation(
        self, data_i: np.ndarray, data_j: np.ndarray
    ) -> np.ndarray:
        """
        Compute cross-correlation function between two time series.
        
        Args:
            data_i (np.ndarray): First time series data
            data_j (np.ndarray): Second time series data
            
        Returns:
            np.ndarray: Normalized cross-correlation function
        """
        data_i_centered = data_i - np.mean(data_i)
        data_j_centered = data_j - np.mean(data_j)

        correlation = signal.correlate(data_i_centered, data_j_centered, mode="full")
        corr_max = np.max(np.abs(correlation))
        return correlation / (corr_max if corr_max > 0 else 1.0)

    def _calculate_theoretical_correlation(
        self, S_ii: Tensor, S_jj: Tensor, coherence: Tensor, M: int
    ) -> np.ndarray:
        """
        Calculate theoretical cross-correlation function via inverse Fourier transform.
        
        Args:
            S_ii (Tensor): Auto-spectral density at point i
            S_jj (Tensor): Auto-spectral density at point j
            coherence (Tensor): Coherence function between points i and j
            M (int): Number of time points
            
        Returns:
            np.ndarray: Theoretical cross-correlation function
        """
        # Convert inputs to PyTorch tensors
        S_ii = torch.as_tensor(S_ii, device=self.device)
        S_jj = torch.as_tensor(S_jj, device=self.device)
        coherence = torch.as_tensor(coherence, device=self.device)

        # Calculate cross-spectral density
        cross_spectrum = torch.sqrt(S_ii * S_jj) * coherence

        # Prepare complete frequency spectrum
        full_spectrum = torch.zeros(M, dtype=torch.complex64, device=self.device)
        N = len(coherence)
        full_spectrum[1 : N + 1] = torch.complex(
            cross_spectrum, torch.zeros_like(cross_spectrum)
        )
        full_spectrum[M - N :] = torch.flip(
            torch.conj(full_spectrum[1 : N + 1]), dims=[0]
        )

        # Execute IFFT to get theoretical cross-correlation function
        theo_correlation = torch.real(torch.fft.ifft(full_spectrum))
        theo_correlation = torch.fft.fftshift(theo_correlation)

        theo_max = torch.max(torch.abs(theo_correlation))
        if theo_max > 0:
            theo_correlation = theo_correlation / theo_max

        return theo_correlation.cpu().numpy()

    def plot_cross_correlation(
        self,
        wind_samples: np.ndarray,
        positions: np.ndarray,
        wind_speeds: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True,
        component: str = "u",
        indices: Optional[Union[int, Tuple[int, int]]] = None,
        downsample: int = 1,
        **kwargs,
    ):
        """
        Plot cross-correlation function and compare with theoretical values.

        Args:
            wind_samples (np.ndarray): Wind speed time series, shape (n_points, n_timesteps)
            positions (np.ndarray): Position coordinates, shape (n_points, 3)
            wind_speeds (np.ndarray): Mean wind speeds, shape (n_points,)
            save_path (Optional[str]): Path to save the image, None if not saving
            show (bool): Whether to display the plot
            component (str): Wind component, 'u' for along-wind, 'w' for vertical
            indices (Optional[Union[int, Tuple[int, int]]]): Indices of two points for cross-correlation
            downsample (int): Downsampling factor to speed up computation
            **kwargs: Additional parameters

        Returns:
            Tuple[int, int]: Indices pair used for computation
        """
        n = wind_samples.shape[0]

        # 应用降采样
        if downsample > 1:
            wind_samples = wind_samples[:, ::downsample]
            dt = self.params["dt"] * downsample
        else:
            dt = self.params["dt"]

        # 确定要使用的点索引
        torch.manual_seed(self.seed)
        if indices is None:
            idx = torch.randint(0, n, (1,)).item()
            indices = idx, idx
        elif isinstance(indices, int):
            indices = (indices, indices)
        elif isinstance(indices, tuple) and len(indices) == 2:
            pass
        else:
            raise ValueError("indices必须是整数或两个整数的元组")

        i, j = indices
        data_i = wind_samples[i]
        data_j = wind_samples[j]

        # 计算实际互相关函数
        correlation = self._compute_correlation(data_i, data_j)
        lags = np.arange(-len(data_i) + 1, len(data_i))
        lag_times = lags * dt

        # 提取位置信息
        x_i, y_i, z_i = positions[i]
        x_j, y_j, z_j = positions[j]
        U_zi, U_zj = wind_speeds[i], wind_speeds[j]

        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]

        frequencies = self.simulator.calculate_simulation_frequency(N, dw)
        S_ii = self.simulator.spectrum.calculate_power_spectrum(
            frequencies, self._to_tensor(z_i, device=self.device), component
        ).cpu().numpy()
        S_jj = self.simulator.spectrum.calculate_power_spectrum(
            frequencies, self._to_tensor(z_j, device=self.device), component
        ).cpu().numpy()
        frequencies = frequencies.cpu().numpy()


        # 计算相干函数
        coherence_list = []
        for freq in frequencies:
            # 使用角频率
            ang_freq = 2 * np.pi * freq
            coh = (
                self.simulator.calculate_coherence(
                    x_i,
                    x_j,
                    y_i,
                    y_j,
                    z_i,
                    z_j,
                    self._to_tensor(ang_freq, device=self.device),
                    U_zi,
                    U_zj,
                    self.params["C_x"],
                    self.params["C_y"],
                    self.params["C_z"],
                )
                .cpu()
                .numpy()
            )
            coherence_list.append(coh)

        coherence = np.array(coherence_list)

        # 计算理论互相关
        theo_correlation = self._calculate_theoretical_correlation(
            S_ii, S_jj, coherence, M
        )
        theo_lags = np.arange(-M // 2, M // 2)
        theo_lag_times = theo_lags * dt

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 8))

        mid = len(correlation) // 2
        range_points = kwargs.get("range_points", len(theo_lag_times) // 2)

        plot_corr = correlation[mid - range_points : mid + range_points + 1]
        plot_lag_times = lag_times[mid - range_points : mid + range_points + 1]

        # 截取相同范围的理论相关函数
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

        return indices
