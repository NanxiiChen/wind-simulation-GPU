from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from .simulator import NumpyWindSimulator


class NumpyWindVisualizer:

    def __init__(self, key=0, simulator: NumpyWindSimulator = None, **kwargs):
        self.seed = key
        np.random.seed(key)
        self.simulator = simulator if simulator else NumpyWindSimulator(key)
        self.params = {}
        for key, value in kwargs.items():
            self.params[key] = value

    def plot_psd(
        self,
        wind_samples: np.ndarray,
        Zs: np.ndarray,
        show_num=5,
        save_path: str = None,
        show=True,
        direction="u",
        indices: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """绘制模拟结果"""
        n = wind_samples.shape[0]

        if indices is None:
            np.random.seed(self.seed)
            self.seed += 1
            indices = np.random.randint(0, n, size=show_num)
        elif isinstance(indices, int):
            indices = (indices,)
        elif isinstance(indices, tuple) and len(indices) == 2:
            pass
        else:
            raise ValueError("indices必须是整数或整数序列")

        ncol = kwargs.get("ncol", 3)
        nrow = (len(indices) + ncol - 1) // ncol
        
        # 计算理论谱
        frequencies_theory = []
        S_theory = []
        for z in Zs:
            freq, spec = self._calculate_theoretical_spectrum(z, direction)
            frequencies_theory.append(freq)
            S_theory.append(spec)

        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 5 * nrow))
        axes = axes.flatten() if nrow > 1 else [axes]
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

            ax.loglog(
                frequencies_theory[i],
                S_theory[i],
                "--",
                color="black",
                linewidth=2,
                label="Theoretical Reference",
            )
            ax.set(
                xlabel="Frequency (Hz)",
                ylabel="Power Spectral Density (m²/s²)",
                title=f"PSD of {direction.upper()} Wind at Point {i+1} (Z={Zs[i]:.2f} m)",
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
        """计算互相关函数"""
        data_i_centered = data_i - np.mean(data_i)
        data_j_centered = data_j - np.mean(data_j)
        correlation = signal.correlate(
            data_i_centered, data_j_centered, mode="full"
        )
        corr_max = np.max(np.abs(correlation))
        return correlation / (corr_max if corr_max > 0 else 1.0) 

    def _calculate_theoretical_correlation(self, S_ii, S_jj, coherence, M):
        """计算理论互相关函数（傅里叶逆变换）"""
        # 计算互谱密度
        cross_spectrum = np.sqrt(S_ii * S_jj) * coherence

        # 准备完整的频谱
        full_spectrum = np.zeros(M, dtype=np.complex128)
        N = len(coherence)
        full_spectrum[1:N+1] = cross_spectrum
        full_spectrum[M-N:] = np.flip(np.conj(cross_spectrum))

        # 执行IFFT获得理论互相关函数
        theo_correlation = np.real(np.fft.ifft(full_spectrum))
        theo_correlation = np.fft.fftshift(theo_correlation)

        theo_max = np.max(np.abs(theo_correlation))
        return theo_correlation / (theo_max if theo_max > 0 else 1.0)

    def _calculate_theoretical_spectrum(
        self, Z: float, direction: str = "u"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算理论功率谱"""
        u_star = self.simulator.calculate_friction_velocity(
            Z,
            self.params["U_d"],
            self.params["z_0"],
            self.params["z_d"],
            self.params["K"],
        )
        dw = self.params["dw"]
        N = self.params["N"]
        frequencies = self.simulator.calculate_simulation_frequency(N, dw)
        f_nondimensional = self.simulator.calculate_f(
            frequencies, Z, self.params["U_d"]
        )
        
        if direction == "u":
            S_theory = self.simulator.calculate_power_spectrum_u(
                frequencies, u_star, f_nondimensional
            )
        else:
            S_theory = self.simulator.calculate_power_spectrum_w(
                frequencies, u_star, f_nondimensional
            )
            
        return frequencies, S_theory

    def plot_cross_correlation(
        self,
        wind_samples,
        positions,
        wind_speeds,
        save_path=None,
        show=True,
        direction="u",
        indices=None,
        downsample=1,
        **kwargs,
    ):
        """绘制互相关函数并与理论值比较"""
        n = wind_samples.shape[0]

        if downsample > 1:
            wind_samples = wind_samples[:, ::downsample]
            dt = self.params["dt"] * downsample
        else:
            dt = self.params["dt"]

        np.random.seed(self.seed)
        self.seed += 1
        if indices is None:
            idx = np.random.randint(0, n)
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

        # 选择合适的谱函数
        spectrum_func = (
            self.simulator.calculate_power_spectrum_u
            if direction == "u"
            else self.simulator.calculate_power_spectrum_w
        )

        # 计算各点的摩阻速度
        u_star_i = self.simulator.calculate_friction_velocity(
            z_i,
            self.params["U_d"],
            self.params["z_0"],
            self.params["z_d"],
            self.params["K"],
        )
        u_star_j = self.simulator.calculate_friction_velocity(
            z_j,
            self.params["U_d"],
            self.params["z_0"],
            self.params["z_d"],
            self.params["K"],
        )

        # 计算无量纲频率
        f_i = self.simulator.calculate_f(frequencies, z_i, self.params["U_d"])
        f_j = self.simulator.calculate_f(frequencies, z_j, self.params["U_d"])

        # 计算自功率谱
        S_ii = spectrum_func(frequencies, u_star_i, f_i)
        S_jj = spectrum_func(frequencies, u_star_j, f_j)

        # 计算空间相关函数
        coherence = np.zeros_like(frequencies)
        for idx, freq in enumerate(frequencies):
            coherence[idx] = self.simulator.calculate_coherence(
                x_i,
                x_j,
                y_i,
                y_j,
                z_i,
                z_j,
                2 * np.pi * freq,  # 使用角频率
                U_zi,
                U_zj,
                self.params["C_x"],
                self.params["C_y"],
                self.params["C_z"],
            )

        theo_correlation = self._calculate_theoretical_correlation(
            S_ii, S_jj, coherence, M
        )
        theo_lags = np.arange(-M // 2, M // 2)
        theo_lag_times = theo_lags * dt

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
            f"Cross Correlation of {direction.upper()} Wind at Points {i+1} and {j+1}\n"
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