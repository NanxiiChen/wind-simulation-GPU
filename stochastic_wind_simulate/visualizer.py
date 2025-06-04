import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, random, vmap
from jax.scipy.linalg import cholesky

from .model import WindSimulator


class WindVisualizer:

    def __init__(self, key=0, simulator: WindSimulator = None, **kwargs):
        self.key = random.PRNGKey(key)
        self.simulator = simulator if simulator else WindSimulator(key)
        self.params = {}
        for key, value in kwargs.items():
            self.params[key] = value

    def plot_psd(
        self, wind_samples, Z, show_num=5, save_path=None, show=True, direction="u"
    ):
        """绘制模拟结果"""
        n = wind_samples.shape[0]  # 点的数量

        # 随机挑选 show_num个点进行绘图
        indices = random.choice(
            self.key, jnp.arange(n), shape=(show_num,), replace=False
        )

        fig, ax = plt.subplots(figsize=(12, 8))

        for i in indices:
            data = wind_samples[i]
            frequencies, psd = jax.scipy.signal.welch(
                data,
                fs=1 / self.params["dt"],
                nperseg=1024,
                scaling="density",
                window="hann",
            )
            ax.loglog(frequencies, psd, label=f"Point {i+1}")

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
        S_u_theory = (
            self.simulator.calculate_power_spectrum_u(
                frequencies, u_star, f_nondimensional
            )
            if direction == "u"
            else self.simulator.calculate_power_spectrum_w(
                frequencies, u_star, f_nondimensional
            )
        )
        ax.loglog(
            frequencies,
            S_u_theory,
            "--",
            color="black",
            linewidth=2,
            label="Theoretical Reference",
        )
        ax.set(
            xlabel="Frequency (Hz)",
            ylabel="Power Spectral Density (m²/s²)",
            title=f"PSD of {direction.upper()} Wind at Z={Z}m",
        )

        plt.grid(True, which="both", ls="-", alpha=0.6)
        plt.legend()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()
    
    @partial(jit, static_argnums=(0,))
    def _compute_correlation(self, data_i, data_j):
        """计算两个数据序列的互相关"""
        data_i_centered = data_i - jnp.mean(data_i)
        data_j_centered = data_j - jnp.mean(data_j)

        # 计算中心化后的互相关
        correlation = jax.scipy.signal.correlate(
            data_i_centered, data_j_centered, mode="full"
        )

        # 使用标准的归一化方法
        std_i = jnp.std(data_i_centered)
        std_j = jnp.std(data_j_centered)
        n = len(data_i)
        correlation = correlation / (n * std_i * std_j)

        return correlation
    

    def plot_cross_correlation(
        self,
        wind_samples,
        positions,
        wind_speeds,
        save_path=None,
        show=True,
        direction="u",
        indices=None,
        **kwargs,
    ):
        n = wind_samples.shape[0]
        dt = self.params["dt"]

        self.key, subkey = random.split(self.key)
        if indices is None:
            idx = random.randint(subkey, (1,), 0, n)
            indices = idx, idx
        elif isinstance(indices, int):
            indices = (indices, indices)
        elif isinstance(indices, tuple) and len(indices) == 2:
            pass
        else:
            raise ValueError("indices must be an int or a tuple of two ints")

        i, j = indices[0], indices[1]
        data_i = wind_samples[i]
        data_j = wind_samples[j]
        correlation = self._compute_correlation(data_i, data_j)

        lags = jnp.arange(-len(data_i) + 1, len(data_i))
        lag_times = lags * dt

        x_i, y_i, z_i = positions[i]
        x_j, y_j, z_j = positions[j]
        U_zi, U_zj = wind_speeds[i], wind_speeds[j]
        distance = jnp.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)

        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]
        frequencies = self.simulator.calculate_simulation_frequency(N, dw)

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

        # 计算相干函数和互谱密度
        # coherence = jnp.array(
        #     [
        #         self.simulator.calculate_coherence(
        #             x_i,
        #             x_j,
        #             y_i,
        #             y_j,
        #             z_i,
        #             z_j,
        #             2 * jnp.pi * freq,
        #             U_zi,
        #             U_zj,  # 使用角频率
        #             self.params["C_x"],
        #             self.params["C_y"],
        #             self.params["C_z"],
        #         )
        #         for freq in frequencies
        #     ]
        # )
        coherence = vmap(
            lambda freq: self.simulator.calculate_coherence(
                x_i,
                x_j,
                y_i,
                y_j,
                z_i,
                z_j,
                2 * jnp.pi * freq,  # 使用角频率
                U_zi,
                U_zj,
                self.params["C_x"],
                self.params["C_y"],
                self.params["C_z"],
            )
        )(frequencies)

        cross_spectrum = jnp.sqrt(S_ii * S_jj) * coherence

        # 为IFFT准备完整的互谱密度函数 (确保是厄米特矩阵)
        full_spectrum = jnp.zeros(M, dtype=jnp.complex64)
        # 正频率部分
        full_spectrum = full_spectrum.at[1 : N + 1].set(cross_spectrum)
        # 负频率部分 (共轭对称)
        full_spectrum = full_spectrum.at[M - N :].set(jnp.flip(cross_spectrum))

        # 执行IFFT得到理论互相关函数
        theo_correlation = jnp.real(jnp.fft.ifft(full_spectrum))
        theo_correlation = jnp.fft.fftshift(theo_correlation)  # 将零延迟移到中间

        # 理论互相关函数时间轴
        theo_lags = jnp.arange(-M // 2, M // 2)
        theo_lag_times = theo_lags * dt

        # 归一化理论互相关函数
        theo_max = jnp.max(jnp.abs(theo_correlation))
        theo_correlation = theo_correlation / (theo_max if theo_max > 0 else 1.0)

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 截取一定范围的实测相关函数 (±200点)
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
