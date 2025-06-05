from typing import Dict, Tuple, Optional, Union, List
import torch
import torch.func as func
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from .simulator import TorchWindSimulator


class TorchWindVisualizer:
    """随机风场可视化器类"""

    def __init__(self, key=0, simulator: Optional[TorchWindSimulator] = None, **kwargs):
        """
        初始化风场可视化器

        参数:
            key: 随机数种子
            simulator: 风场模拟器实例，如果为 None 则创建新实例
            **kwargs: 传递给模拟器的其他参数
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
        将输入安全地转换为设备上的张量

        参数:
            value: 要转换的值（张量或其他类型）
            device: 目标设备，默认使用类的设备

        返回:
            在指定设备上的张量
        """
        if device is None:
            device = self.device

        if isinstance(value, torch.Tensor):
            return value.to(device)
        else:
            return torch.tensor(value, device=device)

    def _calculate_theoretical_spectrum(
        self,
        Z: float,
        direction: str = "u",
    ) -> Tuple[Tensor, Tensor]:
        # 计算理论谱
        u_star = self.simulator.calculate_friction_velocity(
            self._to_tensor(Z, device=self.device),
            self.params["U_d"],
            self.params["z_0"],
            self.params["z_d"],
            self.params["K"],
        )

        dw = self.params["dw"]
        N = self.params["N"]

        frequencies = self.simulator.calculate_simulation_frequency(N, dw)

        f_nondimensional = self.simulator.calculate_f(
            frequencies,
            self._to_tensor(Z, device=self.device),
            self.params["U_d"],
        )

        S_theory = (
            self.simulator.calculate_power_spectrum_u(
                self._to_tensor(frequencies, device=self.device),
                self._to_tensor(u_star, device=self.device),
                self._to_tensor(f_nondimensional, device=self.device),
            )
            if direction == "u"
            else self.simulator.calculate_power_spectrum_w(
                self._to_tensor(frequencies, device=self.device),
                self._to_tensor(u_star, device=self.device),
                self._to_tensor(f_nondimensional, device=self.device),
            )
        )

        return frequencies, S_theory

    def plot_psd(
        self,
        wind_samples: np.ndarray,
        Zs: torch.Tensor,
        show_num: int = 5,
        save_path: Optional[str] = None,
        show: bool = True,
        direction: str = "u",
        indices: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """
        绘制功率谱密度图

        参数:
            wind_samples: 风速时间序列，形状为 (n_points, n_timesteps)
            Zs: 高度，形状为 (n_points,)
            show_num: 要显示的采样点数量
            save_path: 保存图像的路径，如果不保存则为None
            show: 是否显示图形
            direction: 风向，'u'表示顺风向，'w'表示竖向
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
            raise ValueError("indices必须是整数或整数元组")

        ncol = kwargs.get("ncol", 3)
        nrow = (len(indices) + ncol - 1) // ncol

        frequencies_theory, S_theory = func.vmap(
            self._calculate_theoretical_spectrum, in_dims=(0, None)
        )(Zs, direction)
        frequencies_theory = frequencies_theory.cpu().numpy()
        S_theory = S_theory.cpu().numpy()

        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 5 * nrow))
        axes = axes.flatten() if nrow > 1 else [axes]
        # 计算并绘制每个采样点的功率谱密度
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

            # 绘制理论谱线
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
                title=f"PSD of {direction.upper()} Wind at Z={Zs[i].item():.2f} m (Point {i+1})",
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
        """计算互相关函数"""
        data_i_centered = data_i - np.mean(data_i)
        data_j_centered = data_j - np.mean(data_j)

        correlation = signal.correlate(data_i_centered, data_j_centered, mode="full")

        std_i = np.std(data_i)
        std_j = np.std(data_j)
        n = len(data_i)

        return correlation / (n * std_i * std_j)

    def _calculate_theoretical_correlation(
        self, S_ii: Tensor, S_jj: Tensor, coherence: Tensor, M: int
    ) -> np.ndarray:
        """计算理论互相关函数（通过傅里叶逆变换）"""
        # 将输入转换为PyTorch张量
        S_ii = torch.as_tensor(S_ii, device=self.device)
        S_jj = torch.as_tensor(S_jj, device=self.device)
        coherence = torch.as_tensor(coherence, device=self.device)

        # 计算互谱密度
        cross_spectrum = torch.sqrt(S_ii * S_jj) * coherence

        # 准备完整的频谱
        full_spectrum = torch.zeros(M, dtype=torch.complex64, device=self.device)
        N = len(coherence)
        full_spectrum[1 : N + 1] = torch.complex(
            cross_spectrum, torch.zeros_like(cross_spectrum)
        )
        full_spectrum[M - N :] = torch.flip(
            torch.conj(full_spectrum[1 : N + 1]), dims=[0]
        )

        # 执行IFFT获得理论互相关函数
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
        direction: str = "u",
        indices: Optional[Union[int, Tuple[int, int]]] = None,
        downsample: int = 1,
        **kwargs,
    ):
        """
        绘制互相关函数并与理论值比较

        参数:
            wind_samples: 风速时间序列，形状为 (n_points, n_timesteps)
            positions: 位置坐标，形状为 (n_points, 3)
            wind_speeds: 平均风速，形状为 (n_points,)
            save_path: 保存图像的路径，如果不保存则为None
            show: 是否显示图形
            direction: 风向，'u'表示顺风向，'w'表示竖向
            indices: 要计算互相关的两个点的索引，为None时随机选择
            downsample: 降采样因子，用于加快计算
            **kwargs: 额外参数

        返回:
            indices: 使用的索引对
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

        # 计算频率
        frequencies = self.simulator.calculate_simulation_frequency(N, dw).cpu().numpy()
        frequencies_tensor = self._to_tensor(frequencies, device=self.device)

        # 根据风向选择谱函数
        spectrum_func = (
            self.simulator.calculate_power_spectrum_u
            if direction == "u"
            else self.simulator.calculate_power_spectrum_w
        )

        # 计算摩擦速度
        u_star_i = (
            self.simulator.calculate_friction_velocity(
                self._to_tensor(z_i, device=self.device),
                self.params["U_d"],
                self.params["z_0"],
                self.params["z_d"],
                self.params["K"],
            )
            .cpu()
            .numpy()
        )

        u_star_j = (
            self.simulator.calculate_friction_velocity(
                self._to_tensor(z_j, device=self.device),
                self.params["U_d"],
                self.params["z_0"],
                self.params["z_d"],
                self.params["K"],
            )
            .cpu()
            .numpy()
        )

        # 计算无量纲频率
        f_i = (
            self.simulator.calculate_f(
                frequencies_tensor,
                self._to_tensor(z_i, device=self.device),
                self.params["U_d"],
            )
            .cpu()
            .numpy()
        )

        f_j = (
            self.simulator.calculate_f(
                frequencies_tensor,
                self._to_tensor(z_j, device=self.device),
                self.params["U_d"],
            )
            .cpu()
            .numpy()
        )

        # 计算自功率谱
        S_ii_list = []
        S_jj_list = []

        for idx, freq in enumerate(frequencies):
            S_ii_list.append(
                spectrum_func(
                    self._to_tensor(freq, device=self.device),
                    self._to_tensor(u_star_i, device=self.device),
                    self._to_tensor(f_i[idx], device=self.device),
                )
                .cpu()
                .numpy()
            )

            S_jj_list.append(
                spectrum_func(
                    self._to_tensor(freq, device=self.device),
                    self._to_tensor(u_star_j, device=self.device),
                    self._to_tensor(f_j[idx], device=self.device),
                )
                .cpu()
                .numpy()
            )

        S_ii = np.array(S_ii_list)
        S_jj = np.array(S_jj_list)

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
