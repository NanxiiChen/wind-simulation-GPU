from typing import Dict

import torch
import torch.func as func
from torch import Tensor


class TorchWindSimulator:
    """使用 PyTorch 实现的随机风场模拟器类"""

    def __init__(self, key=0):
        """
        初始化风场模拟器

        参数:
        key - 随机数种子
        """
        self.seed = key
        torch.manual_seed(key)

        # 设置计算设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = self._set_default_parameters()

    def _set_default_parameters(self) -> Dict:
        """设置默认风场模拟参数"""
        params = {
            "K": 0.4,  # 无量纲常数
            "H_bar": 10.0,  # 周围建筑物平均高度(m)
            "z_0": 0.05,  # 地表粗糙高度
            "C_x": 16.0,  # x方向衰减系数
            "C_y": 6.0,  # y方向衰减系数
            "C_z": 10.0,  # z方向衰减系数
            "w_up": 5.0,  # 截止频率(Hz)
            "N": 3000,  # 频率分段数
            "M": 6000,  # 时间点数(M=2N)
            "T": 600,  # 模拟时长(s)
            "dt": 0.1,  # 时间步长(s)
            "U_d": 25.0,  # 设计基本风速(m/s)
        }
        params["dw"] = params["w_up"] / params["N"]  # 频率增量
        params["z_d"] = params["H_bar"] - params["z_0"] / params["K"]  # 计算零平面位移

        return params

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

    def update_parameters(self, **kwargs):
        """更新模拟参数"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        # 更新依赖参数
        self.params["dw"] = self.params["w_up"] / self.params["N"]
        self.params["z_d"] = (
            self.params["H_bar"] - self.params["z_0"] / self.params["K"]
        )

    def calculate_friction_velocity(
        self, Z: Tensor, U_d: float, z_0: float, z_d: float, K: float
    ) -> Tensor:
        """计算风的摩阻速度 u_*"""
        return K * U_d / torch.log((Z - z_d) / z_0)

    def calculate_f(self, n: Tensor, Z: Tensor, U_d: float) -> Tensor:
        """计算无量纲频率 f"""
        return n * Z / U_d

    def calculate_power_spectrum_u(
        self, n: Tensor, u_star: Tensor, f: Tensor
    ) -> Tensor:
        """计算顺风向脉动风功率谱密度 S_u(n)"""
        return (u_star**2 / n) * (200 * f / ((1 + 50 * f) ** (5 / 3)))

    def calculate_power_spectrum_w(
        self, n: Tensor, u_star: Tensor, f: Tensor
    ) -> Tensor:
        """计算竖向脉动风功率谱密度 S_w(n)"""
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
        """计算空间相关函数 Coh"""
        # 转换为张量
        C_x_t = self._to_tensor(C_x, device=self.device)
        C_y_t = self._to_tensor(C_y, device=self.device)
        C_z_t = self._to_tensor(C_z, device=self.device)
        
        # 完全使用PyTorch计算
        distance_term = torch.sqrt(
            C_x_t**2 * (x_i - x_j) ** 2
            + C_y_t**2 * (y_i - y_j) ** 2
            + C_z_t**2 * (z_i - z_j) ** 2
        )

        # 使用PyTorch的π常数
        denominator = 2 * torch.pi * (U_zi + U_zj)
        safe_denominator = torch.maximum(
            denominator, 
            torch.tensor(1e-8, device=self.device)
        )

        return torch.exp(-2 * w * distance_term / safe_denominator)

    def calculate_cross_spectrum(
        self, S_ii: Tensor, S_jj: Tensor, coherence: Tensor
    ) -> Tensor:
        """计算互谱密度函数 S_ij"""
        return torch.sqrt(S_ii * S_jj) * coherence

    def calculate_simulation_frequency(self, N: int, dw: float) -> Tensor:
        """计算模拟频率数组"""
        return torch.arange(1, N + 1, device=self.device) * dw - dw / 2

    def build_spectrum_matrix(
        self, positions: Tensor, wind_speeds: Tensor, frequencies: Tensor, spectrum_func
    ) -> Tensor:
        """
        构建互谱密度矩阵 S(w) - 尽可能向量化实现
        """
        positions = torch.as_tensor(positions, device=self.device)
        wind_speeds = torch.as_tensor(wind_speeds, device=self.device)
        frequencies = torch.as_tensor(frequencies, device=self.device)

        n = positions.shape[0]
        num_freqs = len(frequencies)

        # 计算各点的摩阻速度
        u_stars = func.vmap(
            lambda z: self.calculate_friction_velocity(
                z,
                self.params["U_d"],
                self.params["z_0"],
                self.params["z_d"],
                self.params["K"],
            )
        )(positions[:, 2])

        # 计算所有频率点的无量纲频率 - 避免嵌套vmap
        f_values_all = torch.zeros((num_freqs, n), device=self.device)
        for freq_idx in range(num_freqs):
            freq = frequencies[freq_idx]
            f_values_all[freq_idx] = func.vmap(
                lambda z: self.calculate_f(freq, z, self.params["U_d"])
            )(positions[:, 2])

        # 计算所有频率点的功率谱密度 - 避免嵌套vmap
        S_values_all = torch.zeros((num_freqs, n), device=self.device)
        for freq_idx in range(num_freqs):
            freq = frequencies[freq_idx]
            S_values_all[freq_idx] = func.vmap(
                lambda u_star, f_val: spectrum_func(freq, u_star, f_val)
            )(u_stars, f_values_all[freq_idx])

        # 创建互谱密度矩阵 - 完全向量化实现
        S_matrices = torch.zeros(
            (num_freqs, n, n), device=self.device, dtype=torch.float32
        )

        # 将自功率谱放在对角线上
        for freq_idx in range(num_freqs):
            S_matrices[freq_idx].diagonal().copy_(S_values_all[freq_idx])

        # 创建网格以计算所有点对
        x_i = positions[:, 0].unsqueeze(1).expand(n, n)  # [n, n]
        x_j = positions[:, 0].unsqueeze(0).expand(n, n)  # [n, n]
        y_i = positions[:, 1].unsqueeze(1).expand(n, n)  # [n, n]
        y_j = positions[:, 1].unsqueeze(0).expand(n, n)  # [n, n]
        z_i = positions[:, 2].unsqueeze(1).expand(n, n)  # [n, n]
        z_j = positions[:, 2].unsqueeze(0).expand(n, n)  # [n, n]
        U_i = wind_speeds.unsqueeze(1).expand(n, n)  # [n, n]
        U_j = wind_speeds.unsqueeze(0).expand(n, n)  # [n, n]

        # 为每个频率点批量计算互谱
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
        模拟脉动风场

        参数:
        positions - 形状为(n, 3)的数组，每行为(x, y, z)坐标
        wind_speeds - 形状为(n,)的数组，表示各点的平均风速
        direction - 风向，'u'表示顺风向，'w'表示竖向

        返回:
        wind_samples - 形状为(n, M)的数组，表示各点的脉动风时程
        frequencies - 频率数组
        """
        if not isinstance(positions, Tensor):
            positions = torch.from_numpy(positions)
        return self._simulate_fluctuating_wind(positions, wind_speeds, direction)

    def _simulate_fluctuating_wind(self, positions, wind_speeds, direction):
        """风场模拟的内部实现"""
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

        # 构建互谱密度矩阵
        S_matrices = self.build_spectrum_matrix(
            positions, wind_speeds, frequencies, spectrum_func
        )

        # 对每个频率点进行Cholesky分解 - 使用 vmap 代替循环
        def cholesky_with_reg(S):
            return torch.linalg.cholesky(
                S + torch.eye(n, device=self.device) * 1e-12
            )


        H_matrices = func.vmap(cholesky_with_reg)(S_matrices)

        # 修改 _simulate_fluctuating_wind 方法中的 B 矩阵计算部分
        N_int = int(N.item()) if isinstance(N, torch.Tensor) else int(N)
        M_int = int(M.item()) if isinstance(M, torch.Tensor) else int(M)

        # 生成随机相位
        torch.manual_seed(self.seed)  # 确保可重复性
        phi = torch.rand((n, n, N_int), device=self.device) * 2 * torch.pi

        # 初始化 B 矩阵
        B = torch.zeros((n, M_int), dtype=torch.complex64, device=self.device)

        for j in range(n):
            # 使用 einsum 一次计算所有频率点的贡献
            # 注意维度顺序: H_matrices 是 [N, n, n]，我们需要 [:, j, :j+1]
            H_slice = H_matrices[:, j, : j + 1].to(torch.complex64)  # [N, j+1]
            exp_slice = torch.exp(1j * phi[j, : j + 1, :].permute(1, 0))  # [N, j+1]

            # 在维度1上求和 (对应于 i)
            B[j, :N_int] = torch.einsum("li,li->l", H_slice, exp_slice)

        # 计算 FFT
        G = torch.fft.fft(B, dim=1)

        # 计算风场样本 - 可以保留向量化
        p_indices = torch.arange(M, device=self.device)
        exponent = torch.exp(1j * (p_indices * torch.pi / M))
        wind_samples = torch.sqrt(2 * dw) * (G * exponent.unsqueeze(0)).real

        # 转换为NumPy数组以确保与JAX版本一致的输出
        return wind_samples.cpu().numpy(), frequencies.cpu().numpy()
