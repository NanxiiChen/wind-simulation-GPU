import numpy as np
from typing import Dict, Tuple
from scipy.linalg import cholesky


class NumpyWindSimulator:
    """使用 NumPy 实现的随机风场模拟器类 - 与JAX版本逻辑一致"""

    def __init__(self, key=0):
        """初始化风场模拟器"""
        self.seed = key
        np.random.seed(key)
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

    @staticmethod
    def calculate_friction_velocity(Z, U_d, z_0, z_d, K):
        """计算风的摩阻速度 u_*"""
        return K * U_d / np.log((Z - z_d) / z_0)

    @staticmethod
    def calculate_f(n, Z, U_d):
        """计算无量纲频率 f"""
        return n * Z / U_d

    @staticmethod
    def calculate_power_spectrum_u(n, u_star, f):
        """计算顺风向脉动风功率谱密度 S_u(n)"""
        return (u_star**2 / n) * (200 * f / ((1 + 50 * f) ** (5 / 3)))

    @staticmethod
    def calculate_power_spectrum_w(n, u_star, f):
        """计算竖向脉动风功率谱密度 S_w(n)"""
        return (u_star**2 / n) * (6 * f / ((1 + 4 * f) ** 2))

    @staticmethod
    def calculate_coherence(x_i, x_j, y_i, y_j, z_i, z_j, w, U_zi, U_zj, C_x, C_y, C_z):
        """计算空间相关函数 Coh"""
        distance_term = np.sqrt(
            C_x**2 * (x_i - x_j) ** 2
            + C_y**2 * (y_i - y_j) ** 2
            + C_z**2 * (z_i - z_j) ** 2
        )
        # 增加数值稳定性保护，避免除以接近零的值
        denominator = 2 * np.pi * (U_zi + U_zj)
        safe_denominator = np.maximum(denominator, 1e-8)  # 设置安全最小值

        return np.exp(-2 * w * distance_term / safe_denominator)

    @staticmethod
    def calculate_cross_spectrum(S_ii, S_jj, coherence):
        """计算互谱密度函数 S_ij"""
        return np.sqrt(S_ii * S_jj) * coherence

    @staticmethod
    def calculate_simulation_frequency(N, dw):
        """计算模拟频率数组"""
        return np.arange(1, N + 1) * dw - dw / 2

    def build_spectrum_matrix(self, positions, wind_speeds, frequencies, spectrum_func):
        """构建互谱密度矩阵 S(w) - 与JAX版本完全对应"""
        n = positions.shape[0]
        num_freqs = len(frequencies)

        # 计算各点的摩阻速度
        u_stars = self.calculate_friction_velocity(
            positions[:, 2],
            self.params["U_d"], 
            self.params["z_0"], 
            self.params["z_d"], 
            self.params["K"]
        )

        # 计算f值
        f_values_all = np.zeros((num_freqs, n))
        for freq_idx, freq in enumerate(frequencies):
            f_values_all[freq_idx] = self.calculate_f(freq, positions[:, 2], self.params["U_d"])

        # 计算功率谱密度
        S_values_all = np.zeros((num_freqs, n))
        for freq_idx in range(num_freqs):
            S_values_all[freq_idx] = spectrum_func(
                frequencies[freq_idx], u_stars, f_values_all[freq_idx]
            )

        # 创建网格坐标 - 直接对应JAX的实现
        x_i = positions[:, 0][:, np.newaxis].repeat(n, axis=1)  # [n, n]
        x_j = positions[:, 0][np.newaxis, :].repeat(n, axis=0)  # [n, n]
        y_i = positions[:, 1][:, np.newaxis].repeat(n, axis=1)  # [n, n]
        y_j = positions[:, 1][np.newaxis, :].repeat(n, axis=0)  # [n, n]
        z_i = positions[:, 2][:, np.newaxis].repeat(n, axis=1)  # [n, n]
        z_j = positions[:, 2][np.newaxis, :].repeat(n, axis=0)  # [n, n]
        
        U_i = wind_speeds[:, np.newaxis].repeat(n, axis=1)  # [n, n]
        U_j = wind_speeds[np.newaxis, :].repeat(n, axis=0)  # [n, n]
        
        # 初始化结果矩阵
        S_matrices = np.zeros((num_freqs, n, n))
        
        # 对每个频率计算互谱矩阵
        for freq_idx, freq in enumerate(frequencies):
            # 计算相干函数
            coherence = self.calculate_coherence(
                x_i, x_j, y_i, y_j, z_i, z_j, 
                freq, U_i, U_j,
                self.params["C_x"], self.params["C_y"], self.params["C_z"]
            )
            
            # 计算互谱密度
            S_i = S_values_all[freq_idx].reshape(n, 1)  # [n, 1]
            S_j = S_values_all[freq_idx].reshape(1, n)  # [1, n]
            cross_spectrum = np.sqrt(S_i * S_j) * coherence
            
            S_matrices[freq_idx] = cross_spectrum
        
        return S_matrices

    def simulate_wind(self, positions, wind_speeds, direction="u"):
        """模拟脉动风场"""
        np.random.seed(self.seed)
        self.seed += 1
        
        # 转换输入为NumPy数组
        positions = np.asarray(positions, dtype=np.float64)
        wind_speeds = np.asarray(wind_speeds, dtype=np.float64)
        
        return self._simulate_fluctuating_wind(
            positions, wind_speeds, direction
        )

    def _simulate_fluctuating_wind(self, positions, wind_speeds, direction):
        """风场模拟的内部实现 - 与JAX版本对应"""
        n = positions.shape[0]
        N = self.params["N"]
        M = self.params["M"]
        dw = self.params["dw"]

        # 计算频率和选择谱函数
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

        # 对每个频率点进行Cholesky分解
        H_matrices = np.zeros((N, n, n), dtype=np.complex128)
        for i in range(N):
            # 添加小量对角项提高数值稳定性
            S_reg = S_matrices[i] + np.eye(n) * 1e-12
            H_matrices[i] = cholesky(S_reg, lower=True)

        # 生成随机相位 - 与JAX版本相同
        phi = np.random.uniform(0, 2*np.pi, (n, n, N))
        
        # 计算B矩阵
        B = np.zeros((n, M), dtype=np.complex128)
        
        for j in range(n):
            # 创建掩码 - 只使用下三角部分
            mask = np.arange(n) <= j
            
            # 提取当前点的矩阵行
            H_terms = H_matrices[:, j, :]
            H_masked = H_terms * mask
            
            # 计算相位项
            phi_masked = phi[j, :, :] * mask.reshape(n, 1)
            exp_terms = np.exp(1j * phi_masked.T)
            
            # 计算B值
            B_values = np.zeros(N, dtype=np.complex128)
            for freq_idx in range(N):
                B_values[freq_idx] = np.sum(
                    H_masked[freq_idx] * exp_terms[freq_idx] * mask
                )
                
            # 填充B矩阵
            B[j, :N] = B_values
        
        # FFT变换
        G = np.fft.fft(B)
        
        # 计算风场样本
        wind_samples = np.zeros((n, M))
        p_indices = np.arange(M)
        exp_factor = np.exp(1j * (p_indices * np.pi / M))
        
        for j in range(n):
            wind_samples[j] = np.sqrt(2 * dw) * np.real(G[j] * exp_factor)
        
        return wind_samples, frequencies