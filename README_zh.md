# 全并行随机风场模拟框架

[![Gitee](https://img.shields.io/badge/镜像-Gitee-red)](https://gitee.com/nanxi_chen/wind-simulation-gpu)
[![Paper](https://img.shields.io/badge/论文-MSSP-FF6B00)](https://doi.org/10.1016/j.ymssp.2026.114603)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://www.researchgate.net/publication/408456305_Reconciling_the_accuracy-efficiency_trade-off_in_stochastic_wind_field_simulation_a_dual-level_parallel_algorithmic_perspective)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uGForghIGKL-5xTcpMUj0SBOjYeKZ7X6?usp=sharing)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nanxiichen/wind-simulation)
[![README](https://img.shields.io/badge/README-EN-blue)](README.md)

基于 Shinozuka 谐波合成法的高性能平稳/非平稳随机风场模拟框架，为风工程应用提供显著的计算加速。

注意：本仓库的稳定版本位于 `legacy` 分支，该分支已不再维护。`main` 分支包含最新的功能和改进。请根据需求选择合适的分支。

## 概述

随机风场模拟广泛应用于土木与风工程领域，用于分析桥梁、建筑等结构在风荷载作用下的动力响应。本库实现了经典的 Shinozuka 谐波合成法，并结合现代 GPU 并行计算技术实现高效的随机风场生成。

### 核心特性

- **多后端支持**：JAX (jit + vmap)、PyTorch (func.vmap)、NumPy (CPU) — 统一 API
- **平稳 & 非平稳**：非平稳模拟基于演化功率谱密度，作为平稳情况的自然扩展
- **自适应批处理**：自动按频率/时间分块，将内存控制在用户指定的上限之内
- **可插拔风谱**：内置 Kaimal、Panofsky、Teunissen 风谱模型；通过简单子类化即可自定义风谱
- **完善的验证工具**：内置 PSD 和互相关绘图工具

## 安装

### 在线试用

我们提供 Google Colab 和 Kaggle 笔记本，无需本地配置即可快速在线体验。点击下方徽章打开：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uGForghIGKL-5xTcpMUj0SBOjYeKZ7X6?usp=sharing)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nanxiichen/wind-simulation)

### 本地安装

对于实际项目和开发，推荐将包以可编辑模式安装到本地。克隆仓库并安装：

```bash
git clone -b main --depth=1 https://github.com/NanxiiChen/wind-simulation-GPU.git # Github 仓库
git clone -b main --depth=1 https://gitee.com/nanxi_chen/wind-simulation-gpu.git # 或使用 Gitee 仓库
cd wind-simulation-GPU
pip install -e .
```

需要以下之一：`jax[cuda]`（GPU）、`jax[cpu]`（CPU）、`torch` 或 `numpy` + `scipy`。JAX 是推荐的 GPU 和 CPU 后端。

```bash
pip install "jax[cuda12]"  # GPU 版本（CUDA 12）
pip install "jax[cpu]"     # CPU 版本
```

其他依赖：
```bash
pip install ml_collections matplotlib
```

### AI Agent 技能文件

我们还提供了 `SKILL.md` 文件，为 AI 编程助手提供本库架构、设计模式、命令和常见陷阱的完整上下文 — 新用户和开发者无需通读整个代码库即可通过 AI Agent 使用和修改本库。

**Claude Code:**

```bash
mkdir -p .claude/skills/wind-sim
cp SKILL.md .claude/skills/wind-sim/
```

**OpenAI Codex CLI:**

```bash
mkdir -p .codex/skills/wind-sim
cp SKILL.md .codex/skills/wind-sim/
```

Claude Code 会在该目录下工作时自动检测该技能。当提及风场模拟、脚本、基准测试或 `src/stochastic_wind_simulate/` 下的任何文件时会触发。

## 快速开始

### 命令行

所有参数均通过配置文件管理，可通过 ``--config.key=value`` 覆盖任意参数：

```bash
# 平稳模拟（默认配置）
python scripts/simulate.py --config=configs/default.py --config.backend=numpy

# 覆盖参数
python scripts/simulate.py --config=configs/default.py \
    --config.params.N=5000 --config.spatial.n_points=200 \
    --config.backend=jax

# 非平稳模拟
python scripts/simulate.py --config=configs/nonstationary.py

python scripts/simulate.py --config=configs/default.py \
    --config.nonstationary.enabled=True \
    --config.nonstationary.modulation_amplitude=0.5 \
    --config.params.N=1024 --config.spatial.n_points=100

# 频率扩展性基准测试
python scripts/benchmark.py --config=configs/benchmark_freq.py

# 空间点扩展性基准测试
python scripts/benchmark.py --config=configs/benchmark_points.py

# 覆盖基准测试参数
python scripts/benchmark.py --config=configs/benchmark_freq.py \
    --config.backends=jax,numpy --config.test_frequencies=100,500,1000

# 非平稳验证
python scripts/validate.py --config=configs/validate.py

python scripts/validate.py --config=configs/validate.py \
    --config.params.N=512 --config.validation.n_realizations=16
```

### Python API

```python
import numpy as np
from stochastic_wind_simulate import create_simulator, NonstationaryWindSimulator, WindVisualizer

# --- 平稳模拟 ---
sim = create_simulator("jax", "kaimal", seed=42, N=3000, U_d=20.0, w_up=5.0)

positions = np.zeros((100, 3), dtype=np.float32)
positions[:, 0] = np.linspace(0, 1000, 100)
positions[:, 1] = 5.0
positions[:, 2] = 35.0
wind_speeds = np.full(100, 30.0, dtype=np.float32)

samples, freqs = sim.simulate_wind(positions, wind_speeds, component="u",
                                   max_memory_gb=4.0, auto_batch=True)

# --- 非平稳模拟 ---
ns = NonstationaryWindSimulator(sim)  # 包装平稳模拟器
samples_ns, freqs_ns = ns.simulate_nonstationary(
    positions, wind_speeds, component="u",
    mode="chunked-vmap", modulation_amplitude=0.2,
    max_memory_gb=8.0,
)

# --- 可视化（平稳：Welch 方法）---
viz = WindVisualizer(sim)
viz.plot_psd(samples, positions[:, 2], show_num=6, component="u")
viz.plot_cross_correlation(samples, positions, wind_speeds, component="u", indices=(1, 5))

# --- 可视化（非平稳：短时傅里叶变换）---
viz_ns = WindVisualizer(ns)
viz_ns.plot_nonstationary_psd(
    samples_ns[0], height=35.0, wind_speed=30.0, component="u",
    window_size=64, overlap=50, snapshot_count=4,
)
```

### 风谱模型

| 名称 | 键值 | 分量 |
|------|-----|------|
| Kaimal | `"kaimal"` | u, w |
| Panofsky | `"panofsky"` | w |
| Teunissen | `"teunissen"` | u, w |

```python
sim = create_simulator("jax", "teunissen", seed=42, N=3000)
```

### 参数说明

| 参数 | 默认值 | 描述 |
|-----------|---------|------|
| `U_d` | 25.0 | 10 m 高度处参考风速 [m/s] |
| `H_bar` | 10.0 | 平均建筑高度 [m] |
| `z_0` | 0.05 | 地表粗糙高度 [m] |
| `alpha_0` | 0.16 | 地表粗糙指数 |
| `C_x, C_y, C_z` | 16, 6, 10 | Davenport 衰减系数 |
| `w_up` | 5.0 | 截止频率 [Hz] |
| `N` | 3000 | 频率分段数 |

自动计算：`M = 2N`、`T = N/w_up`、`dt = T/M`、`dw = w_up/N`、`z_d = H_bar - z_0/K`。

```python
sim.update_params(U_d=30.0, N=5000)
```

### 配置文件

配置文件使用 `ml_collections.ConfigDict`：

```python
# configs/default.py
from ml_collections import ConfigDict

def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.backend = "jax"
    cfg.spectrum = "kaimal"
    cfg.seed = 42
    cfg.component = "u"

    cfg.params = ConfigDict()
    cfg.params.U_d = 20.0; cfg.params.H_bar = 20.0
    cfg.params.alpha_0 = 0.12; cfg.params.z_0 = 0.01
    cfg.params.w_up = 5.0; cfg.params.N = 3000

    cfg.spatial = ConfigDict()
    cfg.spatial.n_points = 100; cfg.spatial.z = 35.0
    cfg.spatial.wind_speed = 30.0

    cfg.nonstationary = ConfigDict()
    cfg.nonstationary.enabled = False

    cfg.memory = ConfigDict()
    cfg.memory.max_memory_gb = 4.0
    cfg.memory.auto_batch = True

    cfg.visualization = ConfigDict()
    cfg.visualization.show_plots = True

    cfg.output = ConfigDict()
    cfg.output.save_samples = True
    cfg.output.save_dir = "output"
    return cfg
```

CLI 参数通过点号分隔覆盖配置值：

```bash
python scripts/simulate.py --config=configs/default.py \
    --config.spatial.n_points=200 --config.backend=numpy
```

### 自定义风谱

```python
from stochastic_wind_simulate.spectrum import WindSpectrum

class MySpectrum(WindSpectrum):
    def psd_u(self, n, u_star, f):
        return (u_star**2 / n) * (200.0 * f / (1.0 + 50.0 * f)**(5.0 / 3.0))

    def psd_w(self, n, u_star, f):
        return (u_star**2 / n) * (3.36 * f / (1.0 + 10.0 * f)**(5.0 / 3.0))

sim = create_simulator("jax", MySpectrum, seed=42, N=3000)
# 或直接传入类：
from stochastic_wind_simulate import JaxWindSimulator
sim = JaxWindSimulator(key=42, spectrum_type=MySpectrum, N=3000)
```

## 架构

```
src/stochastic_wind_simulate/
├── simulator.py       # _BaseSimulator + Jax/Numpy/TorchWindSimulator
├── nonstationary.py   # NonstationaryWindSimulator（包装平稳模拟器）
├── spectrum.py        # Kaimal、Panofsky、Teunissen 风谱（与后端无关）
├── coherence.py       # Davenport 相干模型
├── params.py          # SimulationParams 数据类
└── visualizer.py      # 平稳（Welch）+ 非平稳（STFT）绘图

configs/               # ml_collections 配置预设
scripts/
├── simulate.py        # 平稳 & 非平稳模拟，支持所有后端
├── benchmark.py       # 频率 & 空间点扩展性基准测试
└── validate.py        # 非平稳 EPSD 验证

examples/
├── basic_usage.py     # 最小平稳示例
├── custom_spectrum.py # 自定义风谱示例
└── nonstationary_custom_psd.py  # 自定义演化 PSD 示例
```

### 设计理念：一份共享算法，三个轻量后端

核心模拟算法（风谱 → 相干性 → CSD → Cholesky → IFFT）在 `_BaseSimulator` 中**仅定义一次**。每个后端仅提供不同的基础操作：

| 基础操作 | JAX | NumPy | Torch |
|-----------|-----|-------|-------|
| `_xp` | `jax.numpy` | `numpy` | `torch` |
| `_jit` | `jax.jit` | identity | identity |
| `_vmap` | `jax.vmap` | 列表推导 | `torch.func.vmap` |
| `_cholesky` | `jax.scipy.linalg.cholesky` | `scipy.linalg.cholesky` | `torch.linalg.cholesky` |
| `_fft` | `jnp.fft.ifft` | `np.fft.ifft` | `torch.fft.ifft` |

### 非平稳：包装器模式

`NonstationaryWindSimulator` 包装了一个特定后端的平稳模拟器，并将所有操作委托给它：

```python
sim = create_simulator("jax", "kaimal", seed=42, N=1024)  # 任意后端
ns  = NonstationaryWindSimulator(sim)
samples, freqs = ns.simulate_nonstationary(...)
```

## 开发者指南

如果你只是想为自己的工作使用或修改本库，请参考上方的[本地安装](#本地安装)。本部分面向希望将修改贡献回本仓库的开发者。

如果你有功能需求、改进想法或有希望实现的新增内容，欢迎[提交 Issue](https://github.com/NanxiiChen/wind-simulation-GPU/issues) — 贡献和建议始终欢迎。

### Fork & 克隆（一次性操作）

在[仓库页面](https://github.com/NanxiiChen/wind-simulation-GPU)点击 **Fork**，然后：

```bash
git clone https://github.com/YOUR_USERNAME/wind-simulation-GPU.git
cd wind-simulation-GPU
git remote add upstream https://github.com/NanxiiChen/wind-simulation-GPU.git
```

### 典型工作流

```bash
# 同步上游更新
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# 在新分支中进行修改
git checkout -b my-feature
# ... 编辑代码 ...
git add .
git commit -m "简要描述"
git push origin my-feature
```

然后前往你 Fork 的 GitHub 仓库，点击 **Compare & pull request**。

## 引用

```bibtex
@article{chen2026mssp,
  title = {Reconciling the accuracy-efficiency trade-off in stochastic wind field simulation: A dual-level parallel algorithmic perspective},
  journal = {Mechanical Systems and Signal Processing},
  volume = {258},
  pages = {114603},
  year = {2026},
  issn = {0888-3270},
  doi = {https://doi.org/10.1016/j.ymssp.2026.114603},
  url = {https://www.sciencedirect.com/science/article/pii/S0888327026007600},
  author = {Nanxi Chen and Guilin Liu and Junrui Zhang and Rujin Ma and Haocheng Chang and Yan Zhu and Xu Qiu and Airong Chen},
  keywords = {Stochastic wind field simulation, Spectral representation, Harmonic synthesis method, Parallel computing, GPU acceleration},
}
```
