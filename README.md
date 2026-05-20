# Fully Parallelized Stochastic Wind Field Simulation Framework

A high-performance framework for stationary and non-stationary stochastic wind field simulation based on Shinozuka's harmonic synthesis method, delivering significant computational speedups for wind engineering applications.

Note: a stable version of this code is on the `main` branch, which is no longer maintained. The `dev` branch contains the latest features and improvements, but may be less stable. Please choose the branch that best suits your needs.

## Overview

Stochastic wind field simulation is extensively utilized in civil and wind engineering for analyzing the dynamic responses of bridges, buildings, and other structures under wind loading. This library implements the classical Shinozuka harmonic synthesis method enhanced with modern GPU parallel computing techniques to achieve efficient stochastic wind field generation.

### Key Features

- **Multi-backend**: JAX (jit + vmap), PyTorch (func.vmap), NumPy (CPU) — all from a unified API
- **Stationary & nonstationary**: Nonstationary simulation with evolutionary PSD, built as a natural extension of the stationary case
- **Adaptive batching**: Automatic frequency/time chunking to keep memory within user-specified limits
- **Pluggable spectra**: Kaimal, Panofsky, Teunissen built-in; custom spectra via simple subclassing
- **Comprehensive validation**: Built-in PSD and cross-correlation plotting tools

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone -b dev --depth=1 https://github.com/NanxiiChen/wind-simulation-GPU.git
cd wind-simulation-GPU
pip install -e .
```

Requires one of: `jax[cuda]` (GPU), `jax[cpu]` (CPU), `torch`, or `numpy` + `scipy`.  JAX is the recommended backend on both GPU and CPU.

```bash
pip install "jax[cuda12]"  # for GPU (with CUDA 12)
pip install "jax[cpu]"   # for CPU
```

### Claude Code Skill

We also provide a `SKILL.md` file for use with [Claude Code](https://docs.anthropic.com/en/docs/claude-code), which gives the agent full context on the library's architecture, design patterns, commands, and common pitfalls. This allows new users and developers to interact with and modify the library through the agent without needing to read the entire codebase.

To install it in your project:

```bash
mkdir -p .claude/skills/wind-sim
cp SKILL.md .claude/skills/wind-sim/
```

Claude Code auto-detects the skill when working in that directory. It triggers when you mention wind simulation, the scripts, benchmarks, or any file under `src/stochastic_wind_simulate/`.

## Quick Start

### Command-line

All parameters live in config files.  Override anything with ``--config.key=value``:

```bash
# Stationary (default config)
python scripts/simulate.py --config=configs/default.py

# Override parameters
python scripts/simulate.py --config=configs/default.py \
    --config.params.N=5000 --config.spatial.n_points=200

# Nonstationary
python scripts/simulate.py --config=configs/nonstationary.py

python scripts/simulate.py --config=configs/default.py \
    --config.nonstationary.enabled=True \
    --config.nonstationary.modulation_amplitude=0.5 \
    --config.params.N=1024 --config.spatial.n_points=100

# Benchmark frequency scaling
python scripts/benchmark.py --config=configs/benchmark_freq.py

# Benchmark point scaling
python scripts/benchmark.py --config=configs/benchmark_points.py

# Override benchmark parameters
python scripts/benchmark.py --config=configs/benchmark_freq.py \
    --config.backends=jax,numpy --config.test_frequencies=100,500,1000

# Validate nonstationary
python scripts/validate.py --config=configs/validate.py

python scripts/validate.py --config=configs/validate.py \
    --config.params.N=512 --config.validation.n_realizations=16
```

### Python API

```python
import numpy as np
from stochastic_wind_simulate import create_simulator, NonstationaryWindSimulator, WindVisualizer

# --- Stationary ---
sim = create_simulator("jax", "kaimal", seed=42, N=3000, U_d=20.0, w_up=5.0)

positions = np.zeros((100, 3), dtype=np.float32)
positions[:, 0] = np.linspace(0, 1000, 100)
positions[:, 1] = 5.0
positions[:, 2] = 35.0
wind_speeds = np.full(100, 30.0, dtype=np.float32)

samples, freqs = sim.simulate_wind(positions, wind_speeds, component="u",
                                   max_memory_gb=4.0, auto_batch=True)

# --- Nonstationary ---
ns = NonstationaryWindSimulator(sim)  # wraps the stationary simulator
samples_ns, freqs_ns = ns.simulate_nonstationary(
    positions, wind_speeds, component="u",
    mode="chunked-vmap", modulation_amplitude=0.2,
    max_memory_gb=8.0,
)

# --- Visualisation (stationary) ---
viz = WindVisualizer(sim)
viz.plot_psd(samples, positions[:, 2], show_num=6, component="u")
viz.plot_cross_correlation(samples, positions, wind_speeds, component="u", indices=(1, 5))

# --- Visualisation (nonstationary: short-time Fourier) ---
viz_ns = WindVisualizer(ns)
viz_ns.plot_nonstationary_psd(
    samples_ns[0], height=35.0, wind_speed=30.0, component="u",
    window_size=64, overlap=50, snapshot_count=4,
)
```

### Wind Spectrum Models

| Name | Key | Components |
|------|-----|------------|
| Kaimal | `"kaimal"` | u, w |
| Panofsky | `"panofsky"` | w |
| Teunissen | `"teunissen"` | u, w |

```python
sim = create_simulator("jax", "teunissen", seed=42, N=3000)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `U_d` | 25.0 | Reference wind speed at 10 m [m/s] |
| `H_bar` | 10.0 | Average building height [m] |
| `z_0` | 0.05 | Surface roughness height [m] |
| `alpha_0` | 0.16 | Surface roughness exponent |
| `C_x, C_y, C_z` | 16, 6, 10 | Davenport decay coefficients |
| `w_up` | 5.0 | Cutoff frequency [Hz] |
| `N` | 3000 | Number of frequency segments |

Computed automatically: `M = 2N`, `T = N/w_up`, `dt = T/M`, `dw = w_up/N`, `z_d = H_bar - z_0/K`.

```python
sim.update_params(U_d=30.0, N=5000)
```

### Configuration Files

Config files use `ml_collections.ConfigDict`:

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

CLI arguments override config values via dotted notation:

```bash
python scripts/simulate.py --config=configs/default.py \
    --config.spatial.n_points=200 --config.backend=numpy
```

### Custom Spectrum

```python
from stochastic_wind_simulate.spectrum import WindSpectrum

class MySpectrum(WindSpectrum):
    def psd_u(self, n, u_star, f):
        return (u_star**2 / n) * (200.0 * f / (1.0 + 50.0 * f)**(5.0 / 3.0))

    def psd_w(self, n, u_star, f):
        return (u_star**2 / n) * (3.36 * f / (1.0 + 10.0 * f)**(5.0 / 3.0))

sim = create_simulator("jax", MySpectrum, seed=42, N=3000)
# or pass the class directly:
from stochastic_wind_simulate import JaxWindSimulator
sim = JaxWindSimulator(key=42, spectrum_type=MySpectrum, N=3000)
```

## Architecture

```
src/stochastic_wind_simulate/
├── simulator.py       # _BaseSimulator + Jax/Numpy/TorchWindSimulator
├── nonstationary.py   # NonstationaryWindSimulator (wraps a stationary sim)
├── spectrum.py        # Kaimal, Panofsky, Teunissen (backend-agnostic)
├── coherence.py       # Davenport coherence model
├── params.py          # SimulationParams dataclass
└── visualizer.py      # Stationary (Welch) + nonstationary (STFT) plots

configs/               # ml_collections config presets
scripts/
├── simulate.py        # Stationary & nonstationary, all backends
├── benchmark.py       # Frequency & point scaling benchmarks
└── validate.py        # Nonstationary EPSD validation

examples/
├── basic_usage.py     # Minimal stationary example
├── custom_spectrum.py # Custom spectrum class
└── nonstationary_custom_psd.py  # Custom evolutionary PSD
```

### Design: One shared algorithm, three thin backends

The core simulation algorithm (spectrum → coherence → CSD → Cholesky → IFFT) is defined **once** in `_BaseSimulator`. Each backend provides only the primitives that differ:

| Primitive | JAX | NumPy | Torch |
|-----------|-----|-------|-------|
| `_xp` | `jax.numpy` | `numpy` | `torch` |
| `_jit` | `jax.jit` | identity | identity |
| `_vmap` | `jax.vmap` | list comprehension | `torch.func.vmap` |
| `_cholesky` | `jax.scipy.linalg.cholesky` | `scipy.linalg.cholesky` | `torch.linalg.cholesky` |
| `_fft` | `jnp.fft.ifft` | `np.fft.ifft` | `torch.fft.ifft` |

### Nonstationary: wrapper pattern

`NonstationaryWindSimulator` wraps a backend-specific stationary simulator and delegates all operations to it:

```python
sim = create_simulator("jax", "kaimal", seed=42, N=1024)  # any backend
ns  = NonstationaryWindSimulator(sim)
samples, freqs = ns.simulate_nonstationary(...)
```

The nonstationary algorithm uses **jit-then-vmap** (for JAX) applied to local functions defined once and reused across chunks — matching the original code's performance characteristics.

## Citation

```bibtex
@article{chen5707657high,
  title={Reconciling the accuracy-efficiency trade-off in stochastic wind field simulation: a dual-level parallel algorithmic perspective},
  author={Chen, Nanxi and Liu, Guilin and Zhang, Junrui and Ma, Rujin and Chang, Haocheng and Zhu, Yan and Qiu, Xu and Chen, Airong},
  journal={Available at SSRN 5707657}
}
```
