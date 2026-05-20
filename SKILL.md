---
name: wind-sim
description: >
  Stochastic wind field simulation library (JAX / PyTorch / NumPy) with
  stationary and nonstationary support.  Use this skill whenever working
  in this repository — it covers both usage (running simulations,
  benchmarks, validation) and development (architecture, adding backends,
  modifying the algorithm, config system).  Trigger on: running scripts,
  modifying any file under src/stochastic_wind_simulate/, adding features,
  debugging performance, or questions about the codebase.
---

# Wind Simulation — Usage & Development Guide

## Environment

```bash
pip install -e .
```

Dependencies: `jax[cuda]` (or `jax[cpu]`), `torch`, or `numpy`+`scipy`. Also `absl-py`, `ml_collections`, `matplotlib` (for scripts). JAX is the recommended backend on both GPU and CPU.

---

## Part 1: Using the library

### Command-line (config-based)

All parameters live in `ml_collections` config files.  Override anything with `--config.key=value`:

```bash
# Stationary
python scripts/simulate.py --config=configs/default.py

# Override any field
python scripts/simulate.py --config=configs/default.py \
    --config.params.N=5000 --config.backend=numpy

# Nonstationary
python scripts/simulate.py --config=configs/nonstationary.py

# Enable nonstationary on the default config
python scripts/simulate.py --config=configs/default.py \
    --config.nonstationary.enabled=True \
    --config.nonstationary.modulation_amplitude=0.3 \
    --config.params.N=1024

# Benchmark (freq or point scaling)
python scripts/benchmark.py --config=configs/benchmark_freq.py
python scripts/benchmark.py --config=configs/benchmark_points.py \
    --config.backends=jax,torch --config.test_sizes=10,50,100

# Validate nonstationary against theoretical EPSD
python scripts/validate.py --config=configs/validate.py \
    --config.params.N=512 --config.validation.n_realizations=16
```

Config files live in `configs/`.  Each defines ALL allowed fields — `ml_collections` rejects CLI overrides for undefined keys.  The config structure:

```python
cfg.backend          # "jax" | "numpy" | "torch"
cfg.spectrum         # "kaimal" | "panofsky" | "teunissen"
cfg.seed             # int
cfg.component        # "u" | "w"
cfg.params           # U_d, H_bar, z_0, alpha_0, w_up, N, (C_x, C_y, C_z)
cfg.spatial          # n_points, z, wind_speed
cfg.nonstationary    # enabled, mode, modulation_amplitude
cfg.memory           # max_memory_gb, auto_batch, freq_batch_size
cfg.visualization    # show_plots, point_index, window_size, overlap, snapshot_count
cfg.output           # save_samples, save_dir
```

### Python API

```python
from stochastic_wind_simulate import (
    create_simulator, NonstationaryWindSimulator, WindVisualizer,
)

# Stationary
sim = create_simulator("jax", "kaimal", seed=42, N=3000, U_d=20.0, w_up=5.0)
samples, freqs = sim.simulate_wind(positions, wind_speeds, component="u",
                                   max_memory_gb=4.0, auto_batch=True)

# Nonstationary
ns = NonstationaryWindSimulator(sim)   # wraps a stationary sim
samples_ns, _ = ns.simulate_nonstationary(
    positions, wind_speeds, component="u",
    mode="chunked-vmap", modulation_amplitude=0.2, max_memory_gb=8.0,
)

# Visualisation — stationary (Welch) vs nonstationary (short-time Fourier)
viz = WindVisualizer(sim)
viz.plot_psd(samples, positions[:, 2], show_num=6, component="u")
viz.plot_cross_correlation(samples, positions, wind_speeds, component="u")

viz_ns = WindVisualizer(ns)
viz_ns.plot_nonstationary_psd(
    samples_ns[0], height=35.0, wind_speed=30.0, component="u",
    window_size=64, overlap=50, snapshot_count=4,
)
```

### Examples (no config needed)

```bash
python examples/basic_usage.py               # minimal stationary
python examples/custom_spectrum.py            # custom spectrum class
python examples/nonstationary_custom_psd.py   # custom evolutionary PSD
```

---

## Part 2: Developing & modifying

### High-level architecture

The simulation algorithm is: **PSD → coherence → cross-spectrum → Cholesky → IFFT**.

It is defined **once** in `_BaseSimulator` (`build_amplitude_matrix`, `_process_amplitude_to_samples`).  Three backend classes — `JaxWindSimulator`, `NumpyWindSimulator`, `TorchWindSimulator` — are each ~40 lines and only provide these **primitives**:

| Primitive | JAX | NumPy | Torch |
|-----------|-----|-------|-------|
| `_xp` | `jax.numpy` | `numpy` | `torch` |
| `_jit(fn)` | `jax.jit(fn)` | `fn` (identity) | `fn` (identity) |
| `_vmap(fn)` | `jax.vmap(fn)` | list-comp loop | `torch.func.vmap(fn)` |
| `_cholesky(m)` | `jax.scipy…cholesky(m, lower=True)` | `scipy…cholesky(m, lower=True)` | `torch.linalg.cholesky(m, upper=False)` |
| `_fft(x, axis)` | `jnp.fft.ifft(x, axis=axis)` | `np.fft.ifft(x, axis=axis)` | `torch.fft.ifft(x, dim=axis)` |

Additional primitives: `_clip_positive`, `_to_complex`, `_asarray`, `_to_numpy`, `_slice_set`, `_eye`, `_zeros_c`, `_arange`, `_random_phases`, `_spec_fn` (pre-JIT spectrum), `_coh_fn` (pre-JIT coherence), `_csd_fn` (pre-JIT cross-spectrum).

### Adding a new backend

Create a new class inheriting `_BaseSimulator`.  Implement `__init__` with all primitives listed above.  See `JaxWindSimulator` as the template.  No other code changes needed — the shared algorithm picks up the new primitives automatically.

### NumpyWindSimulator

### Key design invariants (do not break)

1. **jit first, then vmap.**  `self._jit(local_fn)` compiles the scalar function, then `self._vmap(...)` batches it.  This matches the original JAX performance pattern.  For JAX backends, `_spec_fn`, `_coh_fn`, `_csd_fn` are pre-JIT-compiled at `__init__` time to reduce outer-JIT trace work.

2. **One shared algorithm, three backends.**  `build_amplitude_matrix`, `_process_amplitude_to_samples`, `_simulate_direct`, `_simulate_batched`, `estimate_memory` all live in `_BaseSimulator`.  Never add backend-specific branches to these methods — use a new primitive instead.

3. **Torch `func.vmap` forbids `.item()`.**  Any tensor indexing inside a vmapped function triggers an internal `.item()` call and crashes.  Pre-compute arrays (e.g. modulation factors) and slice them BEFORE entering vmap, passing the pre-sliced chunks through.  This is why `nonstationary.py` pre-computes `mod_factors` at the top.

4. **Torch scalar quirks.**  `torch.minimum(tensor, float)` fails — use `_clip_positive`.  `torch.arange` defaults to CPU — wrap results with `_asarray`.  `torch.matmul(float32_matrix, complex64_vector)` fails — use `_to_complex(H)` first.  FFT uses `dim=` not `axis=`.

5. **Config files define ALL fields.**  `ml_collections` rejects CLI overrides (`--config.x.y=z`) for paths that don't exist in the loaded config.  Every section must be complete.  No silent `.get()` defaults in scripts — missing fields should fail loudly.

6. **Spectrum classes are backend-agnostic.**  They store `self.xp` (array module) and only use it for `self.xp.log()`.  All arithmetic (`+`, `-`, `*`, `/`, `**`) is identical across numpy/jax.numpy/torch.

### Nonstationary pattern

`NonstationaryWindSimulator` wraps a stationary simulator and delegates all backend ops via `__getattr__`.  The `simulate_nonstationary` method uses the same `self._jit` / `s._vmap` pattern as the stationary case.  Modulation factors (`mod_factors`) are pre-computed once as a flat `(M,)` array at the top of the method, then sliced into chunks before entering the double-vmap over frequency and time.

### File map

```
src/stochastic_wind_simulate/
├── simulator.py       # _BaseSimulator + 3 backend classes
├── nonstationary.py   # NonstationaryWindSimulator (wraps a stationary sim)
├── spectrum.py        # Kaimal, Panofsky, Teunissen (backend-agnostic)
├── coherence.py       # Davenport coherence + cross-spectrum functions
├── params.py          # SimulationParams dataclass (auto-computed fields)
└── visualizer.py      # Stationary (Welch) + nonstationary (STFT) plots

configs/               # ml_collections ConfigDict presets
scripts/
├── simulate.py        # Stationary & nonstationary, all backends
├── benchmark.py       # Frequency & point scaling benchmarks
└── validate.py        # Nonstationary EPSD validation (ensemble)
examples/
├── basic_usage.py
├── custom_spectrum.py
└── nonstationary_custom_psd.py
```
