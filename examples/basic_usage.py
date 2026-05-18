"""Basic stationary wind field simulation.

This example demonstrates the simplest workflow:
1. Create a simulator (JAX backend, Kaimal spectrum)
2. Define spatial points and mean wind speeds
3. Run the simulation
4. Plot PSD and cross-correlation

Run:
    python examples/basic_usage.py
"""

import numpy as np
from stochastic_wind_simulate import create_simulator, WindVisualizer

# --- 1. Create simulator ---
# Backend: "jax" (GPU, recommended), "numpy" (CPU), or "torch" (GPU)
# Spectrum: "kaimal", "panofsky", or "teunissen"
sim = create_simulator(
    backend="jax",
    spectrum="kaimal",
    seed=42,
    N=3000,          # number of frequency segments
    U_d=20.0,        # reference wind speed at 10 m  [m/s]
    w_up=5.0,        # cutoff frequency  [Hz]
    H_bar=20.0,      # average building height  [m]
    z_0=0.01,        # surface roughness  [m]
    alpha_0=0.12,    # roughness exponent
)

# --- 2. Define spatial points ---
n_points = 100
positions = np.zeros((n_points, 3), dtype=np.float32)
positions[:, 0] = np.linspace(0, 1000, n_points)   # x: spread along a line
positions[:, 1] = 5.0                                # y: fixed
positions[:, 2] = 35.0                               # z: fixed height

# Mean wind speed at each point (can vary spatially)
wind_speeds = np.full(n_points, 30.0, dtype=np.float32)

# --- 3. Simulate ---
u_samples, frequencies = sim.simulate_wind(
    positions, wind_speeds,
    component="u",              # "u" = along-wind, "w" = vertical
    max_memory_gb=4.0,          # auto-batch if memory exceeds this
    auto_batch=True,
)
print(f"Simulated: {u_samples.shape[0]} points x {u_samples.shape[1]} time steps")

# --- 4. Visualise ---
viz = WindVisualizer(sim, seed=42)
viz.plot_psd(u_samples, positions[:, 2], show_num=5, component="u")
viz.plot_cross_correlation(
    u_samples, positions, wind_speeds,
    component="u", indices=(1, 50),
)
