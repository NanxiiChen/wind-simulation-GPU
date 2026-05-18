"""Nonstationary simulation with a custom evolutionary PSD.

Demonstrates how to supply an ``evolution_psd_generator`` callback
that defines a time-varying power spectrum.  The generator below
applies a Gaussian amplitude modulation that peaks at mid-simulation.

Run:
    python examples/nonstationary_custom_psd.py
"""

import numpy as np
from stochastic_wind_simulate import (
    create_simulator, NonstationaryWindSimulator, WindVisualizer,
)


# --- 1. Define a custom evolutionary PSD generator ---
# The callback receives (freq, heights, wind_speeds, component,
# time_value, time_index, total_time, simulator) and must return
# PSD values — one per spatial point.
def my_evolution_psd(freq, heights, wind_speeds, component,
                     mod_factor, simulator):
    """Time-scaled Kaimal spectrum.

    The PSD is the standard Kaimal spectrum multiplied by the
    square of the modulation factor.  This approximates the effect
    of a time-varying mean wind on turbulence intensity.

    Parameters
    ----------
    freq : scalar
        Frequency [Hz].
    heights : array (n,)
        Heights at each point.
    wind_speeds : array (n,)
        Reference mean wind speeds.
    component : str
        ``"u"`` or ``"w"``.
    mod_factor : scalar or array
        Pre-computed modulation factor at the current time step
        (e.g. ``1.0 + amplitude * sin(2*pi*t/T)``).
    simulator : NonstationaryWindSimulator
        The calling simulator (access spectrum, params, etc.).
    """
    # Standard Kaimal PSD, but with time-modulated mean wind speed
    # so that friction velocity and reduced frequency vary over time
    modulated_wind = wind_speeds * mod_factor
    base_psd = simulator.spectrum(
        freq, heights, component, U_d=modulated_wind,
    )
    return base_psd


# --- 2. Create stationary simulator & wrap with nonstationary ---
sim = create_simulator("jax", "kaimal", seed=42, N=1024, U_d=20.0, w_up=5.0)
ns  = NonstationaryWindSimulator(sim)

# --- 3. Define spatial points ---
n_points = 50
positions = np.zeros((n_points, 3), dtype=np.float32)
positions[:, 0] = np.linspace(0, 1000, n_points)
positions[:, 1] = 5.0
positions[:, 2] = 35.0
wind_speeds = np.full(n_points, 30.0, dtype=np.float32)

# --- 4. Simulate ---
samples, freqs = ns.simulate_nonstationary(
    positions, wind_speeds,
    component="u",
    mode="chunked-vmap",
    evolution_psd_generator=my_evolution_psd,   # custom generator
    max_memory_gb=8.0,
    auto_batch=True,
)
print(f"Simulated: {samples.shape[0]} points x {samples.shape[1]} time steps")

# --- 5. Inspect: time-varying variance ---
window = 64
var_t = np.array([
    np.var(samples[0, i:i + window]) for i in range(0, samples.shape[1] - window, window // 2)
])
dt = sim.params.dt
t_var = np.arange(len(var_t)) * (window // 2) * dt
print(f"Variance ranges from {var_t.min():.1f} to {var_t.max():.1f} "
      f"(period ≈ {sim.params.T:.0f} s)")

# --- 6. Visualise ---
viz = WindVisualizer(sim, seed=42)
viz.plot_psd(samples, positions[:, 2], show_num=5, component="u")
