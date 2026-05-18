"""Stationary simulation with a custom wind spectrum.

Demonstrates how to subclass ``WindSpectrum`` and plug it into
the simulator.  The custom spectrum below combines Kaimal for
the along-wind component and Panofsky for the vertical component.

Run:
    python examples/custom_spectrum.py
"""

import numpy as np
from stochastic_wind_simulate import JaxWindSimulator, WindVisualizer
from stochastic_wind_simulate.spectrum import WindSpectrum


# --- 1. Define a custom spectrum ---
class MySpectrum(WindSpectrum):
    """Custom spectrum: Kaimal (u) + Panofsky (w)."""

    def psd_u(self, n, u_star, f):
        r"""Kaimal along-wind: :math:`S_u(n) = \frac{u_*^2}{n} \frac{200 f}{(1+50f)^{5/3}}`."""
        return (u_star**2 / n) * (200.0 * f / (1.0 + 50.0 * f)**(5.0 / 3.0))

    def psd_w(self, n, u_star, f):
        r"""Panofsky vertical: :math:`S_w(n) = \frac{u_*^2}{n} \frac{6 f}{(1+4f)^2}`."""
        return (u_star**2 / n) * (6.0 * f / (1.0 + 4.0 * f)**2.0)


# --- 2. Create simulator with the custom spectrum ---
# Pass the class itself (not a string) as spectrum_type
sim = JaxWindSimulator(
    key=42,
    spectrum_type=MySpectrum,   # custom class
    N=3000, U_d=20.0, w_up=5.0,
)

# --- 3. Define spatial points ---
n_points = 100
positions = np.zeros((n_points, 3), dtype=np.float32)
positions[:, 0] = np.linspace(0, 1000, n_points)
positions[:, 1] = 5.0
positions[:, 2] = 35.0
wind_speeds = np.full(n_points, 30.0, dtype=np.float32)

# --- 4. Simulate both components ---
u_samples, freqs = sim.simulate_wind(positions, wind_speeds, component="u",
                                     max_memory_gb=4.0, auto_batch=True)
w_samples, _     = sim.simulate_wind(positions, wind_speeds, component="w",
                                     max_memory_gb=4.0, auto_batch=True)
print(f"u: mean={u_samples.mean():.3f}  std={u_samples.std():.3f}")
print(f"w: mean={w_samples.mean():.3f}  std={w_samples.std():.3f}")

# --- 5. Visualise ---
viz = WindVisualizer(sim, seed=42)
viz.plot_psd(u_samples, positions[:, 2], show_num=5, component="u")
viz.plot_psd(w_samples, positions[:, 2], show_num=5, component="w")
