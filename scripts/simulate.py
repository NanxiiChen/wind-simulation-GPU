#!/usr/bin/env python
"""Unified wind field simulation script.

Uses ``ml_collections`` + ``absl.flags``.  All parameters live in config
files under ``configs/`` and can be overridden with dotted notation.

Examples
--------
    python scripts/simulate.py --config=configs/default.py
    python scripts/simulate.py --config=configs/default.py --config.params.N=5000
    python scripts/simulate.py --config=configs/nonstationary.py
    python scripts/simulate.py --config=configs/default.py \
        --config.nonstationary.enabled=True \
        --config.visualization.window_size=128
"""

import logging
import sys
import time
from pathlib import Path

from absl import app
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_collections import ConfigDict, config_flags
from stochastic_wind_simulate import (
    NonstationaryWindSimulator, WindVisualizer, create_simulator,
)

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "Path to config file", lock_config=False,
)
FLAGS = app.flags.FLAGS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = ConfigDict(dict(
    backend="jax", spectrum="kaimal", seed=42, component="u",
    params=dict(U_d=20.0, H_bar=20.0, z_0=0.01, alpha_0=0.12, w_up=5.0, N=3000),
    spatial=dict(n_points=100, z=35.0, wind_speed=30.0),
    nonstationary=dict(enabled=False, mode="chunked-vmap", modulation_amplitude=0.2),
    memory=dict(max_memory_gb=4.0, auto_batch=True, freq_batch_size=None),
    visualization=dict(point_index=0, window_size=64, overlap=50, snapshot_count=4,
                       show_plots=True),
    output=dict(save_samples=True, save_dir="output"),
))


def main(_):
    cfg = _CONFIG.value or _DEFAULT_CONFIG
    backend = cfg.backend
    spectrum = cfg.spectrum
    seed = cfg.seed
    component = cfg.component
    params = cfg.params
    spatial = cfg.spatial
    ns_cfg = cfg.nonstationary
    mem = cfg.memory
    viz_cfg = cfg.visualization
    out = cfg.output

    positions = np.zeros((spatial.n_points, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1000.0, spatial.n_points)
    positions[:, 1] = 5.0
    positions[:, 2] = spatial.z
    wind_speeds = np.full(spatial.n_points, spatial.wind_speed, dtype=np.float32)

    logger.info("%s | %s | %s | points=%d N=%d",
                "nonstationary" if ns_cfg.enabled else "stationary",
                backend, spectrum, spatial.n_points, params.N)

    sim = create_simulator(backend, spectrum, seed=seed, **params.to_dict())
    if ns_cfg.enabled:
        sim = NonstationaryWindSimulator(sim)

    t0 = time.time()
    if ns_cfg.enabled:
        samples, freqs = sim.simulate_nonstationary(
            positions, wind_speeds, component=component,
            mode=ns_cfg.mode,
            modulation_amplitude=ns_cfg.modulation_amplitude,
            max_memory_gb=ns_cfg.get("max_memory_gb", mem.max_memory_gb),
            freq_batch_size=ns_cfg.get("freq_batch_size", mem.get("freq_batch_size")),
            auto_batch=ns_cfg.get("auto_batch", mem.get("auto_batch", True)),
        )
    else:
        samples, freqs = sim.simulate_wind(
            positions, wind_speeds, component=component,
            max_memory_gb=mem.get("max_memory_gb", 4.0),
            freq_batch_size=mem.get("freq_batch_size"),
            auto_batch=mem.get("auto_batch", True),
        )
    logger.info("Done in %.2f s | shape %s", time.time() - t0, samples.shape)

    if out.save_samples:
        d = Path(out.save_dir); d.mkdir(parents=True, exist_ok=True)
        tag = "ns" if ns_cfg.enabled else "s"
        np.save(d / f"samples_{backend}_{tag}.npy", samples)
        logger.info("Saved %s", d / f"samples_{backend}_{tag}.npy")

    if viz_cfg.show_plots or out.save_samples:
        viz = WindVisualizer(sim, seed=seed)
        d = Path(out.save_dir)
        if ns_cfg.enabled:
            pt = viz_cfg.point_index
            viz.plot_nonstationary_psd(
                samples[pt], height=positions[pt, 2], wind_speed=wind_speeds[pt],
                component=component,
                window_size=viz_cfg.window_size, overlap=viz_cfg.overlap,
                modulation_amplitude=ns_cfg.modulation_amplitude,
                snapshot_count=viz_cfg.snapshot_count,
                show=viz_cfg.show_plots,
                save_path=str(d / f"psd_{backend}_ns.png") if out.save_samples else None,
            )
        else:
            viz.plot_psd(samples, positions[:, 2], show_num=6, component=component,
                         show=viz_cfg.show_plots,
                         save_path=str(d / f"psd_{backend}.png") if out.save_samples else None)
            viz.plot_cross_correlation(samples, positions, wind_speeds, component=component,
                                       indices=(1, min(5, spatial.n_points - 1)),
                                       show=viz_cfg.show_plots,
                                       save_path=str(d / f"corr_{backend}.png") if out.save_samples else None)


if __name__ == "__main__":
    app.run(main)
