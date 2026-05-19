#!/usr/bin/env python
"""Unified wind field simulation script.

Supports stationary and nonstationary simulation across all backends.
Parameters are loaded from an ``ml_collections`` config file and can
be overridden via command-line arguments.

Examples
--------
    python scripts/simulate.py
    python scripts/simulate.py --backend numpy --n-points 200
    python scripts/simulate.py --config configs/nonstationary.py --nonstationary
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stochastic_wind_simulate import (
    NonstationaryWindSimulator,
    WindVisualizer,
    create_simulator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _first(*values):
    for v in values:
        if v is not None:
            return v
    return None


def main():
    p = argparse.ArgumentParser(description="Stochastic wind field simulation")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--nonstationary", action="store_true")
    p.add_argument("--backend", type=str, choices=["jax", "numpy", "torch"])
    p.add_argument("--spectrum", type=str, choices=["kaimal", "panofsky", "teunissen"])
    p.add_argument("--seed", type=int)
    p.add_argument("--U_d", type=float); p.add_argument("--H_bar", type=float)
    p.add_argument("--alpha_0", type=float); p.add_argument("--z_0", type=float)
    p.add_argument("--w-up", type=float); p.add_argument("--N", type=int)
    p.add_argument("--n-points", type=int); p.add_argument("--z", type=float)
    p.add_argument("--wind-speed", type=float)
    p.add_argument("--component", type=str, choices=["u", "w"])
    p.add_argument("--max-memory-gb", type=float)
    p.add_argument("--freq-batch-size", type=int)
    p.add_argument("--no-auto-batch", action="store_true")
    p.add_argument("--mode", type=str, choices=["chunked-vmap", "freq-for", "full-vmap"])
    p.add_argument("--modulation-amplitude", type=float)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--output-dir", type=str, default="output")
    args = p.parse_args()

    # Load config
    cfg = {}
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("cfg", args.config)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.get_config().to_dict()

    backend = _first(args.backend, cfg.get("backend"), "jax")
    spectrum = _first(args.spectrum, cfg.get("spectrum"), "kaimal")
    seed = _first(args.seed, cfg.get("seed"), 42)
    nonstationary = args.nonstationary or cfg.get("nonstationary", False)

    params_cfg = cfg.get("params", {})
    param_kw = {}
    for k in ["U_d", "H_bar", "alpha_0", "z_0", "w_up", "N"]:
        v = _first(getattr(args, k, None), params_cfg.get(k))
        if v is not None:
            param_kw[k] = v

    spatial_cfg = cfg.get("spatial", {})
    n_points = _first(args.n_points, spatial_cfg.get("n_points"), 100)
    z_height = _first(args.z, spatial_cfg.get("z", spatial_cfg.get("height")), 35.0)
    wind_speed_val = _first(args.wind_speed, cfg.get("wind", {}).get("speed",
                            spatial_cfg.get("wind_speed")), 30.0)
    component = _first(args.component, cfg.get("component",
                       cfg.get("wind", {}).get("component")), "u")

    mem_cfg = cfg.get("memory", {})
    max_mem = _first(args.max_memory_gb, mem_cfg.get("max_memory_gb"), 4.0)
    freq_bs = _first(args.freq_batch_size, mem_cfg.get("freq_batch_size"))

    # Build positions
    positions = np.zeros((n_points, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1000.0, n_points)
    positions[:, 1] = 5.0
    positions[:, 2] = z_height
    wind_speeds = np.full(n_points, wind_speed_val, dtype=np.float32)

    logger.info("%s | %s | %s | points=%d N=%d",
                "nonstationary" if nonstationary else "stationary",
                backend, spectrum, n_points,
                param_kw.get("N", cfg.get("params", {}).get("N", 3000)))

    # Create simulator
    ns_cfg = cfg.get("nonstationary", {})
    mode = _first(args.mode, ns_cfg.get("mode"), "chunked-vmap")
    mod_amp = _first(args.modulation_amplitude, ns_cfg.get("modulation_amplitude"), 0.2)

    sim = create_simulator(backend, spectrum, seed=seed, **param_kw)
    if nonstationary:
        sim = NonstationaryWindSimulator(sim)

    # Simulate
    t0 = time.time()
    if nonstationary:
        samples, freqs = sim.simulate_nonstationary(
            positions, wind_speeds, component=component,
            mode=mode, modulation_amplitude=mod_amp,
            max_memory_gb=max_mem, freq_batch_size=freq_bs,
            auto_batch=not args.no_auto_batch,
        )
    else:
        samples, freqs = sim.simulate_wind(
            positions, wind_speeds, component=component,
            max_memory_gb=max_mem, freq_batch_size=freq_bs,
            auto_batch=not args.no_auto_batch,
        )
    logger.info("Done in %.2f s | shape %s", time.time() - t0, samples.shape)

    # Save
    if not args.no_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "ns" if nonstationary else "s"
        path = out_dir / f"samples_{backend}_{tag}.npy"
        np.save(path, samples)
        logger.info("Saved %s", path)

    # Visualize
    show = not args.no_show
    if show or not args.no_save:
        viz = WindVisualizer(sim, seed=seed)
        out_dir = Path(args.output_dir)
        if nonstationary:
            viz_cfg = cfg.get("visualization", {})
            pt = viz_cfg.get("point_index", 0)
            viz.plot_nonstationary_psd(
                samples[pt], height=positions[pt, 2], wind_speed=wind_speeds[pt],
                component=component,
                window_size=viz_cfg.get("window_size", 64),
                overlap=viz_cfg.get("overlap", 50),
                modulation_amplitude=mod_amp,
                snapshot_count=viz_cfg.get("snapshot_count", 4),
                show=show,
                save_path=str(out_dir / f"psd_{backend}_ns.png") if not args.no_save else None,
            )
        else:
            viz.plot_psd(samples, positions[:, 2], show_num=6, component=component,
                         show=show, save_path=str(out_dir / f"psd_{backend}.png") if not args.no_save else None)
            viz.plot_cross_correlation(samples, positions, wind_speeds, component=component,
                                       indices=(1, min(5, n_points - 1)), show=show,
                                       save_path=str(out_dir / f"corr_{backend}.png") if not args.no_save else None)


if __name__ == "__main__":
    main()
