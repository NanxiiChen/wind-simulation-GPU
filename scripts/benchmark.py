#!/usr/bin/env python
"""Unified benchmark script.

Uses ``ml_collections`` + ``absl.flags``.  No argparse — everything
through ``--config.key=value``.

Examples
--------
    python scripts/benchmark.py --config=configs/benchmark_freq.py
    python scripts/benchmark.py --config=configs/benchmark_points.py
    python scripts/benchmark.py --config=configs/benchmark_freq.py \
        --config.backends=jax,numpy --config.test_frequencies=100,500,1000
"""

import csv
import logging
import time
from pathlib import Path

from absl import app
import numpy as np


from ml_collections import config_flags
from stochastic_wind_simulate import create_simulator

_CONFIG = config_flags.DEFINE_config_file(
    "config", "configs/benchmark_freq", "Path to config file", lock_config=False,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_case(backend, n_points, n_freqs, use_batching, max_memory_gb, seed):
    positions = np.zeros((n_points, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1000.0, n_points)
    positions[:, 1] = 5.0; positions[:, 2] = 35.0
    wind_speeds = np.full(n_points, 25.0, dtype=np.float32)

    sim = create_simulator(backend, "kaimal", seed=seed, N=n_freqs, w_up=5.0)
    est_mem = sim.estimate_memory(n_points, n_freqs)

    try:
        t0 = time.time()
        sim.simulate_wind(positions, wind_speeds, "u",
                          max_memory_gb=max_memory_gb, auto_batch=use_batching)
        return time.time() - t0, est_mem, use_batching and est_mem > max_memory_gb
    except Exception as e:
        logger.error("%s n=%d N=%d: %s", backend, n_points, n_freqs, e)
        return float("nan"), est_mem, False


def main(_):
    cfg = _CONFIG.value
    params = cfg.params
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def _parse_list(v):
        return v.split(",") if isinstance(v, str) else v

    backends = _parse_list(cfg.backends)
    modes = _parse_list(cfg.modes)

    # --- Frequency scaling ---
    if cfg.bench_type in ("freq", "both"):
        test_freqs = [int(x) for x in _parse_list(cfg.test_frequencies)]
        logger.info("=== Frequency Scaling ===")
        for backend in backends:
            for mode in modes:
                tag = f"freq_{backend}_{mode}_{cfg.n_points}pts"
                with (out_dir / f"benchmark_{tag}.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["N", "time_s", "memory_gb", "used_batching"])
                    for N in test_freqs:
                        t, mem, batched = run_case(
                            backend, cfg.n_points, N, mode == "batched",
                            cfg.max_memory_gb, cfg.seed + N)
                        w.writerow([N, f"{t:.4f}", f"{mem:.4f}", batched])
                        logger.info("  %s %s N=%d: %.4f s", backend, mode, N, t)
                logger.info("Saved %s", out_dir / f"benchmark_{tag}.csv")

    # --- Point scaling ---
    if cfg.bench_type in ("points", "both"):
        test_sizes = [int(x) for x in _parse_list(cfg.test_sizes)]
        logger.info("=== Point Scaling ===")
        for backend in backends:
            for mode in modes:
                tag = f"points_{backend}_{mode}_{cfg.n_frequency}freqs"
                with (out_dir / f"benchmark_{tag}.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["n", "time_s", "memory_gb", "used_batching"])
                    for n in test_sizes:
                        t, mem, batched = run_case(
                            backend, n, cfg.n_frequency, mode == "batched",
                            cfg.max_memory_gb, cfg.seed + n)
                        w.writerow([n, f"{t:.4f}", f"{mem:.4f}", batched])
                        logger.info("  %s %s n=%d: %.4f s", backend, mode, n, t)
                logger.info("Saved %s", out_dir / f"benchmark_{tag}.csv")

    logger.info("Done. Results in %s/", out_dir)


if __name__ == "__main__":
    app.run(main)
