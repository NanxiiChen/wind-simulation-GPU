#!/usr/bin/env python
"""Unified benchmark script for wind field simulators.

Covers both frequency-scaling and point-scaling benchmarks across
all backends.  Loads configuration from ``ml_collections`` config
files and supports CLI overrides.

Examples
--------
.. code-block:: bash

    # Frequency scaling benchmark
    python scripts/benchmark.py --config configs/benchmark_freq.py

    # Point scaling benchmark
    python scripts/benchmark.py --config configs/benchmark_points.py

    # Quick test: single backend, small sizes
    python scripts/benchmark.py --backends jax --freqs 100,500,1000 --n-points 50
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stochastic_wind_simulate import create_simulator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Wind simulator benchmark")

    p.add_argument("--config", type=str, default=None,
                   help="Path to ml_collections config file")

    # Which benchmark to run
    p.add_argument("--bench-type", type=str,
                   choices=["freq", "points", "both"], default="freq",
                   help="Benchmark type")

    # Backends
    p.add_argument("--backends", type=str, nargs="+",
                   choices=["jax", "numpy", "torch", "all"],
                   default=["jax"])
    p.add_argument("--modes", type=str, nargs="+",
                   choices=["batched", "direct", "both"],
                   default=["batched"])

    # Test parameters (override config)
    p.add_argument("--freqs", type=str,
                   help="Comma-separated frequency counts, e.g. 100,500,1000")
    p.add_argument("--sizes", type=str,
                   help="Comma-separated point counts, e.g. 10,50,100")
    p.add_argument("--n-points", type=int, default=None,
                   help="Fixed point count (freq benchmark)")
    p.add_argument("--n-freqs", type=int, default=None,
                   help="Fixed frequency count (points benchmark)")
    p.add_argument("--max-memory-gb", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--output-dir", type=str, default="benchmark_results")

    return p.parse_args()


def run_case(backend, n_points, n_freqs, use_batching, max_memory_gb, seed):
    """Run a single benchmark case and return (time, memory, used_batching)."""
    # Build positions
    positions = np.zeros((n_points, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1000.0, n_points)
    positions[:, 1] = 5.0
    positions[:, 2] = 35.0
    wind_speeds = np.full(n_points, 25.0, dtype=np.float32)

    sim = create_simulator(
        backend, "kaimal", seed=seed,
        N=n_freqs, w_up=5.0 if backend == "jax" else 5.0,
    )

    est_mem = sim.estimate_memory(n_points, n_freqs)
    auto_batch = use_batching
    batch_mem = 0.1 if use_batching else 16.0

    try:
        t0 = time.time()
        samples, freqs = sim.simulate_wind(
            positions, wind_speeds, component="u",
            max_memory_gb=batch_mem, auto_batch=auto_batch,
        )
        elapsed = time.time() - t0
        return elapsed, est_mem, auto_batch and est_mem > max_memory_gb
    except Exception as e:
        logger.error("Case failed (backend=%s, n=%d, N=%d): %s",
                     backend, n_points, n_freqs, e)
        return float("nan"), est_mem, False


def main():
    args = parse_args()

    # Load config
    cfg = {}
    if args.config:
        import importlib.util

        spec = importlib.util.spec_from_file_location("config", args.config)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.get_config().to_dict()

    # Determine backends
    if "all" in args.backends:
        backends = ["jax", "torch", "numpy"]
    else:
        backends = args.backends

    # Determine modes
    if "both" in args.modes:
        modes = ["batched", "direct"]
    else:
        modes = args.modes

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Frequency scaling benchmark ---
    if args.bench_type in ("freq", "both"):
        freq_cfg = cfg.get("test_frequencies", [50, 100, 200, 500, 1000, 2000, 5000])
        if args.freqs:
            test_freqs = [int(x) for x in args.freqs.split(",")]
        else:
            test_freqs = freq_cfg

        n_points = args.n_points or cfg.get("n_points", 100)

        logger.info("=== Frequency Scaling Benchmark ===")
        logger.info("Backends: %s | Fixed points: %d | Freqs: %s",
                    backends, n_points, test_freqs)

        for backend in backends:
            for mode in modes:
                use_batching = (mode == "batched")
                tag = f"freq_{backend}_{mode}_{n_points}pts"
                csv_path = out_dir / f"benchmark_{tag}.csv"

                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["N", "time_s", "memory_gb", "used_batching"])

                    for N in test_freqs:
                        elapsed, mem, batched = run_case(
                            backend, n_points, N, use_batching,
                            args.max_memory_gb, args.seed + N,
                        )
                        writer.writerow([N, f"{elapsed:.4f}", f"{mem:.4f}", batched])
                        logger.info("  %s %s N=%d: %.4f s", backend, mode, N, elapsed)

                logger.info("Results: %s", csv_path)

    # --- Point scaling benchmark ---
    if args.bench_type in ("points", "both"):
        pts_cfg = cfg.get("test_sizes", [2, 10, 25, 50, 100, 200, 500, 1000])
        if args.sizes:
            test_sizes = [int(x) for x in args.sizes.split(",")]
        else:
            test_sizes = pts_cfg

        n_freqs = args.n_freqs or cfg.get("n_frequency", 3000)

        logger.info("=== Point Scaling Benchmark ===")
        logger.info("Backends: %s | Fixed freqs: %d | Points: %s",
                    backends, n_freqs, test_sizes)

        for backend in backends:
            for mode in modes:
                use_batching = (mode == "batched")
                tag = f"points_{backend}_{mode}_{n_freqs}freqs"
                csv_path = out_dir / f"benchmark_{tag}.csv"

                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["n", "time_s", "memory_gb", "used_batching"])

                    for n in test_sizes:
                        elapsed, mem, batched = run_case(
                            backend, n, n_freqs, use_batching,
                            args.max_memory_gb, args.seed + n,
                        )
                        writer.writerow([n, f"{elapsed:.4f}", f"{mem:.4f}", batched])
                        logger.info("  %s %s n=%d: %.4f s", backend, mode, n, elapsed)

                logger.info("Results: %s", csv_path)

    logger.info("Benchmark complete. Results in %s/", out_dir)


if __name__ == "__main__":
    main()
