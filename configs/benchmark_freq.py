"""Frequency-scaling benchmark configuration."""

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.backends = ["jax", "torch", "numpy"]
    cfg.modes = ["batched"]
    cfg.max_memory_gb = 2.0

    # Fixed spatial points
    cfg.n_points = 100

    # Frequency counts to test
    cfg.test_frequencies = [50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000]

    # Simulation parameters
    cfg.params = ConfigDict()
    cfg.params.U_d = 25.0
    cfg.params.w_up = 5.0
    cfg.params.T = 600.0

    cfg.seed = 42
    cfg.output_dir = "benchmark_results"

    return cfg
