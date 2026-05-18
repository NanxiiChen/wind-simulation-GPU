"""Point-scaling benchmark configuration."""

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.backends = ["jax", "torch", "numpy"]
    cfg.modes = ["batched"]
    cfg.max_memory_gb = 2.0

    # Fixed frequency count
    cfg.n_frequency = 3000

    # Point counts to test
    cfg.test_sizes = [2, 10, 25, 50, 100, 200, 500, 1000]

    # Simulation parameters
    cfg.params = ConfigDict()
    cfg.params.U_d = 25.0
    cfg.params.w_up = 1.0
    cfg.params.T = 600.0

    cfg.seed = 42
    cfg.output_dir = "benchmark_results"

    return cfg
