"""Nonstationary validation configuration."""

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.backend = "jax"
    cfg.spectrum = "kaimal"
    cfg.seed = 42

    # Simulation parameters
    cfg.params = ConfigDict()
    cfg.params.U_d = 20.0
    cfg.params.H_bar = 20.0
    cfg.params.alpha_0 = 0.12
    cfg.params.z_0 = 0.01
    cfg.params.w_up = 5.0
    cfg.params.N = 1024

    # Spatial
    cfg.spatial = ConfigDict()
    cfg.spatial.n_points = 100
    cfg.spatial.height = 35.0

    # Wind
    cfg.wind = ConfigDict()
    cfg.wind.speed = 30.0
    cfg.wind.component = "u"

    # Nonstationary
    cfg.nonstationary = ConfigDict()
    cfg.nonstationary.mode = "chunked-vmap"
    cfg.nonstationary.modulation_amplitude = 0.2

    cfg.memory = ConfigDict()
    cfg.memory.max_memory_gb = 8.0

    # Validation
    cfg.validation = ConfigDict()
    cfg.validation.n_realizations = 32
    cfg.validation.point_index = 0
    cfg.validation.window_size = 64
    cfg.validation.overlap = 50
    cfg.validation.skip_low_freq_bins = 1
    cfg.validation.psd_snapshot_count = 4

    cfg.output_dir = "benchmark_results"

    return cfg
