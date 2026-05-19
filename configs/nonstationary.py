"""Default nonstationary simulation configuration."""

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.backend = "jax"
    cfg.spectrum = "kaimal"
    cfg.seed = 42
    cfg.component = "u"

    cfg.params = ConfigDict()
    cfg.params.U_d = 20.0
    cfg.params.H_bar = 20.0
    cfg.params.alpha_0 = 0.12
    cfg.params.z_0 = 0.01
    cfg.params.w_up = 5.0
    cfg.params.N = 1024

    cfg.spatial = ConfigDict()
    cfg.spatial.n_points = 100
    cfg.spatial.z = 35.0
    cfg.spatial.wind_speed = 30.0

    cfg.nonstationary = ConfigDict()
    cfg.nonstationary.enabled = True
    cfg.nonstationary.mode = "chunked-vmap"
    cfg.nonstationary.modulation_amplitude = 0.2
    cfg.nonstationary.max_memory_gb = 8.0
    cfg.nonstationary.auto_batch = True
    cfg.nonstationary.freq_batch_size = None

    cfg.memory = ConfigDict()
    cfg.memory.max_memory_gb = 4.0
    cfg.memory.auto_batch = True

    cfg.visualization = ConfigDict()
    cfg.visualization.point_index = 0
    cfg.visualization.window_size = 64
    cfg.visualization.overlap = 50
    cfg.visualization.snapshot_count = 4
    cfg.visualization.show_plots = True

    cfg.output = ConfigDict()
    cfg.output.save_samples = True
    cfg.output.save_dir = "output"

    return cfg
