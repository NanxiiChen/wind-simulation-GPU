"""Default stationary simulation configuration."""

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.backend = "jax"
    cfg.spectrum = "kaimal"
    cfg.seed = 42
    cfg.component = "u"

    # Physical & numerical parameters
    cfg.params = ConfigDict()
    cfg.params.U_d = 20.0
    cfg.params.H_bar = 20.0
    cfg.params.alpha_0 = 0.12
    cfg.params.z_0 = 0.01
    cfg.params.w_up = 5.0
    cfg.params.N = 3000

    # Spatial layout
    cfg.spatial = ConfigDict()
    cfg.spatial.n_points = 100
    cfg.spatial.z = 35.0
    cfg.spatial.wind_speed = 30.0

    # Nonstationary (disabled by default — enable with --config.nonstationary.enabled=True)
    cfg.nonstationary = ConfigDict()
    cfg.nonstationary.enabled = False
    cfg.nonstationary.mode = "chunked-vmap"
    cfg.nonstationary.modulation_amplitude = 0.2

    # Memory / batching
    cfg.memory = ConfigDict()
    cfg.memory.max_memory_gb = 4.0
    cfg.memory.auto_batch = True
    cfg.memory.freq_batch_size = None  # None = auto

    # Visualisation
    cfg.visualization = ConfigDict()
    cfg.visualization.show_plots = True
    cfg.visualization.point_index = 0
    cfg.visualization.window_size = 64
    cfg.visualization.overlap = 50
    cfg.visualization.snapshot_count = 4

    # Output
    cfg.output = ConfigDict()
    cfg.output.save_samples = True
    cfg.output.save_dir = "output"

    return cfg
