"""Default stationary simulation configuration."""

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    # Backend
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
    cfg.params.N = 3000

    # Spatial configuration
    cfg.spatial = ConfigDict()
    cfg.spatial.n_points = 100
    cfg.spatial.z = 35.0  # Height (m)
    cfg.spatial.x_range = [0.0, 1000.0]  # x min, x max
    cfg.spatial.y = 5.0  # Fixed y coordinate
    cfg.spatial.wind_speed = 30.0  # Constant wind speed at all points

    # Wind component
    cfg.component = "u"

    # Memory / batching
    cfg.memory = ConfigDict()
    cfg.memory.max_memory_gb = 4.0
    cfg.memory.auto_batch = True
    cfg.memory.freq_batch_size = None  # None = auto

    # Output
    cfg.output = ConfigDict()
    cfg.output.save_samples = True
    cfg.output.save_dir = "output"
    cfg.output.show_plots = True

    return cfg
