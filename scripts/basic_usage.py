import sys
import os
import time
import argparse
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stochastic_wind_simulate import get_simulator, get_visualizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Main program entry point."""
    # Create wind field simulator
    # simulator = WindSimulator(key=42)
    arg_parser = argparse.ArgumentParser(description="Wind field simulator example")
    arg_parser.add_argument(
        "--backend",
        type=str,
        choices=["jax", "torch", "numpy"],
        default="jax",
        help="Choose backend library: jax or torch or numpy (default: jax)",
    )
    args = arg_parser.parse_args()
    backend = args.backend
    logging.info(f"Using backend: {backend}")
    simulator = get_simulator(backend=backend, key=42, spectrum_type="kaimal-nd")

    # Update simulator parameters
    simulator.update_parameters(
        U_d=20.0,
        H_bar=20.0,
        alpha_0=0.12,
        z_0=0.01,
        w_up=5.0,
        N=1024,
        M=2048
    )

    # Define simulation point positions and mean wind speeds
    n = 2048  # Number of simulation points
    Z = 30.0  # Height (m)

    positions = np.zeros((n, 3))
    positions[:, 0] = np.linspace(0, 1000, n)
    positions[:, 1] = 5
    positions[:, -1] = Z + 5

    if backend == "jax":
        import jax.numpy as jnp
        positions = jnp.array(positions)

    elif backend == "torch":
        import torch
        positions = torch.from_numpy(positions)
    
    elif backend == "numpy":
        # positions is already a numpy array, no conversion needed
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    wind_speeds = positions[:, 0] * 0.0 + 30.0  # Simulate linearly varying mean wind speed

    start_time = time.time()
    samples, frequencies = simulator.simulate_wind(positions, wind_speeds, component="u", 
                                                   auto_batch=True, max_memory_gb=8.0)
    elapsed_time = time.time() - start_time

    logging.info(f"Simulation completed, elapsed time: {elapsed_time:.2f} seconds")

    visualizer = get_visualizer(backend=backend, key=42, simulator=simulator)
    visualizer.plot_psd(samples, positions[:, -1], show_num=6, show=True, component="u")
    visualizer.plot_cross_correlation(samples, positions, wind_speeds, show=True, component="u", indices=(1, 5))

    if backend == "jax":
        import jax.numpy as jnp
        jnp.save("samples_jax.npy", samples)
    elif backend == "torch":
        import torch
        np.save("samples_torch.npy", samples)
    elif backend == "numpy":
        np.save("samples_numpy.npy", samples)
    else:
        raise ValueError(f"Unsupported backend: {backend}")



if __name__ == "__main__":
    main()
