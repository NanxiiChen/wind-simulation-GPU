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
    simulator = get_simulator(backend=backend, key=42, spectrum_type="teunissen")

    # Update simulator parameters
    simulator.update_parameters(
        U_d=15.0,
        H_bar=15.0,
        alpha_0=0.12,
        z_0=0.01,
        w_up=5.0,
    )

    # Define simulation point positions and mean wind speeds
    n = 100  # Number of simulation points
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


    # Mean wind speed at each point
    wind_speeds = positions[:, 0] * 0.0 + 30.0  # Simulate linearly varying mean wind speed

    # Record start time
    start_time = time.time()

    # Simulate along-wind fluctuating wind
    logging.info("Simulating along-wind fluctuating wind...")
    u_samples, frequencies = simulator.simulate_wind(
        positions, wind_speeds, component="u"
    )

    # Simulate vertical fluctuating wind
    logging.info("Simulating vertical fluctuating wind...")
    w_samples, frequencies = simulator.simulate_wind(
        positions, wind_speeds, component="w"
    )

    # Print computation time
    elapsed_time = time.time() - start_time
    logging.info(f"Simulation completed, elapsed time: {elapsed_time:.2f} seconds")

    # visualizer = WindVisualizer(key=42, **simulator.params)
    visualizer = get_visualizer(backend=backend, key=42, simulator=simulator)
    visualizer.plot_psd(
        u_samples, positions[:, -1], show_num=6, show=True, component="u"
    )
    visualizer.plot_psd(
        w_samples, positions[:, -1], show_num=6, show=True, component="w"
    )

    visualizer.plot_cross_correlation(
        u_samples, positions, wind_speeds, show=True, component="u", indices=(1, 10)
    )
    visualizer.plot_cross_correlation(
        w_samples, positions, wind_speeds, show=True, component="w", indices=(1, 10)
    )

    if backend == "jax":
        import jax.numpy as jnp
        jnp.save("u_samples_jax.npy", u_samples)
        jnp.save("w_samples_jax.npy", w_samples)
    elif backend == "torch":
        import torch
        u_samples = u_samples.detach().cpu().numpy()
        w_samples = w_samples.detach().cpu().numpy()
        np.save("u_samples_torch.npy", u_samples)
        np.save("w_samples_torch.npy", w_samples)
    elif backend == "numpy":
        np.save("u_samples_numpy.npy", u_samples)
        np.save("w_samples_numpy.npy", w_samples)
    else:
        raise ValueError(f"Unsupported backend: {backend}")



if __name__ == "__main__":
    main()
