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
    """Main program entry point demonstrating batched simulation."""
    arg_parser = argparse.ArgumentParser(description="Wind field simulator with batching example")
    arg_parser.add_argument(
        "--backend",
        type=str,
        choices=["jax", "torch", "numpy"],
        default="jax",
        help="Choose backend library: jax or torch or numpy (default: jax)",
    )
    arg_parser.add_argument(
        "--n-points",
        type=int,
        default=200,
        help="Number of simulation points (default: 200)",
    )
    arg_parser.add_argument(
        "--use-batching",
        action="store_true",
        help="Use batched simulation for memory management",
    )
    arg_parser.add_argument(
        "--max-memory",
        type=float,
        default=2.0,
        help="Maximum memory limit in GB (default: 2.0)",
    )
    
    args = arg_parser.parse_args()
    backend = args.backend
    n = args.n_points
    use_batching = args.use_batching
    max_memory = args.max_memory
    
    logging.info(f"Using backend: {backend}")
    logging.info(f"Number of simulation points: {n}")
    logging.info(f"Use batching: {use_batching}")
    
    simulator = get_simulator(backend=backend, key=42, spectrum_type="kaimal-nd")

    # Update simulator parameters
    simulator.update_parameters(
        U_d=20.0,
        H_bar=20.0,
        alpha_0=0.12,
        z_0=0.01,
        w_up=5.0,
    )

    # Define simulation point positions and mean wind speeds
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

    wind_speeds = positions[:, 0] * 0.0 + 30.0  # Constant wind speed

    # Estimate memory requirement (for all backends that support it)
    if hasattr(simulator, 'estimate_memory_requirement'):
        estimated_memory = simulator.estimate_memory_requirement(n, simulator.params["N"])
        logging.info(f"Estimated memory requirement: {estimated_memory:.2f} GB")
    
    start_time = time.time()
    
    if use_batching:
        # Force batching by using a very low memory limit
        logging.info("Using forced batching with low memory limit...")
        # Use a very low memory limit to force batching
        force_batch_memory = 0.1
        samples, frequencies = simulator.simulate_wind(
            positions, wind_speeds, component="u", 
            max_memory_gb=force_batch_memory, auto_batch=True
        )
    else:
        # Use regular simulation with auto-batching based on memory requirements
        logging.info("Using regular simulation with auto-batching...")
        
        # Adjust default memory limits based on backend
        if backend == "numpy":
            # NumPy typically has access to more CPU memory
            default_memory = max(max_memory, 8.0)
        else:
            default_memory = max_memory
            
        samples, frequencies = simulator.simulate_wind(
            positions, wind_speeds, component="u", 
            max_memory_gb=default_memory, auto_batch=True
        )
    
    elapsed_time = time.time() - start_time
    logging.info(f"Simulation completed, elapsed time: {elapsed_time:.2f} seconds")

    # Only visualize a subset for large simulations
    visualize_points = min(6, n)
    visualizer = get_visualizer(backend=backend, key=42, simulator=simulator)
    visualizer.plot_psd(samples, positions[:, -1], show_num=visualize_points, show=True, component="u")

    # plot cross correlations
    visualizer.plot_cross_correlation(samples, positions, wind_speeds, show=True, component="u", indices=(1, 10))

    
    # Save results
    if backend == "jax":
        import jax.numpy as jnp
        jnp.save(f"samples_jax_n{n}_batched_{use_batching}.npy", samples)
    elif backend == "torch":
        import torch
        np.save(f"samples_torch_n{n}_batched_{use_batching}.npy", samples)
    elif backend == "numpy":
        np.save(f"samples_numpy_n{n}_batched_{use_batching}.npy", samples)
    
    logging.info(f"Results saved for {n} points, batching={use_batching}")


if __name__ == "__main__":
    main()
