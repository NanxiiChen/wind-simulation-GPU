import time
import logging
import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stochastic_wind_simulate import get_simulator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    arg_parser = argparse.ArgumentParser(description="Wind field simulator benchmark test")
    arg_parser.add_argument(
        "--backend",
        type=str,
        choices=["jax", "torch", "numpy"],
        default="jax",
        help="Choose backend library: jax or torch (default: jax)",
    )
    args = arg_parser.parse_args()
    backend = args.backend
    logging.info(f"Using backend: {backend}")

    simulator = get_simulator(backend=backend, key=42)
    simulator.update_parameters(
        M=3000, N=1500, T=600, dt=0.2
    )

    with open(f"time_cost_{backend}.txt", "w") as f:
        f.write("n_samples,time_cost(s)\n")

    ns = [2, 5, 10, 30, 50, 100, 150, 200, 300, 500]
    time_costs = []
    Z = 30.0
    for i, n in enumerate(ns):
        positions = np.zeros((n, 3))
        positions[:, 0] = np.linspace(0, 100, n)
        positions[:, -1] = Z

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

        wind_speeds = positions[:, 0] * 0.2 + 25.0  # Simulate linearly varying mean wind speed

        start_time = time.time()
        u_samples, frequencies = simulator.simulate_wind(
            positions, wind_speeds, direction="u"
        )
        w_samples, frequencies = simulator.simulate_wind(
            positions, wind_speeds, direction="w"
        )

        elapsed_time = time.time() - start_time
        time_costs.append(elapsed_time)
        logging.info(f"n_samples: {n}, time cost: {elapsed_time:.4f} seconds, ")
        with open(f"time_cost_{backend}.txt", "a") as f:
            f.write(f"{n},{elapsed_time:.4f}\n")


if __name__ == "__main__":
    main()
