import time
import logging
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stochastic_wind_simulate import get_simulator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    arg_parser = argparse.ArgumentParser(description="风场模拟器基准测试")
    arg_parser.add_argument(
        "--backend",
        type=str,
        choices=["jax", "torch"],
        default="jax",
        help="选择后端库: jax 或 torch (默认: jax)",
    )
    args = arg_parser.parse_args()
    backend = args.backend
    logging.info(f"Using backend: {backend}")

    simulator = get_simulator(backend=backend, key=42)
    with open(f"time_cost_{backend}.txt", "w") as f:
        f.write("n_samples,time_cost(s)\n")

    ns = [2, 5, 10, 30, 50, 100, 150, 200]
    time_costs = []
    Z = 30.0
    for i, n in enumerate(ns):

        if backend == "jax":
            import jax.numpy as jnp

            positions = jnp.zeros((n, 3))  # 初始化位置数组，(x, y, z)
            positions = positions.at[:, 0].set(jnp.linspace(0, 100, n))
            positions = positions.at[:, -1].set(Z)

        elif backend == "torch":
            import torch
            import numpy as np

            positions = torch.zeros((n, 3))  # 初始化位置数组，(x, y, z)
            positions[:, 0] = torch.linspace(0, 100, n)
            positions[:, -1] = Z

        wind_speeds = positions[:, 0] * 0.2 + 25.0  # 模拟线性变化的平均风速

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
