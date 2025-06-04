
import sys
import os
import time
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stochastic_wind_simulate import get_simulator, get_visualizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    """主程序入口"""
    # 创建风场模拟器
    # simulator = WindSimulator(key=42)
    arg_parser = argparse.ArgumentParser(description="风场模拟器示例")
    arg_parser.add_argument(
        "--backend",
        type=str,
        choices=["jax", "torch"],
        default="jax",
        help="选择后端库: jax 或 torch (默认: jax)",
    )
    args = arg_parser.parse_args()
    backend = args.backend
    logging.info(f"使用后端: {backend}")
    simulator = get_simulator(backend=backend, key=42)

    # 更新模拟器参数
    simulator.update_parameters(
        U_d=20.0,
        H_bar=15.0,
    )

    # 定义模拟点的位置和平均风速
    n = 200  # 模拟点数量
    Z = 30.0  # 高度(m)

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

    # 各点平均风速
    wind_speeds = positions[:, 0] * 0.2 + 25.0  # 模拟线性变化的平均风速

    # 记录开始时间
    start_time = time.time()

    # 模拟顺风向脉动风
    logging.info("模拟顺风向脉动风...")
    u_samples, frequencies = simulator.simulate_wind(
        positions, wind_speeds, direction="u"
    )

    # 模拟竖向脉动风
    logging.info("模拟竖向脉动风...")
    w_samples, frequencies = simulator.simulate_wind(
        positions, wind_speeds, direction="w"
    )


    # 打印计算时间
    elapsed_time = time.time() - start_time
    logging.info(f"模拟完成，耗时: {elapsed_time:.2f}秒")

    # visualizer = WindVisualizer(key=42, **simulator.params)
    visualizer = get_visualizer(backend=backend, key=42, **simulator.params)
    visualizer.plot_psd(u_samples, Z, show_num=5, show=True, direction="u")
    visualizer.plot_psd(w_samples, Z, show_num=5, show=True, direction="w")

    visualizer.plot_cross_correlation(
        u_samples, positions, wind_speeds, show=True, direction="u", indices=(1, 1)
    )
    visualizer.plot_cross_correlation(
        w_samples, positions, wind_speeds, show=True, direction="w", indices=(1, 1)
    )


if __name__ == "__main__":
    main()
