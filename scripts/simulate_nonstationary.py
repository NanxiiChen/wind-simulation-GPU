# import os
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["OPENBLAS_NUM_THREADS"] = "24"
import argparse
import csv
import logging
import time
from itertools import product
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from stochastic_wind_simulate.jax_backend.simulator_nonstationary import JaxNonstationaryWindSimulator


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _default_output_path(mode):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("benchmark_results") / f"nonstationary_benchmark_{mode}_{timestamp}.csv"


def build_positions(n_points, height=35.0):
    positions = np.zeros((n_points, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0.0, 1000.0, n_points)
    positions[:, 1] = 5.0
    positions[:, 2] = height
    return jnp.asarray(positions)


def main():
    parser = argparse.ArgumentParser(description="Run JAX nonstationary wind simulation")
    parser.add_argument("--n-points", type=int, nargs="+", default=[100], help="One or more spatial point counts")
    parser.add_argument("--n-freqs", type=int, nargs="+", default=[1024], help="One or more frequency component counts")
    parser.add_argument("--w-up", type=float, default=5.0, help="Cutoff frequency")
    parser.add_argument("--max-memory-gb", type=float, default=8.0, help="Memory budget for nonstationary chunking")
    parser.add_argument("--freq-batch-size", type=int, default=None, help="Manual frequency chunk size")
    parser.add_argument("--time-batch-size", type=int, default=None, help="Manual time chunk size")
    parser.add_argument("--modulation-amplitude", type=float, default=0.2, help="Sinusoidal modulation amplitude")
    parser.add_argument("--component", type=str, default="u", choices=["u", "w"], help="Wind component")
    parser.add_argument(
        "--mode",
        type=str,
        default="chunked-vmap",
        choices=["freq-for", "full-vmap", "chunked-vmap"],
        help="Nonstationary execution mode",
    )
    parser.add_argument("--output-file", type=str, default=None, help="Optional CSV output path under benchmark_results")
    parser.add_argument("--disable-auto-batch", action="store_true", help="Disable nonstationary auto chunking")
    args = parser.parse_args()

    simulator = JaxNonstationaryWindSimulator(key=42, spectrum_type="kaimal-nd")
    output_path = Path(args.output_file) if args.output_file else _default_output_path(args.mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode",
        "n_points",
        "n_freqs",
        "freq_batch",
        "time_batch",
        "n_freq_batches",
        "n_time_batches",
        "estimated_full_memory_gb",
        "chunk_memory_gb",
        "time_cost_s",
        "status",
        "error",
    ]

    combinations = list(product(args.n_points, args.n_freqs))
    logging.info("Running %d nonstationary benchmark case(s)", len(combinations))

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for n_points, n_freqs in combinations:
            logging.info(
                "Running JAX nonstationary simulation for n_points=%d, n_freqs=%d",
                n_points,
                n_freqs,
            )
            simulator.update_parameters(
                U_d=20.0,
                H_bar=20.0,
                alpha_0=0.12,
                z_0=0.01,
                w_up=args.w_up,
                N=n_freqs,
                M=n_freqs * 2,
            )

            positions = build_positions(n_points)
            wind_speeds = jnp.full((n_points,), 30.0, dtype=jnp.float32)
            start_time = time.time()

            row = {
                "mode": args.mode,
                "n_points": n_points,
                "n_freqs": n_freqs,
                "freq_batch": "",
                "time_batch": "",
                "n_freq_batches": "",
                "n_time_batches": "",
                "estimated_full_memory_gb": "",
                "chunk_memory_gb": "",
                "time_cost_s": "",
                "status": "failed",
                "error": "",
            }

            try:
                samples, frequencies = simulator.simulate_wind_nonstationary(
                    positions,
                    wind_speeds,
                    component=args.component,
                    mode=args.mode,
                    modulation_amplitude=args.modulation_amplitude,
                    max_memory_gb=args.max_memory_gb,
                    freq_batch_size=args.freq_batch_size,
                    time_batch_size=args.time_batch_size,
                    auto_batch=not args.disable_auto_batch,
                )
                samples.block_until_ready()
                elapsed = time.time() - start_time

                run_info = getattr(simulator, "last_nonstationary_run_info", {})
                row.update(
                    {
                        "freq_batch": run_info.get("freq_batch_size", ""),
                        "time_batch": run_info.get("time_batch_size", ""),
                        "n_freq_batches": run_info.get("n_freq_batches", ""),
                        "n_time_batches": run_info.get("n_time_batches", ""),
                        "estimated_full_memory_gb": run_info.get("estimated_full_memory_gb", ""),
                        "chunk_memory_gb": run_info.get("chunk_memory_gb", ""),
                        "time_cost_s": f"{elapsed:.6f}",
                        "status": "ok",
                    }
                )
                logging.info(
                    "Completed n_points=%d, n_freqs=%d in %.2f seconds; output shapes: samples=%s, frequencies=%s",
                    n_points,
                    n_freqs,
                    elapsed,
                    samples.shape,
                    frequencies.shape,
                )
            except Exception as exc:
                row["time_cost_s"] = f"{time.time() - start_time:.6f}"
                row["error"] = str(exc)
                logging.exception(
                    "Failed for n_points=%d, n_freqs=%d",
                    n_points,
                    n_freqs,
                )

            writer.writerow(row)
            csv_file.flush()

    logging.info("Benchmark results written to %s", output_path)


if __name__ == "__main__":
    main()