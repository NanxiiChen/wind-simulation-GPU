"""
This script benchmarks the performance of different backends (JAX, PyTorch, NumPy)
It mainly tests the performance of each backend with different frequency segments under fixed sample points.
Modified from benchmarks_points.py to test frequency scaling.
"""

import time
import logging
import argparse
import sys
import os
import numpy as np
from pathlib import Path
import multiprocessing
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stochastic_wind_simulate import get_simulator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_single_case(args_dict):
    """
    Run a single benchmark case in a new process. Args in dict form.
    Returns: dict with time_cost, memory_estimate, freq_batch_size, used_batching
    """
    import time
    import numpy as np
    from stochastic_wind_simulate import get_simulator
    backend = args_dict['backend']
    N = args_dict['N']
    n_points = args_dict['n_points']
    use_batching = args_dict['use_batching']
    max_memory_gb = args_dict['max_memory_gb']
    positions = np.zeros((n_points, 3))
    positions[:, 0] = np.linspace(0, 1000, n_points)
    positions[:, 1] = 5
    positions[:, -1] = 35.0
    if backend == "jax":
        import jax.numpy as jnp
        positions = jnp.array(positions)
    elif backend == "torch":
        import torch
        positions = torch.from_numpy(positions)
    wind_speeds = positions[:, 0] * 0.0 + 25.0
    simulator = get_simulator(backend=backend, key=42, spectrum_type="kaimal-nd")
    simulator.seed = 42 + N
    if backend == "numpy":
        np.random.seed(simulator.seed)
    simulator.update_parameters(N=N, M=2*N, T=600, w_up=5.0)
    memory_estimate = 0.0
    if hasattr(simulator, 'estimate_memory_requirement'):
        memory_estimate = simulator.estimate_memory_requirement(n_points, N)
    actual_batching = False
    freq_batch_size = None
    if use_batching and hasattr(simulator, '_should_use_batching'):
        use_batch, _, auto_freq_batch = simulator._should_use_batching(
            n_points, N, max_memory_gb, None, None, auto_batch=True)
        actual_batching = use_batch
        if use_batch:
            freq_batch_size = auto_freq_batch
    try:
        start_time = time.time()
        if use_batching:
            u_samples, freqs = simulator.simulate_wind(
                positions, wind_speeds, component="u",
                max_memory_gb=max_memory_gb,
                auto_batch=True)
        else:
            u_samples, freqs = simulator.simulate_wind(
                positions, wind_speeds, component="u",
                max_memory_gb=16.0,
                auto_batch=True)
        elapsed_time = time.time() - start_time
        result = dict(time_cost=elapsed_time, memory_estimate=memory_estimate,
                      freq_batch_size=freq_batch_size, used_batching=actual_batching)
    except Exception as e:
        result = dict(time_cost=float('nan'), memory_estimate=memory_estimate,
                      freq_batch_size=freq_batch_size, used_batching=actual_batching)
    return result


def benchmark_backend(backend, frequencies, use_batching=True, max_memory_gb=2.0, n_points=100):
    """
    Benchmark a specific backend with different frequency segments.
    Each case runs in a new process for cold-start isolation.
    """
    logging.info(f"Benchmarking {backend} backend with freq batching={use_batching}, fixed points={n_points}")
    time_costs = []
    memory_estimates = []
    batch_info = []
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    batch_suffix = "batched" if use_batching else "direct"
    output_file = results_dir / f"freq_benchmark_{backend}_{batch_suffix}_{n_points}pts.txt"
    with open(output_file, "w") as f:
        f.write("n_frequencies,time_cost(s),memory_estimate(GB),used_batching,freq_batch_size\n")
    for i, N in enumerate(frequencies):
        logging.info(f"Testing {backend} with {N} frequency segments...")
        args_dict = dict(backend=backend, N=N, n_points=n_points,
                        use_batching=use_batching, max_memory_gb=max_memory_gb)
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            result = pool.apply(run_single_case, (args_dict,))
        time_costs.append(result['time_cost'])
        memory_estimates.append(result['memory_estimate'])
        batch_info.append({
            'freq_batch_size': result['freq_batch_size'],
            'used_batching': result['used_batching']
        })
        logging.info(f"  N={N}: {result['time_cost']:.4f}s, memory={result['memory_estimate']:.2f}GB, "
                    f"freq_batching={result['used_batching']}")
        with open(output_file, "a") as f:
            f.write(f"{N},{result['time_cost']:.4f},{result['memory_estimate']:.4f},"
                   f"{result['used_batching']},{result['freq_batch_size']}\n")
    return time_costs, memory_estimates, batch_info


def create_comparison_report(results, n_points):
    """Create a comparison report of all benchmark results."""
    results_dir = Path("benchmark_results")
    
    with open(results_dir / "freq_benchmark_comparison.txt", "w") as f:
        f.write("Frequency Scaling Benchmark Comparison Report\n")
        f.write(f"Fixed points: {n_points}\n")
        f.write("=" * 50 + "\n\n")
        
        for backend, data in results.items():
            f.write(f"{backend.upper()} Backend Results:\n")
            f.write("-" * 30 + "\n")
            
            for mode in ['batched', 'direct']:
                if mode in data:
                    times = data[mode]['times']
                    memories = data[mode]['memories']
                    
                    # Calculate statistics
                    valid_times = [t for t in times if not np.isnan(t)]
                    if valid_times:
                        avg_time = np.mean(valid_times)
                        total_time = np.sum(valid_times)
                        max_memory = max(memories) if memories else 0
                        
                        f.write(f"  {mode.capitalize()} mode:\n")
                        f.write(f"    Average time: {avg_time:.4f}s\n")
                        f.write(f"    Total time: {total_time:.4f}s\n")
                        f.write(f"    Max memory estimate: {max_memory:.2f}GB\n")
                        f.write(f"    Successful runs: {len(valid_times)}\n\n")
            
            f.write("\n")
        
        # Overall comparison
        f.write("Overall Performance Ranking:\n")
        f.write("-" * 30 + "\n")
        
        backend_avg_times = {}
        for backend, data in results.items():
            for mode in ['batched', 'direct']:
                if mode in data:
                    times = data[mode]['times']
                    valid_times = [t for t in times if not np.isnan(t)]
                    if valid_times:
                        avg_time = np.mean(valid_times)
                        backend_avg_times[f"{backend}_{mode}"] = avg_time
        
        # Sort by average time
        sorted_backends = sorted(backend_avg_times.items(), key=lambda x: x[1])
        
        for i, (backend_mode, avg_time) in enumerate(sorted_backends, 1):
            f.write(f"{i}. {backend_mode}: {avg_time:.4f}s average\n")



def main():
    """Main benchmark function."""
    arg_parser = argparse.ArgumentParser(description="Wind field simulator frequency scaling benchmark")
    arg_parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        choices=["jax", "torch", "numpy", "all"],
        default=["all"],
        help="Choose backend(s) to benchmark (default: all)",
    )
    arg_parser.add_argument(
        "--max-memory",
        type=float,
        default=2.0,
        help="Maximum memory limit in GB for batching (default: 2.0)",
    )
    arg_parser.add_argument(
        "--test-frequencies",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000],
        help="Test frequency segment numbers (N)",
    )
    arg_parser.add_argument(
        "--n-points",
        type=int,
        default=100,
        help="Fixed number of spatial points (default: 100)",
    )
    arg_parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=["batched", "direct", "both"],
        default=["batched"],
        help="Test modes: batched, direct, or both (default: batched)",
    )
    
    args = arg_parser.parse_args()
    
    # Determine which backends to test
    if "all" in args.backends:
        backends = ["jax", "torch", "numpy"]
    else:
        backends = args.backends
    
    # Determine which modes to test
    if "both" in args.modes:
        modes = ["batched", "direct"]
    else:
        modes = args.modes
    
    logging.info(f"Starting frequency scaling benchmark...")
    logging.info(f"Backends: {backends}")
    logging.info(f"Modes: {modes}")
    logging.info(f"Test frequencies: {args.test_frequencies}")
    logging.info(f"Fixed points: {args.n_points}")
    logging.info(f"Max memory: {args.max_memory}GB")
    
    results = {}
    
    for backend in backends:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing {backend.upper()} backend")
        logging.info(f"{'='*50}")
        
        results[backend] = {}
        
        for mode in modes:
            use_batching = (mode == "batched")
            
            try:
                times, memories, batch_info = benchmark_backend(
                    backend, args.test_frequencies, use_batching, 
                    args.max_memory, args.n_points
                )
                
                results[backend][mode] = {
                    'times': times,
                    'memories': memories,
                    'batch_info': batch_info
                }
                
                # Print summary for this backend/mode
                valid_times = [t for t in times if not np.isnan(t)]
                if valid_times:
                    avg_time = np.mean(valid_times)
                    total_time = np.sum(valid_times)
                    logging.info(f"{backend} {mode}: avg={avg_time:.4f}s, total={total_time:.4f}s")
                else:
                    logging.warning(f"{backend} {mode}: No successful runs")
                    
            except Exception as e:
                logging.error(f"Failed to benchmark {backend} in {mode} mode: {e}")
                results[backend][mode] = {
                    'times': [np.nan] * len(args.test_frequencies),
                    'memories': [0.0] * len(args.test_frequencies),
                    'batch_info': [{}] * len(args.test_frequencies)
                }
    
    # Create comparison report
    create_comparison_report(results, args.n_points)
    
    logging.info(f"\nBenchmark completed! Results saved in benchmark_results/")
    logging.info(f"Check freq_benchmark_comparison.txt for detailed comparison.")


if __name__ == "__main__":
    main()
