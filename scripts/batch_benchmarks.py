import time
import logging
import argparse
import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stochastic_wind_simulate import get_simulator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def benchmark_backend(backend, ns, use_batching=True, max_memory_gb=2.0):
    """
    Benchmark a specific backend with optional batching.
    
    Args:
        backend: Backend name ('jax', 'torch', 'numpy')
        ns: List of sample sizes to test
        use_batching: Whether to use batching
        max_memory_gb: Memory limit for batching
        
    Returns:
        List of time costs for each sample size
    """
    logging.info(f"Benchmarking {backend} backend with batching={use_batching}")
    
    simulator = get_simulator(backend=backend, key=42, spectrum_type="kaimal-nd")
    
    # Use smaller parameters for faster benchmarking
    simulator.update_parameters(
        N=3000,     # Reduced frequency points
        M=6000,     # Reduced time points  
        T=600,      # Reduced duration
        w_up=1.0    # Reduced cutoff frequency
    )
    
    time_costs = []
    memory_estimates = []
    batch_info = []
    
    # Create results directory
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create output file
    batch_suffix = "batched" if use_batching else "direct"
    output_file = results_dir / f"batch_benchmark_{backend}_{batch_suffix}.txt"
    
    with open(output_file, "w") as f:
        f.write("n_samples,time_cost(s),memory_estimate(GB),used_batching,point_batch_size,freq_batch_size\n")
    
    Z = 30.0  # Height (m)
    
    for i, n in enumerate(ns):
        logging.info(f"Testing {backend} with {n} points...")
        
        # Create positions
        positions = np.zeros((n, 3))
        positions[:, 0] = np.linspace(0, 1000, n)
        positions[:, 1] = 5
        positions[:, -1] = Z + 5
        
        # Convert to appropriate format for backend
        if backend == "jax":
            import jax.numpy as jnp
            positions = jnp.array(positions)
        elif backend == "torch":
            import torch
            positions = torch.from_numpy(positions)
        elif backend == "numpy":
            # positions is already a numpy array
            pass
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        wind_speeds = positions[:, 0] * 0.0 + 25.0  # Constant wind speed
        
        # Estimate memory requirement
        memory_estimate = 0.0
        if hasattr(simulator, 'estimate_memory_requirement'):
            memory_estimate = simulator.estimate_memory_requirement(n, simulator.params["N"])
        
        # Let the simulator determine if batching is needed and calculate optimal batch sizes
        actual_batching = False
        point_batch_size = None
        freq_batch_size = None
        
        if use_batching and hasattr(simulator, '_should_use_batching'):
            # Use the simulator's built-in logic to determine if batching is needed
            use_batch, auto_point_batch, auto_freq_batch = simulator._should_use_batching(
                n, simulator.params["N"], max_memory_gb, None, None, auto_batch=True
            )
            actual_batching = use_batch
            if use_batch:
                point_batch_size = auto_point_batch
                freq_batch_size = auto_freq_batch
        
        try:
            start_time = time.time()
            
            if use_batching:
                # Use auto-batching with specified memory limit
                u_samples, frequencies = simulator.simulate_wind(
                    positions, wind_speeds, component="u",
                    max_memory_gb=max_memory_gb,
                    auto_batch=True  # Let simulator decide if batching is needed
                )
            else:
                # Use direct simulation with high memory limit to avoid auto-batching
                u_samples, frequencies = simulator.simulate_wind(
                    positions, wind_speeds, component="u",
                    max_memory_gb=16.0,  # High limit to avoid auto-batching
                    auto_batch=True
                )
            
            elapsed_time = time.time() - start_time
            time_costs.append(elapsed_time)
            memory_estimates.append(memory_estimate)
            
            # Determine if batching was actually used by checking memory vs limit
            if use_batching:
                actual_batching = memory_estimate > max_memory_gb
                if actual_batching and hasattr(simulator, 'get_optimal_batch_sizes'):
                    # Get the optimal batch sizes that would have been used
                    point_batch_size, freq_batch_size = simulator.get_optimal_batch_sizes(
                        n, simulator.params["N"], max_memory_gb
                    )
                else:
                    point_batch_size, freq_batch_size = None, None
            else:
                actual_batching = False
                point_batch_size, freq_batch_size = None, None
            
            # Store batch information
            batch_info.append({
                'point_batch_size': point_batch_size,
                'freq_batch_size': freq_batch_size,
                'used_batching': actual_batching
            })
            
            logging.info(f"  n={n}: {elapsed_time:.4f}s, memory={memory_estimate:.2f}GB, "
                        f"batching={actual_batching}")
            
            # Write to file
            with open(output_file, "a") as f:
                f.write(f"{n},{elapsed_time:.4f},{memory_estimate:.4f},"
                       f"{actual_batching},{point_batch_size},{freq_batch_size}\n")
                
        except Exception as e:
            logging.error(f"Error with {backend} backend, n={n}: {e}")
            time_costs.append(np.nan)
            memory_estimates.append(memory_estimate)
            batch_info.append({
                'point_batch_size': point_batch_size,
                'freq_batch_size': freq_batch_size,
                'used_batching': actual_batching
            })
            
            # Write error to file
            with open(output_file, "a") as f:
                f.write(f"{n},NaN,{memory_estimate:.4f},"
                       f"{actual_batching},{point_batch_size},{freq_batch_size}\n")
    
    return time_costs, memory_estimates, batch_info


def create_comparison_report(results):
    """Create a comparison report of all benchmark results."""
    results_dir = Path("benchmark_results")
    
    with open(results_dir / "batch_benchmark_comparison.txt", "w") as f:
        f.write("Batch Processing Benchmark Comparison Report\n")
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
    arg_parser = argparse.ArgumentParser(description="Wind field simulator batch processing benchmark")
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
        "--test-sizes",
        type=int,
        nargs="+",
        default=[2, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000],
        help="Test sample sizes",
    )
    arg_parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=["batched", "direct", "both"],
        default=["batched"],
        help="Test modes: batched, direct, or both (default: both)",
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
    
    logging.info(f"Starting batch processing benchmark...")
    logging.info(f"Backends: {backends}")
    logging.info(f"Modes: {modes}")
    logging.info(f"Test sizes: {args.test_sizes}")
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
                    backend, args.test_sizes, use_batching, args.max_memory
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
                    'times': [np.nan] * len(args.test_sizes),
                    'memories': [0.0] * len(args.test_sizes),
                    'batch_info': [{}] * len(args.test_sizes)
                }
    
    # Create comparison report
    create_comparison_report(results)
    
    logging.info(f"\nBenchmark completed! Results saved in benchmark_results/")
    logging.info(f"Check batch_benchmark_comparison.txt for detailed comparison.")


if __name__ == "__main__":
    main()
