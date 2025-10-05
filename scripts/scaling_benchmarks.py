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


def benchmark_sampling_scaling(backend, sample_sizes, fixed_N=1000):
    """
    Benchmark scaling with sample size while keeping frequency segments fixed.
    
    Args:
        backend: Backend name ('jax', 'torch', 'numpy')
        sample_sizes: List of sample sizes to test
        fixed_N: Fixed number of frequency segments
        
    Returns:
        List of (sample_size, time_cost, memory_estimate, successful)
    """
    logging.info(f"Benchmarking {backend} backend: Sample scaling (N={fixed_N})")
    
    simulator = get_simulator(backend=backend, key=42, spectrum_type="kaimal-nd")
    
    # Set fixed frequency parameters
    simulator.update_parameters(
        N=fixed_N,
        M=fixed_N * 2,
        T=600,
        w_up=1.0
    )
    
    results = []
    
    # Create results directory
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create output file
    output_file = results_dir / f"scaling_sample_{backend}_N{fixed_N}.txt"
    
    with open(output_file, "w") as f:
        f.write("n_samples,time_cost(s),memory_estimate(GB),successful\n")
    
    Z = 30.0  # Height (m)
    
    for n in sample_sizes:
        logging.info(f"Testing {backend} with {n} samples (N={fixed_N})...")
        
        # Create positions
        positions = np.zeros((n, 3))
        positions[:, 0] = np.linspace(0, 1000, n)
        positions[:, 1] = 5
        positions[:, -1] = Z + 5
        
        # Convert to appropriate format for backend
        try:
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
        except ImportError as e:
            raise ImportError(f"Failed to import {backend} backend: {e}. "
                            f"Make sure {backend} is installed in your environment.")
        
        wind_speeds = positions[:, 0] * 0.0 + 25.0  # Constant wind speed
        
        # Estimate memory requirement
        memory_estimate = 0.0
        if hasattr(simulator, 'estimate_memory_requirement'):
            memory_estimate = simulator.estimate_memory_requirement(n, fixed_N)
        
        successful = True
        
        try:
            # Clear GPU cache before each test for PyTorch
            if backend == "torch":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            start_time = time.time()
            
            # Use direct simulation without batching
            u_samples, frequencies = simulator.simulate_wind(
                positions, wind_speeds, component="u",
                max_memory_gb=32.0,  # High limit to avoid auto-batching
                auto_batch=False  # Disable auto-batching
            )
            
            # Ensure PyTorch operations are completed
            if backend == "torch":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            
            logging.info(f"  n={n}: {elapsed_time:.4f}s, memory={memory_estimate:.2f}GB")
            
            # Clear references and cache after each test
            del u_samples, frequencies
            if backend == "torch":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Error with {backend} backend, n={n}: {e}")
            elapsed_time = np.nan
            successful = False
        
        results.append((n, elapsed_time, memory_estimate, successful))
        
        # Write to file
        with open(output_file, "a") as f:
            f.write(f"{n},{elapsed_time:.4f},{memory_estimate:.4f},{successful}\n")
    
    return results


def benchmark_frequency_scaling(backend, frequency_sizes, fixed_n=100):
    """
    Benchmark scaling with frequency segments while keeping sample size fixed.
    
    Args:
        backend: Backend name ('jax', 'torch', 'numpy')
        frequency_sizes: List of frequency segment numbers to test
        fixed_n: Fixed number of sampling points
        
    Returns:
        List of (N, time_cost, memory_estimate, successful)
    """
    logging.info(f"Benchmarking {backend} backend: Frequency scaling (n={fixed_n})")
    
    simulator = get_simulator(backend=backend, key=42, spectrum_type="kaimal-nd")
    
    results = []
    
    # Create results directory
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    # Create output file
    output_file = results_dir / f"scaling_frequency_{backend}_n{fixed_n}.txt"
    
    with open(output_file, "w") as f:
        f.write("N_frequencies,time_cost(s),memory_estimate(GB),successful\n")
    
    Z = 30.0  # Height (m)
    
    # Create fixed positions
    positions = np.zeros((fixed_n, 3))
    positions[:, 0] = np.linspace(0, 1000, fixed_n)
    positions[:, 1] = 5
    positions[:, -1] = Z + 5
    
    # Convert to appropriate format for backend
    try:
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
    except ImportError as e:
        raise ImportError(f"Failed to import {backend} backend: {e}. "
                        f"Make sure {backend} is installed in your environment.")
    
    wind_speeds = positions[:, 0] * 0.0 + 25.0  # Constant wind speed
    
    for N in frequency_sizes:
        logging.info(f"Testing {backend} with N={N} frequencies (n={fixed_n})...")
        
        # Update frequency parameters
        simulator.update_parameters(
            N=N,
            M=N * 2,
            T=600,
            w_up=1.0
        )
        
        # Estimate memory requirement
        memory_estimate = 0.0
        if hasattr(simulator, 'estimate_memory_requirement'):
            memory_estimate = simulator.estimate_memory_requirement(fixed_n, N)
        
        successful = True
        
        try:
            # Clear GPU cache before each test for PyTorch
            if backend == "torch":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            start_time = time.time()
            
            # Use direct simulation without batching
            u_samples, frequencies = simulator.simulate_wind(
                positions, wind_speeds, component="u",
                max_memory_gb=32.0,  # High limit to avoid auto-batching
                auto_batch=False  # Disable auto-batching
            )
            
            # Ensure PyTorch operations are completed
            if backend == "torch":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            
            logging.info(f"  N={N}: {elapsed_time:.4f}s, memory={memory_estimate:.2f}GB")
            
            # Clear references and cache after each test
            del u_samples, frequencies
            if backend == "torch":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Error with {backend} backend, N={N}: {e}")
            elapsed_time = np.nan
            successful = False
        
        results.append((N, elapsed_time, memory_estimate, successful))
        
        # Write to file
        with open(output_file, "a") as f:
            f.write(f"{N},{elapsed_time:.4f},{memory_estimate:.4f},{successful}\n")
    
    return results


def create_single_backend_report(backend, sample_results, frequency_results):
    """Create a report for a single backend's scaling results."""
    results_dir = Path("benchmark_results")
    
    with open(results_dir / f"scaling_{backend}_report.txt", "w") as f:
        f.write(f"{backend.upper()} Backend Scaling Performance Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Sample scaling results
        if sample_results:
            f.write("SAMPLE SCALING RESULTS\n")
            f.write("-" * 30 + "\n")
            
            results = sample_results[backend]
            successful_results = [(n, t, m) for n, t, m, s in results if s and not np.isnan(t)]
            
            if successful_results:
                f.write(f"Successful runs: {len(successful_results)}/{len(results)}\n")
                
                # Detailed results
                f.write(f"\nDetailed Results:\n")
                f.write(f"{'n_samples':<10} {'Time(s)':<10} {'Memory(GB)':<12} {'Status':<10}\n")
                f.write("-" * 50 + "\n")
                
                for n, t, m, s in results:
                    status = "SUCCESS" if s and not np.isnan(t) else "FAILED"
                    time_str = f"{t:.4f}" if not np.isnan(t) else "N/A"
                    f.write(f"{n:<10} {time_str:<10} {m:<12.2f} {status:<10}\n")
                
                # Summary statistics
                total_time = sum(t for _, t, _ in successful_results)
                avg_time = np.mean([t for _, t, _ in successful_results])
                max_memory = max(m for _, _, m in successful_results)
                largest_n = max(n for n, _, _ in successful_results)
                
                f.write(f"\nSummary Statistics:\n")
                f.write(f"  Total computation time: {total_time:.4f}s\n")
                f.write(f"  Average time per run: {avg_time:.4f}s\n")
                f.write(f"  Maximum memory usage: {max_memory:.2f}GB\n")
                f.write(f"  Largest successful n: {largest_n}\n")
                
                # Scaling analysis
                if len(successful_results) >= 3:
                    # Simple linear fit to estimate scaling
                    n_vals = np.array([n for n, _, _ in successful_results])
                    t_vals = np.array([t for _, t, _ in successful_results])
                    
                    # Log-log fit for power law estimation
                    log_n = np.log(n_vals)
                    log_t = np.log(t_vals)
                    coeffs = np.polyfit(log_n, log_t, 1)
                    scaling_exponent = coeffs[0]
                    
                    f.write(f"  Estimated scaling exponent: {scaling_exponent:.2f}\n")
                    f.write(f"  (Time ~ n^{scaling_exponent:.2f})\n")
            else:
                f.write(f"No successful runs\n")
        
        # Frequency scaling results
        if frequency_results:
            f.write(f"\n\nFREQUENCY SCALING RESULTS\n")
            f.write("-" * 30 + "\n")
            
            results = frequency_results[backend]
            successful_results = [(N, t, m) for N, t, m, s in results if s and not np.isnan(t)]
            
            if successful_results:
                f.write(f"Successful runs: {len(successful_results)}/{len(results)}\n")
                
                # Detailed results
                f.write(f"\nDetailed Results:\n")
                f.write(f"{'N_freq':<10} {'Time(s)':<10} {'Memory(GB)':<12} {'Status':<10}\n")
                f.write("-" * 50 + "\n")
                
                for N, t, m, s in results:
                    status = "SUCCESS" if s and not np.isnan(t) else "FAILED"
                    time_str = f"{t:.4f}" if not np.isnan(t) else "N/A"
                    f.write(f"{N:<10} {time_str:<10} {m:<12.2f} {status:<10}\n")
                
                # Summary statistics
                total_time = sum(t for _, t, _ in successful_results)
                avg_time = np.mean([t for _, t, _ in successful_results])
                max_memory = max(m for _, _, m in successful_results)
                largest_N = max(N for N, _, _ in successful_results)
                
                f.write(f"\nSummary Statistics:\n")
                f.write(f"  Total computation time: {total_time:.4f}s\n")
                f.write(f"  Average time per run: {avg_time:.4f}s\n")
                f.write(f"  Maximum memory usage: {max_memory:.2f}GB\n")
                f.write(f"  Largest successful N: {largest_N}\n")
                
                # Scaling analysis
                if len(successful_results) >= 3:
                    # Simple linear fit to estimate scaling
                    N_vals = np.array([N for N, _, _ in successful_results])
                    t_vals = np.array([t for _, t, _ in successful_results])
                    
                    # Log-log fit for power law estimation
                    log_N = np.log(N_vals)
                    log_t = np.log(t_vals)
                    coeffs = np.polyfit(log_N, log_t, 1)
                    scaling_exponent = coeffs[0]
                    
                    f.write(f"  Estimated scaling exponent: {scaling_exponent:.2f}\n")
                    f.write(f"  (Time ~ N^{scaling_exponent:.2f})\n")
            else:
                f.write(f"No successful runs\n")


def create_scaling_comparison_report(sample_results, frequency_results):
    """Create a comparison report of scaling benchmark results."""
    results_dir = Path("benchmark_results")
    
    with open(results_dir / "scaling_benchmark_comparison.txt", "w") as f:
        f.write("Scaling Performance Benchmark Comparison Report\n")
        f.write("=" * 60 + "\n\n")
        
        # Sample scaling results
        f.write("SAMPLE SCALING RESULTS (Fixed N=1000)\n")
        f.write("-" * 40 + "\n")
        for backend, results in sample_results.items():
            f.write(f"\n{backend.upper()} Backend:\n")
            
            successful_results = [(n, t, m) for n, t, m, s in results if s and not np.isnan(t)]
            if successful_results:
                total_time = sum(t for _, t, _ in successful_results)
                avg_time = np.mean([t for _, t, _ in successful_results])
                max_memory = max(m for _, _, m in successful_results)
                
                f.write(f"  Successful runs: {len(successful_results)}/{len(results)}\n")
                f.write(f"  Total time: {total_time:.4f}s\n")
                f.write(f"  Average time: {avg_time:.4f}s\n")
                f.write(f"  Max memory: {max_memory:.2f}GB\n")
                
                # Performance per sample
                largest_successful = max(successful_results, key=lambda x: x[0])
                f.write(f"  Largest successful: n={largest_successful[0]} ({largest_successful[1]:.4f}s)\n")
            else:
                f.write(f"  No successful runs\n")
        
        # Frequency scaling results
        f.write(f"\n\nFREQUENCY SCALING RESULTS (Fixed n=100)\n")
        f.write("-" * 40 + "\n")
        for backend, results in frequency_results.items():
            f.write(f"\n{backend.upper()} Backend:\n")
            
            successful_results = [(N, t, m) for N, t, m, s in results if s and not np.isnan(t)]
            if successful_results:
                total_time = sum(t for _, t, _ in successful_results)
                avg_time = np.mean([t for _, t, _ in successful_results])
                max_memory = max(m for _, _, m in successful_results)
                
                f.write(f"  Successful runs: {len(successful_results)}/{len(results)}\n")
                f.write(f"  Total time: {total_time:.4f}s\n")
                f.write(f"  Average time: {avg_time:.4f}s\n")
                f.write(f"  Max memory: {max_memory:.2f}GB\n")
                
                # Performance per frequency
                largest_successful = max(successful_results, key=lambda x: x[0])
                f.write(f"  Largest successful: N={largest_successful[0]} ({largest_successful[1]:.4f}s)\n")
            else:
                f.write(f"  No successful runs\n")
        
        # Overall performance ranking
        f.write(f"\n\nOVERALL PERFORMANCE RANKING\n")
        f.write("-" * 30 + "\n")
        
        all_avg_times = {}
        
        # Sample scaling ranking
        f.write("\nSample Scaling (avg time per backend):\n")
        sample_ranking = []
        for backend, results in sample_results.items():
            successful_times = [t for _, t, _, s in results if s and not np.isnan(t)]
            if successful_times:
                avg_time = np.mean(successful_times)
                sample_ranking.append((backend, avg_time))
                all_avg_times[f"{backend}_sample"] = avg_time
        
        sample_ranking.sort(key=lambda x: x[1])
        for i, (backend, avg_time) in enumerate(sample_ranking, 1):
            f.write(f"  {i}. {backend}: {avg_time:.4f}s average\n")
        
        # Frequency scaling ranking
        f.write("\nFrequency Scaling (avg time per backend):\n")
        freq_ranking = []
        for backend, results in frequency_results.items():
            successful_times = [t for _, t, _, s in results if s and not np.isnan(t)]
            if successful_times:
                avg_time = np.mean(successful_times)
                freq_ranking.append((backend, avg_time))
                all_avg_times[f"{backend}_frequency"] = avg_time
        
        freq_ranking.sort(key=lambda x: x[1])
        for i, (backend, avg_time) in enumerate(freq_ranking, 1):
            f.write(f"  {i}. {backend}: {avg_time:.4f}s average\n")


def main():
    """Main benchmark function."""
    arg_parser = argparse.ArgumentParser(description="Wind field simulator scaling benchmark")
    arg_parser.add_argument(
        "--backend",
        type=str,
        choices=["jax", "torch", "numpy"],
        required=True,
        help="Choose backend to benchmark (required: jax, torch, or numpy)",
    )
    arg_parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[10, 25, 50, 75, 100, 150, 200, 300, 400, 500],
        help="Sample sizes to test (default: 10 to 500)",
    )
    arg_parser.add_argument(
        "--frequency-sizes",
        type=int,
        nargs="+",
        default=[100, 200, 500, 800, 1000, 1500, 2000, 2500, 3000],
        help="Frequency segment numbers to test (default: 100 to 3000)",
    )
    arg_parser.add_argument(
        "--fixed-N",
        type=int,
        default=1000,
        help="Fixed frequency segments for sample scaling test (default: 1000)",
    )
    arg_parser.add_argument(
        "--fixed-n",
        type=int,
        default=100,
        help="Fixed sample points for frequency scaling test (default: 100)",
    )
    arg_parser.add_argument(
        "--test-type",
        type=str,
        choices=["sample", "frequency", "both"],
        default="both",
        help="Type of scaling test to run (default: both)",
    )
    
    args = arg_parser.parse_args()
    
    # Use single backend
    backend = args.backend
    
    logging.info(f"Starting scaling benchmark...")
    logging.info(f"Backend: {backend}")
    logging.info(f"Test type: {args.test_type}")
    
    if args.test_type in ["sample", "both"]:
        logging.info(f"Sample sizes: {args.sample_sizes} (Fixed N={args.fixed_N})")
    
    if args.test_type in ["frequency", "both"]:
        logging.info(f"Frequency sizes: {args.frequency_sizes} (Fixed n={args.fixed_n})")
    
    sample_results = {}
    frequency_results = {}
    
    # Sample scaling test
    if args.test_type in ["sample", "both"]:
        logging.info(f"\n{'='*60}")
        logging.info(f"SAMPLE SCALING TEST (Fixed N={args.fixed_N})")
        logging.info(f"{'='*60}")
        
        logging.info(f"\nTesting {backend.upper()} backend for sample scaling...")
        
        try:
            results = benchmark_sampling_scaling(backend, args.sample_sizes, args.fixed_N)
            sample_results[backend] = results
            
            # Print summary
            successful_results = [r for r in results if r[3] and not np.isnan(r[1])]
            if successful_results:
                avg_time = np.mean([r[1] for r in successful_results])
                max_n = max([r[0] for r in successful_results])
                logging.info(f"{backend} sample scaling: {len(successful_results)}/{len(results)} successful")
                logging.info(f"Average time: {avg_time:.4f}s, Max n: {max_n}")
            else:
                logging.warning(f"{backend} sample scaling: No successful runs")
                
        except Exception as e:
            logging.error(f"Failed to benchmark {backend} sample scaling: {e}")
            sample_results[backend] = [(n, np.nan, 0.0, False) for n in args.sample_sizes]
    
    # Frequency scaling test
    if args.test_type in ["frequency", "both"]:
        logging.info(f"\n{'='*60}")
        logging.info(f"FREQUENCY SCALING TEST (Fixed n={args.fixed_n})")
        logging.info(f"{'='*60}")
        
        logging.info(f"\nTesting {backend.upper()} backend for frequency scaling...")
        
        try:
            results = benchmark_frequency_scaling(backend, args.frequency_sizes, args.fixed_n)
            frequency_results[backend] = results
            
            # Print summary
            successful_results = [r for r in results if r[3] and not np.isnan(r[1])]
            if successful_results:
                avg_time = np.mean([r[1] for r in successful_results])
                max_N = max([r[0] for r in successful_results])
                logging.info(f"{backend} frequency scaling: {len(successful_results)}/{len(results)} successful")
                logging.info(f"Average time: {avg_time:.4f}s, Max N: {max_N}")
            else:
                logging.warning(f"{backend} frequency scaling: No successful runs")
                
        except Exception as e:
            logging.error(f"Failed to benchmark {backend} frequency scaling: {e}")
            frequency_results[backend] = [(N, np.nan, 0.0, False) for N in args.frequency_sizes]
    
    # Create individual backend report (not comparison)
    if sample_results or frequency_results:
        create_single_backend_report(backend, sample_results, frequency_results)
        logging.info(f"\nBenchmark completed! Results saved in benchmark_results/")
        logging.info(f"Check scaling_{backend}_report.txt for detailed results.")
    else:
        logging.warning("No benchmark results to report.")


if __name__ == "__main__":
    main()
