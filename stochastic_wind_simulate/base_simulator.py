"""
Base simulator class with common functionality for memory management and batching.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np


class BaseWindSimulator(ABC):
    """Base class for wind field simulators with common memory management and batching functionality."""
    
    def __init__(self):
        """Initialize base simulator."""
        self.params = self._set_default_parameters()
    
    @abstractmethod
    def _set_default_parameters(self) -> Dict:
        """Set default wind field simulation parameters. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def simulate_wind(self, positions, wind_speeds, component="u", **kwargs):
        """Simulate wind field. Must be implemented by subclasses."""
        pass
    
    def estimate_memory_requirement(self, n_points, n_frequencies):
        """
        Estimate memory requirement for wind field simulation in GB.
        
        This is a general estimation that works for all backends since they all 
        use similar matrix operations (spectrum matrices, Cholesky decomposition, etc.).
        
        Args:
            n_points: Number of simulation points
            n_frequencies: Number of frequency points
            
        Returns:
            Estimated memory requirement in GB
        """
        # Main matrices: S_matrices (n_freq, n, n), H_matrices (n_freq, n, n), B (n, M)
        # Complex matrices use 2x memory, assuming 32-bit floats (4 bytes)
        dtype_size = 4
        
        S_memory = n_frequencies * n_points * n_points * dtype_size  # Real
        H_memory = n_frequencies * n_points * n_points * dtype_size * 2  # Complex
        B_memory = n_points * (n_frequencies * 2) * dtype_size * 2  # Complex, M = 2*N
        
        # Additional working memory (factor of 2 for safety)
        total_bytes = (S_memory + H_memory + B_memory) * 2
        
        return total_bytes / (1024**3)  # Convert to GB

    def get_optimal_batch_sizes(self, n_points, n_frequencies, max_memory_gb=4.0):
        """
        Calculate optimal batch sizes for points and frequencies to fit within memory limit.
        
        Args:
            n_points: Total number of simulation points
            n_frequencies: Total number of frequency points  
            max_memory_gb: Maximum memory limit in GB
            
        Returns:
            Tuple of (point_batch_size, frequency_batch_size)
        """
        # Start with full sizes and reduce if necessary
        point_batch = n_points
        freq_batch = n_frequencies
        
        # Binary search for optimal batch sizes
        while self.estimate_memory_requirement(point_batch, freq_batch) > max_memory_gb:
            if point_batch > freq_batch:
                point_batch = max(1, point_batch // 2)
            else:
                freq_batch = max(1, freq_batch // 2)
                
            # Prevent infinite loop
            if point_batch == 1 and freq_batch == 1:
                break
        
        return point_batch, freq_batch

    def _should_use_batching(self, n_points, n_frequencies, max_memory_gb, 
                           point_batch_size, freq_batch_size, auto_batch):
        """
        Determine whether batching should be used based on memory requirements and user settings.
        
        Args:
            n_points: Number of simulation points
            n_frequencies: Number of frequency points
            max_memory_gb: Maximum memory limit in GB
            point_batch_size: Manual point batch size (None for auto)
            freq_batch_size: Manual frequency batch size (None for auto)
            auto_batch: If True, automatically determine if batching is needed
            
        Returns:
            Tuple of (use_batching, point_batch_size, freq_batch_size)
        """
        # Estimate memory requirement
        estimated_memory = self.estimate_memory_requirement(n_points, n_frequencies)
        
        # Decide whether to use batching
        use_batching = False
        if auto_batch:
            if estimated_memory > max_memory_gb:
                use_batching = True
        else:
            # Force batching if batch sizes are specified
            if point_batch_size is not None or freq_batch_size is not None:
                use_batching = True
        
        if use_batching:
            # Use provided batch sizes or calculate optimal ones
            if point_batch_size is None or freq_batch_size is None:
                optimal_point_batch, optimal_freq_batch = self.get_optimal_batch_sizes(
                    n_points, n_frequencies, max_memory_gb
                )
                if point_batch_size is None:
                    point_batch_size = optimal_point_batch
                if freq_batch_size is None:
                    freq_batch_size = optimal_freq_batch
        
        return use_batching, point_batch_size, freq_batch_size

    def _get_batch_info(self, n_total, batch_size):
        """
        Calculate batch information for processing.
        
        Args:
            n_total: Total number of items to process
            batch_size: Size of each batch
            
        Returns:
            Number of batches needed
        """
        return (n_total + batch_size - 1) // batch_size

    def _get_batch_range(self, batch_idx, batch_size, n_total):
        """
        Get the start and end indices for a specific batch.
        
        Args:
            batch_idx: Index of the current batch (0-based)
            batch_size: Size of each batch
            n_total: Total number of items
            
        Returns:
            Tuple of (start_index, end_index)
        """
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_total)
        return start_idx, end_idx

    def print_batch_info(self, estimated_memory, max_memory_gb, use_batching, 
                        point_batch_size=None, freq_batch_size=None, 
                        n_point_batches=None, n_freq_batches=None):
        """
        Print informative messages about batching decisions and progress.
        
        Args:
            estimated_memory: Estimated memory requirement in GB
            max_memory_gb: Maximum memory limit in GB
            use_batching: Whether batching is being used
            point_batch_size: Size of point batches (if using batching)
            freq_batch_size: Size of frequency batches (if using batching)
            n_point_batches: Number of point batches (if using batching)
            n_freq_batches: Number of frequency batches (if using batching)
        """
        print(f"Estimated memory requirement: {estimated_memory:.2f} GB")
        
        if use_batching:
            print(f"Using batched simulation with point_batch_size={point_batch_size}, freq_batch_size={freq_batch_size}")
            if n_point_batches and n_freq_batches:
                print(f"Running batched simulation: {n_point_batches} point batches × {n_freq_batches} frequency batches")
        else:
            print("Memory requirement within limit, using direct simulation")

    def print_batch_progress(self, batch_idx, n_batches, batch_type="point", 
                           start_idx=None, end_idx=None):
        """
        Print progress information for batch processing.
        
        Args:
            batch_idx: Current batch index (0-based)
            n_batches: Total number of batches
            batch_type: Type of batch ("point" or "frequency")
            start_idx: Start index of current batch (optional)
            end_idx: End index of current batch (optional)
        """
        if start_idx is not None and end_idx is not None:
            print(f"Processing {batch_type} batch {batch_idx + 1}/{n_batches} "
                  f"({batch_type}s {start_idx}-{end_idx-1})")
        else:
            print(f"Processing {batch_type} batch {batch_idx + 1}/{n_batches}")

    def update_parameters(self, **kwargs):
        """Update simulation parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        # Update dependent parameters
        self.params["dw"] = self.params["w_up"] / self.params["N"]
        self.params["z_d"] = (
            self.params["H_bar"] - self.params["z_0"] / self.params["K"]
        )
        
        # Update spectrum parameters if spectrum exists
        if hasattr(self, 'spectrum') and hasattr(self.spectrum, 'params'):
            self.spectrum.params = self.params
