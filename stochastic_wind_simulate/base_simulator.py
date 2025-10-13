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
    
    def _set_default_parameters(self) -> Dict:
        """
        Set default wind field simulation parameters.
        
        Returns:
            Dict: Dictionary containing default simulation parameters including
                 physical constants, grid specifications, and numerical settings
        """
        params = {
            "K": 0.4,  # Dimensionless constant
            "H_bar": 10.0,  # Average height of surrounding buildings (m)
            "z_0": 0.05,  # Surface roughness height
            "alpha_0": 0.16,  # Surface roughness exponent
            "C_x": 16.0,  # Decay coefficient in x direction
            "C_y": 6.0,  # Decay coefficient in y direction
            "C_z": 10.0,  # Decay coefficient in z direction
            "w_up": 5.0,  # Cutoff frequency (Hz)
            "N": 3000,  # Number of frequency segments
            "z_max": 450.0,  # Maximum height for mean wind speed calculation (m)
            "U_d": 25.0,  # Design basic wind speed (m/s)
        }
        params["M"] = 2 * params["N"]  # Number of time steps
        params["T"] = params["N"] / params["w_up"]  # Total simulation time
        params["dt"] = params["T"] / params["M"]  # Time step
        params["dw"] = params["w_up"] / params["N"]  # Frequency increment
        params["z_d"] = params["H_bar"] - params["z_0"] / params["K"]  # Calculate zero plane displacement
        assert params["dt"] <= 1 / (2 * params["w_up"]), "Time step dt must satisfy the Nyquist criterion."

        return params

    def update_parameters(self, **kwargs):
        """Update simulation parameters."""
        params = self.params.copy()
        for key, value in kwargs.items():
            if key in self.params:
                params[key] = value

        # Update dependent parameters
        params["M"] = 2 * params["N"]  # Number of time steps
        params["T"] = params["N"] / params["w_up"]  # Total simulation time
        params["dt"] = params["T"] / params["M"]  # Time step
        params["dw"] = params["w_up"] / params["N"]  # Frequency increment
        params["z_d"] = params["H_bar"] - params["z_0"] / params["K"]  # Calculate zero plane displacement
        assert params["dt"] <= 1 / (2 * params["w_up"]), "Time step dt must satisfy the Nyquist criterion."
        self.params = params
        self.spectrum.params = self.params  # Update spectrum parameters
    
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
        # Complex matrices use 2x memory, assuming 64-bit floats (8 bytes)
        dtype_size = 4

        S_memory = n_frequencies * n_points * n_points * dtype_size  # Real
        H_memory = n_frequencies * n_points * n_points * dtype_size * 2  # Complex
        B_memory = n_points * (n_frequencies * 2) * dtype_size * 2  # Complex, M = 2*N

        # Additional working memory (factor of 2 for safety)
        total_bytes = (S_memory + H_memory + B_memory) * 2.0
        
        return total_bytes / (1024**3)  # Convert to GB

    def get_optimal_batch_sizes(self, n_points, n_frequencies, max_memory_gb=4.0):
        """
        Calculate optimal frequency batch size to fit within memory limit.
        Note: Spatial batching is not used to preserve complete spatial correlation structure.
        
        Args:
            n_points: Total number of simulation points
            n_frequencies: Total number of frequency points  
            max_memory_gb: Maximum memory limit in GB
            
        Returns:
            Tuple of (point_batch_size, frequency_batch_size) for compatibility
        """
        # Only use frequency batching to preserve spatial correlation integrity
        point_batch = n_points  # Always use all spatial points
        freq_batch = n_frequencies
        
        # Binary search for optimal frequency batch size
        # while self.estimate_memory_requirement(point_batch, freq_batch) > max_memory_gb:
        #     freq_batch = max(1, freq_batch // 2)
                
        #     # Prevent infinite loop
        #     if freq_batch == 1:
        #         break

        backend = self.params.get("backend", "numpy")
        safety_factor = 2.0
        if backend == "torch":
            safety_factor = 5.0
        elif backend == "jax":
            safety_factor = 2.0
        elif backend == "numpy":
            safety_factor = 2.0
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        freq_batch = min(
            freq_batch,
            int(
                max_memory_gb * (1024**3) / ((3*n_points**2 + 4 * n_points) * 4 * safety_factor)
            )
        )
        
        return point_batch, freq_batch

    def _should_use_batching(self, n_points, n_frequencies, max_memory_gb, 
                           point_batch_size, freq_batch_size, auto_batch):
        """
        Determine whether frequency batching should be used based on memory requirements.
        Note: Only frequency batching is supported to preserve spatial correlation structure.
        
        Args:
            n_points: Number of simulation points
            n_frequencies: Number of frequency points
            max_memory_gb: Maximum memory limit in GB
            point_batch_size: Manual point batch size (ignored, kept for compatibility)
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
            # Always use all spatial points to preserve correlation structure
            point_batch_size = n_points
            
            # Use provided frequency batch size or calculate optimal one
            if freq_batch_size is None:
                _, optimal_freq_batch = self.get_optimal_batch_sizes(
                    n_points, n_frequencies, max_memory_gb
                )
                freq_batch_size = optimal_freq_batch
        else:
            # No batching: use all points and frequencies
            point_batch_size = n_points
            freq_batch_size = n_frequencies
        
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
        Print informative messages about frequency batching decisions and progress.
        Note: Only frequency batching is used to preserve spatial correlation structure.
        
        Args:
            estimated_memory: Estimated memory requirement in GB
            max_memory_gb: Maximum memory limit in GB
            use_batching: Whether frequency batching is being used
            point_batch_size: Size of point batches (always equals total points)
            freq_batch_size: Size of frequency batches (if using batching)
            n_point_batches: Number of point batches (always 1)
            n_freq_batches: Number of frequency batches (if using batching)
        """
        print(f"Estimated memory requirement: {estimated_memory:.2f} GB")
        
        if use_batching:
            print(f"Using frequency batching with freq_batch_size={freq_batch_size} (all {point_batch_size} spatial points preserved)")
            if n_freq_batches:
                print(f"Running batched simulation: {n_freq_batches} frequency batches")
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
