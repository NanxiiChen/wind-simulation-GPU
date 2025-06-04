
# Stochastic Wind Field Simulation on GPU

We present a GPU-accelerated Python implementation of the stochastic wind field simulation method based on Shinozuka's harmonic synthesis method. This method is widely used in civil engineering to simulate fluctuating wind fields, particularly for structural analysis of bridges and buildings.

We provide both JAX and PyTorch implementations to leverage the parallel computing capabilities of GPUs, significantly speeding up the simulation process compared to traditional CPU-based methods.

| Num_samples | JAX (s) | PyTorch (s) | CPU (s) |
|-------------|---------|-------------|---------|
| 100        | 2.56    | 2.78        | ??  |


