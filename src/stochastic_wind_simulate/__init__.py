"""Stochastic Wind Field Simulation.

A multi-backend (JAX / NumPy / PyTorch) library for simulating
stationary and nonstationary wind fields using the spectral
representation method.
"""

from .simulator import (
    JaxWindSimulator,
    NumpyWindSimulator,
    TorchWindSimulator,
    create_simulator,
)
from .nonstationary import NonstationaryWindSimulator
from .visualizer import WindVisualizer
from .params import SimulationParams
from .spectrum import get_spectrum

__title__ = "Stochastic Wind Simulation"
__version__ = "0.2.0"
__author__ = "Nanxi Chen"
__email__ = "nxchen@tongji.edu.cn"
__license__ = "GPL-3.0"

__all__ = [
    "JaxWindSimulator",
    "NumpyWindSimulator",
    "TorchWindSimulator",
    "NonstationaryWindSimulator",
    "create_simulator",
    "WindVisualizer",
    "SimulationParams",
    "get_spectrum",
]
