# from .model import WindSimulator
# from .visualizer import WindVisualizer

from .factory import get_simulator, get_visualizer


__title__ = "Stochastic Wind Simulation"
__version__ = "0.1.0"
__author__ = "Nanxi Chen"
__email__ = "nxchen@tongji.edu.cn"
__license__ = "GPL-3.0"
__copyright__ = "Copyright (c) 2025 Nanxi Chen"

__all__ = [
    "__title__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
    "get_simulator",
    "get_visualizer",
]