"""Spatial coherence model and cross-spectral density functions.

All functions are backend-agnostic: they accept an ``xp`` module
and work identically with numpy, jax.numpy, or torch arrays.
"""

from __future__ import annotations

from typing import Any


def davenport_coherence(
    xp: Any,
    x_i, x_j,
    y_i, y_j,
    z_i, z_j,
    freq,
    U_i, U_j,
    C_x: float, C_y: float, C_z: float,
):
    """Davenport spatial coherence function.

    .. math::
        \\operatorname{Coh}(f) = \\exp\\left(
            -2f \\, \\frac{
                \\sqrt{C_x^2 \\Delta x^2 + C_y^2 \\Delta y^2 + C_z^2 \\Delta z^2}
            }{U_i + U_j}
        \\right)

    Parameters
    ----------
    xp:
        Array module.
    x_i, x_j:
        X-coordinates, broadcastable shapes e.g. (n, 1) and (1, n).
    y_i, y_j, z_i, z_j:
        Same pattern for Y and Z.
    freq:
        Frequency (scalar).
    U_i, U_j:
        Mean wind speeds at points i and j, broadcastable like coordinates.
    C_x, C_y, C_z:
        Decay coefficients in x, y, z directions.
    """
    dist = xp.sqrt(
        C_x**2 * (x_i - x_j) ** 2
        + C_y**2 * (y_i - y_j) ** 2
        + C_z**2 * (z_i - z_j) ** 2
    )
    return xp.exp(-2.0 * freq * dist / (U_i + U_j + 1e-8))


def cross_spectrum(xp: Any, S_ii, S_jj, coherence):
    """Cross-spectral density: :math:`S_{ij} = \\sqrt{S_{ii} S_{jj}} \\cdot \\operatorname{Coh}`."""
    return xp.sqrt(S_ii * S_jj) * coherence


def simulation_frequencies(xp: Any, N: int, dw: float):
    """Frequency array for spectral representation: f_l = (l - 0.5) * dw, l = 1..N."""
    return xp.arange(1, N + 1, dtype=xp.float32) * dw - dw / 2.0
