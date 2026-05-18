"""Unified wind spectrum models.

All spectrum classes are backend-agnostic: they accept an ``xp`` module
(numpy, jax.numpy, or torch) at init time and use it for array operations.
Basic arithmetic (+, -, *, /, **) works identically across all three backends.
"""

from __future__ import annotations

from typing import Any, Dict


class WindSpectrum:
    """Base class for wind spectrum models, backend-agnostic.

    Parameters
    ----------
    xp:
        Array module (``numpy``, ``jax.numpy``, or ``torch``).
    params:
        Dictionary of physical parameters.
    """

    def __init__(self, xp: Any, params: Dict[str, float]):
        self.xp = xp
        self.params = params

    def mean_wind_speed(self, z, U_d, alpha_0):
        """Mean wind speed at height z: U_d * (z/10)^alpha_0."""
        return U_d * (z / 10.0) ** alpha_0

    def friction_velocity(self, z, U_d, z_0, z_d, K, alpha_0):
        """Friction velocity: u_* = K * U_z / ln((z - z_d) / z_0)."""
        U_z = self.mean_wind_speed(z, U_d, alpha_0)
        return K * U_z / self.xp.log((z - z_d) / z_0)

    def reduced_frequency(self, n, z, U_d, alpha_0):
        """Reduced frequency: f = n * z / U_z."""
        U_z = self.mean_wind_speed(z, U_d, alpha_0)
        return n * z / U_z

    def __call__(self, freq, heights, component: str, **override):
        """Compute power spectral density at a given frequency and heights.

        Parameters
        ----------
        freq:
            Frequency value (scalar).
        heights:
            Heights at each spatial point (array of shape (n,)).
        component:
            Wind component: ``"u"`` (along-wind) or ``"w"`` (vertical).
        override:
            Optional parameter overrides (e.g. ``U_d=25.0``).

        Returns
        -------
        PSD values, shape (n,).
        """
        p = {**self.params, **override}
        u_star = self.friction_velocity(
            heights, p["U_d"], p["z_0"], p["z_d"], p["K"], p["alpha_0"]
        )
        f = self.reduced_frequency(freq, heights, p["U_d"], p["alpha_0"])
        fn = getattr(self, f"psd_{component}", None)
        if fn is None:
            raise ValueError(
                f"Spectrum {self.__class__.__name__} does not support "
                f"component {component!r}. Supported: {self.supported_components()}"
            )
        return fn(freq, u_star, f)

    def supported_components(self):
        """Return list of supported wind components."""
        return [k[4:] for k in dir(self) if k.startswith("psd_")]


class KaimalSpectrum(WindSpectrum):
    """Kaimal wind spectrum (non-dimensional form)."""

    def psd_u(self, n, u_star, f):
        return (u_star**2 / n) * (200.0 * f / (1.0 + 50.0 * f) ** (5.0 / 3.0))

    def psd_w(self, n, u_star, f):
        return (u_star**2 / n) * (3.36 * f / (1.0 + 10.0 * f) ** (5.0 / 3.0))


class PanofskySpectrum(WindSpectrum):
    """Panofsky wind spectrum (non-dimensional form)."""

    def psd_w(self, n, u_star, f):
        return (u_star**2 / n) * (6.0 * f / (1.0 + 4.0 * f) ** 2.0)


class TeunissenSpectrum(WindSpectrum):
    """Teunissen wind spectrum (non-dimensional form)."""

    def psd_u(self, n, u_star, f):
        return (u_star**2 / n) * (105.0 * f / (0.44 + 33.0 * f) ** (5.0 / 3.0))

    def psd_w(self, n, u_star, f):
        return (u_star**2 / n) * (2.0 * f / (1.0 + 5.3 * f) ** (5.0 / 3.0))


_SPECTRUM_REGISTRY: Dict[str, type] = {
    "kaimal": KaimalSpectrum,
    "panofsky": PanofskySpectrum,
    "teunissen": TeunissenSpectrum,
}


def get_spectrum(name: str) -> type:
    """Look up a spectrum class by name (case-insensitive).

    Supported names: ``"kaimal"``, ``"panofsky"``, ``"teunissen"``.
    """
    key = name.lower().replace("-nd", "").replace("_", "")
    if key not in _SPECTRUM_REGISTRY:
        raise ValueError(
            f"Unknown spectrum type: {name!r}. "
            f"Supported: {list(_SPECTRUM_REGISTRY)}"
        )
    return _SPECTRUM_REGISTRY[key]
