from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit


class WindSpectrum:
    """Base class for wind spectrum."""

    def __init__(self, **kwargs):
        """
        Initialize the Kaimal wind spectrum.
        """
        self.params = self._set_default_parameters()
        self.params.update(kwargs)  # Update with any additional parameters


    def _set_default_parameters(self, **kwargs) -> Dict:
        """Set default parameters for the Kaimal wind spectrum."""
        params = {
            "K": 0.4,  # Dimensionless constant
            "H_bar": 10.0,  # Average height of surrounding buildings (m)
            "z_0": 0.05,  # Surface roughness height
            "alpha_0": 0.16, # Surface roughness exponent
            "w_up": 5.0,  # Cutoff frequency (Hz)
            "N": 3000,  # Number of frequency segments
            "M": 6000,  # Number of time points (M=2N)
            "T": 600,  # Simulation duration (s)
            "dt": 0.1,  # Time step (s)
            "U_d": 25.0,  # Design basic wind speed (m/s)
        }
        params["z_d"] = (
            params["H_bar"] - params["z_0"] / params["K"]
        )  # Calculate zero plane displacement

        return params
    
    def update_parameters(self, **kwargs):
        """Update simulation parameters."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        # Update dependent parameters
        self.params["z_d"] = (
            self.params["H_bar"] - self.params["z_0"] / self.params["K"]
        )
        self.params.update(kwargs)  # Update with any additional parameters

    
    def calculate_mean_wind_speed(self, Z, U_d, alpha_0):
        """Calculate mean wind speed at height Z."""
        return U_d * (Z / 10) ** alpha_0
    

class WindSpectrumNonDimensional(WindSpectrum):
    """Base class for non-dimensional wind spectrum."""

    def __init__(self, **kwargs):
        """
        Initialize the non-dimensional wind spectrum.
        """
        super().__init__(**kwargs)

    
    def calculate_f(self, n, Z, U_d, alpha_0):
        """Calculate the frequency-dependent function f."""
        U_z = self.calculate_mean_wind_speed(Z, U_d, alpha_0)
        return n * Z / U_z
    
    def calculate_friction_velocity(self, Z, U_d, z_0, z_d, K, alpha_0):
        """Calculate wind friction velocity u_*."""
        U_z = self.calculate_mean_wind_speed(Z, U_d, alpha_0)
        return K * U_z / jnp.log((Z - z_d) / z_0)
    
    def calculate_power_spectrum_u(self, n, u_star, f):
        """Calculate along-wind fluctuating wind power spectral density S_u(n)."""
        raise NotImplementedError("Spectrum u are not defined in this class.")
    
    def calculate_power_spectrum_v(self, n, u_star, f):
        """Calculate cross-wind fluctuating wind power spectral density S_v(n)."""
        raise NotImplementedError("Spectrum v are not defined in this class.")

    def calculate_power_spectrum_w(self, n, u_star, f):
        """Calculate vertical fluctuating wind power spectral density S_w(n)."""
        raise NotImplementedError("Spectrum w are not defined in this class.")

    
    @partial(jit, static_argnums=(0,3))
    def calculate_power_spectrum(self, freq, Zs, component, **kwargs):
        u_stars = self.calculate_friction_velocity(
            Zs, 
            self.params["U_d"], 
            self.params["z_0"], 
            self.params["z_d"], 
            self.params["K"], 
            self.params["alpha_0"]
        )
        fs = self.calculate_f(freq, Zs, self.params["U_d"], self.params["alpha_0"])
        spectrum_function = getattr(self, f"calculate_power_spectrum_{component}", None)
        if spectrum_function is None:
            raise ValueError(f"Unsupported component: {component}.")
        spectrum = spectrum_function(freq, u_stars, fs)
        return spectrum
    

class KaimalWindSpectrumNonDimensional(WindSpectrumNonDimensional):
    """Kaimal wind spectrum class."""

    def __init__(self, **kwargs):
        """
        Initialize the Kaimal wind spectrum.
        """
        super().__init__(**kwargs)

    def calculate_power_spectrum_u(self, n, u_star, f):
        """Calculate along-wind fluctuating wind power spectral density S_u(n)."""
        return (u_star**2 / n) * (200 * f / ((1 + 50 * f) ** (5 / 3)))


class PanofskyWindSpectrumNonDimensional(WindSpectrumNonDimensional):
    """Panofsky wind spectrum class."""

    def __init__(self, **kwargs):
        """
        Initialize the Kaimal wind spectrum.
        """
        super().__init__(**kwargs)

    def calculate_power_spectrum_w(self, n, u_star, f):
        """Calculate vertical fluctuating wind power spectral density S_w(n)."""
        return (u_star**2 / n) * (6 * f / ((1 + 4 * f) ** 2))

class TeunissenWindSpectrumNonDimensional(WindSpectrumNonDimensional):
    """Teunissen wind spectrum class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def calculate_power_spectrum_u(self, n, u_star, f):
        """Calculate along-wind fluctuating wind power spectral density S_u(n)."""
        return (u_star**2 / n) * (105 * f / ((0.44 + 33 * f) ** (5 / 3)))

    def calculate_power_spectrum_w(self, n, u_star, f):
        """Calculate vertical fluctuating wind power spectral density S_w(n)."""
        return (u_star**2 / n) * (2 * f / ((1 + 5.3 * f) ** (5 / 3)))



def get_spectrum_class(spectrum_type="kaimal-nd", **kwargs):
    """
    Get the appropriate wind spectrum class based on the specified type.

    Args:
        spectrum_type: Type of wind spectrum ("kaimal" or others)
        key: JAX random number seed

    Returns:
        An instance of the specified wind spectrum class.
    """
    if spectrum_type.lower() == "kaimal-nd":
        return KaimalWindSpectrumNonDimensional
    elif spectrum_type.lower() == "teunissen-nd":
        return TeunissenWindSpectrumNonDimensional
    elif spectrum_type.lower() == "panofsky-nd":
        return PanofskyWindSpectrumNonDimensional
    else:
        raise ValueError(f"Unsupported spectrum type: {spectrum_type}")