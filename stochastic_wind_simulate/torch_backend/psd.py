from typing import Dict
import torch
from torch import Tensor


class WindSpectrum:
    """Base class for wind spectrum."""

    def __init__(self, **kwargs):
        """
        Initialize the wind spectrum.
        """
        self.params = {}
        self.params.update(kwargs)  # Update with any additional parameters
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_tensor(self, value, device=None):
        """Convert input to tensor on device."""
        if device is None:
            device = self.device
        if isinstance(value, torch.Tensor):
            return value.to(device)
        else:
            return torch.tensor(value, device=device)
    
    def calculate_mean_wind_speed(self, Z, U_d, alpha_0):
        """Calculate mean wind speed at height Z."""
        Z = self._to_tensor(Z)
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
        n = self._to_tensor(n)
        Z = self._to_tensor(Z)
        U_z = self.calculate_mean_wind_speed(Z, U_d, alpha_0)
        return n * Z / U_z
    
    def calculate_friction_velocity(self, Z, U_d, z_0, z_d, K, alpha_0):
        """Calculate wind friction velocity u_*."""
        Z = self._to_tensor(Z)
        U_z = self.calculate_mean_wind_speed(Z, U_d, alpha_0)
        return K * U_z / torch.log((Z - z_d) / z_0)
    
    def calculate_power_spectrum_u(self, n, u_star, f):
        """Calculate along-wind fluctuating wind power spectral density S_u(n)."""
        raise NotImplementedError("Spectrum u are not defined in this class.")
    
    def calculate_power_spectrum_v(self, n, u_star, f):
        """Calculate cross-wind fluctuating wind power spectral density S_v(n)."""
        raise NotImplementedError("Spectrum v are not defined in this class.")

    def calculate_power_spectrum_w(self, n, u_star, f):
        """Calculate vertical fluctuating wind power spectral density S_w(n)."""
        raise NotImplementedError("Spectrum w are not defined in this class.")

    def calculate_power_spectrum(self, freq, Zs, component, **kwargs):
        """Calculate power spectrum for given frequency, heights, and component."""
        freq = self._to_tensor(freq)
        Zs = self._to_tensor(Zs)
        
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
        n = self._to_tensor(n)
        u_star = self._to_tensor(u_star)
        f = self._to_tensor(f)
        return (u_star**2 / n) * (200 * f / ((1 + 50 * f) ** (5 / 3)))


class DavenportWindSpectrumNonDimensional(WindSpectrumNonDimensional):
    """Davenport wind spectrum class."""

    def __init__(self, **kwargs):
        """
        Initialize the Davenport wind spectrum.
        """
        super().__init__(**kwargs)

    def calculate_power_spectrum_u(self, n, u_star, f):
        """Calculate along-wind fluctuating wind power spectral density S_u(n)."""
        n = self._to_tensor(n)
        u_star = self._to_tensor(u_star)
        f = self._to_tensor(f)
        return (u_star**2 / n) * (4 * f**2 / ((1 + f**2) ** (4 / 3)))


class PanofskyWindSpectrumNonDimensional(WindSpectrumNonDimensional):
    """Panofsky wind spectrum class."""

    def __init__(self, **kwargs):
        """
        Initialize the Panofsky wind spectrum.
        """
        super().__init__(**kwargs)

    def calculate_power_spectrum_w(self, n, u_star, f):
        """Calculate vertical fluctuating wind power spectral density S_w(n)."""
        n = self._to_tensor(n)
        u_star = self._to_tensor(u_star)
        f = self._to_tensor(f)
        return (u_star**2 / n) * (6 * f / ((1 + 4 * f) ** 2))


class TeunissenWindSpectrumNonDimensional(WindSpectrumNonDimensional):
    """Teunissen wind spectrum class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def calculate_power_spectrum_u(self, n, u_star, f):
        """Calculate along-wind fluctuating wind power spectral density S_u(n)."""
        n = self._to_tensor(n)
        u_star = self._to_tensor(u_star)
        f = self._to_tensor(f)
        return (u_star**2 / n) * (105 * f / ((0.44 + 33 * f) ** (5 / 3)))

    def calculate_power_spectrum_w(self, n, u_star, f):
        """Calculate vertical fluctuating wind power spectral density S_w(n)."""
        n = self._to_tensor(n)
        u_star = self._to_tensor(u_star)
        f = self._to_tensor(f)
        return (u_star**2 / n) * (2 * f / ((1 + 5.3 * f) ** (5 / 3)))


def get_spectrum_class(spectrum_type="kaimal-nd", **kwargs):
    """
    Get the appropriate wind spectrum class based on the specified type.

    Args:
        spectrum_type: Type of wind spectrum ("kaimal-nd", "teunissen-nd", "panofsky-nd")
        **kwargs: Additional parameters

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
