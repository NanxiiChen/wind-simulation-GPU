"""Simulation parameters with validation and dependent-parameter computation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimulationParams:
    """Physical and numerical parameters for wind field simulation.

    Attributes
    ----------
    K:
        Dimensionless constant (von Karman constant).
    H_bar:
        Average height of surrounding buildings [m].
    z_0:
        Surface roughness height [m].
    alpha_0:
        Surface roughness exponent.
    C_x, C_y, C_z:
        Decay coefficients in x, y, z directions.
    w_up:
        Cutoff frequency [Hz].
    N:
        Number of frequency segments.
    U_d:
        Reference basic wind speed [m/s].
    z_max:
        Maximum height for mean wind speed calculation [m].

    Computed
    --------
    M:
        Number of time steps (2 * N).
    T:
        Total simulation time [s].
    dt:
        Time step [s].
    dw:
        Frequency increment [Hz].
    z_d:
        Zero-plane displacement [m].
    """

    # Physical constants
    K: float = 0.4
    H_bar: float = 10.0
    z_0: float = 0.05
    alpha_0: float = 0.16
    C_x: float = 16.0
    C_y: float = 6.0
    C_z: float = 10.0
    w_up: float = 5.0

    # Numerical settings
    N: int = 3000
    z_max: float = 450.0
    U_d: float = 25.0

    # Computed fields (set by __post_init__)
    M: int = field(init=False)
    T: float = field(init=False)
    dt: float = field(init=False)
    dw: float = field(init=False)
    z_d: float = field(init=False)

    def __post_init__(self):
        self._recompute()

    def _recompute(self):
        self.M = 2 * self.N
        self.T = self.N / self.w_up
        self.dt = self.T / self.M
        self.dw = self.w_up / self.N
        self.z_d = self.H_bar - self.z_0 / self.K

        if self.dt > 1.0 / (2.0 * self.w_up):
            raise ValueError(
                f"Nyquist criterion violated: "
                f"dt={self.dt:.6f} > 1/(2*w_up)={1.0/(2.0*self.w_up):.6f}"
            )

    def update(self, **kwargs):
        """Return a new ``SimulationParams`` with updated fields."""
        d = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        d.update(kwargs)
        return SimulationParams(**{k: d[k] for k in self.__dataclass_fields__ if k in d})

    def to_dict(self):
        """Export all fields (including computed) as a plain dict."""
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


def params_from_config(config) -> SimulationParams:
    """Build ``SimulationParams`` from an ``ml_collections.ConfigDict`` or plain dict."""
    if hasattr(config, "to_dict"):
        d = config.to_dict()
    else:
        d = dict(config)
    # Filter to known fields
    known = {f.name for f in SimulationParams.__dataclass_fields__.values()}
    return SimulationParams(**{k: v for k, v in d.items() if k in known})
