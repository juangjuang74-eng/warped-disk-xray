"""
utils/physics.py
Shared physical constants and helper functions for the BP X-ray research suite.
All units in CGS unless noted.
"""

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
G = 6.674e-8          # Gravitational constant [cm^3 g^-1 s^-2]
C = 2.998e10          # Speed of light [cm/s]
M_SUN = 1.989e33      # Solar mass [g]
KEV_TO_ERG = 1.602e-9 # 1 keV in erg

# ── Default BH / disk parameters (Abarr & Krawczynski 2019) ──────────────────
DEFAULT_PARAMS = {
    "spin": 0.9,
    "r_bp": 8.0,       # Break radius [r_g]
    "beta": 15.0,      # Misalignment angle [degrees]
    "phi": 90.0,       # Azimuthal viewing angle [degrees]
    "inclination": 75.0,  # Inner disk inclination [degrees]
    "bh_mass": 10.0,   # BH mass [M_sun]
    "mdot_eddington": 0.5,  # Accretion rate [fraction of Eddington]
}

# ── Parameter bounds for sweep / inference ────────────────────────────────────
PARAM_BOUNDS = {
    "spin":        (0.0,   0.998),
    "r_bp":        (3.0,  50.0),    # r_g
    "beta":        (0.0,  45.0),    # degrees
    "phi":         (0.0, 360.0),    # degrees
    "inclination": (10.0,  85.0),   # degrees
}

# ── Energy grid (matches paper's log(E/keV) from -1 to 1) ────────────────────
ENERGY_BINS = np.logspace(-1, 1, 50)  # keV


def gravitational_radius(mass_solar: float) -> float:
    """Return r_g = GM/c^2 in cm."""
    return G * mass_solar * M_SUN / C**2


def isco_radius(spin: float) -> float:
    """
    Innermost stable circular orbit in units of r_g.
    Uses the Bardeen, Press & Teukolsky (1972) formula.
    Valid for prograde orbits (spin >= 0).
    """
    a = np.clip(spin, 0.0, 0.998)
    z1 = 1 + (1 - a**2) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = np.sqrt(3 * a**2 + z1**2)
    return 3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def outer_disk_inclination(i_in: float, beta: float, phi: float) -> float:
    """
    Compute outer disk inclination (degrees) given inner disk inclination,
    misalignment angle beta, and azimuthal angle phi.
    Eq. 7 from Abarr & Krawczynski (2019).
    """
    i = np.radians(i_in)
    b = np.radians(beta)
    p = np.radians(phi)
    cos_i_out = np.cos(i) * np.cos(b) - np.sin(i) * np.cos(p) * np.sin(b)
    return np.degrees(np.arccos(np.clip(cos_i_out, -1.0, 1.0)))


def polarization_angle_swing(pol_angles: np.ndarray, energies: np.ndarray) -> float:
    """
    Compute the total swing in polarization angle (degrees) across the energy band.
    Mirrors the quantity plotted in Figure 9 of the paper.
    """
    low_e_mask = energies < 1.0   # < 1 keV
    high_e_mask = energies > 3.0  # > 3 keV
    if low_e_mask.sum() == 0 or high_e_mask.sum() == 0:
        return np.nan
    angle_low = np.mean(pol_angles[low_e_mask])
    angle_high = np.mean(pol_angles[high_e_mask])
    return float(np.abs(angle_high - angle_low))
