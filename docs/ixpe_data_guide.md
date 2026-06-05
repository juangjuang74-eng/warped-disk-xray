# IXPE Data Processing Guide

## Download

IXPE public archive: https://heasarc.gsfc.nasa.gov/docs/ixpe/archive/

For the candidate targets from the paper:

| Target       | IXPE obs IDs | Notes |
|--------------|-------------|-------|
| GRO J1655-40 | Search HEASARC | Jet at 85°, binary at 70° |
| Cygnus X-1   | Search HEASARC | Inner disk vs binary ~10–15° offset |
| 4U 1957+11   | Search HEASARC | Soft state, high spin |

## Level 2 → .npy conversion

```python
from astropy.io import fits
import numpy as np

def ixpe_to_spectrum(du1_path, du2_path, du3_path, energy_bins) -> np.ndarray:
    """
    Combine three IXPE detector units into a single polarization spectrum.
    Returns array of shape (N_energy, 3): [energy, pol_frac, pol_angle].
    """
    stokes = {"Q": [], "U": [], "I": []}

    for path in [du1_path, du2_path, du3_path]:
        with fits.open(path) as hdul:
            # Level 2 products contain Stokes parameters per event
            # Column names may vary by pipeline version — check hdul.info()
            energy = hdul["EVENTS"].data["ENERGY"]   # keV
            q      = hdul["EVENTS"].data["Q"]
            u      = hdul["EVENTS"].data["U"]
            i_val  = hdul["EVENTS"].data["I"]
            stokes["Q"].append(q)
            stokes["U"].append(u)
            stokes["I"].append(i_val)

    Q = np.concatenate(stokes["Q"])
    U = np.concatenate(stokes["U"])
    I = np.concatenate(stokes["I"])
    E = np.concatenate([energy] * 3)  # same energy for all three

    # Bin into energy grid
    pol_frac  = np.zeros(len(energy_bins))
    pol_angle = np.zeros(len(energy_bins))
    edges = np.concatenate([[energy_bins[0] * 0.9],
                             (energy_bins[:-1] + energy_bins[1:]) / 2,
                             [energy_bins[-1] * 1.1]])

    for j in range(len(energy_bins)):
        mask = (E >= edges[j]) & (E < edges[j + 1])
        if mask.sum() > 0:
            Qb, Ub, Ib = Q[mask].sum(), U[mask].sum(), I[mask].sum()
            pol_frac[j]  = 100 * np.sqrt(Qb**2 + Ub**2) / (Ib + 1e-30)
            pol_angle[j] = 0.5 * np.degrees(np.arctan2(Ub, Qb)) % 180

    return np.column_stack([energy_bins, pol_frac, pol_angle])
```

## Save in repo format

```python
from src.utils.io import save_spectrum
from src.utils.physics import ENERGY_BINS

spectrum = ixpe_to_spectrum("du1.fits", "du2.fits", "du3.fits", ENERGY_BINS)
save_spectrum(spectrum, {"source": "CygX-1", "obs_id": "XXXXXXX"},
              "data/processed/cygx1_obs1.npy")
```

## IXPE energy range note

IXPE measures **2–8 keV**. The simulation energy grid covers 0.1–10 keV.
When running inference, restrict to the 2–8 keV subset:

```python
mask = (ENERGY_BINS >= 2.0) & (ENERGY_BINS <= 8.0)
observed_ixpe = spectrum[mask, :]
```
