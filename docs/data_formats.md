# Data Formats

## Single Spectrum

Stored as a pair of files:

**`spectrum.npy`** — numpy array, shape `(N_energy, 3)`

| Column | Quantity        | Unit    | Notes                          |
|--------|-----------------|---------|--------------------------------|
| 0      | Energy          | keV     | Log-spaced, 0.1–10 keV         |
| 1      | Pol. fraction   | %       | Range [0, 100]                 |
| 2      | Pol. angle      | degrees | Range [0, 180], IAU convention |

**`spectrum.json`** — companion parameter file

```json
{
  "spin":        0.9,
  "r_bp":        8.0,
  "beta":        15.0,
  "phi":         90.0,
  "inclination": 75.0,
  "disk_type":   "warped"
}
```

`disk_type` options: `"warped"`, `"aligned"`, `"misaligned"`

---

## Dataset (HDF5)

```
dataset.h5
  /spectra    float32  (N, N_energy, 3)
  /params     float32  (N, 5)            [spin, r_bp, beta, phi, inclination]
  attrs:
    param_names   JSON string
```

```python
from src.utils.io import load_dataset_hdf5
X, y = load_dataset_hdf5("data/simulated/sweep_dataset.h5")
```

---

## GRMHD Bridge Output (HDF5)

```
grmhd_polarization.h5
  /time          float64  (N_t,)
  /geometry      float64  (N_t, 5)
  /polarization  float32  (N_t, N_energy, 2)
  /energy_keV    float64  (N_energy,)
```

---

## IXPE Input (FITS)

IXPE Level 2 data: three detector unit files (DU1, DU2, DU3).
See `docs/ixpe_data_guide.md` for conversion to `.npy` format.
