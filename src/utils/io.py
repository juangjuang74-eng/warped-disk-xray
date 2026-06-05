"""
utils/io.py
Load, save, and validate polarization spectra and simulation outputs.
"""

import json
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_spectrum(path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a polarization spectrum from .npy + companion .json.

    Returns
    -------
    spectrum : ndarray, shape (N_energy, 3)
        Columns: [energy_keV, pol_fraction_pct, pol_angle_deg]
    params : dict
        Physical parameters for this simulation.
    """
    path = Path(path)
    spectrum = np.load(path)
    param_path = path.with_suffix(".json")
    params = json.loads(param_path.read_text()) if param_path.exists() else {}
    return spectrum, params


def save_spectrum(spectrum: np.ndarray, params: dict, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, spectrum)
    path.with_suffix(".json").write_text(json.dumps(params, indent=2))


def load_dataset_hdf5(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a full simulation dataset from HDF5.

    Returns
    -------
    X : ndarray, shape (N_samples, N_energy, 3)   — spectra
    y : ndarray, shape (N_samples, N_params)        — parameter vectors
    """
    with h5py.File(path, "r") as f:
        X = f["spectra"][:]
        y = f["params"][:]
    return X, y


def save_dataset_hdf5(
    spectra: np.ndarray,
    params: np.ndarray,
    path: str,
    param_names: Optional[list] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("spectra", data=spectra, compression="gzip")
        f.create_dataset("params", data=params, compression="gzip")
        if param_names:
            f.attrs["param_names"] = json.dumps(param_names)


def validate_spectrum(spectrum: np.ndarray) -> bool:
    """Basic sanity checks on a polarization spectrum array."""
    if spectrum.ndim != 2 or spectrum.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {spectrum.shape}")
    energies, pol_frac, pol_angle = spectrum.T
    if not np.all(np.diff(energies) > 0):
        raise ValueError("Energy axis must be monotonically increasing.")
    if np.any(pol_frac < 0) or np.any(pol_frac > 100):
        raise ValueError("Polarization fraction must be in [0, 100] %.")
    return True
