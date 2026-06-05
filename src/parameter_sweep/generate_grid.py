"""
parameter_sweep/generate_grid.py
Gap 1: Systematic parameter space exploration.

Generates a grid (or Latin Hypercube) of (spin, r_bp, beta, phi, inclination)
parameter combinations and calls the ray-tracing simulator (or its mock) for each.
Results are saved to data/simulated/ as an HDF5 dataset for downstream ML use.

Usage
-----
    python src/parameter_sweep/generate_grid.py --config configs/sweep_config.yaml
    python src/parameter_sweep/generate_grid.py --mode lhs --n_samples 5000
"""

import argparse
import yaml
import numpy as np
from itertools import product
from pathlib import Path
from tqdm import tqdm

from src.utils.physics import PARAM_BOUNDS, ENERGY_BINS, isco_radius, outer_disk_inclination
from src.utils.io import save_dataset_hdf5


# ── Mock simulator (replace with actual ray-tracing call) ─────────────────────

def mock_simulate(spin, r_bp, beta, phi, inclination) -> np.ndarray:
    """
    Placeholder for the real GR ray-tracing code.
    Returns a synthetic (N_energy, 3) spectrum array.

    In production: replace this with a subprocess call to the Fortran/C
    ray-tracing binary, or a Python wrapper around it.
    """
    rng = np.random.default_rng(seed=int(spin * 1e4 + r_bp * 100 + beta))
    i_out = outer_disk_inclination(inclination, beta, phi)

    # Toy model: pol_fraction peaks at intermediate energies, angle rotates
    energies = ENERGY_BINS
    pol_frac = (
        5.0
        + 3.0 * np.sin(np.linspace(0, np.pi, len(energies)))
        + rng.normal(0, 0.3, len(energies))
    )
    pol_angle = (
        90.0
        + (i_out - inclination) * np.linspace(0, 1, len(energies))
        + rng.normal(0, 1.0, len(energies))
    )
    pol_frac = np.clip(pol_frac, 0, 100)
    return np.column_stack([energies, pol_frac, pol_angle])


# ── Grid generation ────────────────────────────────────────────────────────────

def latin_hypercube_sample(n_samples: int, bounds: dict, seed: int = 42) -> np.ndarray:
    """Generate Latin Hypercube samples over the parameter bounds."""
    from scipy.stats import qmc
    keys = list(bounds.keys())
    lo = np.array([bounds[k][0] for k in keys])
    hi = np.array([bounds[k][1] for k in keys])
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    samples = sampler.random(n=n_samples)
    return qmc.scale(samples, lo, hi), keys


def uniform_grid(n_per_dim: int, bounds: dict) -> tuple:
    """Generate a uniform grid over all parameter dimensions."""
    keys = list(bounds.keys())
    axes = [np.linspace(bounds[k][0], bounds[k][1], n_per_dim) for k in keys]
    grid = np.array(list(product(*axes)))
    return grid, keys


def run_sweep(params_array: np.ndarray, param_names: list, output_path: str):
    n = len(params_array)
    all_spectra = []
    all_params = []

    print(f"Running sweep over {n} parameter combinations...")
    for row in tqdm(params_array):
        p = dict(zip(param_names, row))
        spectrum = mock_simulate(**p)
        all_spectra.append(spectrum)
        all_params.append(row)

    X = np.array(all_spectra)   # (N, N_energy, 3)
    y = np.array(all_params)    # (N, N_params)
    save_dataset_hdf5(X, y, output_path, param_names=param_names)
    print(f"Saved {n} spectra → {output_path}")
    return X, y


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sweep_config.yaml")
    parser.add_argument("--mode", choices=["grid", "lhs"], default="lhs")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_per_dim", type=int, default=5)
    parser.add_argument("--output", type=str, default="data/simulated/sweep_dataset.h5")
    args = parser.parse_args()

    # Load config if exists
    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    mode = cfg.get("mode", args.mode)
    n_samples = cfg.get("n_samples", args.n_samples)
    n_per_dim = cfg.get("n_per_dim", args.n_per_dim)
    output = cfg.get("output", args.output)

    if mode == "lhs":
        params_array, param_names = latin_hypercube_sample(n_samples, PARAM_BOUNDS)
    else:
        params_array, param_names = uniform_grid(n_per_dim, PARAM_BOUNDS)

    run_sweep(params_array, param_names, output)


if __name__ == "__main__":
    main()
