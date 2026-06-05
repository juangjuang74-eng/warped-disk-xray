"""
inference/run_mcmc.py
Gap 2: Bayesian inference — recover disk geometry parameters from observed polarization.

Strategy:
  - Likelihood: Gaussian noise model on pol_fraction and pol_angle
  - Prior:      Uniform over PARAM_BOUNDS
  - Forward model: trained emulator (fast) or mock simulator (slow)
  - Sampler: Pyro NUTS (HMC) or fallback emcee

Usage
-----
    python src/inference/run_mcmc.py --spectrum data/processed/sample_obs.npy \
                                     --emulator results/models/emulator_best.pt
"""

import argparse
import numpy as np
import torch
import json
from pathlib import Path

from src.utils.physics import PARAM_BOUNDS, ENERGY_BINS
from src.utils.io import load_spectrum
from src.emulator.model import load_emulator


# ── Likelihood & prior ────────────────────────────────────────────────────────

def log_prior(params: np.ndarray, bounds: dict) -> float:
    """Uniform prior. Returns 0 if in bounds, -inf otherwise."""
    for i, (k, (lo, hi)) in enumerate(bounds.items()):
        if not (lo <= params[i] <= hi):
            return -np.inf
    return 0.0


def log_likelihood(
    params: np.ndarray,
    observed: np.ndarray,
    emulator,
    normalizer,
    sigma_frac: float = 0.5,
    sigma_angle: float = 2.0,
) -> float:
    """
    Gaussian log-likelihood comparing emulated spectrum to observed.

    observed : (N_energy, 2) — [pol_fraction, pol_angle]
    sigma_frac : noise on polarization fraction [%]
    sigma_angle: noise on polarization angle [deg]
    """
    pred = emulator.predict_spectrum(params, normalizer)  # (N_energy, 2)
    residual_frac = (observed[:, 0] - pred[:, 0]) / sigma_frac
    residual_angle = (observed[:, 1] - pred[:, 1]) / sigma_angle
    return -0.5 * (np.sum(residual_frac**2) + np.sum(residual_angle**2))


def log_posterior(params, observed, emulator, normalizer):
    lp = log_prior(params, PARAM_BOUNDS)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, observed, emulator, normalizer)


# ── MCMC with emcee ────────────────────────────────────────────────────────────

def run_emcee(observed: np.ndarray, emulator, normalizer, n_walkers=32, n_steps=2000):
    try:
        import emcee
    except ImportError:
        raise ImportError("Install emcee: pip install emcee")

    n_dim = len(PARAM_BOUNDS)
    bounds_arr = np.array(list(PARAM_BOUNDS.values()))

    # Initialize walkers near the center of parameter space
    p0 = np.random.uniform(
        low=bounds_arr[:, 0],
        high=bounds_arr[:, 1],
        size=(n_walkers, n_dim),
    )

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        log_posterior,
        args=(observed, emulator, normalizer),
    )

    print(f"Running emcee: {n_walkers} walkers × {n_steps} steps...")
    sampler.run_mcmc(p0, n_steps, progress=True)

    # Discard burn-in, thin chain
    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    return flat_samples, sampler


def summarize_posterior(samples: np.ndarray, param_names: list) -> dict:
    """Compute median and 68% credible intervals for each parameter."""
    results = {}
    for i, name in enumerate(param_names):
        lo, mid, hi = np.percentile(samples[:, i], [16, 50, 84])
        results[name] = {"median": mid, "lo_1sigma": lo, "hi_1sigma": hi}
        print(f"  {name:15s}: {mid:.3f}  [{lo:.3f}, {hi:.3f}]")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum", required=True, help="Path to observed spectrum .npy")
    parser.add_argument("--emulator", default="results/models/emulator_best.pt")
    parser.add_argument("--n_walkers", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--output", default="results/reports/inference_result.json")
    args = parser.parse_args()

    # Load data
    spectrum, true_params = load_spectrum(args.spectrum)
    observed = spectrum[:, 1:]  # pol_fraction and pol_angle only

    # Load emulator
    emulator, normalizer = load_emulator(args.emulator)

    # Run MCMC
    samples, sampler = run_emcee(
        observed, emulator, normalizer,
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
    )

    param_names = list(PARAM_BOUNDS.keys())
    print("\nPosterior summary:")
    results = summarize_posterior(samples, param_names)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    np.save(args.output.replace(".json", "_samples.npy"), samples)
    print(f"\nResults saved to {args.output}")

    # Corner plot
    try:
        import corner, matplotlib.pyplot as plt
        fig = corner.corner(samples, labels=param_names, truths=list(true_params.values()) if true_params else None)
        fig.savefig(args.output.replace(".json", "_corner.png"), dpi=150)
        print("Corner plot saved.")
    except ImportError:
        print("Install corner for posterior plots: pip install corner")


if __name__ == "__main__":
    main()
