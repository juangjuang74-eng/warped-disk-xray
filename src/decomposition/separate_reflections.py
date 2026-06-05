"""
decomposition/separate_reflections.py
Gap 4: Separate multi-reflection components in polarization spectra.

The paper identifies three emission components:
  1. Direct emission
  2. Photons reflected only off the inner disk
  3. Photons reflected only off the outer disk
  (4. Small component reflecting off both disks — currently unquantified)

This module uses Non-negative Matrix Factorization (NMF) and a supervised
neural decomposer to isolate these components from total spectra.

Usage
-----
    python src/decomposition/separate_reflections.py \
        --total  data/processed/total_spectrum.npy \
        --output results/figures/decomposition.png
"""

import numpy as np
import argparse
from pathlib import Path
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from typing import Tuple


# ── NMF-based decomposition ───────────────────────────────────────────────────

class NMFDecomposer:
    """
    Non-negative Matrix Factorization decomposer.

    Fits on a collection of total spectra (pol_fraction only — non-negative),
    extracts N_components basis spectra.
    """

    def __init__(self, n_components: int = 4, max_iter: int = 500):
        self.n_components = n_components
        self.model = NMF(
            n_components=n_components,
            init="nndsvda",
            max_iter=max_iter,
            random_state=42,
        )
        self.components_ = None  # (n_components, N_energy)
        self.component_labels = ["direct", "inner_reflect", "outer_reflect", "double_reflect"]

    def fit(self, spectra_collection: np.ndarray) -> "NMFDecomposer":
        """
        spectra_collection : (N_samples, N_energy) — pol_fraction only
        """
        self.model.fit(spectra_collection)
        self.components_ = self.model.components_
        return self

    def decompose(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Decompose a single spectrum.

        spectrum : (N_energy,) — pol_fraction
        Returns  : (n_components,) mixing coefficients
        """
        return self.model.transform(spectrum.reshape(1, -1)).flatten()

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return coefficients @ self.components_


# ── Supervised neural decomposer ──────────────────────────────────────────────

import torch
import torch.nn as nn


class NeuralDecomposer(nn.Module):
    """
    Supervised decomposer: maps total spectrum → (direct, inner_reflect, outer_reflect).

    Training requires simulations with labeled components (available from the
    ray-tracing code when run with component tracking enabled).
    """

    def __init__(self, n_energy: int = 50, n_components: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_energy, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_energy * n_components),
            nn.Softplus(),  # ensure non-negative component spectra
        )
        self.n_energy = n_energy
        self.n_components = n_components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_energy) → (batch, n_components, n_energy)"""
        out = self.net(x)
        return out.view(-1, self.n_components, self.n_energy)


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_decomposition(
    energies: np.ndarray,
    total: np.ndarray,
    components: dict,
    output_path: str,
):
    """Reproduce decomposition plot style from Figure 4 of the paper."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    styles = {
        "total":         ("black",  "-",   "Total"),
        "direct":        ("blue",   "--",  "Direct"),
        "inner_reflect": ("gold",   "--",  "Inner disk reflected"),
        "outer_reflect": ("purple", ":",   "Outer disk reflected"),
        "double_reflect":("gray",   "-.",  "Double reflected"),
    }

    for ax_idx, (col_label, col) in enumerate([("Polarization Fraction [%]", 1),
                                                ("Polarization Angle [°]",    2)]):
        ax = axes[ax_idx]
        for key, arr in {"total": total, **components}.items():
            if key not in styles:
                continue
            color, ls, label = styles[key]
            ax.plot(np.log10(energies), arr[:, col - 1], color=color, ls=ls, label=label)
        ax.set_xlabel("log(Energy/keV)")
        ax.set_ylabel(col_label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved decomposition plot → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", required=True, help="Total spectrum .npy, shape (N_energy, 3)")
    parser.add_argument("--output", default="results/figures/decomposition.png")
    parser.add_argument("--method", choices=["nmf", "neural"], default="nmf")
    args = parser.parse_args()

    from src.utils.io import load_spectrum
    spectrum, _ = load_spectrum(args.total)
    energies = spectrum[:, 0]

    if args.method == "nmf":
        # Unsupervised: just show basis components
        decomposer = NMFDecomposer(n_components=4)
        # For demo, treat the single spectrum as a dataset
        decomposer.fit(spectrum[:, 1:2].T)  # (1, N_energy)
        coeffs = decomposer.decompose(spectrum[:, 1])
        print("NMF mixing coefficients:", dict(zip(decomposer.component_labels, coeffs)))

        # Plot basis components
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, comp in enumerate(decomposer.components_):
            ax.plot(np.log10(energies), comp, label=decomposer.component_labels[i])
        ax.set_xlabel("log(Energy/keV)")
        ax.set_ylabel("NMF component (pol fraction)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Saved → {args.output}")
    else:
        print("Neural decomposer requires pre-trained model. Run training first.")


if __name__ == "__main__":
    main()
