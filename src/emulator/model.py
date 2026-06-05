"""
emulator/model.py
Gap 5: Neural network surrogate emulator for expensive ray-tracing simulations.

Architecture: MLP that maps (spin, r_bp, beta, phi, inclination) → full polarization
spectrum (N_energy × 3 outputs: energy, pol_fraction, pol_angle).

Once trained, inference is ~1000x faster than the ray-tracing code.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class SpectralEmulator(nn.Module):
    """
    Feed-forward MLP emulator.

    Input  : 5 physical parameters (normalized to [0,1])
    Output : flattened spectrum vector of shape (N_energy * 2,)
             — pol_fraction and pol_angle for each energy bin
             (energy bins are fixed and known, so not predicted)
    """

    def __init__(self, n_params: int = 5, n_energy: int = 50, hidden_dims=(256, 512, 512, 256)):
        super().__init__()
        self.n_params = n_params
        self.n_energy = n_energy
        self.output_dim = n_energy * 2   # pol_fraction + pol_angle per bin

        layers = []
        in_dim = n_params
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_params) normalized → returns (batch, n_energy*2)"""
        return self.net(x)

    def predict_spectrum(self, params: np.ndarray, normalizer) -> np.ndarray:
        """
        High-level helper: params (1D array) → (N_energy, 2) array
        with columns [pol_fraction, pol_angle].
        """
        self.eval()
        x = torch.tensor(normalizer.transform(params.reshape(1, -1)), dtype=torch.float32)
        with torch.no_grad():
            out = self.forward(x).numpy().reshape(self.n_energy, 2)
        return out


class ParameterNormalizer:
    """Min-max normalize parameters to [0, 1] using PARAM_BOUNDS."""

    def __init__(self, bounds: dict):
        keys = list(bounds.keys())
        self.lo = np.array([bounds[k][0] for k in keys])
        self.hi = np.array([bounds[k][1] for k in keys])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.lo) / (self.hi - self.lo)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        return X_norm * (self.hi - self.lo) + self.lo


def load_emulator(checkpoint_path: str, device: str = "cpu") -> tuple:
    """Load a saved emulator + normalizer from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SpectralEmulator(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    normalizer = ckpt["normalizer"]
    return model, normalizer
