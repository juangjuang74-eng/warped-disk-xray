"""
emulator/train.py
Gap 5: Training script for the surrogate emulator.

Usage
-----
    python src/emulator/train.py --config configs/emulator_config.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from tqdm import tqdm

from src.emulator.model import SpectralEmulator, ParameterNormalizer
from src.utils.physics import PARAM_BOUNDS
from src.utils.io import load_dataset_hdf5


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    X_raw, y_raw = load_dataset_hdf5(cfg["data_path"])
    # X_raw: (N, N_energy, 3) → we want pol_frac + pol_angle columns only
    targets = X_raw[:, :, 1:].reshape(len(X_raw), -1).astype(np.float32)  # (N, N_energy*2)
    params = y_raw.astype(np.float32)  # (N, 5)

    normalizer = ParameterNormalizer(PARAM_BOUNDS)
    params_norm = normalizer.transform(params).astype(np.float32)

    dataset = TensorDataset(
        torch.tensor(params_norm),
        torch.tensor(targets),
    )
    n_val = int(0.1 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=cfg.get("batch_size", 256), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512)

    # ── Model ──────────────────────────────────────────────────────────────────
    n_energy = X_raw.shape[1]
    model_kwargs = dict(
        n_params=params.shape[1],
        n_energy=n_energy,
        hidden_dims=tuple(cfg.get("hidden_dims", [256, 512, 512, 256])),
    )
    model = SpectralEmulator(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 200)
    )
    criterion = nn.MSELoss()

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    output_dir = Path(cfg.get("output_dir", "results/models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.get("epochs", 200)):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                pred = model(x_batch.to(device))
                val_losses.append(criterion(pred, y_batch.to(device)).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "normalizer": normalizer,
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                output_dir / "emulator_best.pt",
            )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_dir}/emulator_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/emulator_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
