"""
classifier/train_classifier.py
Gap 3: Binary classifier — warped disk vs flat (aligned/misaligned) disk.

This directly addresses Figure 9 in the paper, where the warped disk's
polarization angle swing is indistinguishable from a flat disk by eye.
The classifier quantifies the SNR needed to reliably detect disk warping.

Labels:
  0 = flat (aligned or misaligned, but unwarped)
  1 = warped (Bardeen-Petterson configuration)

Usage
-----
    python src/classifier/train_classifier.py --data data/simulated/classifier_dataset.h5
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

from src.utils.io import load_dataset_hdf5


class DiskClassifier(nn.Module):
    """
    1D-CNN classifier operating on the polarization spectrum.
    Input: (batch, 2, N_energy) — pol_fraction and pol_angle as two channels.
    Output: logit for warped vs flat.
    """

    def __init__(self, n_energy: int = 50):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h).squeeze(-1)


def add_noise(spectra: np.ndarray, snr: float = 20.0) -> np.ndarray:
    """Add Gaussian noise to simulate observational uncertainty."""
    noisy = spectra.copy()
    noisy[:, :, 1] += np.random.normal(0, 100.0 / snr, noisy[:, :, 1].shape)
    noisy[:, :, 2] += np.random.normal(0, 90.0 / snr, noisy[:, :, 2].shape)
    return noisy


def train_classifier(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load warped and flat datasets
    X_warped, _ = load_dataset_hdf5(cfg["warped_data"])
    X_flat, _ = load_dataset_hdf5(cfg["flat_data"])

    snr = cfg.get("snr", 20.0)
    X_warped = add_noise(X_warped, snr)
    X_flat = add_noise(X_flat, snr)

    # Build tensors: use pol_fraction and pol_angle as input channels
    def to_tensor(X, label):
        # X: (N, N_energy, 3) → take cols 1,2 → (N, 2, N_energy)
        t = torch.tensor(X[:, :, 1:].transpose(0, 2, 1), dtype=torch.float32)
        y = torch.full((len(X),), label, dtype=torch.float32)
        return t, y

    Xw, yw = to_tensor(X_warped, 1)
    Xf, yf = to_tensor(X_flat, 0)

    X_all = torch.cat([Xw, Xf])
    y_all = torch.cat([yw, yf])

    dataset = TensorDataset(X_all, y_all)
    n_val = int(0.15 * len(dataset))
    n_test = int(0.15 * len(dataset))
    n_train = len(dataset) - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    model = DiskClassifier(n_energy=X_all.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    output_dir = Path(cfg.get("output_dir", "results/models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(cfg.get("epochs", 100)):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        val_loss = np.mean([
            criterion(model(xb.to(device)), yb.to(device)).item()
            for xb, yb in val_loader
        ])
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / "classifier_best.pt")

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | val_loss={val_loss:.4f}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    model.load_state_dict(torch.load(output_dir / "classifier_best.pt"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device)).cpu().numpy()
            all_preds.extend(logits)
            all_labels.extend(yb.numpy())

    probs = torch.sigmoid(torch.tensor(all_preds)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = np.array(all_labels).astype(int)

    print("\nTest set results:")
    print(classification_report(labels, preds, target_names=["flat", "warped"]))
    print(f"ROC-AUC: {roc_auc_score(labels, probs):.4f}")

    report = {
        "snr": snr,
        "roc_auc": float(roc_auc_score(labels, probs)),
        "classification_report": classification_report(labels, preds, output_dict=True),
    }
    with open(output_dir / "classifier_report.json", "w") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warped_data", default="data/simulated/sweep_dataset.h5")
    parser.add_argument("--flat_data", default="data/simulated/flat_dataset.h5")
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    cfg = {
        "warped_data": args.warped_data,
        "flat_data": args.flat_data,
        "snr": args.snr,
        "epochs": args.epochs,
    }
    train_classifier(cfg)


if __name__ == "__main__":
    main()
