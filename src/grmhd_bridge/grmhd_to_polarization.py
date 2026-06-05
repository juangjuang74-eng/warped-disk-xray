"""
grmhd_bridge/grmhd_to_polarization.py
Gap 6: Pipeline bridging GRMHD simulation outputs to X-ray polarization predictions.

The paper uses a simplified broken-disk geometry instead of full GRMHD outputs.
This module provides a data pipeline to:
  1. Read GRMHD density/velocity fields (HDF5 format from HAMR, H-AMR, BHAC, etc.)
  2. Extract the effective disk warp geometry (r_bp, beta, phi as a function of time)
  3. Feed extracted geometry into the emulator to predict polarization
  4. Generate time-resolved polarization predictions for comparison with IXPE data

Supported GRMHD codes: HAMR (Liska et al. 2018), H-AMR, BHAC
(Extend by adding a new reader class following GRMHDReader interface.)

Usage
-----
    python src/grmhd_bridge/grmhd_to_polarization.py \
        --grmhd  data/raw/hamr_output.h5 \
        --emulator results/models/emulator_best.pt \
        --output results/reports/grmhd_polarization.h5
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from dataclasses import dataclass

from src.emulator.model import load_emulator
from src.utils.physics import ENERGY_BINS


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DiskGeometry:
    """Extracted warp geometry at a single simulation snapshot."""
    time: float          # simulation time [M]
    spin: float          # BH spin parameter
    r_bp: float          # break radius [r_g]
    beta: float          # misalignment angle [degrees]
    phi: float           # azimuthal orientation [degrees]
    inclination: float   # inner disk inclination [degrees]

    def to_array(self) -> np.ndarray:
        return np.array([self.spin, self.r_bp, self.beta, self.phi, self.inclination])


# ── GRMHD readers (abstract interface) ───────────────────────────────────────

class GRMHDReader(ABC):
    """Abstract base class for GRMHD simulation readers."""

    @abstractmethod
    def load(self, path: str) -> dict:
        """Load raw simulation data."""
        ...

    @abstractmethod
    def extract_geometry(self, data: dict, snapshot_idx: int) -> DiskGeometry:
        """Extract disk warp geometry from a simulation snapshot."""
        ...

    @abstractmethod
    def list_snapshots(self, data: dict) -> List[int]:
        """Return list of available snapshot indices."""
        ...


class HAMRReader(GRMHDReader):
    """
    Reader for HAMR (Liska et al. 2018) HDF5 outputs.
    Adjust field names to match your specific HAMR version.
    """

    def load(self, path: str) -> dict:
        with h5py.File(path, "r") as f:
            data = {
                "time":   f["time"][:],
                "rho":    f["rho"][:],     # (N_t, N_r, N_theta, N_phi)
                "v_r":    f["v1"][:],
                "v_th":   f["v2"][:],
                "v_ph":   f["v3"][:],
                "r_grid": f["r"][:],
                "th_grid":f["theta"][:],
                "spin":   float(f.attrs.get("spin", 0.9)),
            }
        return data

    def list_snapshots(self, data: dict) -> List[int]:
        return list(range(len(data["time"])))

    def extract_geometry(self, data: dict, snapshot_idx: int) -> DiskGeometry:
        """
        Estimate r_bp and beta from the density-weighted angular momentum tilt.

        This is a simplified estimator — a full treatment would follow
        Liska et al. (2019) Section 2 for tilt/twist profiles.
        """
        rho = data["rho"][snapshot_idx]         # (N_r, N_theta, N_phi)
        r   = data["r_grid"]
        th  = data["th_grid"]

        # Density-weighted midplane elevation per annulus
        sin_th = np.sin(th)[np.newaxis, :, np.newaxis]
        w = rho * sin_th
        th_weighted = np.sum(w * th[np.newaxis, :, np.newaxis], axis=(1, 2)) / (
            np.sum(w, axis=(1, 2)) + 1e-30
        )

        # Find break radius: where tilt changes most rapidly
        tilt = np.abs(th_weighted - np.pi / 2.0)
        dtilt = np.gradient(tilt, r)
        r_bp_idx = np.argmax(np.abs(dtilt))
        r_bp = float(r[r_bp_idx])

        # Beta: maximum tilt (degrees)
        beta = float(np.degrees(np.max(tilt)))

        # Azimuthal angle: density-weighted mean phi of the warp
        phi_grid = np.linspace(0, 2 * np.pi, rho.shape[2])
        phi_w = np.sum(rho[r_bp_idx] * phi_grid, axis=(0, 1)) / (
            np.sum(rho[r_bp_idx]) + 1e-30
        )
        phi_deg = float(np.degrees(phi_w)) % 360.0

        return DiskGeometry(
            time=float(data["time"][snapshot_idx]),
            spin=data["spin"],
            r_bp=r_bp,
            beta=beta,
            phi=phi_deg,
            inclination=75.0,   # set from observation / system geometry
        )


class MockGRMHDReader(GRMHDReader):
    """Mock reader for testing without real GRMHD data."""

    def load(self, path: str) -> dict:
        return {"time": np.linspace(0, 1000, 20), "spin": 0.9375}

    def list_snapshots(self, data: dict) -> List[int]:
        return list(range(len(data["time"])))

    def extract_geometry(self, data: dict, snapshot_idx: int) -> DiskGeometry:
        t = data["time"][snapshot_idx]
        return DiskGeometry(
            time=t,
            spin=0.9375,
            r_bp=5.0 + 2.0 * np.exp(-t / 500),
            beta=10.0 * np.exp(-t / 800),
            phi=90.0 + 5.0 * np.sin(t / 100),
            inclination=75.0,
        )


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_bridge_pipeline(
    grmhd_path: str,
    emulator_path: str,
    output_path: str,
    reader_type: str = "mock",
):
    # Select reader
    readers = {"hamr": HAMRReader, "mock": MockGRMHDReader}
    reader = readers.get(reader_type, MockGRMHDReader)()

    print(f"Loading GRMHD data from: {grmhd_path}")
    data = reader.load(grmhd_path)
    snapshots = reader.list_snapshots(data)
    print(f"Found {len(snapshots)} snapshots.")

    emulator, normalizer = load_emulator(emulator_path)

    times, geometries, polarizations = [], [], []

    for idx in snapshots:
        geom = reader.extract_geometry(data, idx)
        pred = emulator.predict_spectrum(geom.to_array(), normalizer)  # (N_energy, 2)

        times.append(geom.time)
        geometries.append(geom.to_array())
        polarizations.append(pred)

    times = np.array(times)
    geometries = np.array(geometries)
    polarizations = np.array(polarizations)   # (N_t, N_energy, 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("time",         data=times)
        f.create_dataset("geometry",     data=geometries)
        f.create_dataset("polarization", data=polarizations)
        f.create_dataset("energy_keV",   data=ENERGY_BINS)
        f.attrs["param_names"] = str(["spin", "r_bp", "beta", "phi", "inclination"])

    print(f"Saved time-resolved polarization predictions → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grmhd", default="data/raw/mock_grmhd.h5")
    parser.add_argument("--emulator", default="results/models/emulator_best.pt")
    parser.add_argument("--output", default="results/reports/grmhd_polarization.h5")
    parser.add_argument("--reader", choices=["hamr", "mock"], default="mock")
    args = parser.parse_args()

    run_bridge_pipeline(args.grmhd, args.emulator, args.output, reader_type=args.reader)


if __name__ == "__main__":
    main()
