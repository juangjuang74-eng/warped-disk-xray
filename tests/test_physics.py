"""
tests/test_physics.py
Unit tests for physics utilities.
"""

import numpy as np
import pytest
from src.utils.physics import (
    gravitational_radius,
    isco_radius,
    outer_disk_inclination,
    polarization_angle_swing,
    ENERGY_BINS,
)


def test_gravitational_radius():
    rg = gravitational_radius(10.0)
    assert rg > 0
    assert 1e5 < rg < 1e7  # ~1.48e6 cm for 10 Msun BH


def test_isco_schwarzschild():
    """Non-spinning BH: ISCO should be at 6 r_g."""
    r = isco_radius(0.0)
    assert abs(r - 6.0) < 0.01


def test_isco_maximal_spin():
    """Near-maximally spinning BH: ISCO approaches 1 r_g."""
    r = isco_radius(0.998)
    assert r < 1.5


def test_outer_disk_inclination_aligned():
    """When beta=0, outer inclination equals inner inclination."""
    i_out = outer_disk_inclination(75.0, 0.0, 90.0)
    assert abs(i_out - 75.0) < 0.01


def test_outer_disk_inclination_paper_ob90():
    """Verify against Table 1 of the paper: Ob90 should give i_out=75.52."""
    i_out = outer_disk_inclination(75.0, 15.0, 90.0)
    assert abs(i_out - 75.52) < 0.5


def test_outer_disk_inclination_paper_ob180():
    """Ob180: i_out = 60 degrees."""
    i_out = outer_disk_inclination(75.0, 15.0, 180.0)
    assert abs(i_out - 60.0) < 0.5


def test_polarization_angle_swing():
    energies = ENERGY_BINS
    angles = np.linspace(100, 170, len(energies))
    swing = polarization_angle_swing(angles, energies)
    assert 60 < swing < 80


def test_energy_bins_monotonic():
    assert np.all(np.diff(ENERGY_BINS) > 0)
