"""Tests for the matched streamgapdf profile-screen helpers."""

from argparse import Namespace

import numpy as np

from scripts.run_gd1_streamgapdf_profile_screen import (
    candidate_grid,
    strongest_density_deficit,
)


def test_candidate_grid_separates_mass_and_scale_radius():
    args = Namespace(
        log10_masses=[7.0],
        times=[0.5],
        impact_angles=[0.3],
        scale_factors=[0.5, 2.0],
        impact_param=0.1,
        flyby_velocity=150.0,
    )
    rows = candidate_grid(args)
    assert len(rows) == 2
    assert rows[1]["scale_radius_kpc"] == 4.0 * rows[0]["scale_radius_kpc"]


def test_strongest_density_deficit_recovers_removed_region():
    control = np.repeat(np.arange(0.5, 10.0, 1.0), 20)
    candidate = control[(control < 4.0) | (control > 6.0)]
    result = strongest_density_deficit(candidate, control, (0.0, 10.0), 1.0)
    assert 4.0 <= result["phi1"] <= 6.0
    assert result["fractional_density_deficit"] > 0
