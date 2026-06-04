"""Tests for mask-aware PWB18 GD-1 ingestion."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from src.data.pwb18 import (
    _track_selection_weights,
    compute_pwb18_selection_profile,
    load_pwb18_selection,
)


@pytest.fixture
def masked_fits(tmp_path):
    table = Table()
    table["source_id"] = np.arange(6, dtype=np.int64)
    table["ra"] = np.linspace(140.0, 145.0, 6)
    table["dec"] = np.linspace(30.0, 35.0, 6)
    table["pm_ra_cosdec"] = np.linspace(-8.0, -7.0, 6)
    table["pm_dec"] = np.linspace(-2.0, -1.0, 6)
    table["phi1"] = np.linspace(-50.0, 0.0, 6)
    table["phi2"] = np.linspace(-1.0, 1.0, 6)
    table["pm_mask"] = [True, True, True, True, False, True]
    table["gi_cmd_mask"] = [True, True, False, True, True, True]
    table["stream_track_mask"] = [True, False, True, True, True, True]
    path = tmp_path / "pwb18.fits"
    table.write(path)
    return path


def test_track_selection_applies_all_three_masks(masked_fits):
    table = load_pwb18_selection(masked_fits, selection="track")
    assert table["source_id"].tolist() == [0, 3, 5]
    assert table.meta["selection_expression"] == "pm_mask & gi_cmd_mask & stream_track_mask"
    assert table["stream_track_mask"].tolist() == [True, True, True]


def test_pmcmd_selection_does_not_require_track_mask(masked_fits):
    table = load_pwb18_selection(masked_fits, selection="pmcmd")
    assert table["source_id"].tolist() == [0, 1, 3, 5]
    assert table["stream_track_mask"].tolist() == [True, False, True, True]


def test_pm_aliases_are_standardized(masked_fits):
    table = load_pwb18_selection(masked_fits, selection="track")
    assert "pmra" in table.colnames
    assert "pmdec" in table.colnames
    assert np.allclose(table["pmra"], [-8.0, -7.4, -7.0])


def test_unknown_selection_rejected(masked_fits):
    with pytest.raises(ValueError, match="Unknown PWB18 selection"):
        load_pwb18_selection(masked_fits, selection="made_up")


def test_selection_profile_accounts_for_track_and_offtrack_counts(masked_fits):
    profile = compute_pwb18_selection_profile(
        masked_fits, bin_width_deg=10.0, smooth_sigma_bins=0.0
    )

    assert profile["pmcmd_track_counts"].sum() == 3
    assert profile["pmcmd_offtrack_counts"].sum() == 1
    assert np.all((profile["main_track_signal_fraction"] >= 0.0))
    assert np.all((profile["main_track_signal_fraction"] <= 1.0))


def test_track_weights_map_binned_signal_fraction_to_stars():
    table = Table({"phi1_pwb18": [-1.0, 1.0, 12.0]})
    profile = {
        "bin_edges_pwb18": np.array([-10.0, 0.0, 10.0, 20.0]),
        "main_track_signal_fraction": np.array([0.2, 0.5, 0.9]),
    }

    weights = _track_selection_weights(table, profile)

    assert np.allclose(weights, [0.2, 0.5, 0.9])
