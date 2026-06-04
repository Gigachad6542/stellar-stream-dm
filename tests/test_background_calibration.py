"""Tests for smooth-stream background calibration helpers."""

import pytest

from scripts.run_gd1_background_calibration import subtract_windows
from scripts.run_gd1_background_sensitivity_shortlist import (
    add_event_target_summary,
    aggregate_seed_metrics,
    deduplicate_backgrounds,
)
from src.forward_model.evolve import resolve_spray_model_parameters


def test_subtract_windows_holds_out_both_candidate_regions():
    result = subtract_windows((0.0, 80.0), [(20.0, 30.0), (45.0, 55.0)])
    assert result == [(0.0, 20.0), (30.0, 45.0), (55.0, 80.0)]


def test_resolve_spray_model_parameters_preserves_defaults_and_overrides():
    config = {
        "disruption_age_gyr": 3.0,
        "prog_mass_solar": 5000.0,
    }
    default = resolve_spray_model_parameters(config)
    assert default.stream_age_gyr == 3.0
    assert default.progenitor_mass_msun == 5000.0
    assert default.velocity_dispersion_factor == 0.3

    override = resolve_spray_model_parameters(
        config,
        stream_age_gyr=6.0,
        progenitor_mass_msun=10000.0,
        velocity_dispersion_factor=0.15,
    )
    assert override.stream_age_gyr == 6.0
    assert override.progenitor_mass_msun == 10000.0
    assert override.velocity_dispersion_factor == 0.15


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stream_age_gyr": 0.0},
        {"progenitor_mass_msun": -1.0},
        {"velocity_dispersion_factor": -0.1},
    ],
)
def test_resolve_spray_model_parameters_rejects_unphysical_values(kwargs):
    with pytest.raises(ValueError):
        resolve_spray_model_parameters({}, **kwargs)


def test_deduplicate_backgrounds_uses_physical_parameters_not_labels():
    backgrounds = [
        {
            "name": "a",
            "stream_age_gyr": 9.0,
            "progenitor_mass_msun": 5000.0,
            "velocity_dispersion_factor": 0.3,
        },
        {
            "name": "b",
            "stream_age_gyr": 9.0,
            "progenitor_mass_msun": 5000.0,
            "velocity_dispersion_factor": 0.3,
        },
    ]
    assert deduplicate_backgrounds(backgrounds) == [backgrounds[0]]


def test_aggregate_seed_metrics_requires_all_four_diagnostics():
    def result(value):
        return {
            "metrics": {
                "targets": {
                    "30.7": {
                        "primary_relative_improvement": value,
                        "morphology_relative_improvement": value,
                    },
                    "50.7": {
                        "primary_relative_improvement": value,
                        "morphology_relative_improvement": value,
                    },
                },
                "all_four_improve": value > 0,
            }
        }

    positive = aggregate_seed_metrics([result(0.1), result(0.2)])
    assert positive["all_four_mean_material_5pct"]
    assert positive["all_seeds_all_four_improve"]

    mixed = aggregate_seed_metrics([result(0.1), result(-0.2)])
    assert not mixed["all_four_mean_improve"]


def test_event_target_summary_does_not_require_unrelated_target_to_improve():
    aggregate = {
        "targets": {
            "30.7": {
                "primary_relative_improvement_mean": 0.1,
                "morphology_relative_improvement_mean": 0.2,
                "primary_positive_fraction": 1.0,
                "morphology_positive_fraction": 1.0,
            },
            "50.7": {
                "primary_relative_improvement_mean": -0.4,
                "morphology_relative_improvement_mean": -0.2,
                "primary_positive_fraction": 0.0,
                "morphology_positive_fraction": 0.0,
            },
        }
    }
    result = add_event_target_summary(
        aggregate,
        {"target_label": "canonical_gap_1"},
    )
    assert result["event_target_both_mean_material_5pct"]
    assert result["event_target_both_positive_all_seeds"]
    assert result["other_target_worst_mean_improvement"] == -0.4
