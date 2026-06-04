"""Tests for the constrained GD-1 two-perturbation shortlist."""
from scripts.run_gd1_two_perturbation_shortlist import (
    assess_complexity,
    select_target_representatives,
)


def _row(primary, morphology, balanced, mass):
    return {
        "impact_phi1": 30.7,
        "scale_radius_factor": 1.0,
        "scale_radius_kpc": 0.2,
        "log10_mass": mass,
        "t_since_gyr": 1.0,
        "impact_param_kpc": 0.1,
        "flyby_vel_kms": 200.0,
        "target_label": "canonical_gap_1",
        "compactness_family": "NFW-like",
        "primary_relative_improvement": primary,
        "morphology_relative_improvement": morphology,
        "balanced_relative_improvement": balanced,
    }


def test_select_target_representatives_keeps_three_distinct_objectives():
    rows = [
        _row(0.5, 0.0, 0.0, 7.0),
        _row(0.0, 0.5, 0.0, 8.0),
        _row(0.2, 0.2, 0.2, 9.0),
    ]
    selected = select_target_representatives(rows, 30.7)
    assert {row["role"] for row in selected} == {
        "best_fast_primary",
        "best_fast_morphology",
        "best_fast_balanced",
    }


def test_complexity_guard_requires_material_four_way_improvement():
    pair = {
        "metrics": {
            "balanced_relative_improvement": 0.08,
            "mean_relative_improvement": 0.10,
            "all_four_material_5pct": True,
        }
    }
    singles = [
        {
            "metrics": {
                "balanced_relative_improvement": 0.04,
                "mean_relative_improvement": 0.06,
            }
        }
    ]
    result = assess_complexity(pair, singles)
    assert result["passes_complexity_guard"]

    pair["metrics"]["all_four_material_5pct"] = False
    assert not assess_complexity(pair, singles)["passes_complexity_guard"]
