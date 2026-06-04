"""Interpretation guardrails for the continuous density-profile grid."""
from __future__ import annotations

from scripts.run_density_profile_grid import selection_robustness


def _result(selection: str, family: str = "NFW-like") -> dict:
    return {
        "selection": selection,
        "best_candidate": {
            "impact_phi1": 30.7,
            "compactness_family": family,
            "scale_radius_factor": 1.0,
            "better_than_null": True,
        },
        "target_detected_features": {"30.7": [{"phi1_center": 30.7}]},
    }


def test_single_selection_cannot_be_called_selection_robust():
    summary = selection_robustness([_result("catalog_native")])

    assert not summary["robust_screening_preference"]
    assert not summary["has_selection_comparison"]
    assert summary["n_selections_compared"] == 1


def test_matching_supported_selections_can_pass_screening_guard():
    summary = selection_robustness(
        [_result("catalog_native"), _result("legacy_narrow")]
    )

    assert summary["robust_screening_preference"]
    assert summary["has_selection_comparison"]
