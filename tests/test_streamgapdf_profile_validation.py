"""Tests for the matched two-arm streamgapdf validation contract."""
from argparse import Namespace

import numpy as np

from scripts.run_gd1_streamgapdf_injection_recovery import (
    _neighborhood_tolerance,
    _truth_neighborhood,
    candidate_grid_for_truth,
    fixed_truth_cases,
    impact_params,
    summarize_rows as summarize_injection_rows,
)
from scripts.run_gd1_streamgapdf_null_fpr import summarize_rows as summarize_null_rows
from src.forward_model.profile_validation import (
    PROFILE_FAMILY_NAMES,
    detection_passes,
    profile_family,
    shard_output_path,
    targeting_mask,
)


def _args():
    return Namespace(
        log10_masses=[7.5, 8.0, 8.5],
        times=[0.5, 1.0],
        impact_params=[0.05, 0.2],
        scale_factors=[0.5, 1.0, 3.0, 10.0],
        fixed_impact_angle=0.65,
        flyby_velocity=150.0,
    )


def test_fixed_truth_grid_has_predeclared_48_cells():
    rows = fixed_truth_cases(_args())
    assert len(rows) == 48
    assert len({row["case_id"] for row in rows}) == 48


def test_candidate_grids_hold_or_expand_geometry_as_declared():
    truth = impact_params(8.0, 1.0, 0.65, 0.05, 3.0, 150.0)
    fixed = candidate_grid_for_truth(
        truth,
        "fixed",
        [0.5, 1.0, 3.0, 10.0],
        [-0.15, 0.0, 0.15],
        [-0.5, 0.0, 0.5],
    )
    nuisance = candidate_grid_for_truth(
        truth,
        "nuisance",
        [0.5, 1.0, 3.0, 10.0],
        [-0.15, 0.0, 0.15],
        [-0.5, 0.0, 0.5],
    )
    assert len(fixed) == 4
    assert len(nuisance) == 36
    assert all(row["impact_angle_rad"] == truth["impact_angle_rad"] for row in fixed)


def test_truth_neighborhood_does_not_credit_entire_nuisance_grid():
    truth = impact_params(8.0, 1.0, 0.65, 0.05, 3.0, 150.0)
    exact = impact_params(8.0, 1.0, 0.65, 0.05, 3.0, 150.0)
    adjacent = impact_params(8.0, 1.5, 0.8, 0.05, 3.0, 150.0)
    angle_tolerance = _neighborhood_tolerance([-0.15, 0.0, 0.15])
    time_tolerance = _neighborhood_tolerance([-0.5, 0.0, 0.5])
    assert _truth_neighborhood(exact, truth, angle_tolerance, time_tolerance)
    assert not _truth_neighborhood(adjacent, truth, angle_tolerance, time_tolerance)


def test_profile_family_uses_nearest_predeclared_log_scale():
    assert tuple(profile_family(value) for value in (0.5, 1.0, 3.0, 10.0)) == (
        PROFILE_FAMILY_NAMES
    )
    assert profile_family(0.6) == "compact_0p5x"
    assert profile_family(8.0) == "very_cored_10x"


def test_detection_contract_requires_both_diagnostic_families():
    assert detection_passes(
        {
            "primary_relative_improvement": 0.06,
            "morphology_relative_improvement": 0.07,
        }
    )
    assert not detection_passes(
        {
            "primary_relative_improvement": 0.20,
            "morphology_relative_improvement": 0.04,
        }
    )


def test_targeting_mask_is_deterministic_and_keeps_sparse_support():
    phi1 = np.linspace(0.0, 80.0, 200)
    template = np.concatenate([np.linspace(10.0, 20.0, 40), np.linspace(50.0, 60.0, 80)])
    first = targeting_mask(phi1, template, seed=42)
    second = targeting_mask(phi1, template, seed=42)
    np.testing.assert_array_equal(first, second)
    assert first.sum() >= 20
    assert first.sum() < len(phi1)


def test_injection_and_null_summaries_apply_frozen_gates():
    injection_rows = [
        {
            "truth": {"case_id": "a"},
            "profile_family_recovered": True,
            "recovered_profile_family": "nfw_like_1x",
            "profile_scale_log10_bias": 0.0,
            "detected": True,
            "truth_neighborhood_top10": True,
        },
        {
            "truth": {"case_id": "a"},
            "profile_family_recovered": True,
            "recovered_profile_family": "nfw_like_1x",
            "profile_scale_log10_bias": 0.0,
            "detected": True,
            "truth_neighborhood_top10": True,
        },
    ]
    injection = summarize_injection_rows(injection_rows, "nuisance", 0.7, 0.8, 0.6, 0.8)
    assert injection["gates"]["all_applicable_pass"]

    null_rows = [
        {"anchor": {"case_id": "a"}, "false_positive": False},
        {"anchor": {"case_id": "a"}, "false_positive": False},
    ]
    null = summarize_null_rows(null_rows, 0.05, 0.8)
    assert null["gates"]["all_applicable_pass"]


def test_shard_output_path_prevents_overwrite():
    path = shard_output_path("outputs/profile_validation/result.json", 2, 4)
    assert path.name == "result.shard-02-of-04.json"
