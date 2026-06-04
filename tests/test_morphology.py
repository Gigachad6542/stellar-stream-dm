"""Tests for the sparse-spectroscopy cross-stream morphology diagnostic."""
from __future__ import annotations

import numpy as np

from src.forward_model.morphology import (
    component_morphology_summary,
    conditional_cross_track_js_score,
    reference_track_residuals,
    weighted_quantile,
)


def test_weighted_quantile_follows_dominant_weight():
    value = weighted_quantile(
        np.array([0.0, 10.0]),
        [0.5],
        np.array([100.0, 1.0]),
    )[0]
    assert value < 1.0


def test_reference_track_residuals_remove_track_slope():
    phi1 = np.linspace(-10.0, 10.0, 500)
    phi2 = 0.2 * phi1 + 1.0
    residual = reference_track_residuals(phi1, phi2, phi1, phi2, bin_width_deg=1.0)
    assert np.std(residual) < 0.02


def test_conditional_score_prefers_matching_cross_track_shape():
    rng = np.random.default_rng(4)
    phi1 = rng.uniform(-10.0, 10.0, 3000)
    observed = rng.normal(0.0, 0.2, len(phi1))
    match = observed.copy()
    broad = rng.normal(0.0, 1.0, len(phi1))
    score_match = conditional_cross_track_js_score(
        phi1, match, phi1, observed, (-10.0, 10.0)
    )
    score_broad = conditional_cross_track_js_score(
        phi1, broad, phi1, observed, (-10.0, 10.0)
    )
    assert score_match["active"]
    assert score_match["score"] < score_broad["score"]


def test_component_summary_reports_cocoon_offset():
    delta = np.array([0.0, 0.1, 1.0, 1.1])
    p_thin = np.array([1.0, 1.0, 0.0, 0.0])
    p_cocoon = np.array([0.0, 0.0, 1.0, 1.0])
    summary = component_morphology_summary(delta, p_thin, p_cocoon)
    assert summary["cocoon_fraction"] == 0.5
    assert summary["absolute_component_offset_deg"] > 0.8
