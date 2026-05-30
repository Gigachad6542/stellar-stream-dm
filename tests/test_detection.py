"""Tests for the detection -> timeline handoff (src/forward_model/detection.py).

These cover the model-free path and the config-seeding logic, which need no
trained network. The learned GNN path is exercised only for graceful
degradation (missing checkpoint) so the suite does not depend on large weights.
"""

import json

import numpy as np
import pytest

from src.forward_model.detection import (
    DetectionResult,
    StreamImpactDetector,
    detect_impacts_modelfree,
    seed_config_from_detection,
)
from src.forward_model.pipeline import ForwardModelConfig


# ---------------------------------------------------------------------------
# Model-free detection
# ---------------------------------------------------------------------------

class TestModelFreeDetection:
    def _stream_with_gap(self, gap_center=-10.0, gap_halfwidth=2.0, n=8000, seed=0):
        rng = np.random.default_rng(seed)
        phi1 = rng.uniform(-50, 30, n)
        # Carve a clean gap
        phi1 = phi1[np.abs(phi1 - gap_center) > gap_halfwidth]
        return {"phi1": phi1, "membership_prob": np.ones_like(phi1)}

    def test_detects_clean_gap(self):
        obs = self._stream_with_gap(gap_center=-10.0)
        r = detect_impacts_modelfree(obs, (-50, 30), "GD1", significant_threshold=3.0)
        assert r.gap_detected
        assert r.impact_detected
        # The most significant gap should sit near the carved location.
        assert any(abs(loc - (-10.0)) < 3.0 for loc in r.gap_phi1_locations)

    def test_uniform_stream_no_significant_gap(self):
        rng = np.random.default_rng(1)
        obs = {"phi1": rng.uniform(0, 70, 6000)}
        r = detect_impacts_modelfree(obs, (0, 70), "GD1", significant_threshold=3.0)
        assert not r.gap_detected
        assert not r.impact_detected

    def test_fallback_positions_are_in_frame(self):
        rng = np.random.default_rng(2)
        obs = {"phi1": rng.uniform(5, 65, 6000)}
        r = detect_impacts_modelfree(obs, (5, 65), "GD1")
        # Quartile fallbacks must lie within the observed phi1 extent.
        assert len(r.fallback_phi1_values) == 3
        assert all(5 <= p <= 65 for p in r.fallback_phi1_values)
        # And be monotonically increasing (25 < 50 < 75 percentile).
        assert r.fallback_phi1_values == sorted(r.fallback_phi1_values)

    def test_result_is_json_serializable(self):
        obs = self._stream_with_gap()
        r = detect_impacts_modelfree(obs, (-50, 30), "GD1")
        # Should not raise.
        json.dumps(r.to_dict())


# ---------------------------------------------------------------------------
# Config seeding
# ---------------------------------------------------------------------------

class TestSeedConfig:
    def _base(self):
        return ForwardModelConfig(
            stream_name="GD1",
            t_since_range=(0.5, 10.0),
            impact_phi1_values=[-40.0],
        )

    def test_seeds_phi1_from_gaps(self):
        det = DetectionResult(
            stream_name="GD1", phi1_range=(0.0, 70.0),
            gap_phi1_locations=[12.0, 34.0, 56.0, 60.0],
            fallback_phi1_values=[17.0, 35.0, 52.0],
        )
        cfg = seed_config_from_detection(det, self._base(), max_phi1_seeds=3)
        assert cfg.impact_phi1_values == [12.0, 34.0, 56.0]

    def test_falls_back_to_quartiles_when_no_gaps(self):
        det = DetectionResult(
            stream_name="GD1", phi1_range=(0.0, 70.0),
            gap_phi1_locations=[],
            fallback_phi1_values=[17.0, 35.0, 52.0],
        )
        cfg = seed_config_from_detection(det, self._base())
        assert cfg.impact_phi1_values == [17.0, 35.0, 52.0]
        # Never the out-of-frame base default.
        assert -40.0 not in cfg.impact_phi1_values

    def test_narrows_time_window_around_estimate(self):
        det = DetectionResult(
            stream_name="GD1", phi1_range=(0.0, 70.0),
            t_since_gyr_estimate=7.0,
        )
        cfg = seed_config_from_detection(
            det, self._base(), t_window_gyr=2.0,
            min_t_since_gyr=0.5, max_t_since_gyr=10.0,
        )
        assert cfg.t_since_range == (5.0, 9.0)

    def test_time_window_clipped_to_bounds(self):
        det = DetectionResult(
            stream_name="GD1", phi1_range=(0.0, 70.0),
            t_since_gyr_estimate=9.5,
        )
        cfg = seed_config_from_detection(
            det, self._base(), t_window_gyr=2.0,
            min_t_since_gyr=0.5, max_t_since_gyr=10.0,
        )
        assert cfg.t_since_range[0] == pytest.approx(7.5)
        assert cfg.t_since_range[1] == pytest.approx(10.0)

    def test_keeps_time_range_when_no_estimate(self):
        det = DetectionResult(
            stream_name="GD1", phi1_range=(0.0, 70.0),
            t_since_gyr_estimate=None,
        )
        cfg = seed_config_from_detection(det, self._base())
        assert cfg.t_since_range == (0.5, 10.0)

    def test_does_not_mutate_base(self):
        base = self._base()
        det = DetectionResult(
            stream_name="GD1", phi1_range=(0.0, 70.0),
            gap_phi1_locations=[12.0], t_since_gyr_estimate=5.0,
        )
        _ = seed_config_from_detection(det, base)
        assert base.impact_phi1_values == [-40.0]
        assert base.t_since_range == (0.5, 10.0)


# ---------------------------------------------------------------------------
# Learned detector: graceful degradation
# ---------------------------------------------------------------------------

class TestInputAccuracy:
    """The fix for GNN over-confidence: use real errors, pass NaN through for
    unmeasured features so the detector can impute them to the training mean."""

    def test_obs_to_data_uses_real_errors(self):
        from src.forward_model.gnn_scorer import obs_particles_to_data
        n = 50
        obs = {
            "phi1": np.linspace(0, 50, n), "phi2": np.zeros(n),
            "pm1": np.zeros(n), "pm2": np.zeros(n), "dist": np.full(n, 8.0),
            "vrad": np.full(n, np.nan),               # unmeasured RV
            "e_dist": np.full(n, 0.7), "e_pm1": np.full(n, 0.42),
            "e_pm2": np.full(n, 0.43), "e_vrad": np.full(n, np.nan),
        }
        data = obs_particles_to_data(obs, "GD1")
        x = data.x.numpy()
        # Columns: [phi1,phi2,dist,pm1,pm2,vrad,e_dist,e_pm1,e_pm2,e_vrad,mem,onehot]
        assert np.allclose(x[:, 6], 0.7)     # real e_dist used, not the 0.3 default
        assert np.allclose(x[:, 7], 0.42)    # real e_pm1 used, not the 0.1 default
        assert np.isnan(x[:, 5]).all()       # unmeasured vrad passed through as NaN
        assert np.isnan(x[:, 9]).all()       # unmeasured e_vrad passed through as NaN

    def test_obs_to_data_falls_back_to_defaults(self):
        from src.forward_model.gnn_scorer import obs_particles_to_data
        n = 30
        obs = {"phi1": np.linspace(0, 30, n), "phi2": np.zeros(n),
               "pm1": np.zeros(n), "pm2": np.zeros(n), "dist": np.full(n, 8.0),
               "vrad": np.zeros(n)}
        data = obs_particles_to_data(obs, "GD1")
        x = data.x.numpy()
        assert np.allclose(x[:, 6], 0.3)     # default e_dist when not provided
        assert np.allclose(x[:, 7], 0.1)     # default e_pm1 when not provided


class TestDetectorDegradation:
    def test_missing_checkpoint_is_unavailable(self):
        det = StreamImpactDetector("checkpoints/does_not_exist.pt", device="cpu")
        assert not det.is_available

    def test_detect_returns_modelfree_when_gnn_unavailable(self):
        det = StreamImpactDetector("checkpoints/does_not_exist.pt", device="cpu")
        rng = np.random.default_rng(3)
        phi1 = rng.uniform(-50, 30, 8000)
        phi1 = phi1[np.abs(phi1 - (-10.0)) > 2.0]
        obs = {"phi1": phi1, "membership_prob": np.ones_like(phi1)}
        r = det.detect(obs, (-50, 30), stream_name="GD1", significant_threshold=3.0)
        assert not r.gnn_available
        assert r.gap_detected  # model-free still works
        assert r.p_impact is None
