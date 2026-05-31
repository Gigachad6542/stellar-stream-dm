"""Tests for the GD-1 PWB18(Koposov) <-> I21 frame transform.

The pure-rotation parts run offline; the galstreams-dependent mapping is marked
slow because it initialises the galstreams library.
"""

import numpy as np
import pytest

from src.data.gd1_frames import (
    R_ICRS_TO_GD1,
    koposov_phi_to_icrs,
    pwb18_phi1_to_i21,
    transform_known_gaps_to_i21,
)


class TestKoposovRotation:
    def test_rotation_is_orthonormal(self):
        # A valid rotation matrix: R R^T = I, det = +1.
        assert np.allclose(R_ICRS_TO_GD1 @ R_ICRS_TO_GD1.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R_ICRS_TO_GD1), 1.0, atol=1e-6)

    def test_phi_to_icrs_on_sky(self):
        ra, dec = koposov_phi_to_icrs(-40.0, 0.0)
        assert 0.0 <= float(ra) < 360.0
        assert -90.0 <= float(dec) <= 90.0
        # GD-1 near phi1=-40 lies in the northern sky (ra~150-185, dec~30-45).
        assert 140.0 < float(ra) < 200.0
        assert 20.0 < float(dec) < 60.0

    def test_array_input(self):
        ra, dec = koposov_phi_to_icrs(np.array([-40.0, -20.0, 0.0]))
        assert ra.shape == (3,) and dec.shape == (3,)


@pytest.mark.slow
class TestFrameMapping:
    def test_known_gap_maps_near_data_minimum(self):
        # PWB18 gap_1 at phi1=-40 should map into the I21 data range and sit near
        # the data-driven deepest density minimum (~36).
        phi1_i21 = pwb18_phi1_to_i21(-40.0)
        assert 25.0 < phi1_i21 < 40.0

    def test_monotonic_and_in_range(self):
        pts = np.array([-60.0, -40.0, -20.0, 0.0])
        mapped = np.array([pwb18_phi1_to_i21(p) for p in pts])
        # Monotonic increasing and within the I21 data extent [0.6, 72.8].
        assert np.all(np.diff(mapped) > 0)
        assert mapped.min() > 0.0 and mapped.max() < 85.0

    def test_transform_known_gaps_preserves_pwb18(self):
        gaps = [{"phi1_center": -40.0, "phi1_width": 3.0, "label": "gap_1"}]
        out = transform_known_gaps_to_i21(gaps)
        assert out[0]["phi1_center_pwb18"] == -40.0
        assert 25.0 < out[0]["phi1_center"] < 40.0
        assert out[0]["label"] == "gap_1"
