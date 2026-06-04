"""Tests for multi-epoch fusion and the radial-velocity scoring term.

Network fetchers (Gaia/S5) are not exercised here; only the pure fusion math
and the RV score, which run offline.
"""

import numpy as np
import pytest

from src.data.multi_epoch import (
    GAIA_DR_BASELINES_YR,
    inverse_variance_combine,
    pm_error_scaling_factor,
    fuse_radial_velocities,
)
from src.data.radial_velocity import RadialVelocityData
from src.forward_model.scoring import radial_velocity_score


class TestInverseVariance:
    def test_combine_reduces_error(self):
        # Two equal measurements with equal errors -> error shrinks by sqrt(2).
        v = [np.array([10.0]), np.array([10.0])]
        e = [np.array([2.0]), np.array([2.0])]
        fused, err = inverse_variance_combine(v, e)
        assert fused[0] == pytest.approx(10.0)
        assert err[0] == pytest.approx(2.0 / np.sqrt(2))

    def test_weighting_favours_precise_measurement(self):
        v = [np.array([0.0]), np.array([10.0])]
        e = [np.array([0.1]), np.array([5.0])]   # first is far more precise
        fused, _ = inverse_variance_combine(v, e)
        assert fused[0] < 1.0  # pulled toward the precise value

    def test_missing_handled(self):
        v = [np.array([np.nan, 5.0]), np.array([3.0, np.nan])]
        e = [np.array([np.nan, 1.0]), np.array([1.0, np.nan])]
        fused, err = inverse_variance_combine(v, e)
        assert fused[0] == pytest.approx(3.0)  # only catalog 2 measured star 0
        assert fused[1] == pytest.approx(5.0)  # only catalog 1 measured star 1

    def test_all_missing_is_nan(self):
        v = [np.array([np.nan])]
        e = [np.array([np.nan])]
        fused, err = inverse_variance_combine(v, e)
        assert np.isnan(fused[0]) and np.isnan(err[0])


class TestPMScaling:
    def test_dr4_tighter_than_dr3(self):
        assert pm_error_scaling_factor("DR4") < 1.0
        assert pm_error_scaling_factor("DR5") < pm_error_scaling_factor("DR4")

    def test_dr3_vs_itself_is_one(self):
        assert pm_error_scaling_factor("DR3") == pytest.approx(1.0)

    def test_baselines_increasing(self):
        assert GAIA_DR_BASELINES_YR["DR2"] < GAIA_DR_BASELINES_YR["DR3"] < GAIA_DR_BASELINES_YR["DR4"]


class TestFuseRadialVelocities:
    def test_alignment_and_mask(self):
        member_ids = np.array([100, 200, 300, 400], dtype=np.int64)
        cat = RadialVelocityData(
            source_ids=np.array([200, 400]),
            rv_km_s=np.array([-50.0, 75.0]),
            rv_error_km_s=np.array([1.0, 2.0]),
            survey="S5", snr=np.array([np.nan, np.nan]),
        )
        rv, e_rv, mask, per = fuse_radial_velocities(member_ids, [cat])
        assert mask.tolist() == [0.0, 1.0, 0.0, 1.0]
        assert rv[1] == pytest.approx(-50.0)
        assert rv[3] == pytest.approx(75.0)
        assert np.isnan(rv[0]) and np.isnan(rv[2])
        assert per == {"S5": 2}

    def test_two_surveys_combine(self):
        member_ids = np.array([1, 2], dtype=np.int64)
        c1 = RadialVelocityData(np.array([1]), np.array([10.0]), np.array([1.0]),
                                "Gaia_RVS", np.array([np.nan]))
        c2 = RadialVelocityData(np.array([1]), np.array([12.0]), np.array([1.0]),
                                "S5", np.array([np.nan]))
        rv, e_rv, mask, per = fuse_radial_velocities(member_ids, [c1, c2])
        assert rv[0] == pytest.approx(11.0)              # inverse-variance mean
        assert e_rv[0] == pytest.approx(1.0 / np.sqrt(2))
        assert mask[1] == 0.0
        assert per == {"Gaia_RVS": 1, "S5": 1}


class TestRadialVelocityScore:
    def test_inactive_without_obs_rv(self):
        phi1 = np.linspace(0, 50, 400)
        s, active = radial_velocity_score(
            phi1, np.zeros_like(phi1), phi1, np.full_like(phi1, np.nan), (0, 50),
        )
        assert not active and s == 0.0

    def test_matching_tracks_low_score(self):
        rng = np.random.default_rng(0)
        phi1 = rng.uniform(0, 50, 2000)
        vrad = 100.0 + 0.5 * phi1
        s_match, active = radial_velocity_score(phi1, vrad, phi1, vrad, (0, 50))
        assert active
        assert s_match < 1.0

    def test_constant_offset_insensitive(self):
        # A constant LOS zero-point offset is a frame/convention nuisance and
        # must NOT inflate the score (it is removed before the RMS).
        rng = np.random.default_rng(1)
        phi1 = rng.uniform(0, 50, 2000)
        vrad = 100.0 + 0.5 * phi1
        s_match, _ = radial_velocity_score(phi1, vrad, phi1, vrad, (0, 50))
        s_off, _ = radial_velocity_score(phi1, vrad + 50.0, phi1, vrad, (0, 50))
        assert abs(s_off - s_match) < 1e-6

    def test_differential_track_higher_score(self):
        # A differential (slope) change IS a real perturbation and must raise it.
        rng = np.random.default_rng(1)
        phi1 = rng.uniform(0, 50, 4000)
        vrad = 100.0 + 0.5 * phi1
        s_match, _ = radial_velocity_score(phi1, vrad, phi1, vrad, (0, 50))
        s_slope, _ = radial_velocity_score(phi1, 100.0 + 1.5 * phi1, phi1, vrad, (0, 50))
        assert s_slope > s_match

    def test_uncertain_outlier_is_downweighted(self):
        rng = np.random.default_rng(5)
        phi1 = rng.uniform(0, 50, 4000)
        vrad = 100.0 + 0.5 * phi1
        obs_vrad = vrad.copy()
        obs_error = np.full_like(vrad, 1.0)
        outlier = (phi1 >= 20.0) & (phi1 < 24.0)
        obs_vrad[outlier] += 30.0
        obs_error[outlier] = 100.0

        unweighted, _ = radial_velocity_score(phi1, vrad, phi1, obs_vrad, (0, 50))
        weighted, _ = radial_velocity_score(
            phi1,
            vrad,
            phi1,
            obs_vrad,
            (0, 50),
            obs_vrad_error=obs_error,
        )
        assert weighted < unweighted

    def test_membership_weights_select_thin_component(self):
        rng = np.random.default_rng(8)
        phi1 = rng.uniform(0, 50, 4000)
        thin = 100.0 + 0.5 * phi1
        obs_vrad = thin.copy()
        cocoon = (phi1 >= 20.0) & (phi1 < 24.0)
        obs_vrad[cocoon] += 25.0
        thin_probability = np.ones_like(phi1)
        thin_probability[cocoon] = 0.01

        unweighted, _ = radial_velocity_score(phi1, thin, phi1, obs_vrad, (0, 50))
        weighted, _ = radial_velocity_score(
            phi1,
            thin,
            phi1,
            obs_vrad,
            (0, 50),
            obs_vrad_weight=thin_probability,
        )
        assert weighted < unweighted
