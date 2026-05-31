"""Tests for the Erkal & Belokurov 2015 Plummer velocity kick."""

import numpy as np
import pytest

from src.simulation.subhalo import (
    G_KPC_KMS,
    build_encounter_geometry,
    erkal_plummer_kick,
)


class TestErkalPlummerKick:
    def test_point_mass_limit_matches_2GM_over_bw(self):
        # r_s -> 0, star at the stream impact point x_impact (the subhalo passes
        # at distance b in +z): |p| = b, |dv| = 2GM/(b w), directed +z (toward
        # the subhalo's path).
        M, w, b, rs = 1e8, 200.0, 0.5, 1e-4
        x_impact = np.array([10.0, 0.0, 0.0])
        w_vec = np.array([0.0, w, 0.0])          # subhalo moving in +y
        b_vec = np.array([0.0, 0.0, b])          # subhalo path offset +z (perp to w)
        star = x_impact[None, :]                 # star at the stream impact point
        dv = erkal_plummer_kick(star, x_impact, w_vec, b_vec, M, rs)
        expected = 2.0 * G_KPC_KMS * M / (b * w)
        assert np.linalg.norm(dv[0]) == pytest.approx(expected, rel=1e-3)
        # direction is toward the subhalo path (+z)
        assert dv[0, 2] > 0 and abs(dv[0, 0]) < 1e-6 and abs(dv[0, 1]) < 1e-6

    def test_scales_linearly_with_mass(self):
        x0 = np.array([10.0, 0.0, 0.0]); w = np.array([0, 200.0, 0]); b = np.array([0, 0, 0.5])
        star = np.array([[10.0, 0.0, 0.5]])
        dv1 = erkal_plummer_kick(star, x0, w, b, 1e7, 0.3)
        dv2 = erkal_plummer_kick(star, x0, w, b, 2e7, 0.3)
        assert np.linalg.norm(dv2) == pytest.approx(2 * np.linalg.norm(dv1), rel=1e-6)

    def test_inverse_with_velocity(self):
        x0 = np.array([10.0, 0.0, 0.0]); b = np.array([0, 0, 0.5]); star = np.array([[10.0, 0, 0.5]])
        dv_slow = erkal_plummer_kick(star, x0, np.array([0, 100.0, 0]), b, 1e8, 0.3)
        dv_fast = erkal_plummer_kick(star, x0, np.array([0, 200.0, 0]), b, 1e8, 0.3)
        assert np.linalg.norm(dv_slow) == pytest.approx(2 * np.linalg.norm(dv_fast), rel=1e-6)

    def test_bounded_no_divergence_at_zero_impact(self):
        # Even at b -> 0 the kick is finite (max GM/(w r_s)), unlike 2GM/(bw).
        x0 = np.array([10.0, 0.0, 0.0]); w = np.array([0, 200.0, 0])
        rs, M, wmag = 0.4, 1e8, 200.0
        star = np.array([[10.0, 0.0, 1e-6]])
        dv = erkal_plummer_kick(star, x0, w, np.array([0, 0, 1e-6]), M, rs)
        assert np.all(np.isfinite(dv))
        assert np.linalg.norm(dv) <= G_KPC_KMS * M / (wmag * rs) + 1e-6

    def test_max_kick_at_perp_distance_rs(self):
        # |dv|(d) = 2GM/w d/(d^2+rs^2) peaks at d = rs.
        M, w, rs = 1e8, 200.0, 0.5
        x0 = np.array([0.0, 0.0, 0.0]); wv = np.array([0, w, 0])
        def mag(d):
            star = np.array([[0.0, 0.0, d]])
            return np.linalg.norm(erkal_plummer_kick(star, x0, wv, np.array([0, 0, 0]), M, rs))
        peak = mag(rs)
        assert peak > mag(rs * 0.3) and peak > mag(rs * 3.0)

    def test_kick_perpendicular_to_w(self):
        x0 = np.array([8.0, 1.0, 0.0]); w = np.array([30.0, 200.0, 10.0])
        b = np.array([0.0, 0.0, 0.4])
        stars = np.array([[8.0, 1.0, 0.4], [9.0, 1.5, -0.3], [7.0, 0.0, 0.2]])
        dv = erkal_plummer_kick(stars, x0, w, b, 1e8, 0.3)
        # Each kick is perpendicular to w (impulse is transverse to the path).
        for i in range(len(stars)):
            assert abs(np.dot(dv[i], w)) < 1e-6 * (np.linalg.norm(dv[i]) * np.linalg.norm(w) + 1e-9)


class TestEncounterGeometry:
    def test_default_b_perp_to_w(self):
        w_vec, b_vec = build_encounter_geometry(
            np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), b_kpc=0.5, w_kms=200.0)
        assert abs(np.dot(w_vec, b_vec)) < 1e-6
        assert np.linalg.norm(b_vec) == pytest.approx(0.5, rel=1e-6)
        assert np.linalg.norm(w_vec) == pytest.approx(200.0, rel=1e-6)

    def test_default_w_perp_to_stream(self):
        # Gap-forming default: subhalo velocity ~ perpendicular to the stream tangent.
        t = np.array([1.0, 0.0, 0.0])
        w_vec, _ = build_encounter_geometry(t, np.array([0.0, 1.0, 0.0]), 0.5, 200.0)
        assert abs(np.dot(w_vec / np.linalg.norm(w_vec), t)) < 1e-6

    def test_sampled_geometry_perp_and_magnitudes(self):
        rng = np.random.default_rng(0)
        w_vec, b_vec = build_encounter_geometry(
            np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 0.5, 200.0, rng=rng)
        assert abs(np.dot(w_vec, b_vec)) < 1e-6
        assert np.linalg.norm(w_vec) == pytest.approx(200.0, rel=1e-6)
        assert np.linalg.norm(b_vec) == pytest.approx(0.5, rel=1e-6)
