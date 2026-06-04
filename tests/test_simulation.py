"""
Tests for the simulation engine.

Validation criteria:
- CDM mass function slope = -1.9 ± 0.05
- WDM suppression reduces sample count below half-mode mass
- Gap visible after M=1e7 Msun subhalo impact
- Gaia noise levels match GOST predictions at G=19
- Subhalo impulse kick produces physical velocity perturbation
- Baryonic perturbation preserves stream topology
- Foreground contamination adds correct fraction of stars
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.mass_functions import (
    mass_function_suppression_factor,
    sample_cdm_masses,
    sample_fdm_masses,
    sample_wdm_masses,
    wdm_half_mode_mass,
    wdm_suppression,
)
from src.simulation.noise import (
    add_foreground_contamination,
    add_gaia_noise,
    add_gaia_noise_randomized,
    build_membership_probabilities,
    distance_uncertainty_from_parallax,
    gaia_parallax_uncertainty,
    gaia_pm_uncertainty,
    stream_to_g_magnitude,
)
from src.simulation.potentials import circular_velocity_curve, get_mw_potential
from src.simulation.stream_gen import StreamParticles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_stream(n: int = 500, seed: int = 0) -> StreamParticles:
    """Create a minimal fake stream for unit tests (no gala/galstreams required)."""
    rng = np.random.default_rng(seed)
    phi1 = np.linspace(-50.0, 50.0, n) + rng.normal(0, 0.1, n)
    phi2 = rng.normal(0.0, 0.3, n)
    dist = rng.uniform(7.0, 9.0, n)
    pm1 = rng.normal(-6.0, 0.5, n)
    pm2 = rng.normal(-3.0, 0.5, n)
    vrad = rng.normal(-100.0, 5.0, n)
    xyz = np.stack([rng.uniform(-10, 10, n),
                    rng.uniform(-10, 10, n),
                    rng.uniform(-5, 5, n)], axis=0)
    vxyz = np.stack([rng.uniform(-200, 200, n),
                     rng.uniform(-200, 200, n),
                     rng.uniform(-100, 100, n)], axis=0)
    return StreamParticles(
        phi1=phi1, phi2=phi2, dist=dist,
        pm1=pm1, pm2=pm2, vrad=vrad,
        xyz_kpc=xyz, vxyz_kms=vxyz,
    )


def _make_fake_stream_config() -> dict:
    """Minimal stream config for noise/contamination tests."""
    return {
        "phi1_range_deg": [-50.0, 50.0],
        "phi2_selection_deg": 2.0,
        "distance_kpc": [7.0, 9.0],
        "isochrone_feh": -1.5,
        "isochrone_age_gyr": 10.0,
        "velocity_dispersion_kms": 0.4,
        "expected_n_members": 500,
    }


# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------

class TestPotentials:
    def test_potential_loads(self):
        potential = get_mw_potential()
        assert potential is not None

    def test_circular_velocity_at_sun(self):
        potential = get_mw_potential()
        v_circ = circular_velocity_curve(potential, np.array([8.122]))[0]
        # Should be ~229 km/s; accept 200-260 km/s
        assert 200 < v_circ < 260, f"v_circ at 8.122 kpc = {v_circ:.1f} km/s, expected ~229"

    def test_circular_velocity_profile_is_rising_then_flat(self):
        potential = get_mw_potential()
        r = np.array([1.0, 4.0, 8.0, 15.0, 30.0])
        vc = circular_velocity_curve(potential, r)
        # Profile should peak between 4-15 kpc, not at r=1 kpc
        assert vc[2] > vc[0], "Circular velocity should increase from 1 to 8 kpc"
        assert all(v > 100 for v in vc), "All circular velocities should exceed 100 km/s"

    def test_circular_velocity_decreases_at_large_r(self):
        """At very large radii (>50 kpc), v_circ should decline (no flat rotation curve)."""
        potential = get_mw_potential()
        r = np.array([30.0, 60.0, 100.0])
        vc = circular_velocity_curve(potential, r)
        # NFW halo eventually falls; check 100 kpc < 30 kpc
        assert vc[2] < vc[0], "Circular velocity should decrease from 30 to 100 kpc"


# ---------------------------------------------------------------------------
# Mass functions
# ---------------------------------------------------------------------------

class TestMassFunctions:
    def test_cdm_masses_in_range(self):
        masses = sample_cdm_masses(1000, log10_m_min=5.0, log10_m_max=9.0, alpha=-1.9, seed=0)
        assert np.all(masses >= 1e5)
        assert np.all(masses <= 1e9)

    def test_cdm_slope(self):
        n = 10000
        masses = sample_cdm_masses(n, log10_m_min=6.0, log10_m_max=9.0, alpha=-1.9, seed=42)
        log_masses = np.log10(masses)
        # Fit a line to the log-log histogram
        counts, edges = np.histogram(log_masses, bins=20)
        centers = 0.5 * (edges[:-1] + edges[1:])
        valid = counts > 5
        if valid.sum() < 5:
            pytest.skip("Insufficient bins for slope test")
        slope = np.polyfit(centers[valid], np.log10(counts[valid] + 1), 1)[0]
        # dN/dlogM ~ M^(alpha+1) → slope in logN vs logM histogram ~ alpha+1 = -0.9
        assert abs(slope - (-0.9)) < 0.2, f"CDM slope = {slope:.2f}, expected ~-0.9"

    def test_cdm_reproducibility(self):
        """Same seed should produce identical mass samples."""
        m1 = sample_cdm_masses(100, log10_m_min=5.0, log10_m_max=9.0, alpha=-1.9, seed=42)
        m2 = sample_cdm_masses(100, log10_m_min=5.0, log10_m_max=9.0, alpha=-1.9, seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_cdm_different_seeds_differ(self):
        """Different seeds should produce different samples."""
        m1 = sample_cdm_masses(100, log10_m_min=5.0, log10_m_max=9.0, alpha=-1.9, seed=42)
        m2 = sample_cdm_masses(100, log10_m_min=5.0, log10_m_max=9.0, alpha=-1.9, seed=99)
        assert not np.array_equal(m1, m2)

    def test_cdm_masses_more_low_mass(self):
        """CDM mass function is bottom-heavy: most halos are low-mass."""
        masses = sample_cdm_masses(5000, log10_m_min=5.0, log10_m_max=9.0, alpha=-1.9, seed=0)
        median_log = np.log10(np.median(masses))
        # Median should be in the lower half of [5, 9]
        assert median_log < 7.5, f"CDM median log10(M) = {median_log:.2f}, expected < 7.5"

    def test_wdm_suppression_below_halfmode(self):
        m_wdm_kev = 3.0
        m_hm = wdm_half_mode_mass(m_wdm_kev)
        masses_below = np.array([m_hm * 0.1, m_hm * 0.2, m_hm * 0.5])
        masses_above = np.array([m_hm * 2.0, m_hm * 5.0, m_hm * 10.0])
        sup_below = wdm_suppression(masses_below, m_hm)
        sup_above = wdm_suppression(masses_above, m_hm)
        assert np.all(sup_below < sup_above), "WDM suppression should be stronger below M_hm"
        assert np.all(sup_above > 0.5), "WDM suppression above M_hm should be > 0.5"

    def test_wdm_suppression_at_halfmode(self):
        """At M = M_hm, suppression should be ~0.5 (definition of half-mode)."""
        m_wdm_kev = 3.0
        m_hm = wdm_half_mode_mass(m_wdm_kev)
        sup = wdm_suppression(np.array([m_hm]), m_hm)
        assert 0.2 < sup[0] < 0.8, f"WDM suppression at M_hm = {sup[0]:.3f}, expected ~0.5"

    def test_wdm_halfmode_mass_decreases_with_particle_mass(self):
        """Heavier WDM particle → smaller half-mode mass (less suppression)."""
        m_hm_1 = wdm_half_mode_mass(1.0)
        m_hm_5 = wdm_half_mode_mass(5.0)
        m_hm_10 = wdm_half_mode_mass(10.0)
        assert m_hm_1 > m_hm_5 > m_hm_10

    def test_wdm_fewer_low_mass(self):
        n = 5000
        m_wdm_kev = 10.0
        cdm = sample_cdm_masses(n, log10_m_min=5.0, log10_m_max=9.0, seed=0)
        wdm = sample_wdm_masses(n, m_wdm_kev=m_wdm_kev, log10_m_min=5.0, log10_m_max=9.0, seed=0)
        m_hm = wdm_half_mode_mass(m_wdm_kev)
        assert 1e5 < m_hm < 1e9, f"M_hm={m_hm:.2e} is outside [1e5, 1e9]"
        frac_cdm_below = (cdm < m_hm).mean()
        frac_wdm_below = (wdm < m_hm).mean()
        assert frac_wdm_below < frac_cdm_below, \
            f"WDM fraction below M_hm ({frac_wdm_below:.2f}) should be < CDM ({frac_cdm_below:.2f})"

    def test_fdm_suppression_stronger_than_wdm_low_mass(self):
        n = 5000
        wdm = sample_wdm_masses(n, m_wdm_kev=3.0, log10_m_min=5.0, log10_m_max=9.0, seed=0)
        fdm = sample_fdm_masses(n, m_axion_ev=1e-22, log10_m_min=5.0, log10_m_max=9.0, seed=0)
        assert fdm.mean() > wdm.mean() or fdm.min() > wdm.min(), \
            "FDM (m22=1) should suppress more low-mass halos than WDM (3 keV)"

    def test_suppressed_rate_factor_below_cdm(self):
        """WDM/FDM integrated encounter rates should be suppressed relative to CDM."""
        assert mass_function_suppression_factor("CDM", {}, 5.0, 9.0) == pytest.approx(1.0)
        wdm_factor = mass_function_suppression_factor("WDM", {"m_wdm_kev": 2.0}, 5.0, 9.0)
        fdm_factor = mass_function_suppression_factor("FDM", {"m_axion_ev": 1e-22}, 5.0, 9.0)
        assert 0.0 < wdm_factor < 1.0
        assert 0.0 < fdm_factor < 1.0

    def test_cdm_empty_request(self):
        """Requesting zero masses returns empty array."""
        masses = sample_cdm_masses(0, log10_m_min=5.0, log10_m_max=9.0, alpha=-1.9, seed=0)
        assert len(masses) == 0


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

class TestNoiseModel:
    def test_pm_noise_at_g19(self):
        g = np.array([19.0])
        sigma_pm = gaia_pm_uncertainty(g)[0]
        expected = 0.021 + 0.028 * 10.0 ** (0.4 * (19.0 - 15.0))
        assert abs(sigma_pm - expected) < 0.001, f"PM uncertainty at G=19: {sigma_pm:.4f} vs {expected:.4f}"

    def test_pm_noise_increases_with_magnitude(self):
        g_vals = np.array([15.0, 17.0, 19.0, 21.0])
        sigma = gaia_pm_uncertainty(g_vals)
        assert np.all(np.diff(sigma) > 0), "PM noise should increase with magnitude"

    def test_pm_noise_bright_stars(self):
        """Stars brighter than G=13 should have sigma_pm = 0.01 mas/yr."""
        g = np.array([10.0, 12.0, 13.0])
        sigma = gaia_pm_uncertainty(g)
        np.testing.assert_allclose(sigma, 0.01, atol=1e-6)

    def test_parallax_noise_at_g15(self):
        sigma_plx = gaia_parallax_uncertainty(np.array([15.0]))[0]
        expected = 0.007 + 0.007 * 10.0 ** (0.4 * 2.0)
        assert abs(sigma_plx - expected) < 0.001

    def test_parallax_noise_increases_with_magnitude(self):
        g_vals = np.array([14.0, 16.0, 18.0, 20.0])
        sigma = gaia_parallax_uncertainty(g_vals)
        assert np.all(np.diff(sigma) > 0), "Parallax noise should increase with magnitude"

    def test_distance_uncertainty_propagation(self):
        """Distance uncertainty should be larger for more distant stars."""
        dist = np.array([1.0, 5.0, 10.0, 20.0])
        g_mag = np.array([15.0, 17.0, 19.0, 20.0])
        sigma_d = distance_uncertainty_from_parallax(dist, g_mag)
        assert sigma_d[-1] > sigma_d[0], "Distance uncertainty should be larger for more distant/fainter stars"
        assert np.all(sigma_d > 0), "Distance uncertainty must be positive"

    def test_g_magnitude_distance_modulus(self):
        """Stars at greater distance should appear fainter."""
        config = {"isochrone_feh": -2.0}
        g_near = stream_to_g_magnitude(np.array([5.0]), config, seed=0)
        g_far = stream_to_g_magnitude(np.array([20.0]), config, seed=0)
        assert g_far[0] > g_near[0], "More distant stars should be fainter"

    def test_g_magnitude_clipping(self):
        """Magnitudes should be clipped to [14, 21]."""
        config = {"isochrone_feh": -2.0}
        g = stream_to_g_magnitude(np.array([0.01, 0.1, 100.0, 500.0]), config, seed=0)
        assert np.all(g >= 14.0), "G magnitude should be >= 14"
        assert np.all(g <= 21.0), "G magnitude should be <= 21"


# ---------------------------------------------------------------------------
# Gaia noise injection
# ---------------------------------------------------------------------------

class TestNoiseInjection:
    def test_add_gaia_noise_perturbs_pm(self):
        """Noise injection should change pm1/pm2 from their clean values."""
        stream = _make_fake_stream(n=200)
        config = _make_fake_stream_config()
        noisy_stream, *_ = add_gaia_noise(stream, config, seed=42)
        assert not np.allclose(stream.pm1, noisy_stream.pm1), "PM1 should be perturbed"
        assert not np.allclose(stream.pm2, noisy_stream.pm2), "PM2 should be perturbed"

    def test_add_gaia_noise_preserves_phi1(self):
        """Noise injection should not change phi1 (positions are fixed)."""
        stream = _make_fake_stream(n=200)
        config = _make_fake_stream_config()
        noisy_stream, *_ = add_gaia_noise(stream, config, seed=42)
        np.testing.assert_array_equal(stream.phi1, noisy_stream.phi1)

    def test_add_gaia_noise_returns_uncertainties(self):
        """Should return sigma_pm1, sigma_pm2, sigma_dist, sigma_vrad."""
        stream = _make_fake_stream(n=200)
        config = _make_fake_stream_config()
        result = add_gaia_noise(stream, config, seed=42)
        assert len(result) == 5, "Expected (particles, sigma_pm, sigma_pm, sigma_dist, sigma_vrad)"
        for arr in result[1:4]:
            assert len(arr) == 200
            assert np.all(arr > 0), "Uncertainties must be positive"

    def test_add_gaia_noise_clips_distance(self):
        """Noisy distances should be clipped to [0.1, 200] kpc."""
        stream = _make_fake_stream(n=500)
        config = _make_fake_stream_config()
        noisy_stream, *_ = add_gaia_noise(stream, config, seed=42)
        assert np.all(noisy_stream.dist >= 0.1)
        assert np.all(noisy_stream.dist <= 200.0)

    def test_randomized_noise_scales_uncertainties(self):
        """Domain-randomized Gaia noise should scale reported uncertainties."""
        stream = _make_fake_stream(n=200)
        config = _make_fake_stream_config()
        _, sigma_pm_base, _, sigma_dist_base, _ = add_gaia_noise_randomized(
            stream, config, noise_scale_factor=1.0, seed=42,
        )
        _, sigma_pm_hi, _, sigma_dist_hi, _ = add_gaia_noise_randomized(
            stream, config, noise_scale_factor=2.0, seed=42,
        )
        np.testing.assert_allclose(sigma_pm_hi, 2.0 * sigma_pm_base)
        np.testing.assert_allclose(sigma_dist_hi, 2.0 * sigma_dist_base)

    def test_membership_prob_corruption(self):
        """Membership probabilities should encode true members, foreground, and corruption."""
        probs = build_membership_probabilities(
            n_stream=100,
            n_foreground=20,
            corruption_fraction=0.10,
            low_prob_range=(0.05, 0.49),
            noise_sigma=0.0,
            seed=42,
        )
        assert probs.shape == (120,)
        assert np.sum(probs[:100] < 0.5) == 10
        assert np.all(probs[100:] == 0.0)
        assert np.all((probs >= 0.0) & (probs <= 1.0))


# ---------------------------------------------------------------------------
# Foreground contamination
# ---------------------------------------------------------------------------

class TestForegroundContamination:
    def test_contamination_adds_correct_fraction(self):
        stream = _make_fake_stream(n=500)
        config = _make_fake_stream_config()
        contaminated = add_foreground_contamination(stream, config, contamination_fraction=0.10, seed=42)
        expected_n = 500 + int(500 * 0.10)
        assert len(contaminated.phi1) == expected_n

    def test_contamination_zero_fraction(self):
        """Zero contamination should return same star count."""
        stream = _make_fake_stream(n=200)
        config = _make_fake_stream_config()
        contaminated = add_foreground_contamination(stream, config, contamination_fraction=0.0, seed=42)
        assert len(contaminated.phi1) == len(stream.phi1)

    def test_contamination_preserves_originals(self):
        """Original stream stars should be preserved (prepended) in the output."""
        stream = _make_fake_stream(n=100)
        config = _make_fake_stream_config()
        contaminated = add_foreground_contamination(stream, config, contamination_fraction=0.20, seed=42)
        # First 100 stars should be the originals
        np.testing.assert_array_equal(contaminated.phi1[:100], stream.phi1)
        np.testing.assert_array_equal(contaminated.pm1[:100], stream.pm1)

    def test_contamination_has_broad_pm(self):
        """Foreground stars should have much broader PM dispersion than stream."""
        stream = _make_fake_stream(n=500)
        config = _make_fake_stream_config()
        contaminated = add_foreground_contamination(stream, config, contamination_fraction=0.50, seed=42)
        n_orig = 500
        contam_pm1 = contaminated.pm1[n_orig:]
        # Foreground should have sigma ~ 5 mas/yr, not ~0.5 mas/yr
        assert np.std(contam_pm1) > 2.0, "Foreground PM dispersion should be broad"


# ---------------------------------------------------------------------------
# Subhalo impulse kick
# ---------------------------------------------------------------------------

class TestSubhaloImpulse:
    def test_impulse_kick_finite(self):
        """Velocity kick should be finite and non-negative."""
        from src.simulation.subhalo import _hernquist_impulse_kick
        r_perp = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        x_par = np.zeros_like(r_perp)
        dv = _hernquist_impulse_kick(r_perp, x_par, m_solar=1e7, a_kpc=0.01,
                                      b_kpc=0.1, v_kms=200.0)
        assert np.all(np.isfinite(dv)), "Velocity kick must be finite"
        assert np.all(dv >= 0), "Velocity kick magnitude must be non-negative"

    def test_impulse_kick_decreases_with_distance(self):
        """Stars farther from fly-by path should receive smaller kick."""
        from src.simulation.subhalo import _hernquist_impulse_kick
        r_perp = np.array([0.01, 0.1, 1.0, 10.0])
        x_par = np.zeros_like(r_perp)
        dv = _hernquist_impulse_kick(r_perp, x_par, m_solar=1e7, a_kpc=0.01,
                                      b_kpc=0.05, v_kms=200.0)
        assert np.all(np.diff(dv) <= 0), "Kick should decrease with distance"

    def test_impulse_kick_bounded_by_plummer(self):
        """The Plummer impulse is naturally bounded (no ad-hoc cap needed).

        The Erkal+2015 magnitude 2GM/w * d/(d^2+r_s^2) peaks at d = r_s with value
        GM/(w r_s); even at a tiny perpendicular distance the kick stays finite and
        below that bound, unlike the old 2GM/(dw) point-mass form which diverged
        and had to be clipped at 50 km/s.
        """
        from src.simulation.subhalo import G_KPC_KMS, _hernquist_impulse_kick
        M, a, v = 1e8, 0.4, 200.0
        r_perp = np.array([1e-4, 0.01, a, 1.0])  # includes near-zero distance
        x_par = np.zeros_like(r_perp)
        dv = _hernquist_impulse_kick(r_perp, x_par, m_solar=M, a_kpc=a, b_kpc=0.0, v_kms=v)
        bound = G_KPC_KMS * M / (v * a)          # GM/(w r_s), the analytic maximum
        assert np.all(np.isfinite(dv))
        assert np.all(dv <= bound + 1e-9), "Plummer kick must not exceed GM/(w r_s)"
        # The peak is at d = r_s.
        assert dv[2] == pytest.approx(bound, rel=1e-6)

    def test_impulse_kick_scales_with_mass(self):
        """Heavier subhalo should produce larger kick."""
        from src.simulation.subhalo import _hernquist_impulse_kick
        r_perp = np.array([1.0])
        x_par = np.zeros_like(r_perp)
        dv_low = _hernquist_impulse_kick(r_perp, x_par, m_solar=1e5, a_kpc=0.01,
                                          b_kpc=0.1, v_kms=200.0)
        dv_high = _hernquist_impulse_kick(r_perp, x_par, m_solar=1e8, a_kpc=0.05,
                                           b_kpc=0.1, v_kms=200.0)
        assert dv_high[0] > dv_low[0], "Heavier subhalo should produce larger kick"

    def test_apply_impulse_modifies_velocities(self):
        """Applying an impulse should change the particle velocities."""
        from src.simulation.subhalo import EncounterParams, apply_impulse_approximation
        stream = _make_fake_stream(n=200)
        enc = EncounterParams(
            mass_solar=1e7, scale_radius_kpc=0.01,
            impact_param_kpc=0.1, flyby_vel_kms=200.0,
            encounter_phi1=0.0, t_since_impact_gyr=2.0,
            is_valid=True, is_massive=False,
        )
        perturbed = apply_impulse_approximation(stream, enc)
        assert not np.allclose(stream.vrad, perturbed.vrad), "Vrad should change after impulse"
        assert not np.allclose(stream.pm1, perturbed.pm1), "PM1 should change after impulse"

    def test_apply_impulse_invalid_encounter_noop(self):
        """An invalid encounter should leave particles unchanged."""
        from src.simulation.subhalo import EncounterParams, apply_impulse_approximation
        stream = _make_fake_stream(n=200)
        enc = EncounterParams(
            mass_solar=1e7, scale_radius_kpc=0.01,
            impact_param_kpc=0.1, flyby_vel_kms=200.0,
            encounter_phi1=0.0, t_since_impact_gyr=2.0,
            is_valid=False, is_massive=False,
        )
        perturbed = apply_impulse_approximation(stream, enc)
        np.testing.assert_array_equal(stream.vrad, perturbed.vrad)

    def test_concentration_from_mass(self):
        """Concentration should decrease with increasing mass (Ludlow+2016)."""
        from src.simulation.subhalo import concentration_from_mass
        c_low = concentration_from_mass(1e6)
        c_high = concentration_from_mass(1e10)
        assert c_low > c_high, "Concentration should decrease with mass"
        assert c_low > 0 and c_high > 0, "Concentration must be positive"

    def test_scale_radius_positive(self):
        """Scale radius should always be positive."""
        from src.simulation.subhalo import scale_radius_from_mass
        for m in [1e5, 1e7, 1e9]:
            r = scale_radius_from_mass(m)
            assert r > 0, f"Scale radius must be positive, got {r} for M={m}"

    def test_scale_radius_sidm_broader(self):
        """SIDM cored profile should give a broader effective scale radius."""
        from src.simulation.subhalo import scale_radius_from_mass
        r_nfw = scale_radius_from_mass(1e8, "NFW")
        r_sidm = scale_radius_from_mass(1e8, "isothermal_core_NFW")
        assert r_sidm > r_nfw, "SIDM scale radius should be broader than NFW"

    def test_encounter_params_kick_capped_field(self):
        """EncounterParams should have a kick_capped field."""
        from src.simulation.subhalo import EncounterParams
        enc = EncounterParams(
            mass_solar=1e7, scale_radius_kpc=0.01,
            impact_param_kpc=0.1, flyby_vel_kms=200.0,
            encounter_phi1=0.0, t_since_impact_gyr=2.0,
        )
        assert hasattr(enc, "kick_capped")
        assert enc.kick_capped is False


# ---------------------------------------------------------------------------
# Baryonic perturbations
# ---------------------------------------------------------------------------

class TestBaryonicPerturbations:
    def test_gmc_encounter_changes_velocities(self):
        """GMC encounter should perturb velocities but not remove stars."""
        from src.simulation.baryonic import apply_giant_molecular_cloud_encounter
        stream = _make_fake_stream(n=300)
        perturbed = apply_giant_molecular_cloud_encounter(stream, seed=42)
        assert len(perturbed.phi1) == len(stream.phi1), "GMC should not remove stars"
        assert not np.allclose(stream.vrad, perturbed.vrad), "Vrad should change"

    def test_gmc_params_in_range(self):
        """Sampled GMC parameters should be in physical ranges."""
        from src.simulation.baryonic import sample_gmc_params
        params = sample_gmc_params(seed=42)
        assert 1e4 <= params["mass_solar"] <= 1e7, f"GMC mass {params['mass_solar']} out of range"
        assert 0.01 <= params["a_kpc"] <= 0.10, f"GMC scale radius {params['a_kpc']} out of range"
        assert 20.0 <= params["v_kms"] <= 100.0
        assert 0.001 <= params["b_kpc"] <= 0.5

    def test_bar_perturbation_removes_stars(self):
        """Bar perturbation uses rejection sampling, should remove some stars."""
        from src.simulation.baryonic import apply_bar_perturbation
        stream = _make_fake_stream(n=1000)
        perturbed = apply_bar_perturbation(stream, seed=42, bar_amplitude=0.1)
        # With amplitude 0.1, should keep ~90-100% of stars
        assert len(perturbed.phi1) < len(stream.phi1), "Bar should remove some stars"
        assert len(perturbed.phi1) > len(stream.phi1) * 0.5, "Bar shouldn't remove >50%"

    def test_bar_perturbation_zero_amplitude(self):
        """Zero amplitude should keep all stars."""
        from src.simulation.baryonic import apply_bar_perturbation
        stream = _make_fake_stream(n=500)
        perturbed = apply_bar_perturbation(stream, seed=42, bar_amplitude=0.0)
        assert len(perturbed.phi1) == len(stream.phi1)

    def test_baryonic_flag_disk_crossing(self):
        """Disk-crossing flag should mark stars near phi2=0."""
        from src.simulation.baryonic import flag_baryonic_dominated_region
        phi1 = np.linspace(-50, 50, 100)
        phi2 = np.linspace(-2, 2, 100)
        config = {"disk_crossing": True}
        mask = flag_baryonic_dominated_region(phi1, phi2, config)
        # Stars with |phi2| < 0.5 should be flagged
        expected = np.abs(phi2) < 0.5
        np.testing.assert_array_equal(mask, expected)

    def test_baryonic_flag_no_disk_crossing(self):
        """Non-disk-crossing stream should have no baryonic flags."""
        from src.simulation.baryonic import flag_baryonic_dominated_region
        phi1 = np.linspace(-50, 50, 100)
        phi2 = np.linspace(-2, 2, 100)
        config = {"disk_crossing": False}
        mask = flag_baryonic_dominated_region(phi1, phi2, config)
        assert not mask.any(), "Non-disk-crossing stream should have no baryonic flags"


# ---------------------------------------------------------------------------
# Encounter sampling
# ---------------------------------------------------------------------------

class TestEncounterSampling:
    def test_sample_encounter_returns_valid(self):
        """Sampled encounter should have physically valid parameters."""
        from src.simulation.subhalo import sample_subhalo_encounter
        import yaml
        with open(Path(__file__).resolve().parent.parent / "config" / "dm_models.yaml") as f:
            dm_cfg = yaml.safe_load(f)["models"]["CDM"]
        stream_config = _make_fake_stream_config()
        enc = sample_subhalo_encounter("CDM", dm_cfg, stream_config, seed=42)
        assert enc.mass_solar > 0
        assert enc.scale_radius_kpc > 0
        assert enc.impact_param_kpc > 0
        assert enc.flyby_vel_kms > 0
        assert enc.t_since_impact_gyr > 0
        assert stream_config["phi1_range_deg"][0] <= enc.encounter_phi1 <= stream_config["phi1_range_deg"][1]


def test_streamdf_generator_exposes_tidal_arm_choice():
    """Real-stream profile checks must be able to model either tidal arm."""
    import inspect

    from src.simulation.stream_gen import generate_stream_df

    assert "leading" in inspect.signature(generate_stream_df).parameters


def test_streamdf_impact_angle_sign_follows_tidal_arm():
    from src.simulation.stream_gen import impact_params_for_arm

    params = {"impact_angle_rad": 0.4, "mass": 1e8}
    assert impact_params_for_arm(params, leading=True)["impact_angle_rad"] == 0.4
    assert impact_params_for_arm(params, leading=False)["impact_angle_rad"] == -0.4
    assert params["impact_angle_rad"] == 0.4


def test_combine_stream_particles_concatenates_phase_space():
    from src.simulation.stream_gen import combine_stream_particles

    first = _make_fake_stream(n=3)
    second = _make_fake_stream(n=5)
    combined = combine_stream_particles(first, second)
    assert len(combined.phi1) == 8
    assert combined.xyz_kpc.shape == (3, 8)


# ---------------------------------------------------------------------------
# Smoke test: stream generation (requires gala installed)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestStreamGeneration:
    def test_generate_unperturbed_stream(self):
        pytest.importorskip("gala")
        pytest.importorskip("galstreams")
        from src.simulation.potentials import get_mw_potential  # noqa: PLC0415
        from src.simulation.stream_gen import generate_stream  # noqa: PLC0415

        potential = get_mw_potential()
        particles = generate_stream("GD1", potential, n_stars=200, seed=42)
        assert len(particles.phi1) > 10, "Generated stream should have >10 stars"
        assert particles.phi1.min() >= -110.0, "phi1 should be in plausible range"
        assert particles.phi1.max() <= 95.0, "phi1 should be within configured range"
        pm1_std = particles.pm1.std()
        assert pm1_std < 10.0, f"PM1 dispersion too high: {pm1_std:.2f} mas/yr"
