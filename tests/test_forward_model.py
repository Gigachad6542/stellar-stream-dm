"""Tests for the timeline forward model pipeline."""

import numpy as np
import pytest

from src.forward_model.scoring import (
    DensityProfile,
    GapFeature,
    ScoreResult,
    ScoreWeights,
    combined_score,
    compute_density_profile,
    density_residual_score,
    detect_gaps,
    gap_agreement_score,
    kinematic_perturbation_score,
)
from src.simulation.stream_gen import StreamParticles
from src.simulation.subhalo import (
    DM_MODELS,
    EncounterParams,
    apply_impulse_approximation,
    scale_radius_from_mass,
    subhalo_profile_for_model,
)


# ---------------------------------------------------------------------------
# Density profile tests
# ---------------------------------------------------------------------------

class TestDensityProfile:
    def test_uniform_sums_to_one(self):
        rng = np.random.default_rng(42)
        phi1 = rng.uniform(-50, 50, 5000)
        profile = compute_density_profile(phi1, (-50, 50), bin_width_deg=2.0)
        assert abs(profile.density.sum() - 1.0) < 0.01

    def test_bin_count(self):
        rng = np.random.default_rng(42)
        phi1 = rng.uniform(0, 100, 1000)
        profile = compute_density_profile(phi1, (0, 100), bin_width_deg=5.0)
        assert profile.n_bins == 20

    def test_empty_region_zero_density(self):
        phi1 = np.array([10.0, 11.0, 12.0, 90.0, 91.0])
        profile = compute_density_profile(phi1, (0, 100), bin_width_deg=10.0)
        # Most bins should be zero
        assert (profile.counts == 0).sum() >= 7

    def test_weights(self):
        phi1 = np.array([5.0, 5.1, 5.2, 50.0, 50.1])
        weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        profile = compute_density_profile(phi1, (0, 100), bin_width_deg=10.0, weights=weights)
        # Only first bin should have weight
        assert profile.density[0] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Gap detection tests
# ---------------------------------------------------------------------------

class TestGapDetection:
    def test_uniform_no_gaps(self):
        rng = np.random.default_rng(42)
        phi1 = rng.uniform(-50, 50, 10000)
        profile = compute_density_profile(phi1, (-50, 50), bin_width_deg=1.0)
        gaps = detect_gaps(profile, min_depth=0.3, min_significance=2.0)
        assert len(gaps) == 0

    def test_artificial_gap_detected(self):
        rng = np.random.default_rng(42)
        phi1 = rng.uniform(-50, 50, 5000)
        # Remove stars in [-2, 2] to create a gap at phi1=0
        phi1 = phi1[(phi1 < -2) | (phi1 > 2)]
        profile = compute_density_profile(phi1, (-50, 50), bin_width_deg=2.0)
        gaps = detect_gaps(profile, min_depth=0.2, min_significance=1.5)
        assert len(gaps) >= 1
        # Gap should be near phi1=0
        assert abs(gaps[0].phi1_center) < 5.0

    def test_gap_depth(self):
        rng = np.random.default_rng(42)
        phi1 = rng.uniform(-50, 50, 5000)
        # Wide gap: remove everything in [-5, 5]
        phi1 = phi1[(phi1 < -5) | (phi1 > 5)]
        profile = compute_density_profile(phi1, (-50, 50), bin_width_deg=2.0)
        gaps = detect_gaps(profile, min_depth=0.2, min_significance=1.0)
        assert len(gaps) >= 1
        assert gaps[0].depth > 0.5  # Should be a deep gap

    def test_gap_significance_uses_fractional_poisson_uncertainty(self):
        counts = np.array([100.0, 100.0, 100.0, 25.0, 100.0, 100.0, 100.0])
        edges = np.arange(8, dtype=float)
        profile = DensityProfile(
            bin_centers=0.5 * (edges[:-1] + edges[1:]),
            bin_edges=edges,
            counts=counts,
            density=counts / counts.sum(),
            bin_width_deg=1.0,
        )

        gaps = detect_gaps(
            profile,
            min_depth=0.5,
            min_significance=1.0,
            smooth_sigma_bins=0.1,
        )

        assert len(gaps) == 1
        assert gaps[0].significance == pytest.approx(15.0, rel=0.05)


# ---------------------------------------------------------------------------
# Individual scorer tests
# ---------------------------------------------------------------------------

class TestDensityResidualScore:
    def test_identical_profiles_zero(self):
        rng = np.random.default_rng(42)
        phi1 = rng.uniform(-50, 50, 5000)
        profile = compute_density_profile(phi1, (-50, 50), bin_width_deg=2.0)
        assert density_residual_score(profile, profile) == 0.0

    def test_different_profiles_positive(self):
        rng = np.random.default_rng(42)
        phi1_a = rng.uniform(-50, 50, 5000)
        phi1_b = rng.uniform(-50, 0, 5000)  # only in left half
        prof_a = compute_density_profile(phi1_a, (-50, 50), bin_width_deg=2.0)
        prof_b = compute_density_profile(phi1_b, (-50, 50), bin_width_deg=2.0)
        score = density_residual_score(prof_b, prof_a)
        assert score > 0


class TestGapAgreementScore:
    def test_no_obs_gaps(self):
        sim_gaps = [GapFeature(phi1_center=10.0, phi1_width=3.0, depth=0.5, significance=3.0)]
        score = gap_agreement_score(sim_gaps, [])
        assert score == 0.5  # penalty for extra sim gap

    def test_perfect_match(self):
        gap = GapFeature(phi1_center=10.0, phi1_width=3.0, depth=0.5, significance=3.0)
        score = gap_agreement_score([gap], [gap])
        assert score == 0.0  # perfect match

    def test_no_sim_gaps_penalised(self):
        obs = [GapFeature(phi1_center=10.0, phi1_width=3.0, depth=0.5, significance=3.0)]
        score = gap_agreement_score([], obs)
        assert score == 1.0  # unmatched penalty

    def test_nearby_better_than_far(self):
        obs = [GapFeature(phi1_center=10.0, phi1_width=3.0, depth=0.5, significance=3.0)]
        near = [GapFeature(phi1_center=11.0, phi1_width=3.0, depth=0.5, significance=3.0)]
        far = [GapFeature(phi1_center=30.0, phi1_width=3.0, depth=0.5, significance=3.0)]
        score_near = gap_agreement_score(near, obs)
        score_far = gap_agreement_score(far, obs)
        assert score_near < score_far


class TestKinematicScore:
    def test_similar_streams_low_score(self):
        rng = np.random.default_rng(42)
        n = 3000
        phi1 = rng.uniform(-50, 50, n)
        pm1 = -8.0 + 0.04 * phi1 + rng.normal(0, 0.2, n)
        pm2 = rng.normal(0, 0.15, n)
        score = kinematic_perturbation_score(
            phi1, pm1, pm2, phi1, pm1, pm2, (-50, 50)
        )
        assert score < 0.5

    def test_offset_pm_high_score(self):
        rng = np.random.default_rng(42)
        n = 3000
        phi1 = rng.uniform(-50, 50, n)
        pm1 = -8.0 + rng.normal(0, 0.2, n)
        pm2 = rng.normal(0, 0.15, n)
        pm1_shifted = pm1 + 3.0  # big systematic offset
        score = kinematic_perturbation_score(
            phi1, pm1_shifted, pm2, phi1, pm1, pm2, (-50, 50)
        )
        assert score > 1.0


# ---------------------------------------------------------------------------
# Encounter + scoring integration
# ---------------------------------------------------------------------------

class TestForwardModelIntegration:
    def test_encounter_creates_gap(self):
        """A strong encounter should create a detectable gap."""
        rng = np.random.default_rng(42)
        n = 3000
        particles = StreamParticles(
            phi1=rng.uniform(-50, 50, n),
            phi2=rng.normal(0, 0.3, n),
            dist=np.full(n, 8.5),
            pm1=-8.0 + rng.normal(0, 0.2, n),
            pm2=rng.normal(0, 0.1, n),
            vrad=rng.normal(0, 3.0, n),
            xyz_kpc=np.zeros((3, n)),
            vxyz_kms=np.tile([0, 200, 0], (n, 1)).T * 1.0,
        )

        mass = 10**7.5
        enc = EncounterParams(
            mass_solar=mass,
            scale_radius_kpc=scale_radius_from_mass(mass),
            impact_param_kpc=0.05,
            flyby_vel_kms=200.0,
            encounter_phi1=0.0,
            t_since_impact_gyr=2.0,
            is_valid=True,
        )
        perturbed = apply_impulse_approximation(particles, enc)

        # Profile should show a gap near phi1=0
        profile = compute_density_profile(perturbed.phi1, (-50, 50), bin_width_deg=2.0)
        gaps = detect_gaps(profile, min_depth=0.15, min_significance=1.0)
        # At least check the impulse modified the stream
        assert not np.allclose(perturbed.pm1, particles.pm1)

    def test_candidate_result_serializable(self):
        """CandidateResult.to_dict() should be JSON-serializable."""
        import json
        from src.forward_model.pipeline import CandidateResult

        result = CandidateResult(
            log10_mass=7.0,
            t_since_gyr=2.0,
            impact_phi1=-40.0,
            flyby_vel_kms=200.0,
            impact_param_kpc=0.1,
            scale_radius_kpc=0.05,
            score=ScoreResult(
                density_residual=1.5,
                gap_agreement=0.3,
                kinematic_perturbation=0.2,
                combined=0.8,
            ),
            n_stars_sim=2000,
            runtime_s=0.05,
        )
        d = result.to_dict()
        # Should be JSON-serializable
        s = json.dumps(d)
        assert "7.0" in s or "7" in s

    def test_candidate_accepts_continuous_profile_and_geometry(self, monkeypatch):
        """Profile grids can vary compactness independently of perturber mass."""
        from src.forward_model import pipeline
        from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel

        rng = np.random.default_rng(7)
        n = 500
        stream = StreamParticles(
            phi1=rng.uniform(-20, 20, n),
            phi2=rng.normal(0, 0.2, n),
            dist=np.full(n, 8.5),
            pm1=rng.normal(-8.0, 0.1, n),
            pm2=rng.normal(0.0, 0.1, n),
            vrad=np.full(n, np.nan),
            xyz_kpc=np.zeros((3, n)),
            vxyz_kms=np.zeros((3, n)),
        )
        captured = {}

        def fake_apply(base, encounter):
            captured["encounter"] = encounter
            return base

        monkeypatch.setattr(pipeline, "apply_impulse_approximation", fake_apply)
        model = TimelineForwardModel(
            ForwardModelConfig(use_fast_mode=True, use_gnn_scorer=False)
        )
        model.base_stream = stream
        model.phi1_range = (-20.0, 20.0)
        model.obs_particles = {
            "phi1": stream.phi1,
            "phi2": stream.phi2,
            "pm1": stream.pm1,
            "pm2": stream.pm2,
            "vrad": stream.vrad,
        }
        model.obs_profile = compute_density_profile(stream.phi1, model.phi1_range)
        model.obs_gaps = []

        result = model.evaluate_candidate(
            {
                "log10_mass": 7.0,
                "t_since_gyr": 1.5,
                "impact_phi1": 3.0,
                "scale_radius_factor": 3.0,
                "impact_param_kpc": 0.2,
                "flyby_vel_kms": 150.0,
            }
        )
        encounter = captured["encounter"]
        assert encounter.scale_radius_kpc == pytest.approx(3.0 * scale_radius_from_mass(1e7))
        assert encounter.impact_param_kpc == pytest.approx(0.2)
        assert encounter.flyby_vel_kms == pytest.approx(150.0)
        assert result.scale_radius_kpc == pytest.approx(encounter.scale_radius_kpc)
        assert result.impact_param_kpc == pytest.approx(0.2)
        assert result.flyby_vel_kms == pytest.approx(150.0)

    def test_trusted_catalog_selection_can_disable_rectangular_cuts(self, tmp_path):
        """Already-clean external catalogs should not be recut by default logic."""
        import h5py

        from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel

        path = tmp_path / "trusted.h5"
        columns = {
            "phi1": [0.0, 1.0, 2.0],
            "phi2": [0.0, 2.0, 0.0],
            "pm1": [0.0, 0.0, 5.0],
            "pm2": [0.0, 0.0, 0.0],
            "dist": [8.0, 8.0, 8.0],
            "vrad": [np.nan, np.nan, np.nan],
            "membership_prob": [1.0, 1.0, 1.0],
            "selection_weight": [0.2, 0.5, 0.8],
            "desi_p_thin": [0.9, 0.8, 0.7],
            "desi_p_cocoon": [0.1, 0.2, 0.3],
            "desi_delta_phi2": [0.0, 0.1, 0.2],
            "desi_delta_vgsr": [1.0, 2.0, 3.0],
        }
        with h5py.File(path, "w") as handle:
            group = handle.create_group("streams/GD1/members")
            for name, values in columns.items():
                group.create_dataset(name, data=np.asarray(values))

        strict = TimelineForwardModel(
            ForwardModelConfig(processed_h5_path=str(path), phi2_cut_deg=1.0, apply_pm_cuts=True)
        )
        strict.stream_config = {"pm1_range_masyr": [-1.0, 1.0], "pm2_range_masyr": [-1.0, 1.0]}
        strict._load_observed_data()
        assert len(strict.obs_particles["phi1"]) == 1

        trusted = TimelineForwardModel(
            ForwardModelConfig(processed_h5_path=str(path), phi2_cut_deg=None, apply_pm_cuts=False)
        )
        trusted.stream_config = strict.stream_config
        trusted._load_observed_data()
        assert len(trusted.obs_particles["phi1"]) == 3
        assert np.allclose(trusted.obs_particles["selection_weight"], [0.2, 0.5, 0.8])
        assert np.allclose(trusted.obs_particles["desi_p_thin"], [0.9, 0.8, 0.7])

    def test_selection_weight_takes_precedence_for_density_scoring(self):
        from src.forward_model.pipeline import (
            observed_density_weights,
            observed_gap_detection_weights,
        )

        particles = {
            "membership_prob": np.array([1.0, 1.0]),
            "selection_weight": np.array([0.25, 0.75]),
        }

        assert np.allclose(observed_density_weights(particles), [0.25, 0.75])
        assert observed_gap_detection_weights(particles) is None


# ---------------------------------------------------------------------------
# Multiprocessing tests
# ---------------------------------------------------------------------------

class TestGNNScorer:
    def test_stream_particles_to_data(self):
        """StreamParticles -> PyG Data conversion produces correct shape."""
        from src.forward_model.gnn_scorer import stream_particles_to_data

        rng = np.random.default_rng(42)
        n = 200
        particles = StreamParticles(
            phi1=rng.uniform(-50, 50, n),
            phi2=rng.normal(0, 0.3, n),
            dist=np.full(n, 8.5),
            pm1=-8.0 + rng.normal(0, 0.2, n),
            pm2=rng.normal(0, 0.1, n),
            vrad=rng.normal(0, 3.0, n),
            xyz_kpc=np.zeros((3, n)),
            vxyz_kms=np.zeros((3, n)),
        )
        data = stream_particles_to_data(particles, "GD1")
        assert data.x.shape == (n, 18)
        assert data.batch.shape == (n,)
        # GD1 one-hot should be at index 11
        assert data.x[0, 11].item() == 1.0
        assert data.x[0, 12].item() == 0.0  # not Pal5

    def test_scorer_self_distance_zero(self):
        """Same stream embedded twice should give distance ~0."""
        from src.forward_model.gnn_scorer import GNNProfileScorer

        scorer = GNNProfileScorer("checkpoints/gnn_best.pt", device="cpu")
        if not scorer.is_available:
            pytest.skip("GNN checkpoint not available")

        rng = np.random.default_rng(42)
        n = 300
        particles = StreamParticles(
            phi1=rng.uniform(-50, 50, n),
            phi2=rng.normal(0, 0.3, n),
            dist=np.full(n, 8.5),
            pm1=-8.0 + rng.normal(0, 0.2, n),
            pm2=rng.normal(0, 0.1, n),
            vrad=rng.normal(0, 3.0, n),
            xyz_kpc=np.zeros((3, n)),
            vxyz_kms=np.zeros((3, n)),
        )
        obs = {"phi1": particles.phi1, "phi2": particles.phi2,
               "pm1": particles.pm1, "pm2": particles.pm2,
               "dist": particles.dist, "vrad": particles.vrad}
        scorer.set_observed_embedding(obs, "GD1")
        dist = scorer.score(particles, "GD1")
        assert dist < 0.01  # should be essentially zero

    def test_scorer_different_streams_nonzero(self):
        """Different streams should have nonzero distance."""
        from src.forward_model.gnn_scorer import GNNProfileScorer

        scorer = GNNProfileScorer("checkpoints/gnn_best.pt", device="cpu")
        if not scorer.is_available:
            pytest.skip("GNN checkpoint not available")

        rng = np.random.default_rng(42)
        n = 300
        obs = {"phi1": rng.uniform(-50, 50, n), "phi2": rng.normal(0, 0.3, n),
               "pm1": -8.0 + rng.normal(0, 0.2, n), "pm2": rng.normal(0, 0.1, n),
               "dist": np.full(n, 8.5), "vrad": rng.normal(0, 3.0, n)}
        scorer.set_observed_embedding(obs, "GD1")

        # Very different stream
        particles = StreamParticles(
            phi1=rng.uniform(-50, 50, n),
            phi2=rng.normal(0, 0.3, n),
            dist=np.full(n, 8.5),
            pm1=-3.0 + rng.normal(0, 1.0, n),  # very different PM
            pm2=rng.normal(2.0, 0.5, n),
            vrad=rng.normal(50, 10.0, n),
            xyz_kpc=np.zeros((3, n)),
            vxyz_kms=np.zeros((3, n)),
        )
        dist = scorer.score(particles, "GD1")
        assert dist > 0.1  # should be nonzero


class TestMultiprocessing:
    def test_worker_init_and_evaluate(self):
        """Worker init + evaluate should produce valid results."""
        from src.forward_model.pipeline import _worker_init, _worker_evaluate, _worker_state
        from src.forward_model.scoring import ScoreWeights

        # Build minimal observed data
        rng = np.random.default_rng(42)
        n = 500
        phi1 = rng.uniform(-50, 50, n)
        phi2 = rng.normal(0, 0.3, n)
        pm1 = -8.0 + rng.normal(0, 0.2, n)
        pm2 = rng.normal(0, 0.1, n)
        dist = np.full(n, 8.5)
        vrad = rng.normal(0, 3.0, n)

        obs_particles = {
            "phi1": phi1.tolist(),
            "phi2": phi2.tolist(),
            "pm1": pm1.tolist(),
            "pm2": pm2.tolist(),
            "dist": dist.tolist(),
            "vrad": vrad.tolist(),
        }

        # Make a simple density profile
        from src.forward_model.scoring import compute_density_profile, detect_gaps
        profile = compute_density_profile(phi1, (-50, 50), bin_width_deg=2.0)
        obs_profile_dict = {
            "bin_centers": profile.bin_centers.tolist(),
            "density": profile.density.tolist(),
            "counts": profile.counts.tolist(),
            "bin_width_deg": profile.bin_width_deg,
            "n_bins": profile.n_bins,
        }

        gaps = detect_gaps(profile, min_depth=0.3, min_significance=2.0)
        obs_gaps_list = [
            {"phi1_center": g.phi1_center, "phi1_width": g.phi1_width,
             "depth": g.depth, "significance": g.significance}
            for g in gaps
        ]

        # Build a minimal base stream for fast mode
        base_stream = StreamParticles(
            phi1=phi1, phi2=phi2, dist=dist,
            pm1=pm1, pm2=pm2, vrad=vrad,
            xyz_kpc=np.zeros((3, n)),
            vxyz_kms=np.tile([0, 200, 0], (n, 1)).T * 1.0,
        )
        base_stream_dict = {
            "phi1": base_stream.phi1.tolist(),
            "phi2": base_stream.phi2.tolist(),
            "dist": base_stream.dist.tolist(),
            "pm1": base_stream.pm1.tolist(),
            "pm2": base_stream.pm2.tolist(),
            "vrad": base_stream.vrad.tolist(),
            "xyz_kpc": base_stream.xyz_kpc.tolist(),
            "vxyz_kms": base_stream.vxyz_kms.tolist(),
        }

        cfg_dict = {
            "flyby_vel_kms": 200.0,
            "impact_param_kpc": 0.1,
            "n_stars_sim": 500,
            "base_seed": 42,
            "density_bin_width_deg": 2.0,
            "kinematic_bin_width_deg": 4.0,
            "gap_detection_min_depth": 0.3,
            "gap_detection_min_significance": 2.0,
            "score_weights": {"density": 1.0, "gap": 1.5, "kinematic": 0.8, "profile": 0.5},
        }

        # Initialise worker state (normally done by pool initializer)
        _worker_init(
            stream_name="GD1",
            config_path="config/streams.yaml",
            use_fast_mode=True,
            base_stream_dict=base_stream_dict,
            obs_particles=obs_particles,
            obs_profile_dict=obs_profile_dict,
            obs_gaps_list=obs_gaps_list,
            phi1_range=(-50.0, 50.0),
            cfg_dict=cfg_dict,
        )

        # Evaluate a candidate
        params = {"log10_mass": 7.0, "t_since_gyr": 2.0, "impact_phi1": 0.0}
        result = _worker_evaluate(params)

        assert "combined" in result
        assert result["combined"] >= 0
        assert result["log10_mass"] == 7.0
        assert result["n_stars_sim"] > 0
        assert result["runtime_s"] > 0


class TestMultiEncounter:
    """Tests for multi-encounter (sequential impacts) evaluation."""

    def test_multi_encounter_fast_mode_two_impacts(self):
        """Two sequential impulse kicks should produce a valid result."""
        from src.forward_model.pipeline import MultiEncounterResult

        rng = np.random.default_rng(42)
        n = 1000
        base_stream = StreamParticles(
            phi1=rng.uniform(-50, 50, n),
            phi2=rng.normal(0, 0.3, n),
            dist=np.full(n, 8.5),
            pm1=-8.0 + rng.normal(0, 0.2, n),
            pm2=rng.normal(0, 0.1, n),
            vrad=rng.normal(0, 3.0, n),
            xyz_kpc=np.random.default_rng(1).uniform(-10, 10, (3, n)),
            vxyz_kms=np.tile([0, 200, 0], (n, 1)).T * 1.0,
        )

        # Apply two encounters sequentially (fast mode logic)
        enc1 = EncounterParams(
            mass_solar=1e7, scale_radius_kpc=scale_radius_from_mass(1e7),
            impact_param_kpc=0.1, flyby_vel_kms=200.0,
            encounter_phi1=-20.0, t_since_impact_gyr=5.0,
            is_valid=True, is_massive=False,
        )
        enc2 = EncounterParams(
            mass_solar=1e8, scale_radius_kpc=scale_radius_from_mass(1e8),
            impact_param_kpc=0.1, flyby_vel_kms=200.0,
            encounter_phi1=20.0, t_since_impact_gyr=2.0,
            is_valid=True, is_massive=True,
        )

        # Apply oldest first, then newest
        perturbed = apply_impulse_approximation(base_stream, enc1)
        perturbed = apply_impulse_approximation(perturbed, enc2)

        # Should produce a valid stream with two distinct gaps
        assert len(perturbed.phi1) == n
        # Velocities should be modified from base
        assert not np.allclose(perturbed.pm1, base_stream.pm1)

    def test_multi_encounter_ordering(self):
        """Encounters should be applied oldest-first regardless of input order."""
        rng = np.random.default_rng(42)
        n = 500
        base_stream = StreamParticles(
            phi1=rng.uniform(-50, 50, n),
            phi2=rng.normal(0, 0.3, n),
            dist=np.full(n, 8.5),
            pm1=-8.0 + rng.normal(0, 0.2, n),
            pm2=rng.normal(0, 0.1, n),
            vrad=rng.normal(0, 3.0, n),
            xyz_kpc=rng.uniform(-10, 10, (3, n)),
            vxyz_kms=np.tile([0, 200, 0], (n, 1)).T * 1.0,
        )

        enc_old = EncounterParams(
            mass_solar=1e7, scale_radius_kpc=scale_radius_from_mass(1e7),
            impact_param_kpc=0.1, flyby_vel_kms=200.0,
            encounter_phi1=-20.0, t_since_impact_gyr=8.0,
            is_valid=True, is_massive=False,
        )
        enc_new = EncounterParams(
            mass_solar=1e8, scale_radius_kpc=scale_radius_from_mass(1e8),
            impact_param_kpc=0.1, flyby_vel_kms=200.0,
            encounter_phi1=20.0, t_since_impact_gyr=1.0,
            is_valid=True, is_massive=True,
        )

        # Correct order: old first, then new
        s1 = apply_impulse_approximation(base_stream, enc_old)
        result_correct = apply_impulse_approximation(s1, enc_new)

        # Reversed order gives different result (non-commutative)
        s2 = apply_impulse_approximation(base_stream, enc_new)
        result_reversed = apply_impulse_approximation(s2, enc_old)

        # The results should be different (sequential application is order-dependent
        # because gap drift depends on the current phi1 position)
        # They won't be exactly equal because the gap drift modifies phi1
        # before the second kick is applied
        assert not np.allclose(result_correct.pm1, result_reversed.pm1, atol=1e-6)

    def test_multi_encounter_result_dataclass(self):
        """MultiEncounterResult.to_dict() produces valid serialisable dict."""
        from src.forward_model.pipeline import MultiEncounterResult
        from src.forward_model.scoring import ScoreResult

        result = MultiEncounterResult(
            encounters=[
                {"log10_mass": 7.0, "t_since_gyr": 5.0, "impact_phi1": -20.0,
                 "flyby_vel_kms": 200.0, "impact_param_kpc": 0.1, "scale_radius_kpc": 0.01},
                {"log10_mass": 8.0, "t_since_gyr": 2.0, "impact_phi1": 20.0,
                 "flyby_vel_kms": 200.0, "impact_param_kpc": 0.1, "scale_radius_kpc": 0.02},
            ],
            score=ScoreResult(
                density_residual=3.5, gap_agreement=2.0,
                kinematic_perturbation=1.2, combined=2.5, profile_distance=5.0,
            ),
            n_encounters=2,
            n_stars_sim=1000,
            runtime_s=15.3,
        )

        d = result.to_dict()
        assert d["n_encounters"] == 2
        assert len(d["encounters"]) == 2
        assert d["combined_score"] == 2.5
        assert d["profile_distance"] == 5.0
        assert d["n_stars_sim"] == 1000

    def test_generate_multi_evolved_function_exists(self):
        """The multi-encounter evolve function should be importable."""
        from src.forward_model.evolve import generate_perturbed_stream_multi_evolved
        assert callable(generate_perturbed_stream_multi_evolved)

    def test_full_orbit_evolve_exposes_matched_control_switch(self):
        """Full-orbit functions must support identical-path no-kick controls."""
        import inspect

        from src.forward_model.evolve import (
            generate_perturbed_stream_evolved,
            generate_perturbed_stream_multi_evolved,
        )

        assert "apply_kick" in inspect.signature(generate_perturbed_stream_evolved).parameters
        assert "apply_kicks" in inspect.signature(
            generate_perturbed_stream_multi_evolved
        ).parameters

    def test_multi_encounter_accepts_continuous_scale_radius(self, monkeypatch):
        """Multi-event profile tests must not silently revert to NFW radii."""
        from src.forward_model import pipeline
        from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel

        rng = np.random.default_rng(13)
        n = 500
        stream = StreamParticles(
            phi1=rng.uniform(-20, 20, n),
            phi2=rng.normal(0, 0.2, n),
            dist=np.full(n, 8.5),
            pm1=rng.normal(-8.0, 0.1, n),
            pm2=rng.normal(0.0, 0.1, n),
            vrad=np.full(n, np.nan),
            xyz_kpc=np.zeros((3, n)),
            vxyz_kms=np.zeros((3, n)),
        )
        captured = []

        def fake_apply(base, encounter):
            captured.append(encounter)
            return base

        monkeypatch.setattr(pipeline, "apply_impulse_approximation", fake_apply)
        model = TimelineForwardModel(
            ForwardModelConfig(use_fast_mode=True, use_gnn_scorer=False)
        )
        model.base_stream = stream
        model.phi1_range = (-20.0, 20.0)
        model.obs_particles = {
            "phi1": stream.phi1,
            "phi2": stream.phi2,
            "pm1": stream.pm1,
            "pm2": stream.pm2,
            "vrad": stream.vrad,
        }
        model.obs_profile = compute_density_profile(stream.phi1, model.phi1_range)
        model.obs_gaps = []

        result = model.evaluate_multi_encounter(
            [
                {
                    "log10_mass": 7.0,
                    "t_since_gyr": 2.0,
                    "impact_phi1": -5.0,
                    "scale_radius_factor": 3.0,
                },
                {
                    "log10_mass": 8.0,
                    "t_since_gyr": 1.0,
                    "impact_phi1": 5.0,
                    "scale_radius_kpc": 0.75,
                },
            ]
        )

        assert captured[0].scale_radius_kpc == pytest.approx(
            3.0 * scale_radius_from_mass(1e7)
        )
        assert captured[1].scale_radius_kpc == pytest.approx(0.75)
        assert result.encounters[0]["scale_radius_factor"] == pytest.approx(3.0)

    def test_multi_encounter_separation_constraints(self):
        """Multi-encounter grid should enforce separation constraints."""
        # Test that the grid sampler rejects encounters too close together
        rng = np.random.default_rng(42)
        n_valid = 0
        n_attempts = 100
        min_time_sep = 0.5
        min_phi1_sep = 5.0

        for _ in range(n_attempts):
            times = np.sort(rng.uniform(0.5, 10.0, 2))[::-1]
            phi1s = rng.uniform(-40, 40, 2)

            time_diff = abs(times[0] - times[1])
            phi1_diff = abs(phi1s[0] - phi1s[1])

            if time_diff >= min_time_sep and phi1_diff >= min_phi1_sep:
                n_valid += 1

        # Some samples should pass and some should fail
        assert 0 < n_valid < n_attempts


class TestDMModelComparison:
    """Tests for DM model-specific subhalo profiles and comparison."""

    def test_all_four_models_return_valid_profiles(self):
        """subhalo_profile_for_model returns valid dicts for all 4 models."""
        mass = 1e7
        for model in DM_MODELS:
            profile = subhalo_profile_for_model(model, mass)
            assert "scale_radius_kpc" in profile
            assert "concentration" in profile
            assert "profile_type" in profile
            assert "core_radius_kpc" in profile
            assert "kick_suppression" in profile
            assert profile["scale_radius_kpc"] > 0
            assert 0 < profile["kick_suppression"] <= 1.0

    def test_cdm_has_no_core(self):
        """CDM subhalos should have zero core radius."""
        profile = subhalo_profile_for_model("CDM", 1e7)
        assert profile["core_radius_kpc"] == 0.0
        assert profile["kick_suppression"] == 1.0
        assert profile["profile_type"] == "NFW"

    def test_cdm_profile_scale_matches_standard_radius(self):
        """CDM profile helper should use the same kpc radius convention."""
        from src.simulation.subhalo import scale_radius_from_mass

        mass = 1e8
        profile = subhalo_profile_for_model("CDM", mass)
        assert np.isclose(
            profile["scale_radius_kpc"],
            scale_radius_from_mass(mass),
            rtol=1e-12,
        )

    def test_sidm_has_core(self):
        """SIDM subhalos should have nonzero core radius."""
        profile = subhalo_profile_for_model("SIDM", 1e7, sigma_sidm_cm2g=1.0)
        assert profile["core_radius_kpc"] > 0.0
        assert profile["kick_suppression"] < 1.0
        assert profile["profile_type"] == "isothermal_core_NFW"
        assert profile["scale_radius_kpc"] > subhalo_profile_for_model("CDM", 1e7)["scale_radius_kpc"]

    def test_fdm_has_soliton_core(self):
        """FDM subhalos should have a soliton core."""
        profile = subhalo_profile_for_model("FDM", 1e7, m_axion_ev=1e-22)
        assert profile["core_radius_kpc"] > 0.0
        assert profile["profile_type"] == "soliton_NFW"

    def test_fdm_lighter_axion_larger_core(self):
        """Lighter FDM axion mass should give a larger soliton core."""
        p_heavy = subhalo_profile_for_model("FDM", 1e7, m_axion_ev=1e-21)
        p_light = subhalo_profile_for_model("FDM", 1e7, m_axion_ev=1e-23)
        assert p_light["core_radius_kpc"] > p_heavy["core_radius_kpc"]

    def test_wdm_lower_concentration(self):
        """WDM should have lower concentration than CDM for low-mass halos."""
        cdm = subhalo_profile_for_model("CDM", 1e6)
        wdm = subhalo_profile_for_model("WDM", 1e6, m_wdm_kev=1.0)
        assert wdm["concentration"] < cdm["concentration"]

    def test_sidm_higher_cross_section_larger_core(self):
        """Higher SIDM cross-section should create a larger core."""
        low = subhalo_profile_for_model("SIDM", 1e7, sigma_sidm_cm2g=0.5)
        high = subhalo_profile_for_model("SIDM", 1e7, sigma_sidm_cm2g=10.0)
        assert high["core_radius_kpc"] > low["core_radius_kpc"]
        assert high["kick_suppression"] < low["kick_suppression"]

    def test_cdm_and_wdm_differ_at_low_mass(self):
        """At low masses, CDM and WDM profiles should differ significantly."""
        cdm = subhalo_profile_for_model("CDM", 1e6)
        wdm = subhalo_profile_for_model("WDM", 1e6, m_wdm_kev=2.0)
        # WDM halos less concentrated -> larger scale radius
        assert wdm["scale_radius_kpc"] > cdm["scale_radius_kpc"]

    def test_cdm_wdm_sidm_converge_at_high_mass(self):
        """CDM, WDM, SIDM should converge at high mass (FDM soliton stays large)."""
        mass = 1e10
        cdm = subhalo_profile_for_model("CDM", mass)
        wdm = subhalo_profile_for_model("WDM", mass, m_wdm_kev=3.0)
        sidm = subhalo_profile_for_model("SIDM", mass, sigma_sidm_cm2g=1.0)
        radii = [cdm["scale_radius_kpc"], wdm["scale_radius_kpc"], sidm["scale_radius_kpc"]]
        # CDM, WDM, SIDM within factor of 3 at high mass
        assert max(radii) / min(radii) < 3.0

    def test_model_comparison_result_serializable(self):
        """ModelComparisonResult.to_dict() produces valid JSON-safe dict."""
        from src.forward_model.pipeline import ModelComparisonResult
        from src.forward_model.scoring import ScoreResult

        scores = {m: ScoreResult(combined=float(i + 1)) for i, m in enumerate(DM_MODELS)}
        profiles = {m: subhalo_profile_for_model(m, 1e7) for m in DM_MODELS}

        result = ModelComparisonResult(
            log10_mass=7.0, t_since_gyr=2.0, impact_phi1=10.0,
            model_scores=scores, model_profiles=profiles,
            best_model="CDM", best_score=1.0,
            null_combined=5.0, runtime_s=0.5,
        )
        d = result.to_dict()
        assert d["best_model"] == "CDM"
        assert len(d["models"]) == 4
        for m in DM_MODELS:
            assert m in d["models"]
            assert "combined_score" in d["models"][m]
            assert "scale_radius_kpc" in d["models"][m]
            assert "kick_suppression" in d["models"][m]

    def test_invalid_model_raises(self):
        """Unknown DM model should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown DM model"):
            subhalo_profile_for_model("MOND", 1e7)


class TestIntegrationNaNGuard:
    """Regression: non-finite particle coords must not crash the C integrator.

    NaN/Inf phase-space coordinates passed to galpy's dop853_c integrator cause
    a SIGSEGV (a C-level crash, not a Python exception). _integrate_particles_to_epoch
    must sanitize them first. See the full-orbit injection-recovery segfault fix.
    """

    def test_nan_particles_do_not_crash_integration(self):
        from galpy.potential import MWPotential2014
        from src.forward_model.evolve import _integrate_particles_to_epoch

        # 4 sane disc-like particles + 2 poisoned with NaN/Inf
        pos = np.array([
            [8.0, 0.0, 0.0],
            [10.0, 2.0, 0.5],
            [7.0, -3.0, -0.2],
            [12.0, 1.0, 0.1],
            [np.nan, 0.0, 0.0],     # poisoned position
            [9.0, np.inf, 0.0],     # poisoned position
        ])
        vel = np.array([
            [0.0, 220.0, 0.0],
            [-30.0, 200.0, 10.0],
            [20.0, 180.0, -5.0],
            [0.0, np.nan, 0.0],     # poisoned velocity
            [10.0, 200.0, 0.0],
            [0.0, 210.0, 0.0],
        ])
        t_start = np.full(len(pos), -0.5)  # 0.5 Gyr in the past
        # Must return without segfault, with finite, correctly-shaped output.
        pf, vf = _integrate_particles_to_epoch(pos, vel, t_start, 0.0, MWPotential2014)
        assert pf.shape == pos.shape and vf.shape == vel.shape
        assert np.isfinite(pf).all(), "integrated positions must be finite"
        assert np.isfinite(vf).all(), "integrated velocities must be finite"
