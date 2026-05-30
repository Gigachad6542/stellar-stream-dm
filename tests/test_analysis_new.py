"""
Tests for newly added analysis and inference modules:
    - power_spectrum_baseline.py
    - suppression_scale.py
    - diagnostics.py
    - splits.py
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Power Spectrum Baseline tests
# ---------------------------------------------------------------------------

class TestPowerSpectrumBaseline:
    """Tests for src/analysis/power_spectrum_baseline.py"""

    def test_compute_density_1d_basic(self):
        from src.analysis.power_spectrum_baseline import compute_density_1d

        rng = np.random.default_rng(42)
        phi1 = rng.uniform(-50, 10, size=1000)
        centers, density = compute_density_1d(phi1, n_bins=64)

        assert len(centers) == 64
        assert len(density) == 64
        assert density.sum() == 1000  # total count preserved
        assert density.dtype == np.float64

    def test_compute_density_1d_custom_range(self):
        from src.analysis.power_spectrum_baseline import compute_density_1d

        phi1 = np.linspace(-30, 30, 500)
        centers, density = compute_density_1d(phi1, n_bins=128, phi1_range=(-40, 40))

        assert centers[0] > -40 and centers[-1] < 40
        assert len(centers) == 128

    def test_detrend_polynomial(self):
        from src.analysis.power_spectrum_baseline import detrend_density

        # Linear trend + noise
        density = np.linspace(10, 100, 200) + np.random.randn(200) * 2
        residual = detrend_density(density, method="polynomial", poly_order=1)

        # Residual should have near-zero mean and be much smaller than original
        assert abs(np.mean(residual)) < 5.0
        assert np.std(residual) < np.std(density)

    def test_detrend_smooth(self):
        from src.analysis.power_spectrum_baseline import detrend_density

        density = np.ones(100) * 50.0
        density[40:50] = 20.0  # gap
        residual = detrend_density(density, method="smooth", smooth_sigma_bins=20.0)

        # The gap should be visible as a negative residual
        assert np.min(residual[40:50]) < -10.0

    def test_compute_power_spectrum_shape(self):
        from src.analysis.power_spectrum_baseline import compute_power_spectrum

        rng = np.random.default_rng(123)
        phi1 = rng.uniform(-60, 10, size=2000)
        result = compute_power_spectrum(phi1, n_bins=128)

        # rfft of 128 points gives 65 values; minus DC = 64
        assert len(result.frequencies) == 64
        assert len(result.power) == 64
        assert result.n_stars == 2000
        # Normalized power sums to 1
        assert abs(result.power.sum() - 1.0) < 1e-6

    def test_power_spectrum_detects_periodic_signal(self):
        from src.analysis.power_spectrum_baseline import compute_power_spectrum

        # Create stream with periodic gaps every 10 degrees
        phi1_base = np.linspace(-60, 60, 5000)
        # Remove stars in periodic gaps
        gap_mask = np.cos(2 * np.pi * phi1_base / 10.0) > 0.5
        phi1 = phi1_base[~gap_mask]

        result = compute_power_spectrum(phi1, n_bins=256, phi1_range=(-60, 60))

        # Should have a peak near frequency = 1/10 = 0.1 cycles/deg
        peak_idx = np.argmax(result.power)
        peak_freq = result.frequencies[peak_idx]
        assert 0.05 < peak_freq < 0.2  # should be near 0.1

    def test_power_spectrum_empty_input(self):
        from src.analysis.power_spectrum_baseline import compute_power_spectrum

        result = compute_power_spectrum(np.array([1.0, 2.0, 3.0]), n_bins=64)
        # Too few stars — should return empty
        assert len(result.power) == 0

    def test_power_spectrum_summary_vector_length(self):
        from src.analysis.power_spectrum_baseline import power_spectrum_summary_vector

        rng = np.random.default_rng(42)
        phi1 = rng.uniform(-50, 10, size=1000)
        bp = power_spectrum_summary_vector(phi1, n_bandpowers=8)

        assert bp.shape == (8,)
        assert not np.any(np.isnan(bp))

    def test_power_ratio_perturbed_exceeds_smooth(self):
        from src.analysis.power_spectrum_baseline import compute_power_ratio

        rng = np.random.default_rng(42)
        # Smooth stream
        smooth = rng.uniform(-50, 10, size=2000)
        # Perturbed: add gaps
        perturbed = smooth.copy()
        mask = (perturbed > -20) & (perturbed < -17)
        perturbed = perturbed[~mask]

        result = compute_power_ratio(perturbed, smooth, n_bins=128, phi1_range=(-50, 10))

        assert result.power_ratio is not None
        # Ratio should exceed 1 somewhere (perturbation adds power)
        assert np.max(result.power_ratio) > 1.0

    def test_chi_squared_same_distribution(self):
        from src.analysis.power_spectrum_baseline import chi_squared_power_spectrum

        rng = np.random.default_rng(42)
        observed = rng.uniform(-50, 10, size=1000)
        sims = [rng.uniform(-50, 10, size=1000) for _ in range(50)]

        result = chi_squared_power_spectrum(observed, sims, n_bins=64, n_bandpowers=5)

        # Same distribution: chi2 should be reasonable (not enormous)
        assert result["chi2"] < 50  # 5 DOF; extreme would be > 50
        assert result["p_value"] > 0.001

    def test_detect_power_excess_with_gaps(self):
        from src.analysis.power_spectrum_baseline import detect_power_excess

        rng = np.random.default_rng(42)
        smooth = rng.uniform(-50, 10, size=3000)
        # Add multiple gaps
        perturbed = smooth.copy()
        mask = ((perturbed > -30) & (perturbed < -27)) | ((perturbed > -10) & (perturbed < -7))
        perturbed = perturbed[~mask]

        result = detect_power_excess(perturbed, smooth, n_bins=128, phi1_range=(-50, 10))

        assert result["mean_ratio"] > 1.0


# ---------------------------------------------------------------------------
# Suppression Scale Inference tests
# ---------------------------------------------------------------------------

class TestSuppressionScale:
    """Tests for src/inference/suppression_scale.py"""

    def test_M_hm_to_wdm_mass_known_value(self):
        from src.inference.suppression_scale import M_hm_to_wdm_mass

        # M_hm = 1.7e10 Msun should give m_wdm = 1 keV (by definition of the formula)
        m_wdm = M_hm_to_wdm_mass(np.log10(1.7e10))
        assert abs(m_wdm - 1.0) < 0.01

    def test_M_hm_to_wdm_mass_monotonic(self):
        from src.inference.suppression_scale import M_hm_to_wdm_mass

        # Higher M_hm -> lower m_wdm (more suppression -> lighter particle)
        m1 = M_hm_to_wdm_mass(7.0)
        m2 = M_hm_to_wdm_mass(8.0)
        m3 = M_hm_to_wdm_mass(9.0)
        assert m1 > m2 > m3

    def test_M_hm_to_fdm_mass(self):
        from src.inference.suppression_scale import M_hm_to_fdm_mass

        # M_hm = 1.5e8 -> m_axion = 1e-22 eV (by definition)
        m_axion = M_hm_to_fdm_mass(np.log10(1.5e8))
        assert abs(m_axion - 1e-22) / 1e-22 < 0.01

    def test_M_hm_to_fdm_mass_monotonic(self):
        from src.inference.suppression_scale import M_hm_to_fdm_mass

        m1 = M_hm_to_fdm_mass(7.0)
        m2 = M_hm_to_fdm_mass(8.0)
        assert m1 > m2  # lower M_hm -> heavier axion

    def test_infer_suppression_no_suppression(self):
        from src.inference.suppression_scale import infer_suppression_from_counts

        # If n_impacts matches CDM expectation, M_hm should be at the lower bound
        n_impacts = np.ones(100) * 5.0
        n_cdm = 5.0
        log10_M_sub = np.ones(100) * 7.0

        log10_M_hm = infer_suppression_from_counts(n_impacts, n_cdm, log10_M_sub)

        # No suppression: M_hm should be at/near the minimum (5.0)
        assert np.median(log10_M_hm) < 6.0

    def test_infer_suppression_strong_suppression(self):
        from src.inference.suppression_scale import infer_suppression_from_counts

        # Few impacts -> strong suppression -> high M_hm
        n_impacts = np.ones(100) * 1.0
        n_cdm = 8.0
        log10_M_sub = np.ones(100) * 7.0

        log10_M_hm = infer_suppression_from_counts(n_impacts, n_cdm, log10_M_sub)

        # Strong suppression (1/8 = 12.5% of CDM): M_hm should be significantly
        # above the lower bound of 5.0 (indicating real suppression detected)
        assert np.median(log10_M_hm) > 6.0
        # And should be above the no-suppression case
        n_impacts_cdm = np.ones(100) * 8.0  # same as CDM expectation
        log10_M_hm_null = infer_suppression_from_counts(n_impacts_cdm, n_cdm, log10_M_sub)
        assert np.median(log10_M_hm) > np.median(log10_M_hm_null)

    def test_compute_upper_limit(self):
        from src.inference.suppression_scale import compute_upper_limit

        rng = np.random.default_rng(42)
        samples = rng.normal(7.5, 0.5, 10000)

        result = compute_upper_limit(samples, confidence_level=0.95)

        assert result["log10_M_hm_upper"] > result["log10_M_hm_median"]
        assert result["log10_M_hm_median"] == pytest.approx(7.5, abs=0.1)
        assert result["confidence_level"] == 0.95

    def test_compute_upper_limit_constraining(self):
        from src.inference.suppression_scale import compute_upper_limit

        # Tight posterior well below prior boundary
        samples = np.ones(1000) * 7.0 + np.random.randn(1000) * 0.2
        result = compute_upper_limit(samples)
        assert result["is_constraining"] is True

    def test_compute_upper_limit_unconstraining(self):
        from src.inference.suppression_scale import compute_upper_limit

        # Flat posterior near the boundary
        samples = np.random.uniform(8.5, 10.0, 1000)
        result = compute_upper_limit(samples)
        assert result["is_constraining"] is False


# ---------------------------------------------------------------------------
# Data Splits tests
# ---------------------------------------------------------------------------

class TestDataSplits:
    """Tests for src/data/splits.py"""

    def test_basic_split_sizes(self):
        from src.data.splits import compute_split_indices

        splits = compute_split_indices(1000)

        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 1000
        assert len(splits["train"]) == pytest.approx(700, abs=5)
        assert len(splits["val"]) == pytest.approx(150, abs=5)
        assert len(splits["test"]) == pytest.approx(150, abs=5)

    def test_no_overlap_between_splits(self):
        from src.data.splits import compute_split_indices

        splits = compute_split_indices(500)

        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_all_indices_covered(self):
        from src.data.splits import compute_split_indices

        splits = compute_split_indices(200)

        all_idx = np.concatenate([splits["train"], splits["val"], splits["test"]])
        assert set(all_idx) == set(range(200))

    def test_stratified_split_balance(self):
        from src.data.splits import compute_split_indices

        # 4 models, 100 each
        labels = np.repeat([0, 1, 2, 3], 100)
        splits = compute_split_indices(400, dm_model_labels=labels)

        # Each split should have ~equal representation of each model
        for split_name, idx in splits.items():
            split_labels = labels[idx]
            for model_id in range(4):
                count = np.sum(split_labels == model_id)
                expected = len(idx) / 4
                assert count == pytest.approx(expected, abs=5), \
                    f"Model {model_id} in {split_name}: got {count}, expected ~{expected}"

    def test_reproducibility_with_seed(self):
        from src.data.splits import compute_split_indices

        s1 = compute_split_indices(1000, seed=42)
        s2 = compute_split_indices(1000, seed=42)

        np.testing.assert_array_equal(s1["train"], s2["train"])
        np.testing.assert_array_equal(s1["val"], s2["val"])
        np.testing.assert_array_equal(s1["test"], s2["test"])

    def test_different_seed_gives_different_split(self):
        from src.data.splits import compute_split_indices

        s1 = compute_split_indices(1000, seed=42)
        s2 = compute_split_indices(1000, seed=99)

        # Very unlikely to be identical
        assert not np.array_equal(s1["train"], s2["train"])

    def test_custom_ratios(self):
        from src.data.splits import compute_split_indices

        ratios = {"train": 0.80, "val": 0.10, "test": 0.10}
        splits = compute_split_indices(1000, split_ratios=ratios)

        assert len(splits["train"]) == pytest.approx(800, abs=5)

    def test_invalid_ratios_raises(self):
        from src.data.splits import compute_split_indices

        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_split_indices(100, split_ratios={"train": 0.5, "val": 0.2, "test": 0.1})

    def test_split_hash_deterministic(self):
        from src.data.splits import compute_split_indices, get_split_hash

        s1 = compute_split_indices(500, seed=42)
        s2 = compute_split_indices(500, seed=42)

        assert get_split_hash(s1) == get_split_hash(s2)

    def test_split_hash_changes_with_seed(self):
        from src.data.splits import compute_split_indices, get_split_hash

        s1 = compute_split_indices(500, seed=42)
        s2 = compute_split_indices(500, seed=99)

        assert get_split_hash(s1) != get_split_hash(s2)

    def test_save_load_roundtrip(self, tmp_path):
        from src.data.splits import compute_split_indices, save_split_indices, load_split_indices

        splits = compute_split_indices(300, seed=7)
        save_path = tmp_path / "test_split.npz"
        save_split_indices(splits, save_path)

        loaded = load_split_indices(save_path)

        np.testing.assert_array_equal(splits["train"], loaded["train"])
        np.testing.assert_array_equal(splits["val"], loaded["val"])
        np.testing.assert_array_equal(splits["test"], loaded["test"])

    def test_get_subset_indices_single(self):
        from src.data.splits import compute_split_indices, get_subset_indices

        splits = compute_split_indices(100, seed=42)
        train_idx = get_subset_indices(splits, "train")

        np.testing.assert_array_equal(train_idx, np.sort(splits["train"]))

    def test_get_subset_indices_combined(self):
        from src.data.splits import compute_split_indices, get_subset_indices

        splits = compute_split_indices(100, seed=42)
        combined = get_subset_indices(splits, ["train", "val"])

        expected = np.sort(np.concatenate([splits["train"], splits["val"]]))
        np.testing.assert_array_equal(combined, expected)


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """Tests for src/inference/diagnostics.py"""

    def test_ood_score_in_distribution(self):
        from src.inference.diagnostics import _compute_ood_score

        rng = np.random.default_rng(42)
        training = rng.normal(0, 1, (100, 128))
        # A point near the center should have low OOD score
        center_point = np.zeros(128)
        score = _compute_ood_score(center_point, training)

        # An outlier should have higher score
        outlier = np.ones(128) * 10.0
        outlier_score = _compute_ood_score(outlier, training)

        assert outlier_score > score

    def test_detect_misspecification_no_signals(self):
        from src.inference.diagnostics import detect_misspecification

        rng = np.random.default_rng(42)
        training = rng.normal(0, 1, (200, 128))
        obs = rng.normal(0, 1, 128)  # in-distribution

        result = detect_misspecification(
            obs, training,
            ppc_results=None,
            coverage_empirical=0.89,  # close to 0.90 nominal
        )

        assert result.evidence_level == "none"
        assert result.is_misspecified is False

    def test_detect_misspecification_ood(self):
        from src.inference.diagnostics import detect_misspecification

        rng = np.random.default_rng(42)
        training = rng.normal(0, 1, (200, 128))
        obs = np.ones(128) * 10.0  # far from training

        result = detect_misspecification(obs, training)

        # Should flag OOD
        assert result.ood_score > 0
        assert "ood" in result.details or result.evidence_level != "none"

    def test_embedding_ood_report_basic(self):
        from src.inference.diagnostics import embedding_ood_report

        rng = np.random.default_rng(42)
        training = rng.normal(0, 1, (500, 128))

        observed = {
            "GD1": rng.normal(0, 1, 128),       # in-distribution
            "weird_stream": np.ones(128) * 15.0,  # outlier
        }

        report = embedding_ood_report(observed, training)

        assert "GD1" in report
        assert "weird_stream" in report
        assert report["weird_stream"]["is_ood"] == True  # noqa: E712
        # GD1 should probably be in-distribution
        assert report["GD1"]["percentile"] < 99.0


# ---------------------------------------------------------------------------
# Model comparison (updated) tests
# ---------------------------------------------------------------------------

class TestModelComparisonUpdated:
    """Tests for the updated model_comparison.py with binary comparison."""

    def test_suppressed_vs_unsuppressed_cdm_favored(self):
        from src.analysis.model_comparison import compute_suppressed_vs_unsuppressed

        # CDM has highest evidence
        log_evidences = {"CDM": -100.0, "WDM": -105.0, "FDM": -106.0, "SIDM": -104.0}
        result = compute_suppressed_vs_unsuppressed(log_evidences)

        assert result["log10_BF_suppressed_vs_CDM"] < 0  # favors CDM
        assert result["p_suppressed"] < 0.5

    def test_suppressed_vs_unsuppressed_suppression_favored(self):
        from src.analysis.model_comparison import compute_suppressed_vs_unsuppressed

        # Suppressed models have higher evidence
        log_evidences = {"CDM": -110.0, "WDM": -100.0, "FDM": -101.0, "SIDM": -102.0}
        result = compute_suppressed_vs_unsuppressed(log_evidences)

        assert result["log10_BF_suppressed_vs_CDM"] > 0  # favors suppression
        assert result["p_suppressed"] > 0.5

    def test_suppressed_vs_unsuppressed_requires_reference(self):
        from src.analysis.model_comparison import compute_suppressed_vs_unsuppressed

        with pytest.raises(ValueError, match="unsuppressed reference"):
            compute_suppressed_vs_unsuppressed({"WDM": -100.0, "FDM": -101.0})

    def test_suppressed_vs_unsuppressed_needs_suppressed_model(self):
        from src.analysis.model_comparison import compute_suppressed_vs_unsuppressed

        with pytest.raises(ValueError, match="suppressed model"):
            compute_suppressed_vs_unsuppressed({"CDM": -100.0, "SIDM": -101.0})
