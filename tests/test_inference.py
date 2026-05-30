"""
Tests for the inference pipeline.

Validates:
- Prior construction from config
- Posterior combination (product of log-posteriors)
- Credible interval computation
- Log evidence estimation
- Gap detection on synthetic density profiles
- Gap classification
- Model comparison Bayes factor computation
- Model posterior normalization
- Visualization helpers
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.gap_catalog import classify_gaps, detect_gaps, gaps_to_dataframe
from src.analysis.model_comparison import (
    combine_log_evidences_across_streams,
    compute_bayes_factors,
    compute_model_posterior,
)
from src.inference.posteriors import (
    combine_posteriors_product,
    compute_credible_intervals,
    get_param_names,
)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

class TestPriors:
    def test_prior_loads_cdm(self):
        torch = pytest.importorskip("torch")
        from src.inference.sbi_pipeline import build_prior  # noqa: PLC0415
        prior = build_prior("CDM", config_path="config/dm_models.yaml")
        samples = prior.sample((100,))
        # CDM has 2 inferred parameters: log10_M_sub_mean, n_impacts
        assert samples.shape[1] == 2, f"CDM prior should have 2 params, got {samples.shape[1]}"

    def test_prior_loads_wdm(self):
        torch = pytest.importorskip("torch")
        from src.inference.sbi_pipeline import build_prior  # noqa: PLC0415
        prior = build_prior("WDM", config_path="config/dm_models.yaml")
        samples = prior.sample((10,))
        assert samples.shape[1] == 2  # WDM has 2 inferred params

    def test_prior_loads_fdm(self):
        torch = pytest.importorskip("torch")
        from src.inference.sbi_pipeline import build_prior  # noqa: PLC0415
        prior = build_prior("FDM", config_path="config/dm_models.yaml")
        samples = prior.sample((10,))
        assert samples.shape[1] == 2

    def test_prior_loads_sidm(self):
        torch = pytest.importorskip("torch")
        from src.inference.sbi_pipeline import build_prior  # noqa: PLC0415
        prior = build_prior("SIDM", config_path="config/dm_models.yaml")
        samples = prior.sample((10,))
        assert samples.shape[1] == 2

    def test_prior_samples_in_bounds(self):
        torch = pytest.importorskip("torch")
        from src.inference.sbi_pipeline import build_prior  # noqa: PLC0415
        prior = build_prior("CDM", config_path="config/dm_models.yaml")
        samples = prior.sample((1000,)).numpy()
        # log10_M_sub_mean: [5, 9]
        assert np.all(samples[:, 0] >= 5.0)
        assert np.all(samples[:, 0] <= 9.0)


# ---------------------------------------------------------------------------
# Parameter name helpers
# ---------------------------------------------------------------------------

class TestParamNames:
    def test_cdm_param_names(self):
        names = get_param_names("CDM")
        assert isinstance(names, list)
        assert len(names) > 0
        assert "n_impacts" in names

    def test_wdm_param_names(self):
        names = get_param_names("WDM")
        assert isinstance(names, list)
        assert "n_impacts" in names

    def test_all_models_have_n_impacts(self):
        """All DM models should include n_impacts as an inferred parameter."""
        for model in ["CDM", "WDM", "FDM", "SIDM"]:
            names = get_param_names(model)
            assert "n_impacts" in names, f"{model} missing n_impacts parameter"


# ---------------------------------------------------------------------------
# Posterior combination
# ---------------------------------------------------------------------------

class TestPosteriorCombination:
    def _make_samples(self, mean: float, std: float, n: int = 500, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"log10_M_sub_mean": rng.normal(mean, std, n)})

    def test_combine_two_streams_tightens_posterior(self):
        s1 = self._make_samples(7.0, 0.5)
        s2 = self._make_samples(7.0, 0.5)
        result = combine_posteriors_product([s1, s2], ["log10_M_sub_mean"])
        comb = result["combined"]["log10_M_sub_mean"]
        grid, log_post = comb["grid"], comb["log_posterior"]
        post = np.exp(log_post - log_post.max())
        # Product of two Gaussians → half the variance → grid should be more peaked
        width_combined = np.sum(post > post.max() * np.exp(-0.5)) * (grid[1] - grid[0])
        width_single = 2 * 0.5  # ±1 sigma for N(7, 0.5)
        assert width_combined < width_single, "Combined posterior should be narrower than single-stream"

    def test_combine_seven_streams(self):
        """Combining 7 streams should produce a very tight posterior."""
        samples = [self._make_samples(7.0, 0.5, seed=i) for i in range(7)]
        result = combine_posteriors_product(samples, ["log10_M_sub_mean"])
        comb = result["combined"]["log10_M_sub_mean"]
        grid, log_post = comb["grid"], comb["log_posterior"]
        post = np.exp(log_post - log_post.max())
        width = np.sum(post > post.max() * np.exp(-0.5)) * (grid[1] - grid[0])
        assert width < 0.5, f"7-stream combined posterior should be very narrow, got width={width:.3f}"

    def test_combine_inconsistent_streams(self):
        """Streams with different means should produce a broader or bimodal posterior."""
        s1 = self._make_samples(6.0, 0.3, seed=0)
        s2 = self._make_samples(8.0, 0.3, seed=1)
        result = combine_posteriors_product([s1, s2], ["log10_M_sub_mean"])
        # Should still produce a valid result
        comb = result["combined"]["log10_M_sub_mean"]
        assert len(comb["grid"]) > 0
        assert len(comb["log_posterior"]) > 0

    def test_combine_preserves_param_names(self):
        s1 = self._make_samples(7.0, 0.5)
        result = combine_posteriors_product([s1], ["log10_M_sub_mean"])
        assert result["param_names"] == ["log10_M_sub_mean"]


# ---------------------------------------------------------------------------
# Credible intervals
# ---------------------------------------------------------------------------

class TestCredibleIntervals:
    def test_credible_intervals_coverage(self):
        rng = np.random.default_rng(0)
        samples = pd.DataFrame({
            "log10_M_sub_mean": rng.normal(7.0, 0.3, 2000),
            "n_impacts": rng.uniform(0, 10, 2000),
        })
        ci = compute_credible_intervals(samples, ci_level=0.9)
        assert "log10_M_sub_mean" in ci.index
        assert ci.loc["log10_M_sub_mean", "lower"] < 7.0 < ci.loc["log10_M_sub_mean", "upper"]

    def test_credible_intervals_symmetric_normal(self):
        """For a symmetric normal, credible interval should be roughly symmetric."""
        rng = np.random.default_rng(42)
        samples = pd.DataFrame({"x": rng.normal(0.0, 1.0, 10000)})
        ci = compute_credible_intervals(samples, ci_level=0.9)
        lower = ci.loc["x", "lower"]
        upper = ci.loc["x", "upper"]
        # Should be roughly ±1.645 for 90% CI of N(0,1)
        assert abs(lower + 1.645) < 0.15, f"Lower bound {lower:.3f} should be ~-1.645"
        assert abs(upper - 1.645) < 0.15, f"Upper bound {upper:.3f} should be ~+1.645"

    def test_credible_intervals_columns(self):
        """Output should have mean, median, lower, upper, std columns."""
        rng = np.random.default_rng(0)
        samples = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.uniform(-1, 1, 500)})
        ci = compute_credible_intervals(samples, ci_level=0.9)
        for col in ["mean", "median", "lower", "upper", "std"]:
            assert col in ci.columns, f"Missing column: {col}"
        assert len(ci) == 2  # two parameters

    def test_credible_intervals_wider_for_wider_distribution(self):
        """CI should be wider for higher-variance distribution."""
        rng = np.random.default_rng(0)
        narrow = pd.DataFrame({"x": rng.normal(0, 0.1, 5000)})
        wide = pd.DataFrame({"x": rng.normal(0, 2.0, 5000)})
        ci_narrow = compute_credible_intervals(narrow, ci_level=0.9)
        ci_wide = compute_credible_intervals(wide, ci_level=0.9)
        w_narrow = ci_narrow.loc["x", "upper"] - ci_narrow.loc["x", "lower"]
        w_wide = ci_wide.loc["x", "upper"] - ci_wide.loc["x", "lower"]
        assert w_wide > w_narrow, "Wider distribution should have wider CI"

    def test_credible_intervals_68_vs_90(self):
        """68% CI should be narrower than 90% CI."""
        rng = np.random.default_rng(0)
        samples = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
        ci_68 = compute_credible_intervals(samples, ci_level=0.68)
        ci_90 = compute_credible_intervals(samples, ci_level=0.90)
        w_68 = ci_68.loc["x", "upper"] - ci_68.loc["x", "lower"]
        w_90 = ci_90.loc["x", "upper"] - ci_90.loc["x", "lower"]
        assert w_90 > w_68, "90% CI should be wider than 68% CI"


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

class TestGapDetection:
    def _make_stream_with_gap(self, n: int = 2000, gap_phi1: float = -40.0,
                               gap_depth: float = 0.7, seed: int = 0) -> np.ndarray:
        """Synthetic stream with a density gap at gap_phi1."""
        rng = np.random.default_rng(seed)
        phi1 = rng.uniform(-100.0, 20.0, n)
        # Remove stars in the gap
        in_gap = np.abs(phi1 - gap_phi1) < 3.0
        gap_mask = ~(in_gap & (rng.random(n) < gap_depth))
        return phi1[gap_mask]

    def test_detects_planted_gap(self):
        phi1 = self._make_stream_with_gap(n=3000, gap_phi1=-40.0, gap_depth=0.8)
        gaps = detect_gaps(phi1, phi1_range=(-100.0, 20.0), gap_sigma=2.0)
        assert len(gaps) > 0, "Should detect at least one gap"
        centers = [g.phi1_center for g in gaps]
        assert any(abs(c - (-40.0)) < 5.0 for c in centers), \
            f"Planted gap at -40 not found; detected centers: {centers}"

    def test_no_gaps_smooth_stream(self):
        rng = np.random.default_rng(42)
        phi1 = rng.uniform(-100.0, 20.0, 3000)
        gaps = detect_gaps(phi1, phi1_range=(-100.0, 20.0), gap_sigma=4.0)
        assert len(gaps) <= 2, f"Expected <=2 false gaps, found {len(gaps)}"

    def test_deeper_gap_has_higher_significance(self):
        """A deeper gap should have higher significance."""
        phi1_deep = self._make_stream_with_gap(n=3000, gap_phi1=-40.0, gap_depth=0.9, seed=0)
        phi1_shallow = self._make_stream_with_gap(n=3000, gap_phi1=-40.0, gap_depth=0.3, seed=0)
        gaps_deep = detect_gaps(phi1_deep, phi1_range=(-100.0, 20.0), gap_sigma=1.5)
        gaps_shallow = detect_gaps(phi1_shallow, phi1_range=(-100.0, 20.0), gap_sigma=1.5)
        # Find the gap near -40
        sig_deep = max(
            (g.significance for g in gaps_deep if abs(g.phi1_center - (-40.0)) < 10.0),
            default=0.0
        )
        sig_shallow = max(
            (g.significance for g in gaps_shallow if abs(g.phi1_center - (-40.0)) < 10.0),
            default=0.0
        )
        # Deep gap might not be detected if depth=0.3 doesn't meet threshold
        if sig_shallow > 0:
            assert sig_deep >= sig_shallow, "Deeper gap should have >= significance"

    def test_gap_to_dataframe(self):
        """Gap list should convert to DataFrame."""
        phi1 = self._make_stream_with_gap(n=3000, gap_phi1=-40.0, gap_depth=0.8)
        gaps = detect_gaps(phi1, phi1_range=(-100.0, 20.0), gap_sigma=2.0)
        if len(gaps) > 0:
            df = gaps_to_dataframe(gaps, stream_name="TestStream")
            assert isinstance(df, pd.DataFrame)
            assert "stream" in df.columns
            assert "phi1_center_deg" in df.columns
            assert len(df) == len(gaps)

    def test_gap_classification_runs(self):
        phi1 = self._make_stream_with_gap(n=2000, gap_phi1=-40.0)
        gaps = detect_gaps(phi1, phi1_range=(-100.0, 20.0), gap_sigma=2.0)
        rng = np.random.default_rng(0)
        dm_samples = pd.DataFrame({
            "log10_M_sub_mean": rng.normal(7.0, 0.5, 500),
            "n_impacts": rng.poisson(3, 500).astype(float),
        })
        gaps = classify_gaps(gaps, dm_samples, None)
        for gap in gaps:
            assert 0.0 <= gap.p_dm_subhalo <= 1.0
            assert 0.0 <= gap.p_baryonic <= 1.0
            assert 0.0 <= gap.p_noise <= 1.0
            total = gap.p_dm_subhalo + gap.p_baryonic + gap.p_noise
            assert abs(total - 1.0) < 0.01, f"Probabilities don't sum to 1: {total}"

    def test_gap_classification_probabilities_in_range(self):
        """All classification probabilities should be in [0, 1]."""
        phi1 = self._make_stream_with_gap(n=2000, gap_phi1=-40.0, gap_depth=0.9)
        gaps = detect_gaps(phi1, phi1_range=(-100.0, 20.0), gap_sigma=2.0)
        rng = np.random.default_rng(0)
        dm_samples = pd.DataFrame({
            "log10_M_sub_mean": rng.normal(7.0, 0.5, 500),
            "n_impacts": rng.poisson(3, 500).astype(float),
        })
        gaps = classify_gaps(gaps, dm_samples, None)
        for gap in gaps:
            for attr in ["p_dm_subhalo", "p_baryonic", "p_noise"]:
                val = getattr(gap, attr)
                assert 0.0 <= val <= 1.0, f"Gap {attr} = {val} out of [0, 1]"

    def test_gap_classification_without_n_impacts(self):
        """classify_gaps should work when n_impacts column is missing (fallback to 1)."""
        phi1 = self._make_stream_with_gap(n=2000, gap_phi1=-40.0)
        gaps = detect_gaps(phi1, phi1_range=(-100.0, 20.0), gap_sigma=2.0)
        rng = np.random.default_rng(0)
        dm_samples = pd.DataFrame({"log10_M_sub_mean": rng.normal(7.0, 0.5, 500)})
        gaps = classify_gaps(gaps, dm_samples, None)
        for gap in gaps:
            total = gap.p_dm_subhalo + gap.p_baryonic + gap.p_noise
            assert abs(total - 1.0) < 0.01

    def test_gap_classification_high_mass_favors_dm(self):
        """Posterior with high masses should produce higher P(DM) than low masses."""
        # Use a very deep gap to ensure detection above 3σ
        phi1 = self._make_stream_with_gap(n=5000, gap_phi1=-40.0, gap_depth=0.8)
        gaps = detect_gaps(phi1, phi1_range=(-100.0, 20.0), gap_sigma=2.0)
        significant_gaps = [g for g in gaps if g.significance >= 3.0]
        if len(significant_gaps) == 0:
            pytest.skip("No gap detected above 3-sigma")
        rng = np.random.default_rng(0)
        # High mass posterior → deep gap is plausible → high P(DM)
        high_mass = pd.DataFrame({
            "log10_M_sub_mean": rng.normal(8.0, 0.2, 500),
            "n_impacts": np.full(500, 2.0),
        })
        # Low mass posterior → deep gap is unlikely → lower P(DM)
        low_mass = pd.DataFrame({
            "log10_M_sub_mean": rng.normal(5.0, 0.2, 500),
            "n_impacts": np.full(500, 2.0),
        })
        from copy import deepcopy
        gaps_high = classify_gaps(deepcopy(significant_gaps), high_mass, None)
        gaps_low = classify_gaps(deepcopy(significant_gaps), low_mass, None)
        # At least one gap should show higher P(DM) for higher masses
        any_higher = any(
            gh.p_dm_subhalo > gl.p_dm_subhalo
            for gh, gl in zip(gaps_high, gaps_low)
        )
        assert any_higher, "High-mass posterior should produce higher P(DM)"


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

class TestModelComparison:
    def test_bayes_factors_correct(self):
        log_evidences = {"CDM": 100.0, "WDM": 95.0, "FDM": 90.0, "SIDM": 102.0}
        bf_df = compute_bayes_factors(log_evidences, reference_model="CDM")
        assert len(bf_df) == 4
        cdm_row = bf_df[bf_df["model"] == "CDM"]
        assert float(cdm_row["log10_BF_vs_CDM"].iloc[0]) == pytest.approx(0.0, abs=1e-6)

    def test_bayes_factors_positive_for_preferred(self):
        """Model with highest evidence should have positive BF vs CDM."""
        log_evidences = {"CDM": 100.0, "WDM": 105.0, "FDM": 90.0, "SIDM": 98.0}
        bf_df = compute_bayes_factors(log_evidences, reference_model="CDM")
        wdm_row = bf_df[bf_df["model"] == "WDM"]
        assert float(wdm_row["log10_BF_vs_CDM"].iloc[0]) > 0

    def test_model_posterior_sums_to_one(self):
        log_evidences = {"CDM": 100.0, "WDM": 98.0, "FDM": 95.0, "SIDM": 101.0}
        post_df = compute_model_posterior(log_evidences)
        total = post_df["posterior_probability"].sum()
        assert abs(total - 1.0) < 1e-6

    def test_model_posterior_highest_for_best_evidence(self):
        """Model with highest evidence should have highest posterior probability."""
        log_evidences = {"CDM": 100.0, "WDM": 98.0, "FDM": 95.0, "SIDM": 105.0}
        post_df = compute_model_posterior(log_evidences)
        best = post_df.loc[post_df["posterior_probability"].idxmax(), "model"]
        assert best == "SIDM", f"Expected SIDM to be best, got {best}"

    def test_model_posterior_equal_evidences(self):
        """Equal evidences should give equal posteriors."""
        log_evidences = {"CDM": 50.0, "WDM": 50.0, "FDM": 50.0, "SIDM": 50.0}
        post_df = compute_model_posterior(log_evidences)
        for _, row in post_df.iterrows():
            assert abs(row["posterior_probability"] - 0.25) < 1e-6

    def test_combine_evidences_across_streams(self):
        per_stream = {
            "GD1": {"CDM": 50.0, "WDM": 48.0},
            "Pal5": {"CDM": 30.0, "WDM": 29.0},
        }
        combined = combine_log_evidences_across_streams(per_stream)
        assert combined["CDM"] == pytest.approx(80.0)
        assert combined["WDM"] == pytest.approx(77.0)

    def test_combine_evidences_all_streams(self):
        """Combining across multiple streams should sum log evidences."""
        per_stream = {
            f"stream_{i}": {"CDM": 10.0 + i, "WDM": 9.0 + i}
            for i in range(7)
        }
        combined = combine_log_evidences_across_streams(per_stream)
        expected_cdm = sum(10.0 + i for i in range(7))
        expected_wdm = sum(9.0 + i for i in range(7))
        assert combined["CDM"] == pytest.approx(expected_cdm)
        assert combined["WDM"] == pytest.approx(expected_wdm)

    def test_bayes_factors_ranking(self):
        """Model with highest evidence should have highest Bayes factor."""
        log_evidences = {"CDM": 50.0, "WDM": 55.0, "FDM": 45.0, "SIDM": 52.0}
        bf_df = compute_bayes_factors(log_evidences, reference_model="CDM")
        wdm_bf = float(bf_df[bf_df["model"] == "WDM"]["log10_BF_vs_CDM"].iloc[0])
        fdm_bf = float(bf_df[bf_df["model"] == "FDM"]["log10_BF_vs_CDM"].iloc[0])
        assert wdm_bf > fdm_bf, "WDM (higher evidence) should have higher BF than FDM"


# ---------------------------------------------------------------------------
# Visualization (smoke tests)
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_plot_density_profile_runs(self):
        """Density profile plot should not crash."""
        import matplotlib
        matplotlib.use("Agg")
        from src.analysis.visualization import plot_density_profile  # noqa: PLC0415
        rng = np.random.default_rng(0)
        phi1 = rng.uniform(-100.0, 20.0, 2000)
        fig = plot_density_profile(phi1, stream_name="TestStream", phi1_range=(-100.0, 20.0))
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_corner_runs(self):
        """Corner plot should not crash."""
        import matplotlib
        matplotlib.use("Agg")
        from src.analysis.visualization import plot_corner  # noqa: PLC0415
        samples = pd.DataFrame({
            "log10_M_sub_mean": np.random.randn(500) + 7.0,
            "n_impacts": np.random.uniform(0, 10, 500),
        })
        fig = plot_corner(samples, title="Test Corner Plot")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
