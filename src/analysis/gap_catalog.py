"""
Gap detection and classification in stellar stream density profiles.

Detects density underdensities using a matched filter (Erkal+2017 approach),
assigns significance, and classifies each gap as DM subhalo / GMC / bar / noise.

The classification uses the trained posterior:
    P(DM) = fraction of posterior samples with M_sub > threshold
    P(baryonic) = 1 - P(DM) when baryonic model is preferred
    P(noise) = when gap significance < 3 sigma
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

log = logging.getLogger(__name__)


@dataclass
class Gap:
    """A detected density underdensity in a stellar stream."""
    phi1_center: float          # gap center [deg]
    phi1_width: float           # gap width [deg]
    significance: float         # detection significance [sigma]
    depth: float                # fractional depth (1 - N_gap / N_smooth)
    p_dm_subhalo: float = 0.0   # posterior probability: dark matter subhalo
    p_baryonic: float = 0.0     # posterior probability: baryonic (GMC or bar)
    p_noise: float = 0.0        # posterior probability: noise fluctuation
    log10_mass_dm_50: float = np.nan   # 50th percentile of inferred subhalo mass
    log10_mass_dm_lo: float = np.nan   # 16th percentile
    log10_mass_dm_hi: float = np.nan   # 84th percentile


# ---------------------------------------------------------------------------
# 1D density profile
# ---------------------------------------------------------------------------

def compute_density_profile(
    phi1: np.ndarray,
    n_bins: int = 200,
    phi1_range: tuple[float, float] = (-100.0, 20.0),
    smooth_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin and smooth the stellar density along phi1.

    Args:
        phi1: Stream longitude array [deg].
        n_bins: Number of phi1 bins.
        phi1_range: (min, max) of phi1.
        smooth_sigma: Gaussian smoothing sigma in bins.

    Returns:
        (bin_centers, smoothed_density) arrays.
    """
    bins = np.linspace(phi1_range[0], phi1_range[1], n_bins + 1)
    counts, _ = np.histogram(phi1, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    smooth = gaussian_filter1d(counts.astype(float), sigma=smooth_sigma)
    return centers, smooth


# ---------------------------------------------------------------------------
# Matched filter gap detection (Erkal+2017 prescription)
# ---------------------------------------------------------------------------

def detect_gaps(
    phi1: np.ndarray,
    n_bins: int = 200,
    phi1_range: tuple[float, float] = (-100.0, 20.0),
    smooth_sigma: float = 3.0,
    gap_sigma: float = 2.0,
    min_gap_width_bins: int = 3,
    max_gap_width_bins: int = 20,
) -> list[Gap]:
    """Detect gaps using a matched filter on the smoothed density profile.

    Method:
    1. Compute smooth density model (wide Gaussian smoothing as background).
    2. Residuals = observed - smooth_background.
    3. Find negative peaks in residuals with significance > gap_sigma.
    4. Compute gap width from the half-depth contour.

    Args:
        phi1: Star phi1 positions [deg].
        smooth_sigma: Smoothing sigma for gap search [bins].
        gap_sigma: Minimum significance threshold for gap detection.
        min_gap_width_bins: Minimum gap width in bins.
        max_gap_width_bins: Maximum gap width in bins.

    Returns:
        List of Gap objects sorted by phi1_center.
    """
    centers, density = compute_density_profile(phi1, n_bins, phi1_range, smooth_sigma=1.0)
    bin_width_deg = (phi1_range[1] - phi1_range[0]) / n_bins

    # Background model: broad smoothing to capture large-scale profile
    background = gaussian_filter1d(density, sigma=15.0)

    # Residuals and noise estimate
    residuals = density - background
    noise = np.std(residuals[background > background.max() * 0.1])
    if noise < 1e-6:
        return []

    # Find negative peaks (gaps)
    inverted = -residuals
    peak_idx, props = find_peaks(
        inverted,
        height=gap_sigma * noise,
        width=(min_gap_width_bins, max_gap_width_bins),
        prominence=gap_sigma * noise * 0.5,
    )

    gaps = []
    for i, idx in enumerate(peak_idx):
        significance = float(inverted[idx] / noise)
        depth_frac = float(inverted[idx] / (background[idx] + 1e-6))
        width_bins = float(props["widths"][i]) if "widths" in props else min_gap_width_bins
        width_deg = width_bins * bin_width_deg

        gaps.append(Gap(
            phi1_center=float(centers[idx]),
            phi1_width=width_deg,
            significance=significance,
            depth=depth_frac,
        ))

    gaps.sort(key=lambda g: g.phi1_center)
    log.info("Detected %d gaps (significance > %.1f sigma)", len(gaps), gap_sigma)
    return gaps


# ---------------------------------------------------------------------------
# Gap classification using the posterior
# ---------------------------------------------------------------------------

def _gap_depth_log_likelihood(
    depth: float,
    width_deg: float,
    log10_masses: np.ndarray,
    n_impacts: np.ndarray,
) -> np.ndarray:
    """Compute per-sample log-likelihood of an observed gap under the DM model.

    Gap depth scales as M_sub^0.5 (Erkal+2016 analytic result) and gap width
    scales as dv/v_orb * t * f_geom.  For a single dominant impact:

        depth_pred(M) = min(1, 0.3 * (M / 1e7)^0.5)

    We model the likelihood as Gaussian:
        p(depth | M, n) = N(depth_pred, sigma_depth)

    where sigma_depth captures unresolved systematics (Poisson shot noise in
    the density profile, background subtraction uncertainty).

    Args:
        depth: Observed gap depth fraction (0-1).
        width_deg: Observed gap width in degrees.
        log10_masses: [S] posterior samples of log10(M_sub / Msun).
        n_impacts: [S] posterior samples of n_impacts.

    Returns:
        [S] log-likelihood values.
    """
    masses = 10.0 ** log10_masses
    # Predicted depth: power-law scaling, capped at 1.0
    depth_pred = np.minimum(1.0, 0.3 * (masses / 1.0e7) ** 0.5)
    # Wider gaps require higher mass; penalise mismatch
    width_factor = np.exp(-0.5 * ((width_deg - 3.0) / 5.0) ** 2)  # prior ~3 deg
    # Multiple impacts: each reduces expected depth per gap (energy spread)
    n_eff = np.maximum(n_impacts, 1.0)
    depth_pred_adj = depth_pred / np.sqrt(n_eff)
    # Gaussian likelihood
    sigma_depth = 0.15 + 0.05 * depth  # heteroscedastic noise model
    log_lik = -0.5 * ((depth - depth_pred_adj) / sigma_depth) ** 2
    log_lik -= np.log(sigma_depth * np.sqrt(2.0 * np.pi))
    return log_lik + np.log(width_factor + 1e-300)


def _baryonic_gap_log_likelihood(
    depth: float,
    width_deg: float,
    baryonic_samples: "pd.DataFrame | None",
) -> float:
    """Log-likelihood of an observed gap under the baryonic perturbation model.

    Baryonic perturbations (GMCs, bar) produce shallower, wider features.
    GMC masses ~1e4-1e7 Msun produce depth ~ 0.01-0.10 (vs. 0.05-0.50 for DM).
    Bar resonances produce periodic undulations with depth < 0.05.

    Uses a simple parametric model:
        p(depth | baryonic) = TruncatedNormal(mu=0.05, sigma=0.04, [0, 1])

    Returns:
        Scalar log-likelihood.
    """
    # Baryonic gaps are typically shallow and broad
    mu_depth = 0.05
    sigma_depth = 0.04
    log_lik = -0.5 * ((depth - mu_depth) / sigma_depth) ** 2
    log_lik -= np.log(sigma_depth * np.sqrt(2.0 * np.pi))

    # Baryonic gaps tend to be wider (5-15 deg for bar, 2-5 deg for GMC)
    if width_deg > 2.0:
        log_lik += 0.3  # mild preference for wider gaps
    return float(log_lik)


def classify_gaps(
    gaps: list[Gap],
    dm_posterior_samples: "pd.DataFrame | None",
    baryonic_posterior_samples: "pd.DataFrame | None",
    dm_mass_param: str = "log10_M_sub_mean",
    n_impacts_param: str = "n_impacts",
    significance_noise_threshold: float = 3.0,
    dm_mass_threshold_solar: float = 1.0e6,
    prior_p_dm: float = 0.4,
    prior_p_baryonic: float = 0.3,
    prior_p_noise: float = 0.3,
) -> list[Gap]:
    """Assign posterior probabilities to each gap's origin via Bayes factors.

    .. deprecated::
        This heuristic classifier is superseded by ``gap_classifier.py`` which
        uses SBI-based gap origin classification with a trained Random Forest
        and isotonic calibration. Prefer ``gap_classifier.GapOriginClassifier``
        for publication results. This function is retained for backward
        compatibility and quick exploratory analysis only.

    Three-hypothesis comparison:
        H_DM: gap caused by dark matter subhalo fly-by
        H_bar: gap caused by baryonic perturbation (GMC or bar resonance)
        H_noise: gap is a statistical fluctuation in the density profile

    For each gap with significance >= threshold:
        1. Compute log-likelihood of the observed (depth, width) under H_DM
           using the per-sample DM posterior (marginalised over M_sub, n_impacts).
        2. Compute log-likelihood under H_bar using a parametric baryonic model.
        3. Compute P(noise) from the significance (survival function of normal).
        4. Combine via Bayes' theorem with configurable model priors.

    The DM log-likelihood is marginalised over the full posterior:
        p(gap | H_DM) = (1/S) * sum_i p(gap | theta_i)
    which gives the correct Bayesian evidence integral via Monte Carlo.

    Args:
        gaps: Detected gaps from detect_gaps().
        dm_posterior_samples: Samples from the DM model posterior.
        baryonic_posterior_samples: Samples from the baryonic model posterior
            (currently unused; reserved for future per-gap baryonic SBI).
        dm_mass_param: Column name for log10(M_sub) in dm_posterior_samples.
        n_impacts_param: Column name for n_impacts in dm_posterior_samples.
        significance_noise_threshold: Below this, classify as noise.
        dm_mass_threshold_solar: Subhalo mass above which a detection is "DM".
        prior_p_dm: Prior probability for H_DM (default 0.4).
        prior_p_baryonic: Prior probability for H_bar (default 0.3).
        prior_p_noise: Prior probability for H_noise (default 0.3).

    Returns:
        Gaps with p_dm_subhalo, p_baryonic, p_noise, and mass estimates filled in.
    """
    from scipy.stats import norm as _norm  # noqa: PLC0415

    log_m_threshold = np.log10(dm_mass_threshold_solar)

    for gap in gaps:
        # ── Low-significance gaps: noise by definition ──────────────────────
        if gap.significance < significance_noise_threshold:
            gap.p_noise = 1.0
            gap.p_dm_subhalo = 0.0
            gap.p_baryonic = 0.0
            continue

        # ── P(noise) from Gaussian significance ─────────────────────────────
        # Survival function: P(fluctuation >= observed sigma)
        p_noise_from_sig = float(_norm.sf(gap.significance))
        # Scale to be comparable with model priors
        log_p_noise = np.log(max(p_noise_from_sig, 1e-20)) + np.log(prior_p_noise)

        # ── DM log-evidence (marginalised over posterior) ────────────────────
        if dm_posterior_samples is not None and dm_mass_param in dm_posterior_samples.columns:
            masses = dm_posterior_samples[dm_mass_param].values
            n_impacts = dm_posterior_samples.get(n_impacts_param, pd.Series(np.ones(len(masses)))).values

            # Per-sample log-likelihood
            log_liks = _gap_depth_log_likelihood(gap.depth, gap.phi1_width, masses, n_impacts)
            # Log-mean-exp: log(1/S * sum(exp(ll))) = logsumexp(ll) - log(S)
            max_ll = np.max(log_liks)
            log_evidence_dm = max_ll + np.log(np.mean(np.exp(log_liks - max_ll)))
            log_p_dm = log_evidence_dm + np.log(prior_p_dm)

            # Mass estimates from posterior (regardless of classification)
            gap.log10_mass_dm_50 = float(np.percentile(masses, 50))
            gap.log10_mass_dm_lo = float(np.percentile(masses, 16))
            gap.log10_mass_dm_hi = float(np.percentile(masses, 84))
        else:
            log_p_dm = np.log(prior_p_dm)  # uninformative

        # ── Baryonic log-evidence ────────────────────────────────────────────
        log_evidence_bar = _baryonic_gap_log_likelihood(
            gap.depth, gap.phi1_width, baryonic_posterior_samples,
        )
        log_p_bar = log_evidence_bar + np.log(prior_p_baryonic)

        # ── Normalise to posterior model probabilities ────────────────────────
        log_probs = np.array([log_p_dm, log_p_bar, log_p_noise])
        log_probs -= np.max(log_probs)  # shift for numerical stability
        probs = np.exp(log_probs)
        probs /= probs.sum()

        gap.p_dm_subhalo = float(probs[0])
        gap.p_baryonic = float(probs[1])
        gap.p_noise = float(probs[2])

    return gaps


# ---------------------------------------------------------------------------
# Output catalog
# ---------------------------------------------------------------------------

def gaps_to_dataframe(gaps: list[Gap], stream_name: str) -> pd.DataFrame:
    """Convert list of Gap objects to a DataFrame for the gap catalog."""
    records = []
    for gap in gaps:
        records.append({
            "stream": stream_name,
            "phi1_center_deg": gap.phi1_center,
            "phi1_width_deg": gap.phi1_width,
            "significance_sigma": gap.significance,
            "depth_fraction": gap.depth,
            "p_dm_subhalo": gap.p_dm_subhalo,
            "p_baryonic": gap.p_baryonic,
            "p_noise": gap.p_noise,
            "log10_mass_dm_50": gap.log10_mass_dm_50,
            "log10_mass_dm_16": gap.log10_mass_dm_lo,
            "log10_mass_dm_84": gap.log10_mass_dm_hi,
        })
    return pd.DataFrame(records)
