"""
Model-independent suppression scale inference.

Instead of attempting to discriminate between WDM, FDM, and SIDM (which have
degenerate suppression shapes at Gaia DR3 sensitivity), this module directly
infers the half-mode mass M_hm — the mass scale below which the subhalo mass
function is suppressed relative to CDM.

This is the paper's primary constraint:
    "We constrain M_hm < X M_sun at 95% CL"

which can then be mapped to particle mass limits for any specific DM model:
    WDM:   m_wdm > (M_hm / 1.7e10)^(-1/3.33) keV
    FDM:   m_axion > (M_hm / 1.5e8)^(-2/3) * 1e-22 eV
    SIDM:  (constraint on sigma/m via core-formation threshold)

Scientific justification:
    - Banik+2021: "The sharp feature in the stream constrains the subhalo mass
      function turnover scale" (not the model identity)
    - Dalal+2022: "Current data constrain the free-streaming scale, not the
      specific particle physics model"
    - Nadler+2021: satellite-count constraints quote M_hm, not model identity

This module provides:
    1. Prior construction for M_hm (log-uniform over [1e6, 1e10] M_sun)
    2. Mapping from n_impacts posterior to M_hm constraint (count-based)
    3. Upper limit computation (one-sided 95% CI)
    4. Particle mass conversion utilities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SuppressionConstraint:
    """Result of suppression-scale inference for one or more streams."""
    log10_M_hm_median: float          # posterior median
    log10_M_hm_upper_95: float        # 95% upper limit (one-sided)
    log10_M_hm_lower_5: float         # 5% lower bound
    is_detected: bool                  # True if M_hm constrained away from prior boundary
    detection_bayes_factor: float      # log10 BF for suppression vs CDM
    n_streams_combined: int
    method: str                        # "count_based" or "power_spectrum"
    # Particle mass limits (derived from M_hm upper limit)
    m_wdm_lower_kev: float = 0.0      # WDM thermal relic lower limit
    m_axion_lower_ev: float = 0.0     # FDM axion mass lower limit


# ---------------------------------------------------------------------------
# Half-mode mass from the count-based approach
# ---------------------------------------------------------------------------

def infer_suppression_from_counts(
    n_impacts_posterior: np.ndarray,
    n_impacts_cdm_expected: float,
    log10_M_sub_posterior: np.ndarray,
    log10_M_min: float = 5.0,
    log10_M_max: float = 9.0,
    alpha: float = -1.9,
) -> np.ndarray:
    """Convert posterior on (n_impacts, M_sub) to posterior on M_hm.

    Method:
        Under CDM, the expected number of impacts above M_min is:
            N_CDM = integral(M_min, M_max) dN/dM dM

        Under a suppressed model with half-mode mass M_hm:
            N_sup = integral(M_min, M_max) f_sup(M/M_hm) * dN/dM dM

        where f_sup(x) = (1 + x^{-beta})^{gamma/beta} is the suppression filter.

        Given the posterior on n_impacts, we can solve for M_hm:
            n_impacts / N_CDM = N_sup(M_hm) / N_CDM = suppression_fraction(M_hm)

        This gives a posterior on M_hm by transforming each posterior sample.

    Args:
        n_impacts_posterior: [S] posterior samples of n_impacts.
        n_impacts_cdm_expected: Expected n_impacts under CDM (from CDM posterior mode
            or from theory: integral of mass function * encounter rate).
        log10_M_sub_posterior: [S] posterior samples of log10(M_sub).
        log10_M_min: Minimum subhalo mass in simulations [log10 Msun].
        log10_M_max: Maximum subhalo mass in simulations [log10 Msun].
        alpha: CDM mass function slope.

    Returns:
        [S] posterior samples of log10(M_hm) [Msun].
        Samples where n_impacts >= N_CDM are set to log10_M_min (no suppression).
    """
    # Suppression fraction: n_obs / n_cdm
    # Clamp to [0, 1] — values > 1 mean "consistent with CDM" (no suppression)
    frac = np.clip(n_impacts_posterior / max(n_impacts_cdm_expected, 0.1), 0.0, 1.0)

    # Invert the suppression fraction to get M_hm.
    # For the Lovell+2014 filter with beta=2.7, gamma=-0.99:
    #   f_sup(x) = (1 + x^-2.7)^(0.99/2.7) ≈ (1 + x^-2.7)^0.367
    # We need: average f_sup over the mass function to equal frac.
    #
    # Analytic approximation (valid for M_hm in [1e6, 1e9]):
    #   <f_sup> ≈ 1 - (M_hm / M_ref)^{alpha+1} / (alpha+1) * correction
    # For simplicity, use the numerical inversion:
    log10_M_hm = _invert_suppression_fraction(
        frac, log10_M_min, log10_M_max, alpha,
    )

    return log10_M_hm


def _invert_suppression_fraction(
    frac: np.ndarray,
    log10_M_min: float,
    log10_M_max: float,
    alpha: float,
    beta: float = 2.7,
    gamma: float = -0.99,
    n_grid: int = 200,
) -> np.ndarray:
    """Numerically invert suppression fraction to half-mode mass.

    Builds a lookup table of suppression_fraction(M_hm) and interpolates.
    """
    # Grid of M_hm values
    log10_M_hm_grid = np.linspace(log10_M_min, log10_M_max + 1.0, n_grid)
    M_hm_grid = 10.0 ** log10_M_hm_grid

    # Mass integration grid (the subhalo masses that contribute to n_impacts)
    n_mass = 100
    log10_M = np.linspace(log10_M_min, log10_M_max, n_mass)
    M = 10.0 ** log10_M
    dlog10M = log10_M[1] - log10_M[0]

    # CDM mass function (unnormalized): dN/dlog10M ~ M^(alpha+1)
    dNdlogM_cdm = M ** (alpha + 1.0)
    N_cdm_total = np.sum(dNdlogM_cdm) * dlog10M

    # For each M_hm, compute the suppressed count fraction
    frac_grid = np.zeros(n_grid)
    for i, M_hm in enumerate(M_hm_grid):
        # Suppression filter: f(M) = (1 + (M_hm/M)^beta)^(gamma/beta)
        x = M_hm / M
        f_sup = (1.0 + x ** beta) ** (gamma / beta)
        N_sup = np.sum(dNdlogM_cdm * f_sup) * dlog10M
        frac_grid[i] = N_sup / N_cdm_total

    # frac_grid should be monotonically decreasing with increasing M_hm
    # (higher M_hm = more suppression = lower fraction)
    # Ensure monotonicity for interpolation
    # Actually: higher M_hm means suppression at higher masses, so MORE suppression
    # frac_grid is decreasing. We need to invert: given frac, find M_hm.

    # Interpolate: frac -> log10_M_hm
    # Need to flip because np.interp requires increasing x
    frac_sorted = frac_grid[::-1]
    log10_M_hm_sorted = log10_M_hm_grid[::-1]

    # Clamp frac to valid interpolation range
    frac_clamped = np.clip(frac, frac_sorted[0], frac_sorted[-1])
    log10_M_hm_result = np.interp(frac_clamped, frac_sorted, log10_M_hm_sorted)

    # Samples with frac >= 1 (no suppression) get M_hm = M_min (= no constraint)
    log10_M_hm_result[frac >= 0.99] = log10_M_min

    return log10_M_hm_result


# ---------------------------------------------------------------------------
# Upper limit computation
# ---------------------------------------------------------------------------

def compute_upper_limit(
    log10_M_hm_samples: np.ndarray,
    confidence_level: float = 0.95,
) -> dict:
    """Compute one-sided upper limit on M_hm.

    For suppression constraints, we report: M_hm < X at Y% CL.
    This is a one-sided credible interval (the scientific question is
    "how low can M_hm be?" = "how much suppression is allowed?").

    Args:
        log10_M_hm_samples: Posterior samples of log10(M_hm / Msun).
        confidence_level: Confidence level for upper limit (default 0.95).

    Returns:
        Dict with upper_limit, median, is_constraining, posterior_summary.
    """
    upper = float(np.quantile(log10_M_hm_samples, confidence_level))
    median = float(np.median(log10_M_hm_samples))
    lower = float(np.quantile(log10_M_hm_samples, 1.0 - confidence_level))

    # "Constraining" if the upper limit is below the prior upper boundary
    # (i.e., the data actually rules out some parameter space)
    prior_upper = 10.0  # log10(1e10 Msun) from config
    is_constraining = upper < prior_upper - 0.5  # at least half a dex below boundary

    return {
        "log10_M_hm_upper": upper,
        "log10_M_hm_median": median,
        "log10_M_hm_lower": lower,
        "confidence_level": confidence_level,
        "is_constraining": is_constraining,
    }


# ---------------------------------------------------------------------------
# Particle mass conversions
# ---------------------------------------------------------------------------

def M_hm_to_wdm_mass(log10_M_hm: float) -> float:
    """Convert half-mode mass to WDM thermal relic mass.

    Lovell+2014: M_hm = 1.7e10 * (m_wdm / keV)^{-3.33} M_sun
    => m_wdm = (M_hm / 1.7e10)^{-1/3.33} keV

    An UPPER limit on M_hm gives a LOWER limit on m_wdm.
    """
    M_hm = 10.0 ** log10_M_hm
    m_wdm_kev = (M_hm / 1.7e10) ** (-1.0 / 3.33)
    return m_wdm_kev


def M_hm_to_fdm_mass(log10_M_hm: float) -> float:
    """Convert half-mode mass to FDM axion mass.

    Hui+2017: M_Jeans ~ 1.5e8 * (m_axion / 1e-22 eV)^{-1.5} M_sun
    => m_axion = (M_hm / 1.5e8)^{-2/3} * 1e-22 eV

    An UPPER limit on M_hm gives a LOWER limit on m_axion.
    """
    M_hm = 10.0 ** log10_M_hm
    m_axion_ev = (M_hm / 1.5e8) ** (-2.0 / 3.0) * 1e-22
    return m_axion_ev


def M_hm_to_sidm_cross_section(log10_M_hm: float, v_max_kms: float = 30.0) -> float:
    """Estimate SIDM cross-section constraint from half-mode mass.

    SIDM doesn't directly suppress the mass function, but cored density
    profiles reduce tidal stripping survival below a threshold mass.
    Approximate: subhalos with M < M_hm are disrupted by core collapse
    if sigma/m > sigma_crit(M_hm, v_max).

    Kaplinghat+2016 scaling:
        sigma_crit ~ 10 * (M_hm / 1e8)^{-0.5} * (v_max / 30 km/s) cm^2/g

    Returns sigma/m upper limit in cm^2/g.
    """
    M_hm = 10.0 ** log10_M_hm
    sigma_m = 10.0 * (M_hm / 1e8) ** (-0.5) * (v_max_kms / 30.0)
    return sigma_m


# ---------------------------------------------------------------------------
# Full suppression constraint pipeline
# ---------------------------------------------------------------------------

def run_suppression_inference(
    per_stream_posteriors: dict[str, dict],
    config_path: str = "config/dm_models.yaml",
    streams_config_path: str = "config/streams.yaml",
    confidence_level: float = 0.95,
) -> SuppressionConstraint:
    """Run full suppression scale inference across all primary streams.

    Args:
        per_stream_posteriors: Dict mapping stream_name -> {
            "n_impacts_samples": np.ndarray,
            "log10_M_sub_samples": np.ndarray,
            "n_impacts_cdm_expected": float,  # from CDM posterior mode
        }
        config_path: Path to dm_models.yaml.
        streams_config_path: Path to streams.yaml.
        confidence_level: CI level for upper limit.

    Returns:
        SuppressionConstraint with combined multi-stream result.
    """
    with open(config_path) as f:
        dm_cfg = yaml.safe_load(f)
    with open(streams_config_path) as f:
        streams_cfg = yaml.safe_load(f)

    sup_cfg = dm_cfg.get("suppression_scale_inference", {})
    log10_M_min = 5.0  # from CDM config
    log10_M_max = 9.0
    alpha = -1.9

    # Combine per-stream M_hm posteriors (product of independent posteriors)
    all_log10_M_hm = []

    for stream_name, post_data in per_stream_posteriors.items():
        # Skip forecast-only streams from the primary constraint
        stream_cfg = streams_cfg.get("streams", {}).get(stream_name, {})
        tier = stream_cfg.get("analysis_tier", "primary")
        if tier == "forecast":
            log.info("Skipping forecast-only stream %s from primary constraint", stream_name)
            continue

        n_impacts = post_data["n_impacts_samples"]
        log10_M_sub = post_data["log10_M_sub_samples"]
        n_cdm = post_data["n_impacts_cdm_expected"]

        # Convert to M_hm posterior
        log10_M_hm_stream = infer_suppression_from_counts(
            n_impacts, n_cdm, log10_M_sub, log10_M_min, log10_M_max, alpha,
        )
        all_log10_M_hm.append(log10_M_hm_stream)
        log.info(
            "Stream %s: M_hm posterior median = %.1f, 95%% upper = %.1f",
            stream_name, np.median(log10_M_hm_stream),
            np.quantile(log10_M_hm_stream, confidence_level),
        )

    if not all_log10_M_hm:
        log.warning("No streams available for suppression inference")
        return SuppressionConstraint(
            log10_M_hm_median=9.0, log10_M_hm_upper_95=10.0, log10_M_hm_lower_5=5.0,
            is_detected=False, detection_bayes_factor=0.0,
            n_streams_combined=0, method="count_based",
        )

    # Combine: product of posteriors on a grid (like combine_posteriors_product)
    # For 1D parameter, direct histogram multiplication works well.
    combined_log10_M_hm = _combine_1d_posteriors(all_log10_M_hm, log10_M_min, log10_M_max + 1.0)

    # Compute upper limit
    limit = compute_upper_limit(combined_log10_M_hm, confidence_level)

    # Detection: Bayes factor for suppression (M_hm < 9) vs no suppression (M_hm >= 9)
    frac_suppressed = np.mean(combined_log10_M_hm < 9.0)
    frac_unsuppressed = np.mean(combined_log10_M_hm >= 9.0)
    if frac_unsuppressed > 0:
        log10_bf = np.log10(max(frac_suppressed, 1e-10) / max(frac_unsuppressed, 1e-10))
    else:
        log10_bf = 5.0  # effectively infinite

    # Particle mass conversions from the upper limit
    m_wdm = M_hm_to_wdm_mass(limit["log10_M_hm_upper"])
    m_axion = M_hm_to_fdm_mass(limit["log10_M_hm_upper"])

    result = SuppressionConstraint(
        log10_M_hm_median=limit["log10_M_hm_median"],
        log10_M_hm_upper_95=limit["log10_M_hm_upper"],
        log10_M_hm_lower_5=limit["log10_M_hm_lower"],
        is_detected=limit["is_constraining"],
        detection_bayes_factor=float(log10_bf),
        n_streams_combined=len(all_log10_M_hm),
        method="count_based",
        m_wdm_lower_kev=m_wdm,
        m_axion_lower_ev=m_axion,
    )

    log.info(
        "Combined suppression constraint (N=%d streams): "
        "log10(M_hm) < %.1f at %d%% CL => m_WDM > %.1f keV, m_axion > %.1e eV",
        result.n_streams_combined, result.log10_M_hm_upper_95,
        int(confidence_level * 100), result.m_wdm_lower_kev, result.m_axion_lower_ev,
    )

    return result


def _combine_1d_posteriors(
    sample_lists: list[np.ndarray],
    lo: float,
    hi: float,
    n_grid: int = 500,
) -> np.ndarray:
    """Combine independent 1D posteriors via KDE product on a grid.

    Returns samples from the combined posterior (importance-weighted resampling).
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    grid = np.linspace(lo, hi, n_grid)
    log_combined = np.zeros(n_grid)

    for samples in sample_lists:
        # Clamp to grid range
        s = np.clip(samples, lo + 0.01, hi - 0.01)
        if len(s) < 5:
            continue
        kde = gaussian_kde(s)
        log_combined += np.log(kde(grid) + 1e-300)

    # Normalize
    log_combined -= log_combined.max()
    combined_pdf = np.exp(log_combined)
    combined_pdf /= np.trapz(combined_pdf, grid)

    # Resample from the combined PDF
    cdf = np.cumsum(combined_pdf) * (grid[1] - grid[0])
    cdf /= cdf[-1]
    n_resample = 10000
    u = np.random.uniform(0, 1, n_resample)
    resampled = np.interp(u, cdf, grid)

    return resampled
