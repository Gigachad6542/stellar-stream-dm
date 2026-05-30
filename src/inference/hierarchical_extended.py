"""
Extended hierarchical Bayesian model with free mass function slope alpha.

This extends the base hierarchical model (hierarchical.py) to jointly infer:
  - log10(M_hm): half-mode mass (suppression scale)
  - alpha: mass function slope (base model assumes -1.9 fixed)

The CDM mass function is dN/dM ~ M^alpha, where alpha = -1.9 in the standard
Springel+2008 prediction. Allowing alpha to vary tests robustness to the assumed
mass function shape and could reveal deviations from CDM predictions.

Joint model:
    log10(M_hm) ~ Uniform(4, 10)
    alpha ~ Normal(-1.9, 0.2)  [informative prior centered on CDM prediction]
    Per stream s:
        n_impacts_s ~ Poisson(rate(M_hm, alpha, L_s, T_s, D_s))

References:
    - Springel+2008: alpha = -1.9 from Aquarius CDM simulation
    - Garrison-Kimmel+2017: alpha varies -1.85 to -1.95 across simulations
    - Despali+2018: concentration-dependent alpha variations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


def expected_rate_variable_alpha(
    log10_M_hm: float,
    alpha: float,
    stream_length_deg: float,
    stream_age_gyr: float,
    stream_distance_kpc: float = 15.0,
    m_sub_min_log10: float = 5.0,
    m_sub_max_log10: float = 9.0,
    n_grid: int = 100,
) -> float:
    """Compute expected impact rate with variable mass function slope.

    Like expected_subhalo_rate but alpha is a free parameter instead of fixed -1.9.

    Args:
        log10_M_hm: log10 of half-mode mass in solar masses.
        alpha: Mass function slope (dN/dM ~ M^alpha). Standard CDM: -1.9.
        stream_length_deg: Angular length of the stream (degrees).
        stream_age_gyr: Age of the stream (Gyr).
        stream_distance_kpc: Distance from Galactic center (kpc).

    Returns:
        Expected number of detectable subhalo encounters.
    """
    log_m_grid = np.linspace(m_sub_min_log10, m_sub_max_log10, n_grid)
    m_grid = 10.0 ** log_m_grid
    M_hm = 10.0 ** log10_M_hm

    # Variable mass function: dN/dlog10M ~ M^(alpha + 1)
    alpha_log = alpha + 1.0  # convert from dN/dM to dN/dlogM
    dn_dlogm_cdm = m_grid ** alpha_log

    # Suppression transfer function
    transfer = 1.0 / (1.0 + (M_hm / m_grid) ** 2.0)

    # Suppressed mass function
    dn_dlogm = dn_dlogm_cdm * transfer

    # Geometric and time factors
    stream_length_kpc = stream_length_deg * (np.pi / 180.0) * stream_distance_kpc
    geometric_factor = stream_length_kpc / 10.0
    time_factor = stream_age_gyr / 5.0

    # Integrate
    dlogm = (m_sub_max_log10 - m_sub_min_log10) / n_grid
    n_total = np.sum(dn_dlogm) * dlogm

    # Normalize to CDM: ~2.1 impacts for GD-1 with this alpha
    # (Bonaca+2019 found 2 gaps + 1 spur; previous value of 5.0 overpredicted)
    cdm_integral = np.sum(dn_dlogm_cdm) * dlogm
    normalization = 2.1 / cdm_integral if cdm_integral > 0 else 1.0

    rate = normalization * n_total * geometric_factor * time_factor
    return max(rate, 0.01)


@dataclass
class ExtendedHierarchicalResult:
    """Result of extended hierarchical inference with free alpha."""
    log10_M_hm_samples: np.ndarray
    alpha_samples: np.ndarray
    log10_M_hm_median: float
    alpha_median: float
    log10_M_hm_upper_95: float
    log10_M_hm_hdi_68: tuple[float, float]
    alpha_hdi_68: tuple[float, float]
    n_streams: int
    n_effective_samples: int
    r_hat_max: float
    backend: str
    correlation: float  # Pearson correlation between M_hm and alpha samples


def run_extended_hierarchical(
    per_stream_n_impacts: dict[str, int],
    per_stream_properties: dict[str, dict],
    n_walkers: int = 64,
    n_steps: int = 15000,
    n_burnin: int = 3000,
    seed: int = 42,
) -> ExtendedHierarchicalResult:
    """Run extended hierarchical inference with free alpha.

    Uses emcee with 2D parameter space: [log10_M_hm, alpha].

    Args:
        per_stream_n_impacts: stream_name -> observed count.
        per_stream_properties: stream_name -> {length_deg, age_gyr, distance_kpc}.
        n_walkers: Number of emcee walkers.
        n_steps: Total MCMC steps.
        n_burnin: Burn-in steps to discard.

    Returns:
        ExtendedHierarchicalResult with posterior samples and diagnostics.
    """
    import emcee
    from scipy.special import gammaln

    stream_names = list(per_stream_n_impacts.keys())
    n_impacts_obs = np.array([per_stream_n_impacts[s] for s in stream_names])
    lengths = np.array([per_stream_properties[s]["length_deg"] for s in stream_names])
    ages = np.array([per_stream_properties[s]["age_gyr"] for s in stream_names])
    dists = np.array([per_stream_properties[s].get("distance_kpc", 15.0) for s in stream_names])

    def log_prior(theta):
        log10_M_hm, alpha = theta
        if not (4.0 <= log10_M_hm <= 10.0):
            return -np.inf
        if not (-2.5 <= alpha <= -1.3):
            return -np.inf
        # Informative prior on alpha: Normal(-1.9, 0.2)
        lp_alpha = -0.5 * ((alpha - (-1.9)) / 0.2) ** 2
        return lp_alpha

    def log_likelihood(theta, n_obs, lengths, ages, dists):
        log10_M_hm, alpha = theta
        ll = 0.0
        for s in range(len(n_obs)):
            rate = expected_rate_variable_alpha(
                log10_M_hm, alpha, lengths[s], ages[s], dists[s]
            )
            n = n_obs[s]
            ll += n * np.log(max(rate, 1e-10)) - rate - gammaln(n + 1)
        return ll

    def log_posterior(theta, n_obs, lengths, ages, dists):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, n_obs, lengths, ages, dists)

    ndim = 2
    rng = np.random.default_rng(seed)
    # Initialize: M_hm around 7, alpha around -1.9
    p0 = np.column_stack([
        7.0 + 0.5 * rng.standard_normal(n_walkers),
        -1.9 + 0.1 * rng.standard_normal(n_walkers),
    ])

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior,
        args=(n_impacts_obs, lengths, ages, dists),
    )

    log.info("Running extended emcee: %d walkers x %d steps (ndim=2)...", n_walkers, n_steps)
    sampler.run_mcmc(p0, n_steps, progress=False)

    # Discard burn-in
    chain = sampler.get_chain(discard=n_burnin, flat=True)
    M_hm_samples = chain[:, 0]
    alpha_samples = chain[:, 1]
    acceptance = float(np.mean(sampler.acceptance_fraction))

    # Diagnostics
    half = len(M_hm_samples) // 2
    between_var = np.var([M_hm_samples[:half].mean(), M_hm_samples[half:].mean()])
    within_var = (np.var(M_hm_samples[:half]) + np.var(M_hm_samples[half:])) / 2.0
    r_hat = float(np.sqrt((within_var + between_var) / max(within_var, 1e-10)))

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        n_eff = int(min(len(chain) / tau[0], len(chain) / tau[1]))
    except Exception:
        n_eff = int(len(chain) / 10.0)

    # Summary statistics
    M_hm_median = float(np.median(M_hm_samples))
    alpha_median = float(np.median(alpha_samples))
    M_hm_upper_95 = float(np.percentile(M_hm_samples, 95))

    # HDI
    sorted_M = np.sort(M_hm_samples)
    n_s = len(sorted_M)
    ci_w = int(0.68 * n_s)
    widths = sorted_M[ci_w:] - sorted_M[:n_s - ci_w]
    best = int(np.argmin(widths))
    M_hm_hdi = (float(sorted_M[best]), float(sorted_M[best + ci_w]))

    sorted_a = np.sort(alpha_samples)
    widths_a = sorted_a[ci_w:] - sorted_a[:n_s - ci_w]
    best_a = int(np.argmin(widths_a))
    alpha_hdi = (float(sorted_a[best_a]), float(sorted_a[best_a + ci_w]))

    # Correlation
    corr = float(np.corrcoef(M_hm_samples, alpha_samples)[0, 1])

    log.info("Extended inference complete:")
    log.info("  log10(M_hm): median=%.3f, 95%% upper=%.3f, HDI=[%.3f, %.3f]",
             M_hm_median, M_hm_upper_95, *M_hm_hdi)
    log.info("  alpha: median=%.3f, HDI=[%.3f, %.3f]", alpha_median, *alpha_hdi)
    log.info("  M_hm-alpha correlation: %.3f", corr)
    log.info("  r_hat=%.4f, n_eff=%d, acceptance=%.3f", r_hat, n_eff, acceptance)

    return ExtendedHierarchicalResult(
        log10_M_hm_samples=M_hm_samples,
        alpha_samples=alpha_samples,
        log10_M_hm_median=M_hm_median,
        alpha_median=alpha_median,
        log10_M_hm_upper_95=M_hm_upper_95,
        log10_M_hm_hdi_68=M_hm_hdi,
        alpha_hdi_68=alpha_hdi,
        n_streams=len(stream_names),
        n_effective_samples=n_eff,
        r_hat_max=r_hat,
        backend="emcee",
        correlation=corr,
    )
