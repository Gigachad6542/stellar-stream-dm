"""
Hierarchical Bayesian multi-stream model for M_hm inference.

Replaces the statistically incorrect KDE posterior product (which assumes
independence with no shared parameters) with a proper hierarchical model
where multiple streams share a common half-mode mass M_hm but have different
exposure (stream length, age, distance from Galactic center).

Model:
    M_hm ~ Uniform(4, 10)   [log10 M_sun]
    Per stream s:
        n_impacts_s ~ Poisson(rate(M_hm, L_s, T_s))

where rate(M_hm, L_s, T_s) integrates the suppressed subhalo mass function
above the impact threshold for stream s.

This approach:
- Correctly accounts for the shared parameter (M_hm)
- Handles streams with different lengths and ages
- Propagates per-stream n_impacts uncertainty properly
- Gives a single joint posterior on M_hm

Backends:
    Primary: NumPyro NUTS (JAX-based HMC)
    Fallback: emcee (ensemble sampler)

References:
    - Bonaca+2019: Impact rate scaling with stream properties
    - Erkal+2016: Subhalo encounter rate formalism
    - Banik+2021: Stream sensitivity to subhalo mass function
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical model: expected subhalo encounter rate
# ---------------------------------------------------------------------------

def expected_subhalo_rate(
    log10_M_hm: float,
    stream_length_deg: float,
    stream_age_gyr: float,
    stream_distance_kpc: float = 15.0,
    m_sub_min_log10: float = 5.0,
    m_sub_max_log10: float = 9.0,
    n_grid: int = 100,
) -> float:
    """Compute expected number of detectable subhalo impacts for a given M_hm.

    Integrates the CDM-like subhalo mass function with a suppression cutoff
    at M_hm, weighted by the stream's geometric cross-section and age.

    The CDM mass function is dN/dM ~ M^(-1.9) (Springel+2008).
    Suppression: multiply by a transfer function T(M, M_hm) that goes to 0
    below M_hm (half-mode mass).

    Args:
        log10_M_hm: log10 of half-mode mass in solar masses.
        stream_length_deg: Angular length of the stream (degrees).
        stream_age_gyr: Age of the stream (Gyr).
        stream_distance_kpc: Distance from Galactic center (kpc).
        m_sub_min_log10: Minimum subhalo mass to consider.
        m_sub_max_log10: Maximum subhalo mass to consider.
        n_grid: Integration grid points.

    Returns:
        Expected number of detectable subhalo encounters.
    """
    # Integration grid in log-space
    log_m_grid = np.linspace(m_sub_min_log10, m_sub_max_log10, n_grid)
    m_grid = 10.0 ** log_m_grid
    M_hm = 10.0 ** log10_M_hm

    # CDM mass function: dN/dlog10M ~ M^(-0.9) (converted from dN/dM ~ M^(-1.9))
    dn_dlogm_cdm = m_grid ** (-0.9)

    # Suppression transfer function: smooth step at M_hm
    # T(M) = [1 + (M_hm/M)^2]^(-1) — goes to 0 below M_hm, 1 above
    # This matches the WDM transfer function form from Schneider+2012
    transfer = 1.0 / (1.0 + (M_hm / m_grid) ** 2.0)

    # Suppressed mass function
    dn_dlogm = dn_dlogm_cdm * transfer

    # Impact rate: depends on stream cross-section and time
    # Geometric factor: longer streams intercept more subhalos
    # Rate ~ L_stream * T_stream * integral(suppressed dN/dlogM)
    # Normalization: tuned so CDM (M_hm=0) gives ~2-3 impacts for GD-1-like streams
    # Bonaca+2019 identified 2 gaps + 1 spur in GD-1; we adopt 2.1 as the CDM
    # baseline, consistent with Bonaca+2019 and Banik+2021 estimates.
    # Previous value of 5.0 overpredicted total impacts by ~2.4x.
    stream_length_kpc = stream_length_deg * (np.pi / 180.0) * stream_distance_kpc
    geometric_factor = stream_length_kpc / 10.0  # normalize to ~10 kpc reference
    time_factor = stream_age_gyr / 5.0  # normalize to ~5 Gyr reference

    # Integrate suppressed mass function
    dlogm = (m_sub_max_log10 - m_sub_min_log10) / n_grid
    n_total = np.sum(dn_dlogm) * dlogm

    # CDM normalization: ~2.1 impacts for a GD-1-like stream in CDM
    cdm_integral = np.sum(dn_dlogm_cdm) * dlogm
    normalization = 2.1 / cdm_integral if cdm_integral > 0 else 1.0

    rate = normalization * n_total * geometric_factor * time_factor
    return max(rate, 0.01)  # floor to prevent log(0) issues


# ---------------------------------------------------------------------------
# Hierarchical model results
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalResult:
    """Result of hierarchical Bayesian inference on M_hm."""
    log10_M_hm_samples: np.ndarray     # [n_samples] posterior samples
    log10_M_hm_median: float
    log10_M_hm_upper_95: float         # 95% upper limit
    log10_M_hm_lower_5: float          # 5% lower limit
    log10_M_hm_hdi_68: tuple[float, float]  # 68% highest density interval
    n_streams: int
    n_effective_samples: int
    r_hat_max: float                    # Gelman-Rubin statistic (should be < 1.05)
    backend: str                        # "numpyro" or "emcee"
    stream_names: list[str] = field(default_factory=list)
    per_stream_rates: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

def _numpyro_hierarchical_model(
    n_impacts_obs: np.ndarray,
    stream_lengths: np.ndarray,
    stream_ages: np.ndarray,
    stream_distances: np.ndarray,
    log10_M_hm_prior_low: float = 4.0,
    log10_M_hm_prior_high: float = 10.0,
):
    """NumPyro probabilistic model for hierarchical M_hm inference.

    Args:
        n_impacts_obs: [n_streams] observed impact counts.
        stream_lengths: [n_streams] stream lengths in degrees.
        stream_ages: [n_streams] stream ages in Gyr.
        stream_distances: [n_streams] distances from GC in kpc.
        log10_M_hm_prior_low/high: Uniform prior bounds on log10(M_hm).
    """
    import jax.numpy as jnp  # noqa: PLC0415
    import numpyro  # noqa: PLC0415
    import numpyro.distributions as dist  # noqa: PLC0415

    # Shared parameter: log10(M_hm)
    log10_M_hm = numpyro.sample(
        "log10_M_hm",
        dist.Uniform(log10_M_hm_prior_low, log10_M_hm_prior_high),
    )

    # Per-stream likelihood
    n_streams = len(n_impacts_obs)
    for s in range(n_streams):
        rate = expected_subhalo_rate(
            float(log10_M_hm),
            float(stream_lengths[s]),
            float(stream_ages[s]),
            float(stream_distances[s]),
        )
        numpyro.sample(
            f"n_impacts_{s}",
            dist.Poisson(jnp.clip(rate, 0.01, 100.0)),
            obs=n_impacts_obs[s],
        )


def run_hierarchical_numpyro(
    n_impacts_obs: np.ndarray,
    stream_lengths: np.ndarray,
    stream_ages: np.ndarray,
    stream_distances: np.ndarray,
    n_warmup: int = 500,
    n_samples: int = 2000,
    n_chains: int = 4,
    seed: int = 42,
) -> dict:
    """Run hierarchical inference using NumPyro NUTS.

    Args:
        n_impacts_obs: [n_streams] observed impact counts.
        stream_lengths: [n_streams] angular lengths in degrees.
        stream_ages: [n_streams] ages in Gyr.
        stream_distances: [n_streams] distance from GC in kpc.
        n_warmup: NUTS warmup steps.
        n_samples: Posterior samples per chain.
        n_chains: Number of MCMC chains.
        seed: Random seed.

    Returns:
        Dict with 'samples', 'r_hat', 'n_eff', 'divergences'.
    """
    import jax  # noqa: PLC0415
    import jax.random as random  # noqa: PLC0415
    import numpyro  # noqa: PLC0415
    from numpyro.infer import MCMC, NUTS  # noqa: PLC0415

    numpyro.set_host_device_count(min(n_chains, 4))
    rng_key = random.PRNGKey(seed)

    kernel = NUTS(
        _numpyro_hierarchical_model,
        target_accept_prob=0.85,
        max_tree_depth=10,
    )
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)

    log.info("Running NumPyro NUTS: %d chains x %d samples (warmup=%d)...",
             n_chains, n_samples, n_warmup)
    mcmc.run(
        rng_key,
        n_impacts_obs=n_impacts_obs,
        stream_lengths=stream_lengths,
        stream_ages=stream_ages,
        stream_distances=stream_distances,
    )

    samples = mcmc.get_samples()
    log10_M_hm_samples = np.array(samples["log10_M_hm"])

    # Diagnostics
    from numpyro.diagnostics import summary  # noqa: PLC0415
    site_summary = summary(mcmc.get_samples(group_by_chain=True))
    r_hat = float(site_summary["log10_M_hm"]["r_hat"])
    n_eff = float(site_summary["log10_M_hm"]["n_eff"])
    n_divergent = int(mcmc.get_extra_fields().get("diverging", np.array([])).sum())

    log.info("NUTS complete: n_eff=%.0f, r_hat=%.4f, divergences=%d",
             n_eff, r_hat, n_divergent)

    return {
        "samples": log10_M_hm_samples,
        "r_hat": r_hat,
        "n_eff": n_eff,
        "divergences": n_divergent,
    }


# ---------------------------------------------------------------------------
# emcee fallback
# ---------------------------------------------------------------------------

def run_hierarchical_emcee(
    n_impacts_obs: np.ndarray,
    stream_lengths: np.ndarray,
    stream_ages: np.ndarray,
    stream_distances: np.ndarray,
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burnin: int = 1000,
    seed: int = 42,
) -> dict:
    """Run hierarchical inference using emcee ensemble sampler (fallback).

    Used when JAX/NumPyro are unavailable (e.g., Windows without CUDA JAX).

    Returns:
        Dict with 'samples', 'r_hat' (approximate), 'n_eff', 'acceptance_rate'.
    """
    import emcee  # noqa: PLC0415
    from scipy.special import gammaln  # noqa: PLC0415

    def log_prior(log10_M_hm):
        if 4.0 <= log10_M_hm <= 10.0:
            return 0.0  # log-uniform
        return -np.inf

    def log_likelihood(log10_M_hm, n_obs, lengths, ages, dists):
        ll = 0.0
        for s in range(len(n_obs)):
            rate = expected_subhalo_rate(log10_M_hm, lengths[s], ages[s], dists[s])
            # Poisson log-likelihood: n*log(rate) - rate - log(n!)
            n = n_obs[s]
            ll += n * np.log(max(rate, 1e-10)) - rate - gammaln(n + 1)
        return ll

    def log_posterior(theta, n_obs, lengths, ages, dists):
        log10_M_hm = theta[0]
        lp = log_prior(log10_M_hm)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(log10_M_hm, n_obs, lengths, ages, dists)

    ndim = 1
    rng = np.random.default_rng(seed)
    # Initialize walkers around center of prior
    p0 = 7.0 + 0.5 * rng.standard_normal((n_walkers, ndim))

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior,
        args=(n_impacts_obs, stream_lengths, stream_ages, stream_distances),
    )

    log.info("Running emcee: %d walkers x %d steps (burnin=%d)...",
             n_walkers, n_steps, n_burnin)
    sampler.run_mcmc(p0, n_steps, progress=False)

    # Discard burn-in and flatten
    chain = sampler.get_chain(discard=n_burnin, flat=True)[:, 0]
    acceptance = float(np.mean(sampler.acceptance_fraction))

    # Approximate r_hat from split chains
    half = len(chain) // 2
    chain1, chain2 = chain[:half], chain[half:]
    between_var = np.var([chain1.mean(), chain2.mean()])
    within_var = (np.var(chain1) + np.var(chain2)) / 2.0
    r_hat_approx = float(np.sqrt((within_var + between_var) / max(within_var, 1e-10)))

    # Approximate n_eff via autocorrelation
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        n_eff = float(len(chain) / tau[0]) if tau[0] > 0 else float(len(chain))
    except Exception:
        n_eff = float(len(chain)) / 10.0  # conservative fallback

    log.info("emcee complete: n_eff~%.0f, r_hat~%.4f, acceptance=%.3f",
             n_eff, r_hat_approx, acceptance)

    return {
        "samples": chain,
        "r_hat": r_hat_approx,
        "n_eff": n_eff,
        "acceptance_rate": acceptance,
    }


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def run_hierarchical_inference(
    per_stream_n_impacts: dict[str, int],
    per_stream_properties: dict[str, dict],
    backend: str = "numpyro",
    **kwargs,
) -> HierarchicalResult:
    """Run hierarchical Bayesian inference on M_hm from multiple streams.

    This is the main entry point. It tries the requested backend and falls back
    to emcee if NumPyro is unavailable.

    Args:
        per_stream_n_impacts: Dict mapping stream_name -> observed n_impacts.
        per_stream_properties: Dict mapping stream_name -> {
            'length_deg': float, 'age_gyr': float, 'distance_kpc': float
        }.
        backend: "numpyro" (preferred) or "emcee".
        **kwargs: Passed to the backend runner (n_samples, n_chains, etc.).

    Returns:
        HierarchicalResult with posterior samples and diagnostics.
    """
    stream_names = list(per_stream_n_impacts.keys())
    n_streams = len(stream_names)

    n_impacts_obs = np.array([per_stream_n_impacts[s] for s in stream_names], dtype=np.float64)
    stream_lengths = np.array([per_stream_properties[s]["length_deg"] for s in stream_names])
    stream_ages = np.array([per_stream_properties[s]["age_gyr"] for s in stream_names])
    stream_distances = np.array([per_stream_properties[s].get("distance_kpc", 15.0)
                                  for s in stream_names])

    log.info("Hierarchical inference: %d streams, backend=%s", n_streams, backend)
    log.info("  Streams: %s", stream_names)
    log.info("  n_impacts: %s", dict(zip(stream_names, n_impacts_obs.astype(int).tolist())))

    # Try requested backend, fall back to emcee
    result_dict = None
    used_backend = backend

    if backend == "numpyro":
        try:
            result_dict = run_hierarchical_numpyro(
                n_impacts_obs, stream_lengths, stream_ages, stream_distances, **kwargs,
            )
        except ImportError as e:
            log.warning("NumPyro unavailable (%s), falling back to emcee", e)
            used_backend = "emcee"
        except Exception as e:
            log.warning("NumPyro failed (%s), falling back to emcee", e)
            used_backend = "emcee"

    if result_dict is None:
        result_dict = run_hierarchical_emcee(
            n_impacts_obs, stream_lengths, stream_ages, stream_distances, **kwargs,
        )

    samples = result_dict["samples"]

    # Compute summary statistics
    median = float(np.median(samples))
    upper_95 = float(np.percentile(samples, 95))
    lower_5 = float(np.percentile(samples, 5))

    # 68% HDI (highest density interval)
    from scipy.stats import gaussian_kde  # noqa: PLC0415
    try:
        sorted_samples = np.sort(samples)
        n_s = len(sorted_samples)
        ci_width = int(0.68 * n_s)
        widths = sorted_samples[ci_width:] - sorted_samples[:n_s - ci_width]
        best_start = int(np.argmin(widths))
        hdi_68 = (float(sorted_samples[best_start]), float(sorted_samples[best_start + ci_width]))
    except Exception:
        hdi_68 = (float(np.percentile(samples, 16)), float(np.percentile(samples, 84)))

    # Per-stream expected rates at the median M_hm
    per_stream_rates = {}
    for i, name in enumerate(stream_names):
        rate = expected_subhalo_rate(median, stream_lengths[i], stream_ages[i], stream_distances[i])
        per_stream_rates[name] = rate

    return HierarchicalResult(
        log10_M_hm_samples=samples,
        log10_M_hm_median=median,
        log10_M_hm_upper_95=upper_95,
        log10_M_hm_lower_5=lower_5,
        log10_M_hm_hdi_68=hdi_68,
        n_streams=n_streams,
        n_effective_samples=int(result_dict["n_eff"]),
        r_hat_max=result_dict["r_hat"],
        backend=used_backend,
        stream_names=stream_names,
        per_stream_rates=per_stream_rates,
    )
