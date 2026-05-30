"""
Observational noise injection to match Gaia DR3 characteristics.

Adds realistic proper motion uncertainties, distance uncertainties,
and foreground stellar contamination to simulated streams.

Noise model from the Gaia GOST performance tables (Lindegren+2021).
"""

from __future__ import annotations

import logging

import numpy as np

from .stream_gen import StreamParticles

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaia DR3 noise model
# ---------------------------------------------------------------------------

def gaia_pm_uncertainty(g_mag: np.ndarray) -> np.ndarray:
    """Proper motion uncertainty [mas/yr] as a function of Gaia G magnitude.

    GOST performance model (Lindegren+2021):
        sigma_pm ~ 0.021 + 0.028 * 10^{0.4*(G-15)}  [mas/yr]   for G > 13
        sigma_pm ~ 0.010   [mas/yr]                              for G <= 13

    Args:
        g_mag: Array of Gaia G magnitudes.

    Returns:
        sigma_pm array [mas/yr].
    """
    sigma = np.where(
        g_mag > 13.0,
        0.021 + 0.028 * 10.0 ** (0.4 * (g_mag - 15.0)),
        0.010,
    )
    return sigma


def gaia_parallax_uncertainty(g_mag: np.ndarray) -> np.ndarray:
    """Parallax uncertainty [mas] as a function of G magnitude.

    sigma_plx ~ 0.007 + 0.007 * 10^{0.4*(G-13)}   [mas]
    """
    return 0.007 + 0.007 * 10.0 ** (0.4 * (g_mag - 13.0))


def distance_uncertainty_from_parallax(dist_kpc: np.ndarray, g_mag: np.ndarray) -> np.ndarray:
    """Distance uncertainty [kpc] propagated from parallax uncertainty.

    sigma_dist = sigma_plx / plx^2   (in consistent units)
    For dist >> 1/sigma_plx, parallax is uninformative; use 15% fractional.
    """
    plx = 1.0 / dist_kpc  # mas (when dist in kpc)
    sigma_plx = gaia_parallax_uncertainty(g_mag)
    snr = plx / sigma_plx
    # Parallax uninformative below SNR=3; use 15% fractional uncertainty
    sigma_dist = np.where(snr > 3.0, sigma_plx / plx ** 2, 0.15 * dist_kpc)
    return sigma_dist


def stream_to_g_magnitude(
    dist_kpc: np.ndarray, stream_config: dict, seed: int = 0,
) -> np.ndarray:
    """Assign approximate Gaia G magnitudes to stream stars.

    Uses a simple absolute magnitude relation for old metal-poor main sequence
    stars at the stream's characteristic turnoff.
    M_G_turnoff ~ 4.5 for a 10 Gyr, [Fe/H] = -2 population (rough approximation).
    """
    M_G_turnoff = 4.5 - 0.5 * (stream_config.get("isochrone_feh", -2.0) + 2.0)
    scatter = 2.0  # 2-magnitude range around turnoff
    rng = np.random.default_rng(seed)
    M_G = rng.uniform(M_G_turnoff - 0.5, M_G_turnoff + scatter, size=len(dist_kpc))
    g_mag = M_G + 5.0 * np.log10(dist_kpc * 1000.0) - 5.0  # distance modulus
    return np.clip(g_mag, 14.0, 21.0)


# ---------------------------------------------------------------------------
# Main noise injection
# ---------------------------------------------------------------------------

def add_gaia_noise(
    particles: StreamParticles,
    stream_config: dict,
    seed: int = 0,
) -> tuple[StreamParticles, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add Gaia DR3-realistic noise to simulated stream particles.

    Perturbs pm1, pm2, vrad, dist with Gaussian noise scaled by the
    Gaia noise model at each star's apparent magnitude.

    Args:
        particles: Input clean stream particles.
        stream_config: Stream config dict for magnitude estimation.
        seed: Random seed.

    Returns:
        Noisy StreamParticles and error arrays (sigma_pm1, sigma_pm2, sigma_dist).
    """
    return add_gaia_noise_randomized(
        particles,
        stream_config,
        noise_scale_factor=1.0,
        seed=seed,
    )


def add_gaia_noise_randomized(
    particles: StreamParticles,
    stream_config: dict,
    noise_scale_factor: float = 1.0,
    seed: int = 0,
) -> tuple[StreamParticles, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add Gaia-like noise with a per-simulation uncertainty scale factor.

    Domain-randomized simulations use this to expose the model to Gaia error
    levels spanning under- and over-estimated uncertainties. A factor of 1.0 is
    exactly equivalent to :func:`add_gaia_noise`.

    Args:
        particles: Input clean stream particles.
        stream_config: Stream config dict for magnitude estimation.
        noise_scale_factor: Multiplicative factor applied to pm, distance, and
            radial-velocity uncertainties. Must be positive.
        seed: Random seed.

    Returns:
        Noisy StreamParticles and error arrays (sigma_pm1, sigma_pm2, sigma_dist,
        sigma_vrad).
    """
    if noise_scale_factor <= 0:
        raise ValueError(f"noise_scale_factor must be positive, got {noise_scale_factor}")

    rng = np.random.default_rng(seed)
    n = len(particles.phi1)

    g_mag = stream_to_g_magnitude(particles.dist, stream_config, seed=seed + 7000)
    sigma_pm = gaia_pm_uncertainty(g_mag) * noise_scale_factor
    sigma_dist = distance_uncertainty_from_parallax(particles.dist, g_mag) * noise_scale_factor

    # Add Gaussian noise
    pm1_noisy = particles.pm1 + rng.normal(0.0, sigma_pm)
    pm2_noisy = particles.pm2 + rng.normal(0.0, sigma_pm)
    dist_noisy = particles.dist + rng.normal(0.0, sigma_dist)
    dist_noisy = np.clip(dist_noisy, 0.1, 200.0)

    # Radial velocity: ~1-5 km/s uncertainty for RVS stars, large for faint
    rvs_faint_limit = 16.2  # Gaia RVS practical limit
    rv_available = g_mag < rvs_faint_limit
    sigma_vrad = np.where(rv_available, 1.5 * noise_scale_factor, np.nan)
    rv_sigma_for_draw = np.where(rv_available, 1.5 * noise_scale_factor, 1.0)
    vrad_noisy = np.where(
        rv_available,
        particles.vrad + rng.normal(0.0, rv_sigma_for_draw),
        np.nan,
    )

    return (
        StreamParticles(
            phi1=particles.phi1,
            phi2=particles.phi2,
            dist=dist_noisy,
            pm1=pm1_noisy,
            pm2=pm2_noisy,
            vrad=vrad_noisy,
            xyz_kpc=particles.xyz_kpc,
            vxyz_kms=particles.vxyz_kms,
        ),
        sigma_pm,
        sigma_pm,  # sigma_pm2 (same model for both)
        sigma_dist,
        sigma_vrad,
    )


def build_membership_probabilities(
    n_stream: int,
    n_foreground: int = 0,
    corruption_fraction: float = 0.0,
    low_prob_range: tuple[float, float] = (0.05, 0.49),
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Create domain-randomized membership probabilities for simulated stars.

    True stream members start near probability 1, foreground stars start near 0.
    A random fraction of true members can be deliberately corrupted below 0.5,
    matching real catalog membership mistakes. Optional Gaussian noise is then
    applied to every probability and clipped to [0, 1].
    """
    if n_stream < 0 or n_foreground < 0:
        raise ValueError("n_stream and n_foreground must be non-negative")
    if not 0.0 <= corruption_fraction <= 1.0:
        raise ValueError("corruption_fraction must be in [0, 1]")
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")

    rng = np.random.default_rng(seed)
    probs = np.concatenate([
        np.ones(n_stream, dtype=np.float32),
        np.zeros(n_foreground, dtype=np.float32),
    ])

    n_corrupt = int(round(n_stream * corruption_fraction))
    if n_corrupt > 0:
        low, high = low_prob_range
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError("low_prob_range must satisfy 0 <= low <= high <= 1")
        idx = rng.choice(n_stream, size=n_corrupt, replace=False)
        probs[idx] = rng.uniform(low, high, size=n_corrupt).astype(np.float32)

    if noise_sigma > 0 and len(probs) > 0:
        probs = probs + rng.normal(0.0, noise_sigma, size=len(probs)).astype(np.float32)

    return np.clip(probs, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Foreground contamination
# ---------------------------------------------------------------------------

def add_foreground_contamination(
    particles: StreamParticles,
    stream_config: dict,
    contamination_fraction: float = 0.10,
    seed: int = 0,
) -> StreamParticles:
    """Add foreground MW disk stars to the stream sample.

    Foreground stars have:
    - phi1 uniformly distributed in the stream range
    - phi2 uniformly distributed within phi2_selection_deg
    - Proper motions drawn from a broad MW disk distribution
    - membership_prob flagged as 0 (but we don't use labels for this)

    The contamination_fraction is relative to the current star count.

    Returns a new StreamParticles with contaminating stars appended.
    """
    rng = np.random.default_rng(seed)
    n_cont = int(len(particles.phi1) * contamination_fraction)
    if n_cont == 0:
        return particles

    phi1_range = stream_config["phi1_range_deg"]
    phi2_half = stream_config["phi2_selection_deg"]
    dist_range = stream_config["distance_kpc"]

    phi1_c = rng.uniform(phi1_range[0], phi1_range[1], n_cont)
    phi2_c = rng.uniform(-phi2_half, phi2_half, n_cont)
    dist_c = rng.uniform(dist_range[0] * 0.5, dist_range[1] * 1.5, n_cont)
    # MW disk proper motions: broad distribution
    pm1_c = rng.normal(-2.0, 5.0, n_cont)
    pm2_c = rng.normal(-1.0, 5.0, n_cont)
    vrad_c = rng.normal(0.0, 50.0, n_cont)

    return StreamParticles(
        phi1=np.concatenate([particles.phi1, phi1_c]),
        phi2=np.concatenate([particles.phi2, phi2_c]),
        dist=np.concatenate([particles.dist, dist_c]),
        pm1=np.concatenate([particles.pm1, pm1_c]),
        pm2=np.concatenate([particles.pm2, pm2_c]),
        vrad=np.concatenate([particles.vrad, vrad_c]),
        xyz_kpc=np.concatenate([particles.xyz_kpc, np.zeros((3, n_cont))], axis=1),
        vxyz_kms=np.concatenate([particles.vxyz_kms, np.zeros((3, n_cont))], axis=1),
    )
