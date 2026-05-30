"""
Subhalo mass function sampling for each dark matter model.

All samplers return masses in solar masses drawn from the relevant
theoretical mass function. Uses inverse-CDF or rejection sampling.

References:
    CDM/WDM: Springel+2008, Lovell+2014
    FDM: Hui+2017, Schive+2014
    SIDM: same mass function as CDM
"""

from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CDM  —  power-law dN/dM ~ M^alpha
# ---------------------------------------------------------------------------

def sample_cdm_masses(
    n: int,
    log10_m_min: float = 5.0,
    log10_m_max: float = 9.0,
    alpha: float = -1.9,
    seed: int = 0,
) -> np.ndarray:
    """Draw n subhalo masses [Msun] from a CDM power-law mass function.

    dN/dM ~ M^alpha  =>  dN/dlogM ~ M^(alpha+1)

    Uses inverse-CDF sampling; exact for the pure power law.

    Args:
        n: Number of masses to draw. If 0, returns empty array.
        log10_m_min: Log10 of minimum mass [Msun].
        log10_m_max: Log10 of maximum mass [Msun]. Must be > log10_m_min.
        alpha: Power-law slope (CDM prediction: -1.9).
        seed: Random seed for reproducibility.

    Returns:
        Array of n masses in solar masses, drawn from dN/dM ~ M^alpha.

    Raises:
        ValueError: If log10_m_min >= log10_m_max or n < 0.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return np.array([], dtype=np.float64)
    if log10_m_min >= log10_m_max:
        raise ValueError(f"log10_m_min ({log10_m_min}) must be < log10_m_max ({log10_m_max})")

    rng = np.random.default_rng(seed)
    m_min = 10.0 ** log10_m_min
    m_max = 10.0 ** log10_m_max
    # Inverse-CDF: integrate dN/dM ~ M^alpha → ∫M^alpha dM = M^(alpha+1)/(alpha+1)
    # So the relevant exponent for the CDF is alpha+1 (not alpha+2).
    beta = alpha + 1.0  # exponent of the integrated CDF in M

    if np.abs(beta) < 1e-6:
        # alpha = -1: dN/dM ~ M^{-1} => dN/dlogM = constant (flat in log M)
        log_masses = rng.uniform(log10_m_min, log10_m_max, size=n)
        return 10.0 ** log_masses

    u = rng.uniform(0.0, 1.0, size=n)
    masses = (u * (m_max ** beta - m_min ** beta) + m_min ** beta) ** (1.0 / beta)
    return masses


# ---------------------------------------------------------------------------
# WDM  —  CDM power-law × half-mode suppression filter (Lovell+2014)
# ---------------------------------------------------------------------------

def wdm_half_mode_mass(m_wdm_kev: float) -> float:
    """Half-mode mass [Msun] as a function of WDM particle mass [keV].

    Lovell+2014 fitting function:
        M_hm = 1.7e10 * (m_wdm / 1 keV)^{-3.33}   [Msun]

    Args:
        m_wdm_kev: WDM particle mass in keV. Must be > 0.

    Returns:
        Half-mode mass in solar masses.

    Raises:
        ValueError: If m_wdm_kev <= 0.
    """
    if m_wdm_kev <= 0:
        raise ValueError(f"WDM particle mass must be positive, got {m_wdm_kev} keV")
    return 1.7e10 * (m_wdm_kev ** (-3.33))


def wdm_suppression(m: np.ndarray, m_hm: float) -> np.ndarray:
    """WDM suppression factor at each mass M (Lovell+2014 Eq. 3).

    f(M) = (1 + (M_hm / M)^2.7)^{-0.99/2.7}
    """
    return (1.0 + (m_hm / m) ** 2.7) ** (-0.99 / 2.7)


def sample_wdm_masses(
    n: int,
    m_wdm_kev: float,
    log10_m_min: float = 5.0,
    log10_m_max: float = 9.0,
    alpha: float = -1.9,
    seed: int = 0,
) -> np.ndarray:
    """Draw n subhalo masses [Msun] from a WDM-suppressed mass function.

    Uses rejection sampling against the CDM envelope.
    """
    rng = np.random.default_rng(seed)
    m_hm = wdm_half_mode_mass(m_wdm_kev)

    accepted = []
    # Over-sample by a factor to reduce iterations
    batch = max(n * 10, 1000)
    max_iters = 200
    for _ in range(max_iters):
        if len(accepted) >= n:
            break
        proposal = sample_cdm_masses(batch, log10_m_min, log10_m_max, alpha, seed=rng.integers(0, 2**32))
        suppression = wdm_suppression(proposal, m_hm)
        u = rng.uniform(0.0, 1.0, size=batch)
        accepted.extend(proposal[u < suppression].tolist())
        batch = max(batch, 1000)
    else:
        log.warning("WDM rejection sampling: %d iters (m_wdm=%.2f keV), got %d/%d",
                     max_iters, m_wdm_kev, len(accepted), n)

    return np.array(accepted[:n]) if accepted else np.array([10**log10_m_min] * n)


# ---------------------------------------------------------------------------
# FDM  —  CDM × Jeans mass suppression (Hui+2017)
# ---------------------------------------------------------------------------

def fdm_jeans_mass(m_axion_ev: float) -> float:
    """Characteristic Jeans mass [Msun] for fuzzy DM.

    Hui+2017 approximation:
        M_J ~ 1.5e8 * (m_axion / 1e-22 eV)^{-1.5}   [Msun]
    """
    m22 = m_axion_ev / 1.0e-22
    return 1.5e8 * (m22 ** (-1.5))


def fdm_suppression(m: np.ndarray, m_j: float) -> np.ndarray:
    """FDM suppression below the Jeans mass (sharp cutoff approximation)."""
    return (1.0 + (m_j / m) ** 3.0) ** (-1.0)


def sample_fdm_masses(
    n: int,
    m_axion_ev: float,
    log10_m_min: float = 5.0,
    log10_m_max: float = 9.0,
    alpha: float = -1.9,
    seed: int = 0,
) -> np.ndarray:
    """Draw n subhalo masses [Msun] from an FDM-suppressed mass function."""
    rng = np.random.default_rng(seed)
    m_j = fdm_jeans_mass(m_axion_ev)

    accepted = []
    batch = max(n * 10, 1000)
    max_iters = 200
    for _ in range(max_iters):
        if len(accepted) >= n:
            break
        proposal = sample_cdm_masses(batch, log10_m_min, log10_m_max, alpha, seed=rng.integers(0, 2**32))
        suppression = fdm_suppression(proposal, m_j)
        u = rng.uniform(0.0, 1.0, size=batch)
        accepted.extend(proposal[u < suppression].tolist())
    else:
        log.warning("FDM rejection sampling: %d iters (m_axion=%.2e eV), got %d/%d",
                     max_iters, m_axion_ev, len(accepted), n)

    return np.array(accepted[:n]) if accepted else np.array([10**log10_m_min] * n)


# ---------------------------------------------------------------------------
# SIDM  —  same mass function as CDM (density profiles differ, not the counts)
# ---------------------------------------------------------------------------

def sample_sidm_masses(
    n: int,
    log10_m_min: float = 5.0,
    log10_m_max: float = 9.0,
    alpha: float = -1.9,
    seed: int = 0,
) -> np.ndarray:
    """Draw n subhalo masses for SIDM (identical mass function to CDM)."""
    return sample_cdm_masses(n, log10_m_min, log10_m_max, alpha, seed)


# ---------------------------------------------------------------------------
# Encounter rate
# ---------------------------------------------------------------------------

def compute_encounter_rate(
    stream_age_gyr: float,
    stream_length_kpc: float,
    stream_width_pc: float,
    log10_m_min: float,
    log10_m_max: float,
    alpha: float = -1.9,
    rho_sub_msun_kpc3: float = 5.0e3,  # subhalo mass density [Msun/kpc³]; ~1% of MW DM density
) -> float:
    """Estimate expected number of subhalo impacts during stream lifetime.

    Based on Yoon+2011 / Erkal+2016 §2 cross-section formalism:
        N_enc = n_sub × σ_cross × v_sub × T_age

    where n_sub is the number density of subhalos above M_min [kpc^-3],
    σ_cross is the geometric cross-section of the stream [kpc²],
    v_sub is the typical subhalo speed [kpc/Gyr], and T_age is the stream age [Gyr].

    Calibrated to give ~2-5 expected encounters for CDM (Erkal+2016 §2):
    rho_sub ≈ 5e3 Msun/kpc³ corresponds to ~1% of the MW DM density
    in substructure, consistent with N-body simulations (Springel+2008).

    Returns:
        Expected number of impacts (float; used as Poisson rate).
    """
    m_min = 10.0 ** log10_m_min
    m_max = 10.0 ** log10_m_max
    v_sub_kms = 150.0  # typical subhalo speed [km/s]

    # Number density of subhalos [kpc^-3]: integrate dN/dM from m_min to m_max
    # Normalised so that ∫M dN/dM dM = rho_sub, giving dN/dM = rho_sub/m_ref² * (M/m_ref)^alpha
    m_ref = np.sqrt(m_min * m_max)
    beta = alpha + 2.0  # = alpha + 2 (exponent of the CDF integral)
    if np.abs(beta) < 1e-6:
        n_sub = rho_sub_msun_kpc3 / m_ref * np.log(m_max / m_min)
    else:
        n_sub = rho_sub_msun_kpc3 / (m_ref * beta) * (
            (m_max / m_ref) ** beta - (m_min / m_ref) ** beta
        )

    # Geometric cross section [kpc²]: stream tube projected along orbit
    stream_area_kpc2 = stream_length_kpc * (stream_width_pc / 1000.0) * 2.0

    # Gravitational focusing factor at typical subhalo mass 1e7 Msun
    # (Erkal & Belokurov 2015 Eq. 2)
    G_kpc_kms2 = 4.3009e-6  # G in kpc (km/s)^2 / Msun
    m_sub = 1.0e7  # Msun (representative mass for cross-section)
    r_sub_kpc = (3.0 * m_sub / (4.0 * np.pi * 1.0e8)) ** (1.0 / 3.0)  # ~0.29 kpc
    v_esc_kms = np.sqrt(2.0 * G_kpc_kms2 * m_sub / r_sub_kpc)
    v_rel = np.sqrt(v_sub_kms ** 2 + v_esc_kms ** 2)
    sigma_geo_kpc2 = np.pi * r_sub_kpc ** 2 * (v_rel / v_sub_kms) ** 2

    # Rate = n_sub [kpc^-3] × σ [kpc²] × v [kpc/Gyr] × T [Gyr]
    # Unit conversion: 1 km/s = 1.022 kpc/Gyr  (NOT 1022; previous code was off by 1000×)
    v_sub_kpc_per_gyr = v_sub_kms * 1.022

    rate = n_sub * sigma_geo_kpc2 * v_sub_kpc_per_gyr * stream_age_gyr

    return max(float(rate), 0.05)  # floor: even FDM/WDM have non-zero CDM background


def mass_function_suppression_factor(
    dm_model: str,
    dm_params: dict,
    log10_m_min: float = 5.0,
    log10_m_max: float = 9.0,
    alpha: float = -1.9,
    n_grid: int = 512,
) -> float:
    """Integrated subhalo-count suppression relative to CDM.

    The encounter-rate normalization is computed for an unsuppressed CDM power
    law. Suppressed models should produce fewer low-mass encounters, so this
    returns the weighted integral of the WDM/FDM filter divided by the CDM
    integral over the same mass range.
    """
    if dm_model in {"CDM", "SIDM"}:
        return 1.0

    log_m = np.linspace(log10_m_min, log10_m_max, n_grid)
    masses = 10.0 ** log_m
    weights = masses ** (alpha + 1.0)  # dN/dlogM

    if dm_model == "WDM":
        suppression = wdm_suppression(masses, wdm_half_mode_mass(dm_params["m_wdm_kev"]))
    elif dm_model == "FDM":
        suppression = fdm_suppression(masses, fdm_jeans_mass(dm_params["m_axion_ev"]))
    else:
        raise ValueError(f"Unknown DM model: {dm_model}")

    denom = np.trapezoid(weights, log_m)
    if denom <= 0 or not np.isfinite(denom):
        return 1.0
    factor = np.trapezoid(weights * suppression, log_m) / denom
    return float(np.clip(factor, 0.01, 1.0))


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------

def sample_masses(
    dm_model: str,
    n: int,
    dm_params: dict,
    log10_m_min: float = 5.0,
    log10_m_max: float = 9.0,
    seed: int = 0,
) -> np.ndarray:
    """Dispatch to the correct mass function sampler based on DM model name."""
    if dm_model == "CDM":
        return sample_cdm_masses(n, log10_m_min, log10_m_max,
                                 alpha=dm_params.get("alpha", -1.9), seed=seed)
    elif dm_model == "WDM":
        return sample_wdm_masses(n, dm_params["m_wdm_kev"], log10_m_min, log10_m_max, seed=seed)
    elif dm_model == "FDM":
        return sample_fdm_masses(n, dm_params["m_axion_ev"], log10_m_min, log10_m_max, seed=seed)
    elif dm_model == "SIDM":
        return sample_sidm_masses(n, log10_m_min, log10_m_max, seed=seed)
    else:
        raise ValueError(f"Unknown DM model: {dm_model}")
