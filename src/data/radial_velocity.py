"""
Radial velocity integration for SDSS-V and 4MOST spectroscopic surveys.

This module extends the stellar stream pipeline to incorporate line-of-sight
radial velocities from ground-based spectroscopic surveys. Gaia DR3 provides
radial velocities only for stars brighter than G ~ 14 (RVS instrument), missing
most stream members at d > 10 kpc. SDSS-V (Milky Way Mapper) and 4MOST (4-meter
Multi-Object Spectroscopic Telescope) will provide RVs to G ~ 19-20, covering
the full stream membership.

Adding the 6th phase-space dimension (v_los) provides:
  1. Direct measurement of velocity perturbations from subhalo flybys
  2. Resolution of the distance-velocity degeneracy in proper motions
  3. Full 3D velocity reconstruction for orbital phase estimation
  4. Improved membership assignment (streams are cold in velocity space)

Node feature extension:
  - Without RV: 18 features (phi1, phi2, dist, pm1, pm2, vrad=0, errors, membership, stream_id)
  - With RV: vrad field populated from spectroscopic measurement
  - Mixed mode: some stars have RV, others don't → imputation/masking strategy

Surveys:
  - SDSS-V Milky Way Mapper: Northern sky, R~22000, G < 19.5
    Expected RV precision: ~1-2 km/s for stream stars
    Timeline: Ongoing (2020-2025+), DR1 expected ~2025

  - 4MOST: Southern sky, R~5000-20000, G < 20.5
    Expected RV precision: ~2-5 km/s (low-res) to ~1 km/s (high-res)
    Timeline: First light 2024, survey start 2025

References:
  - Kollmeier+2017: SDSS-V survey design
  - de Jong+2019: 4MOST survey overview
  - Li+2019: Radial velocity follow-up of stream members
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RadialVelocityData:
    """Container for radial velocity measurements from spectroscopic surveys."""
    source_ids: np.ndarray          # Gaia source IDs
    rv_km_s: np.ndarray             # Radial velocity (km/s)
    rv_error_km_s: np.ndarray       # RV uncertainty (km/s)
    survey: str                     # "SDSS-V", "4MOST", "Gaia_RVS", or "combined"
    snr: np.ndarray                 # Spectral SNR per star
    teff_k: Optional[np.ndarray] = None   # Effective temperature (K)
    logg: Optional[np.ndarray] = None     # Surface gravity
    feh: Optional[np.ndarray] = None      # Metallicity [Fe/H]


class RadialVelocityProcessor:
    """Process and integrate radial velocities into the stream graph pipeline.

    Handles three scenarios:
      1. Full RV coverage: all stars have spectroscopic RV
      2. Partial coverage: some stars have RV (Gaia bright + spectroscopic subset)
      3. No RV: fall back to RV=0 with large uncertainty (current default)

    The processor implements a masking strategy for partial coverage:
      - Stars with RV: use measured value and uncertainty
      - Stars without RV: set RV to stream mean and flag with mask feature
      - The GNN learns to weight RV information by the mask

    This avoids the bias of imputing zeros (which would look like the
    reflex-corrected solar motion) and allows the network to learn different
    patterns when RV information is vs. isn't available.
    """

    def __init__(
        self,
        rv_precision_floor_km_s: float = 0.5,
        max_rv_error_km_s: float = 20.0,
        membership_rv_sigma: float = 3.0,
    ):
        """
        Args:
            rv_precision_floor: Minimum RV uncertainty (floor for very bright stars).
            max_rv_error: Maximum acceptable RV error; above this, treat as missing.
            membership_rv_sigma: Sigma-clipping for RV-based membership refinement.
        """
        self.rv_precision_floor = rv_precision_floor_km_s
        self.max_rv_error = max_rv_error_km_s
        self.membership_rv_sigma = membership_rv_sigma

    def cross_match(
        self,
        gaia_source_ids: np.ndarray,
        rv_data: RadialVelocityData,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cross-match Gaia source IDs with spectroscopic RV catalogue.

        Args:
            gaia_source_ids: [N] source IDs from the stream membership list.
            rv_data: External RV measurements.

        Returns:
            Tuple of (rv_values[N], rv_errors[N], rv_mask[N]) where mask=1
            means RV is measured, mask=0 means imputed.
        """
        N = len(gaia_source_ids)
        rv_values = np.zeros(N)
        rv_errors = np.full(N, 999.0)  # Large error = unknown
        rv_mask = np.zeros(N, dtype=np.float32)

        # Build lookup from RV catalogue
        rv_lookup = {sid: (rv, err) for sid, rv, err
                     in zip(rv_data.source_ids, rv_data.rv_km_s, rv_data.rv_error_km_s)}

        matched = 0
        for i, sid in enumerate(gaia_source_ids):
            if sid in rv_lookup:
                rv, err = rv_lookup[sid]
                if err < self.max_rv_error:
                    rv_values[i] = rv
                    rv_errors[i] = max(err, self.rv_precision_floor)
                    rv_mask[i] = 1.0
                    matched += 1

        # Impute missing: use stream median RV (if enough stars have it)
        if matched > 5:
            stream_rv_median = np.median(rv_values[rv_mask > 0])
            stream_rv_std = np.std(rv_values[rv_mask > 0])
            rv_values[rv_mask == 0] = stream_rv_median
            rv_errors[rv_mask == 0] = max(stream_rv_std * 3, 50.0)  # Conservative
        else:
            rv_values[rv_mask == 0] = 0.0
            rv_errors[rv_mask == 0] = 999.0

        coverage = matched / N if N > 0 else 0.0
        log.info("RV cross-match: %d/%d stars matched (%.1f%% coverage) from %s",
                 matched, N, 100 * coverage, rv_data.survey)

        return rv_values, rv_errors, rv_mask

    def refine_membership(
        self,
        rv_values: np.ndarray,
        rv_errors: np.ndarray,
        rv_mask: np.ndarray,
        current_membership_prob: np.ndarray,
    ) -> np.ndarray:
        """Refine stream membership probabilities using RV information.

        Stars with measured RV that deviate significantly from the stream's
        systemic velocity are likely non-members (field star contamination).

        Args:
            rv_values: [N] radial velocities (km/s).
            rv_errors: [N] RV uncertainties (km/s).
            rv_mask: [N] binary mask (1 = measured).
            current_membership_prob: [N] existing membership probabilities.

        Returns:
            Updated membership probabilities [N].
        """
        updated_prob = current_membership_prob.copy()

        # Only refine stars with measured RV
        has_rv = rv_mask > 0
        if has_rv.sum() < 5:
            return updated_prob

        # Compute stream systemic velocity (robust median)
        stream_rvs = rv_values[has_rv]
        stream_median = np.median(stream_rvs)
        stream_mad = np.median(np.abs(stream_rvs - stream_median))
        stream_sigma = 1.4826 * stream_mad  # MAD to sigma conversion

        if stream_sigma < 1.0:
            stream_sigma = 1.0  # Floor at 1 km/s

        # Sigma-clip: penalise stars far from systemic velocity
        for i in range(len(rv_values)):
            if rv_mask[i] > 0:
                deviation = abs(rv_values[i] - stream_median) / stream_sigma
                if deviation > self.membership_rv_sigma:
                    # Reduce membership probability
                    penalty = np.exp(-0.5 * (deviation - self.membership_rv_sigma) ** 2)
                    updated_prob[i] *= penalty

        n_penalised = np.sum(updated_prob < current_membership_prob)
        log.info("RV membership refinement: %d stars penalised (stream v_sys=%.1f km/s, sigma=%.1f km/s)",
                 n_penalised, stream_median, stream_sigma)

        return updated_prob

    def build_extended_node_features(
        self,
        base_features: np.ndarray,
        rv_values: np.ndarray,
        rv_errors: np.ndarray,
        rv_mask: np.ndarray,
    ) -> np.ndarray:
        """Build extended node feature matrix including RV data.

        The base pipeline uses 18 features per node. This extends to 20:
          - Index 5 (vrad): replaced with actual measured RV (or imputed)
          - Index 18 (new): RV uncertainty (normalised)
          - Index 19 (new): RV mask (1=measured, 0=imputed)

        Args:
            base_features: [N, 18] standard node features.
            rv_values: [N] radial velocities (km/s).
            rv_errors: [N] RV uncertainties (km/s).
            rv_mask: [N] binary mask.

        Returns:
            Extended features [N, 20].
        """
        N = base_features.shape[0]
        extended = np.zeros((N, 20), dtype=np.float32)

        # Copy base features
        extended[:, :18] = base_features

        # Replace vrad (index 5) with measured/imputed RV
        extended[:, 5] = rv_values

        # Add RV error (normalised by typical precision)
        extended[:, 18] = rv_errors / 5.0  # Normalise: 5 km/s is typical

        # Add RV mask
        extended[:, 19] = rv_mask

        return extended


def simulate_sdss5_coverage(
    phi1: np.ndarray,
    phi2: np.ndarray,
    g_mag: np.ndarray,
    stream_name: str,
    seed: int = 42,
) -> RadialVelocityData:
    """Simulate expected SDSS-V Milky Way Mapper coverage for a stream.

    Based on SDSS-V target selection: priority for known stream members
    with G < 19.5 in the northern hemisphere (dec > -20).

    Args:
        phi1, phi2: Stream coordinates.
        g_mag: Gaia G-band magnitudes.
        stream_name: Name for logging.
        seed: Random seed.

    Returns:
        Simulated RadialVelocityData.
    """
    rng = np.random.default_rng(seed)
    N = len(phi1)

    # SDSS-V selection: G < 19.5, with completeness depending on magnitude
    completeness = np.zeros(N)
    bright = g_mag < 17.0
    medium = (g_mag >= 17.0) & (g_mag < 18.5)
    faint = (g_mag >= 18.5) & (g_mag < 19.5)
    completeness[bright] = 0.95
    completeness[medium] = 0.70
    completeness[faint] = 0.40

    # Random selection based on completeness
    observed = rng.random(N) < completeness
    n_obs = observed.sum()

    # Generate mock source IDs
    source_ids = np.arange(N)[observed]

    # RV precision depends on magnitude and SNR
    # Bright stars: ~1 km/s; faint: ~3-5 km/s
    rv_errors = np.where(
        g_mag[observed] < 17.0, 1.0 + 0.5 * rng.standard_normal(n_obs).clip(-0.3, 1),
        np.where(g_mag[observed] < 18.5, 2.0 + rng.standard_normal(n_obs).clip(-0.5, 2),
                 4.0 + 2.0 * rng.standard_normal(n_obs).clip(-1, 3))
    )
    rv_errors = np.abs(rv_errors)

    # Mock RVs: stream systemic + random scatter
    stream_vsys = rng.uniform(-100, 100)  # Unknown systemic velocity
    rv_values = stream_vsys + rng.normal(0, 2.0, n_obs)  # Stream dispersion ~2 km/s

    # SNR
    snr = 10.0 ** (0.4 * (19.5 - g_mag[observed])) * 5.0  # Rough scaling

    log.info("Simulated SDSS-V coverage for %s: %d/%d stars (%.0f%%)",
             stream_name, n_obs, N, 100 * n_obs / N)

    return RadialVelocityData(
        source_ids=source_ids,
        rv_km_s=rv_values,
        rv_error_km_s=rv_errors,
        survey="SDSS-V",
        snr=snr,
    )


def simulate_4most_coverage(
    phi1: np.ndarray,
    phi2: np.ndarray,
    g_mag: np.ndarray,
    stream_name: str,
    seed: int = 43,
) -> RadialVelocityData:
    """Simulate expected 4MOST coverage for a stream.

    4MOST operates from the southern hemisphere with wider field but
    lower spectral resolution for faint targets.

    Args:
        phi1, phi2: Stream coordinates.
        g_mag: Gaia G-band magnitudes.
        stream_name: Name for logging.
        seed: Random seed.

    Returns:
        Simulated RadialVelocityData.
    """
    rng = np.random.default_rng(seed)
    N = len(phi1)

    # 4MOST selection: G < 20.5, wider coverage but lower completeness
    completeness = np.zeros(N)
    bright = g_mag < 17.0
    medium = (g_mag >= 17.0) & (g_mag < 19.0)
    faint = (g_mag >= 19.0) & (g_mag < 20.5)
    completeness[bright] = 0.90
    completeness[medium] = 0.60
    completeness[faint] = 0.30

    observed = rng.random(N) < completeness
    n_obs = observed.sum()

    source_ids = np.arange(N)[observed]

    # 4MOST RV precision (low-res mode for faint targets)
    rv_errors = np.where(
        g_mag[observed] < 17.0, 1.5 + 0.5 * np.abs(rng.standard_normal(n_obs)),
        np.where(g_mag[observed] < 19.0, 3.0 + np.abs(rng.standard_normal(n_obs)),
                 5.0 + 3.0 * np.abs(rng.standard_normal(n_obs)))
    )

    stream_vsys = rng.uniform(-100, 100)
    rv_values = stream_vsys + rng.normal(0, 2.5, n_obs)

    snr = 10.0 ** (0.4 * (20.5 - g_mag[observed])) * 3.0

    log.info("Simulated 4MOST coverage for %s: %d/%d stars (%.0f%%)",
             stream_name, n_obs, N, 100 * n_obs / N)

    return RadialVelocityData(
        source_ids=source_ids,
        rv_km_s=rv_values,
        rv_error_km_s=rv_errors,
        survey="4MOST",
        snr=snr,
    )


def quantify_rv_improvement(
    stream_length_deg: float,
    stream_age_gyr: float,
    n_stars: int,
    rv_coverage_fraction: float,
    rv_precision_km_s: float,
) -> dict:
    """Estimate the constraining power improvement from adding RV data.

    Quantifies how much tighter M_hm constraints become when radial velocities
    are available, based on information-theoretic arguments.

    The key insight: velocity perturbations from subhalo impacts have a
    characteristic dipolar pattern (Erkal+2016). Without RV, we only see
    the 2D projected perturbation in proper motions. With RV, we see the
    full 3D velocity field, which:
      - Resolves projection ambiguities
      - Provides independent confirmation of density gaps
      - Enables direct mass estimation from velocity kick amplitude

    Args:
        stream_length_deg: Angular stream length.
        stream_age_gyr: Stream age.
        n_stars: Total number of stream member stars.
        rv_coverage_fraction: Fraction of stars with RV measurements (0-1).
        rv_precision_km_s: Typical RV measurement precision.

    Returns:
        Dict with improvement metrics.
    """
    # Baseline: proper-motion only (2D velocity info)
    # With PM only: ~0.5-1 km/s effective velocity precision at 15 kpc
    pm_precision_eff_km_s = 0.5 * 4.74 * 15.0  # 0.5 mas/yr at 15 kpc ~ 35 km/s
    # Actually proper motion is much better: 0.05 mas/yr → ~3.5 km/s
    pm_precision_eff_km_s = 3.5

    # Information content scales as 1/sigma^2 per star
    pm_info_per_star = 2.0 / pm_precision_eff_km_s ** 2  # 2 PM components
    rv_info_per_star = 1.0 / rv_precision_km_s ** 2  # 1 RV component

    # Total information with and without RV
    n_rv_stars = int(n_stars * rv_coverage_fraction)
    total_info_pm_only = n_stars * pm_info_per_star
    total_info_with_rv = total_info_pm_only + n_rv_stars * rv_info_per_star

    # Information improvement factor
    info_ratio = total_info_with_rv / total_info_pm_only

    # Constraint improvement: sigma scales as 1/sqrt(info)
    constraint_improvement = np.sqrt(info_ratio)

    # Effective improvement in M_hm constraint (log-scale)
    # M_hm posterior width roughly scales inversely with sqrt(info)
    mhm_improvement_dex = np.log10(constraint_improvement)

    # Detection threshold improvement
    # With 3D velocities, we can detect fainter perturbations
    detection_snr_gain = np.sqrt(1.0 + rv_coverage_fraction * (pm_precision_eff_km_s / rv_precision_km_s) ** 2)

    return {
        "info_ratio": round(float(info_ratio), 2),
        "constraint_improvement_factor": round(float(constraint_improvement), 2),
        "mhm_improvement_dex": round(float(mhm_improvement_dex), 3),
        "detection_snr_gain": round(float(detection_snr_gain), 2),
        "n_stars_with_rv": n_rv_stars,
        "rv_coverage_percent": round(100 * rv_coverage_fraction, 1),
        "effective_velocity_dimensions": 2 + rv_coverage_fraction,  # 2 PM + partial RV
    }
