"""
Scoring functions for the timeline forward model.

Each scorer compares a simulated (perturbed) stream against the real observed
stream and returns a scalar score where **lower is better** (residual / distance).

All scorers operate on 1D density profiles binned in phi1, plus optional
kinematic fields (pm1, pm2, vrad).  They are vectorised over the bin axis
and avoid Python loops over stars.

Scorer summary
--------------
density_residual_score
    Binned linear density |sim - obs| in phi1, weighted by Poisson uncertainty.

gap_agreement_score
    Depth/width/location match for detected density dips (gaps).

kinematic_perturbation_score
    pm1 and pm2 track residuals near the gap region.

profile_feature_distance
    Euclidean distance between summary profile feature vectors (reuses the
    existing 157-dim feature extraction from the detector pipeline).

combined_score
    Weighted sum of all individual scores with configurable weights.

References
----------
Erkal & Belokurov 2015, MNRAS 450, 1136
Bonaca+2019, ApJ 881, L37  (GD-1 gap/spur analysis)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DensityProfile:
    """1D density profile of a stream binned in phi1."""
    bin_centers: np.ndarray    # [n_bins] phi1 bin centers [deg]
    bin_edges: np.ndarray      # [n_bins+1] phi1 bin edges [deg]
    counts: np.ndarray         # [n_bins] star counts per bin
    density: np.ndarray        # [n_bins] normalised density (counts / total)
    bin_width_deg: float       # uniform bin width [deg]

    @property
    def n_bins(self) -> int:
        return len(self.bin_centers)


@dataclass
class GapFeature:
    """A detected density gap (local minimum) in a stream."""
    phi1_center: float        # gap center [deg]
    phi1_width: float         # gap FWHM [deg]
    depth: float              # fractional depth: 1 - (min_density / baseline)
    significance: float       # depth / Poisson uncertainty


@dataclass
class ScoreResult:
    """Container for all scores from a single candidate evaluation."""
    density_residual: float = 0.0
    gap_agreement: float = 0.0
    kinematic_perturbation: float = 0.0
    radial_velocity: float = 0.0
    profile_distance: float = 0.0
    combined: float = 0.0
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Density profile construction
# ---------------------------------------------------------------------------

def compute_density_profile(
    phi1: np.ndarray,
    phi1_range: tuple[float, float],
    bin_width_deg: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> DensityProfile:
    """Bin star positions into a 1D density profile along phi1.

    Args:
        phi1: Star longitudes [deg], shape [N].
        phi1_range: (min, max) phi1 extent for binning.
        bin_width_deg: Width of each bin in degrees.
        weights: Optional per-star weights (e.g. membership probability).

    Returns:
        DensityProfile with normalised density.
    """
    n_bins = max(1, int(np.ceil((phi1_range[1] - phi1_range[0]) / bin_width_deg)))
    bin_edges = np.linspace(phi1_range[0], phi1_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    counts, _ = np.histogram(phi1, bins=bin_edges, weights=weights)
    total = max(counts.sum(), 1.0)
    density = counts / total

    return DensityProfile(
        bin_centers=bin_centers,
        bin_edges=bin_edges,
        counts=counts.astype(np.float64),
        density=density,
        bin_width_deg=float(bin_edges[1] - bin_edges[0]),
    )


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def detect_gaps(
    profile: DensityProfile,
    min_depth: float = 0.3,
    min_significance: float = 2.0,
    smooth_sigma_bins: float = 2.0,
) -> list[GapFeature]:
    """Detect density gaps in a profile using smoothed local-minimum finding.

    Algorithm:
        1. Smooth the density with a Gaussian kernel (sigma = smooth_sigma_bins).
        2. Compute a running baseline (median in a 15-bin window).
        3. Identify local minima below baseline * (1 - min_depth).
        4. Estimate FWHM by walking outward from each minimum.
        5. Filter by significance (depth / Poisson noise).

    Args:
        profile: Input density profile.
        min_depth: Minimum fractional depth (0 = no gap, 1 = empty).
        min_significance: Minimum depth/noise ratio.
        smooth_sigma_bins: Gaussian smoothing width in bin units.

    Returns:
        List of GapFeature, sorted by significance (descending).
    """
    from scipy.ndimage import gaussian_filter1d, median_filter

    density = profile.density.copy()
    n = len(density)

    if n < 5:
        return []

    # Smooth
    smoothed = gaussian_filter1d(density, sigma=smooth_sigma_bins, mode="nearest")

    # Running baseline: median over a wide window
    window = min(15, n // 2) | 1  # ensure odd
    baseline = median_filter(density, size=window, mode="nearest")
    baseline = np.maximum(baseline, 1e-10)

    # Poisson uncertainty per bin
    sigma_poisson = np.sqrt(np.maximum(profile.counts, 1.0)) / max(profile.counts.sum(), 1.0)

    # Find local minima in smoothed profile
    gaps = []
    for i in range(1, n - 1):
        if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
            depth = 1.0 - smoothed[i] / baseline[i]
            if depth < min_depth:
                continue
            significance = depth / max(sigma_poisson[i], 1e-10)
            if significance < min_significance:
                continue

            # Estimate FWHM: walk outward to half-depth level
            half_level = baseline[i] * (1.0 - depth / 2.0)
            left = i
            while left > 0 and smoothed[left] < half_level:
                left -= 1
            right = i
            while right < n - 1 and smoothed[right] < half_level:
                right += 1
            width_deg = (right - left) * profile.bin_width_deg

            gaps.append(GapFeature(
                phi1_center=float(profile.bin_centers[i]),
                phi1_width=max(width_deg, profile.bin_width_deg),
                depth=float(depth),
                significance=float(significance),
            ))

    # Sort by significance, descending
    gaps.sort(key=lambda g: g.significance, reverse=True)
    return gaps


def find_density_minima(
    profile: DensityProfile,
    top_k: int = 5,
    smooth_sigma_bins: float = 2.0,
    min_prominence_frac: float = 0.03,
) -> list[GapFeature]:
    """Return the most prominent density minima for *localisation*, ranked.

    Unlike ``detect_gaps`` (which applies a hard depth/significance cut tuned for
    clean simulated gaps), this always returns the real local minima of the
    observed profile, ranked by prominence. It is used to seed the forward-model
    phi1 grid from the actual most-depleted regions, even when a stream's gaps
    are shallow or diluted by contamination (as in the current GD-1 catalog,
    whose deepest dips are only ~10-15%).

    Each returned ``GapFeature`` still carries its depth (vs a running baseline)
    and Poisson significance so callers can judge how real each minimum is.
    """
    from scipy.ndimage import gaussian_filter1d, median_filter
    from scipy.signal import find_peaks

    density = profile.density
    n = len(density)
    if n < 5:
        return []

    smoothed = gaussian_filter1d(density, sigma=smooth_sigma_bins, mode="nearest")
    window = min(15, n // 2) | 1
    baseline = np.maximum(median_filter(density, size=window, mode="nearest"), 1e-10)
    sigma_poisson = np.sqrt(np.maximum(profile.counts, 1.0)) / max(profile.counts.sum(), 1.0)

    # Find minima as peaks of the inverted, smoothed density, ranked by prominence.
    inv = smoothed.max() - smoothed
    prom = min_prominence_frac * (smoothed.max() - smoothed.min() + 1e-12)
    idx, props = find_peaks(inv, prominence=prom)
    if len(idx) == 0:
        # Fall back to the single global minimum.
        idx = np.array([int(np.argmin(smoothed))])
        props = {"prominences": np.array([float(smoothed.max() - smoothed.min())])}

    feats = []
    for j, i in enumerate(idx):
        depth = float(1.0 - smoothed[i] / baseline[i])
        sig = float(max(depth, 0.0) / max(sigma_poisson[i], 1e-10))
        # estimate width at half-prominence
        half = baseline[i] * (1.0 - max(depth, 0.0) / 2.0)
        left = i
        while left > 0 and smoothed[left] < half:
            left -= 1
        right = i
        while right < n - 1 and smoothed[right] < half:
            right += 1
        feats.append((float(props["prominences"][j]), GapFeature(
            phi1_center=float(profile.bin_centers[i]),
            phi1_width=max((right - left) * profile.bin_width_deg, profile.bin_width_deg),
            depth=max(depth, 0.0),
            significance=sig,
        )))
    feats.sort(key=lambda t: t[0], reverse=True)
    return [g for _, g in feats[:top_k]]


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

def density_residual_score(
    sim_profile: DensityProfile,
    obs_profile: DensityProfile,
) -> float:
    """Poisson-weighted L1 residual between simulated and observed density profiles.

    Score = sum_i |sim_i - obs_i| / max(sqrt(obs_counts_i / N_obs), epsilon)

    Lower is better. Returns 0.0 for a perfect match.
    """
    if sim_profile.n_bins != obs_profile.n_bins:
        raise ValueError(
            f"Bin count mismatch: sim={sim_profile.n_bins} vs obs={obs_profile.n_bins}"
        )

    # Poisson weight: inverse of Poisson noise in each bin
    n_obs_total = max(obs_profile.counts.sum(), 1.0)
    sigma = np.sqrt(np.maximum(obs_profile.counts, 1.0)) / n_obs_total
    sigma = np.maximum(sigma, 1e-6)

    residual = np.abs(sim_profile.density - obs_profile.density)
    weighted = residual / sigma

    return float(np.mean(weighted))


def gap_agreement_score(
    sim_gaps: list[GapFeature],
    obs_gaps: list[GapFeature],
    phi1_tolerance_deg: float = 5.0,
) -> float:
    """Score how well simulated gaps match observed gaps.

    For each observed gap, find the nearest simulated gap within phi1_tolerance.
    Penalise:
        - Location offset (|phi1_sim - phi1_obs| / tolerance)
        - Depth mismatch (|depth_sim - depth_obs|)
        - Width mismatch (|width_sim - width_obs| / max(widths))
        - Unmatched observed gaps (penalty = 1.0 each)
        - Extra simulated gaps (penalty = 0.5 each, softer because
          substructure predictions may have real features the data misses)

    Returns a scalar score; lower is better. 0.0 = perfect match.
    """
    if not obs_gaps:
        # No observed gaps: penalise any strong simulated gaps
        return 0.5 * len(sim_gaps)

    total_penalty = 0.0
    matched_sim = set()

    for og in obs_gaps:
        best_penalty = 1.0  # unmatched penalty
        best_idx = -1

        for j, sg in enumerate(sim_gaps):
            if j in matched_sim:
                continue
            dphi = abs(sg.phi1_center - og.phi1_center)
            if dphi > phi1_tolerance_deg:
                continue

            # Per-gap penalty components
            loc_pen = dphi / phi1_tolerance_deg
            depth_pen = abs(sg.depth - og.depth)
            width_max = max(sg.phi1_width, og.phi1_width, 0.1)
            width_pen = abs(sg.phi1_width - og.phi1_width) / width_max

            penalty = 0.4 * loc_pen + 0.4 * depth_pen + 0.2 * width_pen
            if penalty < best_penalty:
                best_penalty = penalty
                best_idx = j

        if best_idx >= 0:
            matched_sim.add(best_idx)
        total_penalty += best_penalty

    # Penalise unmatched simulated gaps (softer)
    unmatched_sim = len(sim_gaps) - len(matched_sim)
    total_penalty += 0.5 * unmatched_sim

    # Normalise by number of observed gaps
    return float(total_penalty / max(len(obs_gaps), 1))


def kinematic_perturbation_score(
    sim_phi1: np.ndarray,
    sim_pm1: np.ndarray,
    sim_pm2: np.ndarray,
    obs_phi1: np.ndarray,
    obs_pm1: np.ndarray,
    obs_pm2: np.ndarray,
    phi1_range: tuple[float, float],
    bin_width_deg: float = 2.0,
) -> float:
    """Score the match of proper-motion tracks between simulation and observation.

    Bins both datasets in phi1 and computes the median PM in each bin. The
    score is the RMS difference across bins, weighted equally for pm1 and pm2.

    This captures the kinematic "kink" signature near gaps: a subhalo fly-by
    perturbs both the density AND the velocity field.

    Lower is better. Returns 0.0 for a perfect PM track match.
    """
    n_bins = max(1, int(np.ceil((phi1_range[1] - phi1_range[0]) / bin_width_deg)))
    edges = np.linspace(phi1_range[0], phi1_range[1], n_bins + 1)

    resid_pm1 = []
    resid_pm2 = []

    for i in range(n_bins):
        sim_mask = (sim_phi1 >= edges[i]) & (sim_phi1 < edges[i + 1])
        obs_mask = (obs_phi1 >= edges[i]) & (obs_phi1 < edges[i + 1])

        if sim_mask.sum() < 3 or obs_mask.sum() < 3:
            continue

        resid_pm1.append(np.median(sim_pm1[sim_mask]) - np.median(obs_pm1[obs_mask]))
        resid_pm2.append(np.median(sim_pm2[sim_mask]) - np.median(obs_pm2[obs_mask]))

    if not resid_pm1:
        return 10.0  # no overlap: large penalty

    rms_pm1 = float(np.sqrt(np.mean(np.array(resid_pm1) ** 2)))
    rms_pm2 = float(np.sqrt(np.mean(np.array(resid_pm2) ** 2)))

    return 0.5 * rms_pm1 + 0.5 * rms_pm2


def radial_velocity_score(
    sim_phi1: np.ndarray,
    sim_vrad: np.ndarray,
    obs_phi1: np.ndarray,
    obs_vrad: np.ndarray,
    phi1_range: tuple[float, float],
    bin_width_deg: float = 4.0,
    min_bins: int = 2,
) -> tuple[float, bool]:
    """Score the match of the radial-velocity (line-of-sight) track.

    Radial velocity is the phase-space dimension most directly tied to the
    "rewind": line-of-sight velocity errors dominate backward orbit integration.
    It is empty in the base Gaia membership tables and only becomes available
    once real spectroscopic RVs are fused in (see ``src/data/multi_epoch.py``).

    Only observed bins with a finite RV are compared, so streams without RV
    coverage leave this term inactive. Returns ``(score, active)`` where
    ``active`` is False when there is insufficient observed RV to compare.

    Lower is better; the score is the RMS of binned-median vrad residuals.
    """
    obs_vrad = np.asarray(obs_vrad, dtype=np.float64)
    obs_phi1 = np.asarray(obs_phi1, dtype=np.float64)
    finite = np.isfinite(obs_vrad)
    if finite.sum() < 5:
        return 0.0, False

    obs_phi1 = obs_phi1[finite]
    obs_vrad = obs_vrad[finite]

    n_bins = max(1, int(np.ceil((phi1_range[1] - phi1_range[0]) / bin_width_deg)))
    edges = np.linspace(phi1_range[0], phi1_range[1], n_bins + 1)

    resid = []
    for i in range(n_bins):
        sim_mask = (sim_phi1 >= edges[i]) & (sim_phi1 < edges[i + 1]) & np.isfinite(sim_vrad)
        obs_mask = (obs_phi1 >= edges[i]) & (obs_phi1 < edges[i + 1])
        if sim_mask.sum() < 3 or obs_mask.sum() < 3:
            continue
        resid.append(np.median(sim_vrad[sim_mask]) - np.median(obs_vrad[obs_mask]))

    if len(resid) < min_bins:
        return 0.0, False

    resid = np.array(resid)
    # Remove the overall median offset: an absolute line-of-sight velocity zero
    # point is a frame/convention nuisance (e.g. heliocentric sim vrad vs a
    # survey's GSR vlos), not a subhalo signal. Scoring the *residual* RV track
    # after subtracting the common offset isolates the differential perturbation.
    resid = resid - np.median(resid)
    return float(np.sqrt(np.mean(resid ** 2))), True


# ---------------------------------------------------------------------------
# Combined scorer
# ---------------------------------------------------------------------------

@dataclass
class ScoreWeights:
    """Relative weights for combining individual scores."""
    density: float = 1.0
    gap: float = 1.5          # gap morphology is the primary observable
    kinematic: float = 0.8
    radial_velocity: float = 0.8   # only active when real RVs are present
    profile: float = 0.5


def combined_score(
    sim_profile: DensityProfile,
    obs_profile: DensityProfile,
    sim_gaps: list[GapFeature],
    obs_gaps: list[GapFeature],
    sim_phi1: np.ndarray,
    sim_pm1: np.ndarray,
    sim_pm2: np.ndarray,
    obs_phi1: np.ndarray,
    obs_pm1: np.ndarray,
    obs_pm2: np.ndarray,
    phi1_range: tuple[float, float],
    weights: Optional[ScoreWeights] = None,
    density_bin_width: float = 1.0,
    kinematic_bin_width: float = 2.0,
    sim_vrad: Optional[np.ndarray] = None,
    obs_vrad: Optional[np.ndarray] = None,
    rv_bin_width: float = 4.0,
) -> ScoreResult:
    """Compute all individual scores and a weighted combination.

    Args:
        sim_profile, obs_profile: Density profiles.
        sim_gaps, obs_gaps: Detected gap features.
        sim_phi1/pm1/pm2, obs_phi1/pm1/pm2: Star-level kinematics.
        phi1_range: Extent for kinematic binning.
        weights: ScoreWeights controlling relative importance.
        density_bin_width: Bin width for density comparison.
        kinematic_bin_width: Bin width for PM comparison.

    Returns:
        ScoreResult with individual and combined scores.
    """
    if weights is None:
        weights = ScoreWeights()

    s_density = density_residual_score(sim_profile, obs_profile)
    s_gap = gap_agreement_score(sim_gaps, obs_gaps)
    s_kin = kinematic_perturbation_score(
        sim_phi1, sim_pm1, sim_pm2,
        obs_phi1, obs_pm1, obs_pm2,
        phi1_range, kinematic_bin_width,
    )

    # Radial-velocity term: only active when real observed RVs are supplied.
    s_rv, rv_active = 0.0, False
    if sim_vrad is not None and obs_vrad is not None:
        s_rv, rv_active = radial_velocity_score(
            sim_phi1, sim_vrad, obs_phi1, obs_vrad, phi1_range, rv_bin_width,
        )

    # Weighted combination over the active terms (RV only when present), so
    # adding RV coverage does not rescale the score for streams without it.
    w_total = weights.density + weights.gap + weights.kinematic
    s_weighted = (
        weights.density * s_density
        + weights.gap * s_gap
        + weights.kinematic * s_kin
    )
    if rv_active:
        w_total += weights.radial_velocity
        s_weighted += weights.radial_velocity * s_rv
    s_combined = s_weighted / max(w_total, 1e-6)

    return ScoreResult(
        density_residual=s_density,
        gap_agreement=s_gap,
        kinematic_perturbation=s_kin,
        radial_velocity=s_rv,
        profile_distance=0.0,  # computed separately if profile features available
        combined=s_combined,
        details={
            "weights": {
                "density": weights.density,
                "gap": weights.gap,
                "kinematic": weights.kinematic,
                "radial_velocity": weights.radial_velocity,
                "profile": weights.profile,
            },
            "rv_active": rv_active,
            "n_obs_gaps": len(obs_gaps),
            "n_sim_gaps": len(sim_gaps),
        },
    )
