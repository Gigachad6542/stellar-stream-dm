"""Cross-stream morphology diagnostics for sparse spectroscopic catalogs.

The primary density-profile likelihood uses the homogeneous PWB18 spatial
catalog. DESI spectroscopy is sparse and non-uniform along the stream, so it
must not be treated as an along-stream density measurement. It can still test
the conditional cross-stream morphology, p(delta_phi2 | phi1), provided the
result is kept diagnostic until the spectroscopic targeting selection is
explicitly modeled.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray | list[float],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return weighted quantiles for finite values with positive weights."""
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("quantiles must lie in [0, 1]")

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != values.shape:
            raise ValueError("weights must match values shape")

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.full(quantiles.shape, np.nan, dtype=np.float64)

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(quantiles, cumulative, values)


def reference_track_residuals(
    phi1: np.ndarray,
    phi2: np.ndarray,
    reference_phi1: np.ndarray,
    reference_phi2: np.ndarray,
    bin_width_deg: float = 2.0,
) -> np.ndarray:
    """Subtract an interpolated reference-track median from cross-stream positions."""
    phi1 = np.asarray(phi1, dtype=np.float64)
    phi2 = np.asarray(phi2, dtype=np.float64)
    reference_phi1 = np.asarray(reference_phi1, dtype=np.float64)
    reference_phi2 = np.asarray(reference_phi2, dtype=np.float64)
    if bin_width_deg <= 0:
        raise ValueError("bin_width_deg must be positive")

    valid_reference = np.isfinite(reference_phi1) & np.isfinite(reference_phi2)
    if valid_reference.sum() < 3:
        raise ValueError("reference track requires at least three finite points")
    lo = float(np.min(reference_phi1[valid_reference]))
    hi = float(np.max(reference_phi1[valid_reference]))
    n_bins = max(2, int(np.ceil((hi - lo) / bin_width_deg)))
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        mask = (
            valid_reference
            & (reference_phi1 >= edges[i])
            & (reference_phi1 < edges[i + 1])
        )
        if mask.sum() >= 3:
            medians[i] = np.median(reference_phi2[mask])

    finite_track = np.isfinite(medians)
    if finite_track.sum() < 2:
        raise ValueError("reference track has insufficient populated bins")
    track = np.interp(
        phi1,
        centers[finite_track],
        medians[finite_track],
        left=medians[finite_track][0],
        right=medians[finite_track][-1],
    )
    return phi2 - track


def conditional_cross_track_js_score(
    sim_phi1: np.ndarray,
    sim_delta_phi2: np.ndarray,
    obs_phi1: np.ndarray,
    obs_delta_phi2: np.ndarray,
    phi1_range: tuple[float, float],
    obs_weight: Optional[np.ndarray] = None,
    phi1_bin_width_deg: float = 4.0,
    phi2_range: tuple[float, float] = (-3.0, 3.0),
    phi2_bin_width_deg: float = 0.2,
    min_obs_stars: int = 3,
    min_sim_stars: int = 8,
) -> dict:
    """Compare conditional cross-stream distributions with JS distance.

    Each populated phi1 bin is normalized independently before comparison.
    This removes DESI's strongly non-uniform along-stream targeting from the
    diagnostic, but it does not prove that targeting is independent of phi2.
    """
    if phi1_bin_width_deg <= 0 or phi2_bin_width_deg <= 0:
        raise ValueError("bin widths must be positive")
    if phi1_range[1] <= phi1_range[0] or phi2_range[1] <= phi2_range[0]:
        raise ValueError("ranges must be increasing")

    sim_phi1 = np.asarray(sim_phi1, dtype=np.float64)
    sim_delta_phi2 = np.asarray(sim_delta_phi2, dtype=np.float64)
    obs_phi1 = np.asarray(obs_phi1, dtype=np.float64)
    obs_delta_phi2 = np.asarray(obs_delta_phi2, dtype=np.float64)
    if obs_weight is None:
        obs_weight = np.ones_like(obs_phi1)
    else:
        obs_weight = np.asarray(obs_weight, dtype=np.float64)
        if obs_weight.shape != obs_phi1.shape:
            raise ValueError("obs_weight must match obs_phi1 shape")

    sim_valid = np.isfinite(sim_phi1) & np.isfinite(sim_delta_phi2)
    obs_valid = (
        np.isfinite(obs_phi1)
        & np.isfinite(obs_delta_phi2)
        & np.isfinite(obs_weight)
        & (obs_weight > 0)
    )
    if sim_valid.sum() < min_sim_stars or obs_valid.sum() < min_obs_stars:
        return {"score": np.nan, "active": False, "n_bins": 0}

    sim_center = weighted_quantile(sim_delta_phi2[sim_valid], [0.5])[0]
    obs_center = weighted_quantile(
        obs_delta_phi2[obs_valid], [0.5], obs_weight[obs_valid]
    )[0]
    sim_delta_phi2 = sim_delta_phi2 - sim_center
    obs_delta_phi2 = obs_delta_phi2 - obs_center

    n_phi1 = max(1, int(np.ceil((phi1_range[1] - phi1_range[0]) / phi1_bin_width_deg)))
    phi1_edges = np.linspace(phi1_range[0], phi1_range[1], n_phi1 + 1)
    n_phi2 = max(2, int(np.ceil((phi2_range[1] - phi2_range[0]) / phi2_bin_width_deg)))
    phi2_edges = np.linspace(phi2_range[0], phi2_range[1], n_phi2 + 1)

    distances = []
    for i in range(n_phi1):
        sim_mask = (
            sim_valid
            & (sim_phi1 >= phi1_edges[i])
            & (sim_phi1 < phi1_edges[i + 1])
        )
        obs_mask = (
            obs_valid
            & (obs_phi1 >= phi1_edges[i])
            & (obs_phi1 < phi1_edges[i + 1])
        )
        if sim_mask.sum() < min_sim_stars or obs_mask.sum() < min_obs_stars:
            continue

        sim_hist, _ = np.histogram(sim_delta_phi2[sim_mask], bins=phi2_edges)
        obs_hist, _ = np.histogram(
            obs_delta_phi2[obs_mask],
            bins=phi2_edges,
            weights=obs_weight[obs_mask],
        )
        # A small Jeffreys-style pseudocount keeps the divergence finite while
        # remaining negligible relative to populated bins.
        sim_prob = sim_hist.astype(np.float64) + 0.5
        obs_prob = obs_hist.astype(np.float64) + 0.5
        sim_prob /= sim_prob.sum()
        obs_prob /= obs_prob.sum()
        midpoint = 0.5 * (sim_prob + obs_prob)
        js = 0.5 * np.sum(sim_prob * np.log(sim_prob / midpoint))
        js += 0.5 * np.sum(obs_prob * np.log(obs_prob / midpoint))
        distances.append(np.sqrt(max(float(js), 0.0)))

    if not distances:
        return {"score": np.nan, "active": False, "n_bins": 0}
    return {
        "score": float(np.mean(distances)),
        "active": True,
        "n_bins": len(distances),
        "per_bin_distance": distances,
    }


def component_morphology_summary(
    delta_phi2: np.ndarray,
    p_thin: np.ndarray,
    p_cocoon: np.ndarray,
) -> dict:
    """Summarize the observed thin/cocoon cross-stream decomposition."""
    delta_phi2 = np.asarray(delta_phi2, dtype=np.float64)
    p_thin = np.clip(np.asarray(p_thin, dtype=np.float64), 0.0, None)
    p_cocoon = np.clip(np.asarray(p_cocoon, dtype=np.float64), 0.0, None)
    if not (delta_phi2.shape == p_thin.shape == p_cocoon.shape):
        raise ValueError("component arrays must have matching shapes")

    def location_width(weight: np.ndarray) -> tuple[float, float]:
        q16, q50, q84 = weighted_quantile(delta_phi2, [0.16, 0.5, 0.84], weight)
        return float(q50), float(0.5 * (q84 - q16))

    thin_location, thin_width = location_width(p_thin)
    cocoon_location, cocoon_width = location_width(p_cocoon)
    thin_sum = float(np.nansum(p_thin))
    cocoon_sum = float(np.nansum(p_cocoon))
    total = thin_sum + cocoon_sum
    return {
        "thin_weight_sum": thin_sum,
        "cocoon_weight_sum": cocoon_sum,
        "cocoon_fraction": cocoon_sum / total if total > 0 else np.nan,
        "thin_location_deg": thin_location,
        "thin_width_deg": thin_width,
        "cocoon_location_deg": cocoon_location,
        "cocoon_width_deg": cocoon_width,
        "absolute_component_offset_deg": abs(cocoon_location - thin_location),
    }
