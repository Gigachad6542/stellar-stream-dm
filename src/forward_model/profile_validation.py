"""Shared helpers for matched two-arm streamgapdf validation.

The validation layer deliberately reuses the project's existing density, gap,
kinematic, radial-velocity, and conditional cross-track morphology scorers. It
adds only an observational selection approximation:

* the smooth PWB18 track/off-track footprint ratio controls along-stream
  acceptance without copying the real GD-1 density features; and
* the DESI along-stream targeting distribution controls which synthetic stars
  supply sparse radial velocities and conditional morphology.

These helpers are diagnostic infrastructure, not a calibrated likelihood.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from src.data.gd1_frames import pwb18_phi1_to_i21
from src.forward_model.morphology import (
    conditional_cross_track_js_score,
    reference_track_residuals,
)
from src.forward_model.scoring import (
    ScoreWeights,
    combined_score,
    compute_density_profile,
    detect_gaps,
    find_density_minima,
)
from src.simulation.stream_gen import StreamParticles


PROFILE_SCALE_FACTORS = (0.5, 1.0, 3.0, 10.0)
PROFILE_FAMILY_NAMES = (
    "compact_0p5x",
    "nfw_like_1x",
    "cored_3x",
    "very_cored_10x",
)


@dataclass(frozen=True)
class GD1SelectionModel:
    """Approximate real-catalog selection without copying real stream features."""

    footprint_phi1_i21: np.ndarray
    footprint_acceptance: np.ndarray
    desi_phi1_i21: np.ndarray
    fused_h5: str
    desi_h5: str


def finite(value: Any) -> Any:
    """Convert numpy values and non-finite floats into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def profile_family(
    scale_factor: float,
    reference_factors: Iterable[float] = PROFILE_SCALE_FACTORS,
) -> str:
    """Assign a continuous scale factor to the nearest predeclared family."""
    factors = np.asarray(tuple(reference_factors), dtype=np.float64)
    if scale_factor <= 0 or np.any(factors <= 0):
        raise ValueError("Scale factors must be positive")
    index = int(np.argmin(np.abs(np.log(float(scale_factor)) - np.log(factors))))
    if tuple(factors) == PROFILE_SCALE_FACTORS:
        return PROFILE_FAMILY_NAMES[index]
    return f"nearest_{factors[index]:g}x"


def load_gd1_selection_model(
    fused_h5: str | Path,
    desi_h5: str | Path,
    mws=None,
) -> GD1SelectionModel:
    """Load PWB18 footprint and DESI targeting summaries for synthetic tests."""
    fused_h5 = Path(fused_h5)
    desi_h5 = Path(desi_h5)
    with h5py.File(fused_h5, "r") as handle:
        profile = handle["streams/GD1/selection_profile"]
        centers_pwb18 = profile["bin_centers_pwb18"][:].astype(np.float64)
        footprint = profile["footprint_ratio"][:].astype(np.float64)
    with h5py.File(desi_h5, "r") as handle:
        desi_phi1 = handle["streams/GD1/members/phi1"][:].astype(np.float64)

    centers_i21 = np.asarray(
        pwb18_phi1_to_i21(centers_pwb18, mws=mws),
        dtype=np.float64,
    )
    valid = np.isfinite(centers_i21) & np.isfinite(footprint) & (footprint >= 0)
    if valid.sum() < 2:
        raise ValueError("PWB18 footprint has insufficient finite support")
    centers_i21 = centers_i21[valid]
    footprint = footprint[valid]
    order = np.argsort(centers_i21)
    centers_i21 = centers_i21[order]
    footprint = footprint[order]
    maximum = float(np.max(footprint))
    acceptance = footprint / maximum if maximum > 0 else np.ones_like(footprint)
    return GD1SelectionModel(
        footprint_phi1_i21=centers_i21,
        footprint_acceptance=np.clip(acceptance, 0.0, 1.0),
        desi_phi1_i21=desi_phi1[np.isfinite(desi_phi1)],
        fused_h5=str(fused_h5),
        desi_h5=str(desi_h5),
    )


def subset_stream(stream: StreamParticles, selection: np.ndarray) -> StreamParticles:
    """Return a phase-space-consistent subset of StreamParticles."""
    selection = np.asarray(selection)
    return StreamParticles(
        phi1=stream.phi1[selection],
        phi2=stream.phi2[selection],
        dist=stream.dist[selection],
        pm1=stream.pm1[selection],
        pm2=stream.pm2[selection],
        vrad=stream.vrad[selection],
        xyz_kpc=stream.xyz_kpc[:, selection],
        vxyz_kms=stream.vxyz_kms[:, selection],
    )


def footprint_select(
    stream: StreamParticles,
    selection_model: GD1SelectionModel | None,
    seed: int,
    min_stars: int = 20,
) -> StreamParticles:
    """Apply the smooth PWB18 footprint-ratio acceptance to a synthetic stream."""
    if selection_model is None:
        return stream
    probability = np.interp(
        stream.phi1,
        selection_model.footprint_phi1_i21,
        selection_model.footprint_acceptance,
        left=0.0,
        right=0.0,
    )
    mask = np.random.default_rng(seed).random(len(stream.phi1)) < probability
    if int(mask.sum()) < min_stars:
        raise ValueError(
            f"PWB18 footprint left only {int(mask.sum())} stars; need at least {min_stars}"
        )
    return subset_stream(stream, mask)


def targeting_mask(
    phi1: np.ndarray,
    template_phi1: np.ndarray,
    seed: int,
    bin_width_deg: float = 4.0,
    min_stars: int = 20,
) -> np.ndarray:
    """Sample a DESI-like along-stream targeting pattern without copying morphology."""
    phi1 = np.asarray(phi1, dtype=np.float64)
    template_phi1 = np.asarray(template_phi1, dtype=np.float64)
    valid_template = template_phi1[np.isfinite(template_phi1)]
    if valid_template.size < 2:
        return np.ones(len(phi1), dtype=bool)
    lo = float(min(np.nanmin(phi1), np.nanmin(valid_template)))
    hi = float(max(np.nanmax(phi1), np.nanmax(valid_template)))
    n_bins = max(2, int(np.ceil((hi - lo) / bin_width_deg)))
    edges = np.linspace(lo, hi, n_bins + 1)
    counts = np.histogram(valid_template, bins=edges)[0].astype(np.float64)
    maximum = float(np.max(counts))
    if maximum <= 0:
        return np.ones(len(phi1), dtype=bool)
    bin_index = np.clip(np.searchsorted(edges, phi1, side="right") - 1, 0, n_bins - 1)
    probability = counts[bin_index] / maximum
    mask = np.random.default_rng(seed).random(len(phi1)) < probability
    if int(mask.sum()) < min(min_stars, len(phi1)):
        order = np.argsort(probability)[::-1]
        mask[order[: min(min_stars, len(phi1))]] = True
    return mask


def make_synthetic_observation(
    truth: StreamParticles,
    selection_model: GD1SelectionModel | None,
    footprint_seed: int,
    targeting_seed: int,
) -> tuple[StreamParticles, StreamParticles, dict[str, np.ndarray]]:
    """Apply real-like selection and create scorer-compatible observed arrays."""
    primary = footprint_select(truth, selection_model, footprint_seed)
    if selection_model is None:
        sparse_mask = np.ones(len(primary.phi1), dtype=bool)
    else:
        sparse_mask = targeting_mask(
            primary.phi1,
            selection_model.desi_phi1_i21,
            targeting_seed,
        )
    morphology = subset_stream(primary, sparse_mask)
    observed = {
        "phi1": np.asarray(primary.phi1),
        "phi2": np.asarray(primary.phi2),
        "pm1": np.asarray(primary.pm1),
        "pm2": np.asarray(primary.pm2),
        "dist": np.asarray(primary.dist),
        "vrad": np.where(sparse_mask, primary.vrad, np.nan),
        "membership_prob": np.ones(len(primary.phi1), dtype=np.float64),
    }
    return primary, morphology, observed


def localize_observed_feature(
    observed_phi1: np.ndarray,
    search_range: tuple[float, float] = (20.0, 70.0),
    bin_width_deg: float = 2.0,
) -> float:
    """Localize the most prominent observed density minimum before grid search."""
    profile = compute_density_profile(observed_phi1, search_range, bin_width_deg)
    minima = find_density_minima(profile, top_k=1)
    if not minima:
        raise ValueError("Could not localize a density minimum")
    return float(minima[0].phi1_center)


def _primary_score(
    stream: StreamParticles,
    observed: dict[str, np.ndarray],
    score_range: tuple[float, float],
    weights: ScoreWeights,
    density_bin_width_deg: float,
    kinematic_bin_width_deg: float,
):
    sim_profile = compute_density_profile(
        stream.phi1,
        score_range,
        bin_width_deg=density_bin_width_deg,
    )
    obs_profile = compute_density_profile(
        observed["phi1"],
        score_range,
        bin_width_deg=density_bin_width_deg,
    )
    sim_gaps = detect_gaps(sim_profile)
    obs_gaps = detect_gaps(obs_profile)
    return combined_score(
        sim_profile=sim_profile,
        obs_profile=obs_profile,
        sim_gaps=sim_gaps,
        obs_gaps=obs_gaps,
        sim_phi1=stream.phi1,
        sim_pm1=stream.pm1,
        sim_pm2=stream.pm2,
        obs_phi1=observed["phi1"],
        obs_pm1=observed["pm1"],
        obs_pm2=observed["pm2"],
        phi1_range=score_range,
        weights=weights,
        density_bin_width=density_bin_width_deg,
        kinematic_bin_width=kinematic_bin_width_deg,
        sim_vrad=stream.vrad,
        obs_vrad=observed["vrad"],
    )


def score_candidate_against_synthetic_observation(
    candidate: StreamParticles,
    control: StreamParticles,
    observed_primary: StreamParticles,
    observed_morphology: StreamParticles,
    observed_arrays: dict[str, np.ndarray],
    target_center: float,
    score_half_window_deg: float = 12.0,
    density_bin_width_deg: float = 2.0,
    kinematic_bin_width_deg: float = 2.0,
    morphology_phi1_bin_width_deg: float = 4.0,
    morphology_phi2_bin_width_deg: float = 0.2,
    morphology_phi2_max_deg: float = 3.0,
    track_bin_width_deg: float = 2.0,
    score_weights: ScoreWeights | None = None,
) -> dict[str, Any]:
    """Score an impact candidate and its matched null against synthetic data."""
    del observed_primary  # retained in the public signature for audit clarity
    weights = score_weights or ScoreWeights(
        density=1.0,
        gap=1.5,
        kinematic=0.8,
        radial_velocity=0.2,
        profile=0.0,
    )
    score_range = (
        float(target_center) - float(score_half_window_deg),
        float(target_center) + float(score_half_window_deg),
    )
    primary = _primary_score(
        candidate,
        observed_arrays,
        score_range,
        weights,
        density_bin_width_deg,
        kinematic_bin_width_deg,
    )
    primary_null = _primary_score(
        control,
        observed_arrays,
        score_range,
        weights,
        density_bin_width_deg,
        kinematic_bin_width_deg,
    )

    candidate_delta = reference_track_residuals(
        candidate.phi1,
        candidate.phi2,
        control.phi1,
        control.phi2,
        bin_width_deg=track_bin_width_deg,
    )
    control_delta = reference_track_residuals(
        control.phi1,
        control.phi2,
        control.phi1,
        control.phi2,
        bin_width_deg=track_bin_width_deg,
    )
    observed_delta = reference_track_residuals(
        observed_morphology.phi1,
        observed_morphology.phi2,
        control.phi1,
        control.phi2,
        bin_width_deg=track_bin_width_deg,
    )
    morphology = conditional_cross_track_js_score(
        candidate.phi1,
        candidate_delta,
        observed_morphology.phi1,
        observed_delta,
        score_range,
        phi1_bin_width_deg=morphology_phi1_bin_width_deg,
        phi2_range=(-morphology_phi2_max_deg, morphology_phi2_max_deg),
        phi2_bin_width_deg=morphology_phi2_bin_width_deg,
    )
    morphology_null = conditional_cross_track_js_score(
        control.phi1,
        control_delta,
        observed_morphology.phi1,
        observed_delta,
        score_range,
        phi1_bin_width_deg=morphology_phi1_bin_width_deg,
        phi2_range=(-morphology_phi2_max_deg, morphology_phi2_max_deg),
        phi2_bin_width_deg=morphology_phi2_bin_width_deg,
    )
    primary_relative = (
        float(primary_null.combined - primary.combined)
        / max(abs(float(primary_null.combined)), 1e-12)
    )
    morphology_relative = np.nan
    if morphology["active"] and morphology_null["active"]:
        morphology_relative = (
            float(morphology_null["score"] - morphology["score"])
            / max(abs(float(morphology_null["score"])), 1e-12)
        )
    relative = np.asarray([primary_relative, morphology_relative], dtype=np.float64)
    return {
        "score_range": list(score_range),
        "primary_score": float(primary.combined),
        "primary_null_score": float(primary_null.combined),
        "primary_relative_improvement": primary_relative,
        "density_residual": float(primary.density_residual),
        "gap_agreement": float(primary.gap_agreement),
        "kinematic_perturbation": float(primary.kinematic_perturbation),
        "radial_velocity": float(primary.radial_velocity),
        "morphology_score": float(morphology["score"]),
        "morphology_null_score": float(morphology_null["score"]),
        "morphology_relative_improvement": float(morphology_relative),
        "morphology_active": bool(morphology["active"] and morphology_null["active"]),
        "morphology_n_bins": int(morphology["n_bins"]),
        "balanced_relative_improvement": (
            float(np.min(relative)) if np.all(np.isfinite(relative)) else np.nan
        ),
        "mean_relative_improvement": (
            float(np.mean(relative)) if np.all(np.isfinite(relative)) else np.nan
        ),
    }


def detection_passes(metrics: dict[str, Any], threshold: float = 0.05) -> bool:
    """Predeclared impact decision: both local diagnostic families improve."""
    values = np.asarray(
        [
            metrics["primary_relative_improvement"],
            metrics["morphology_relative_improvement"],
        ],
        dtype=np.float64,
    )
    return bool(np.all(np.isfinite(values)) and np.all(values >= threshold))


def shard_output_path(path: str | Path, shard_index: int, n_shards: int) -> Path:
    """Return a collision-free output path for one deterministic shard."""
    path = Path(path)
    if n_shards <= 1:
        return path
    return path.with_name(
        f"{path.stem}.shard-{shard_index:02d}-of-{n_shards:02d}{path.suffix}"
    )


def expand_input_paths(values: Iterable[str]) -> list[Path]:
    """Expand explicit paths and wildcard patterns consistently on Windows."""
    output: list[Path] = []
    for value in values:
        path = Path(value)
        if any(char in value for char in "*?["):
            output.extend(sorted(path.parent.glob(path.name)))
        else:
            output.append(path)
    return output


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
