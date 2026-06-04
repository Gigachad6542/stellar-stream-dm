#!/usr/bin/env python
"""Screen GD-1 encounter compactness with a continuous density-profile grid.

This runner separates perturber mass from internal scale radius.  It is a
screening experiment, not a calibrated posterior: the fast impulse mode is
used by default, and score differences must survive selection-function,
multi-seed, and full-orbit checks before they can support a physical claim.

The default run deliberately compares two observational selections:

``catalog_native``
    Uses the input catalog's published/native selection.  No rectangular phi2
    or proper-motion cuts are re-applied.

``legacy_narrow``
    Reproduces the historical pipeline selection (|phi2| < 1 deg plus fixed
    proper-motion boxes).  This selection produces the phi1~50 deg density
    feature and is retained only as a sensitivity comparison.

Outputs:
    outputs/profile_grid/GD1/density_profile_grid_<tag>.json
    outputs/profile_grid/GD1/density_profile_grid_<tag>.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forward_model.pipeline import (
    ForwardModelConfig,
    TimelineForwardModel,
    observed_density_weights,
    observed_gap_detection_weights,
)
from src.forward_model.evolve import generate_perturbed_stream_evolved
from src.forward_model.scoring import ScoreWeights, compute_density_profile, detect_gaps
from src.simulation.subhalo import EncounterParams, scale_radius_from_mass


LOG = logging.getLogger("density_profile_grid")


def compactness_family(scale_factor: float) -> str:
    """Human-readable profile family for a radius relative to standard NFW."""
    if scale_factor < 0.75:
        return "compact"
    if scale_factor <= 1.5:
        return "NFW-like"
    return "broadened/cored"


def target_label(phi1: float) -> str:
    """Name the two literature-motivated GD-1 morphology targets."""
    if abs(float(phi1) - 30.7) <= 1.0:
        return "canonical_gap_1"
    if abs(float(phi1) - 50.7) <= 1.0:
        return "spur_region"
    return "data_or_user_target"


def finite(value: Any) -> Any:
    """Convert numpy values and non-finite floats into JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def selection_config(name: str) -> dict[str, Any]:
    """Return observational filtering settings for a named sensitivity case."""
    if name in {"catalog_native", "trusted"}:
        return {
            "phi2_cut_deg": None,
            "apply_pm_cuts": False,
            "description": (
                "Input catalog's native/published selection; no rectangular "
                "phi2 or proper-motion cuts re-applied. The legacy name "
                "'trusted' is retained as a backward-compatible alias."
            ),
        }
    if name == "legacy_narrow":
        return {
            "phi2_cut_deg": 1.0,
            "apply_pm_cuts": True,
            "description": (
                "Historical |phi2|<1 deg and fixed proper-motion boxes; retained "
                "only to measure selection sensitivity."
            ),
        }
    raise ValueError(f"Unknown selection: {name}")


def candidate_grid(args: argparse.Namespace) -> Iterable[dict[str, float]]:
    """Build the Cartesian encounter/profile grid."""
    for logm, t_since, phi1, impact_b, flyby_v, factor in itertools.product(
        args.log10_masses,
        args.times,
        args.phi1,
        args.impact_params,
        args.flyby_velocities,
        args.scale_factors,
    ):
        yield {
            "log10_mass": float(logm),
            "t_since_gyr": float(t_since),
            "impact_phi1": float(phi1),
            "impact_param_kpc": float(impact_b),
            "flyby_vel_kms": float(flyby_v),
            "scale_radius_factor": float(factor),
            "score_half_window_deg": float(args.score_half_window),
        }


def result_row(selection: str, params: dict[str, float], result, null_score: float) -> dict[str, Any]:
    """Flatten one candidate result with the explicit compactness parameter."""
    factor = float(params["scale_radius_factor"])
    nfw_radius = scale_radius_from_mass(10.0 ** params["log10_mass"])
    return {
        "selection": selection,
        "target_label": target_label(result.impact_phi1),
        "compactness_family": compactness_family(factor),
        "scale_radius_factor": factor,
        "nfw_scale_radius_kpc": nfw_radius,
        "scale_radius_kpc": result.scale_radius_kpc,
        "log10_mass": result.log10_mass,
        "t_since_gyr": result.t_since_gyr,
        "impact_phi1": result.impact_phi1,
        "impact_param_kpc": result.impact_param_kpc,
        "flyby_vel_kms": result.flyby_vel_kms,
        "score_half_window_deg": params["score_half_window_deg"],
        "local_null_score": null_score,
        "combined_score": result.score.combined,
        "delta_vs_null": null_score - result.score.combined,
        "better_than_null": result.score.combined < null_score,
        "density_residual": result.score.density_residual,
        "gap_agreement": result.score.gap_agreement,
        "kinematic_perturbation": result.score.kinematic_perturbation,
        "radial_velocity": result.score.radial_velocity,
        "rv_active": bool(result.score.details.get("rv_active", False)),
        "rv_error_weighted": bool(result.score.details.get("rv_error_weighted", False)),
        "rv_membership_weighted": bool(
            result.score.details.get("rv_membership_weighted", False)
        ),
        "n_stars_sim": result.n_stars_sim,
        "runtime_s": result.runtime_s,
    }


def summarize_group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Summarize score behavior by one profile/grid parameter."""
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)

    summary = []
    for value, group in grouped.items():
        ordered = sorted(group, key=lambda r: r["delta_vs_null"], reverse=True)
        scores = np.asarray([r["combined_score"] for r in ordered], dtype=float)
        deltas = np.asarray([r["delta_vs_null"] for r in ordered], dtype=float)
        summary.append(
            {
                key: value,
                "n_candidates": len(group),
                "n_better_than_null": int(np.sum(deltas > 0)),
                "best_score": float(scores.min()),
                "median_score": float(np.median(scores)),
                "best_delta_vs_null": float(deltas.max()),
                "best_candidate": ordered[0],
            }
        )
    return sorted(summary, key=lambda r: r["best_delta_vs_null"], reverse=True)


def run_selection(args: argparse.Namespace, selection: str) -> dict[str, Any]:
    """Prepare one observational selection and evaluate the full profile grid."""
    selection_cfg = selection_config(selection)
    cfg = ForwardModelConfig(
        stream_name="GD1",
        config_path="config/streams.yaml",
        processed_h5_path=args.h5,
        impact_phi1_values=list(args.phi1),
        n_stars_sim=args.n_stars,
        base_seed=args.seed,
        phi2_cut_deg=selection_cfg["phi2_cut_deg"],
        membership_prob_min=0.5,
        apply_pm_cuts=selection_cfg["apply_pm_cuts"],
        density_bin_width_deg=args.density_bin_width,
        kinematic_bin_width_deg=args.kinematic_bin_width,
        gap_detection_min_depth=args.gap_min_depth,
        gap_detection_min_significance=args.gap_min_significance,
        score_weights=ScoreWeights(
            density=1.0,
            gap=1.5,
            kinematic=0.8,
            radial_velocity=args.rv_weight,
            profile=0.0,
        ),
        use_gnn_scorer=False,
        use_fast_mode=not args.full,
        n_workers=1,
        output_dir="outputs/profile_grid",
    )

    LOG.info("Preparing %s selection", selection)
    model = TimelineForwardModel(cfg)
    model.prepare()
    global_null_score = float(model.null_score.combined)
    local_null_scores = {
        float(phi1): model.evaluate_null_window(float(phi1), args.score_half_window)
        for phi1 in args.phi1
    }
    local_observed_gaps = {}
    target_detected_features = {}
    local_rv_counts = {}
    for phi1 in args.phi1:
        score_range = (
            max(model.phi1_range[0], float(phi1) - args.score_half_window),
            min(model.phi1_range[1], float(phi1) + args.score_half_window),
        )
        rv = model.obs_particles.get("vrad")
        if rv is None:
            local_rv_counts[str(float(phi1))] = 0
        else:
            local_mask = (
                (model.obs_particles["phi1"] >= score_range[0])
                & (model.obs_particles["phi1"] <= score_range[1])
                & np.isfinite(rv)
            )
            local_rv_counts[str(float(phi1))] = int(local_mask.sum())
        profile = compute_density_profile(
            model.obs_particles["phi1"],
            score_range,
            bin_width_deg=cfg.density_bin_width_deg,
            weights=observed_density_weights(model.obs_particles),
        )
        gaps = detect_gaps(
            compute_density_profile(
                model.obs_particles["phi1"],
                score_range,
                bin_width_deg=cfg.density_bin_width_deg,
                weights=observed_gap_detection_weights(model.obs_particles),
            ),
            min_depth=cfg.gap_detection_min_depth,
            min_significance=cfg.gap_detection_min_significance,
        )
        local_observed_gaps[str(float(phi1))] = [
            {
                "phi1_center": gap.phi1_center,
                "phi1_width": gap.phi1_width,
                "depth": gap.depth,
                "significance": gap.significance,
            }
            for gap in gaps
        ]
        target_detected_features[str(float(phi1))] = [
            {
                "phi1_center": gap.phi1_center,
                "phi1_width": gap.phi1_width,
                "depth": gap.depth,
                "significance": gap.significance,
            }
            for gap in model.obs_gaps
            if abs(float(gap.phi1_center) - float(phi1)) <= args.target_tolerance
        ]
    grid = list(candidate_grid(args))

    rows = []
    matched_control_streams = {}
    matched_control_scores = {}
    t0 = time.perf_counter()
    for index, params in enumerate(grid, start=1):
        result = model.evaluate_candidate(params)
        if args.full:
            # Full-orbit candidates must be compared to a no-kick control that
            # follows the identical spray realization, impact epoch, particle
            # split, and numerical integration path. The ordinary base stream
            # uses a different generator/path and is not a valid full-orbit null.
            t_since = float(params["t_since_gyr"])
            if t_since not in matched_control_streams:
                mass = 10.0 ** float(params["log10_mass"])
                encounter = EncounterParams(
                    mass_solar=mass,
                    scale_radius_kpc=scale_radius_from_mass(mass)
                    * float(params["scale_radius_factor"]),
                    impact_param_kpc=float(params["impact_param_kpc"]),
                    flyby_vel_kms=float(params["flyby_vel_kms"]),
                    encounter_phi1=float(params["impact_phi1"]),
                    t_since_impact_gyr=t_since,
                    is_valid=True,
                    is_massive=mass > 1e8,
                )
                matched_control_streams[t_since] = generate_perturbed_stream_evolved(
                    stream_name=model.cfg.stream_name,
                    potential=model.potential,
                    encounter=encounter,
                    n_stars=model.cfg.n_stars_sim,
                    seed=model.cfg.base_seed,
                    config_path=model.cfg.config_path,
                    mws=model._mws,
                    apply_kick=False,
                )
            control_key = (t_since, float(params["impact_phi1"]))
            if control_key not in matched_control_scores:
                center = float(params["impact_phi1"])
                score_range = (
                    max(model.phi1_range[0], center - float(args.score_half_window)),
                    min(model.phi1_range[1], center + float(args.score_half_window)),
                )
                matched_control_scores[control_key] = model._score_stream_vs_obs(
                    matched_control_streams[t_since],
                    phi1_range=score_range,
                )
            null_score = float(matched_control_scores[control_key].combined)
        else:
            null_score = float(local_null_scores[params["impact_phi1"]].combined)
        rows.append(result_row(selection, params, result, null_score))
        if index % max(1, len(grid) // 10) == 0 or index == len(grid):
            best = max(rows, key=lambda r: r["delta_vs_null"])
            LOG.info(
                "%s %d/%d; best score %.4f (delta vs null %+0.4f, %s %.2fx)",
                selection,
                index,
                len(grid),
                best["combined_score"],
                best["delta_vs_null"],
                best["compactness_family"],
                best["scale_radius_factor"],
            )

    rows.sort(key=lambda r: r["delta_vs_null"], reverse=True)
    elapsed = time.perf_counter() - t0
    return {
        "selection": selection,
        "selection_description": selection_cfg["description"],
        "n_observed_stars": len(model.obs_particles["phi1"]),
        "phi1_range": list(model.phi1_range),
        "observed_gaps": [
            {
                "phi1_center": gap.phi1_center,
                "phi1_width": gap.phi1_width,
                "depth": gap.depth,
                "significance": gap.significance,
            }
            for gap in model.obs_gaps
        ],
        "global_null_score": global_null_score,
        "global_null_components": asdict(model.null_score),
        "local_null_scores": {
            str(phi1): asdict(score) for phi1, score in local_null_scores.items()
        },
        "null_policy": (
            "Candidate-specific matched no-kick full-orbit control at the same "
            "impact epoch and local score window."
            if args.full
            else "Ordinary unperturbed base stream in each local score window."
        ),
        "matched_full_orbit_control_scores": {
            f"t={time_gyr:g},phi1={phi1:g}": asdict(score)
            for (time_gyr, phi1), score in matched_control_scores.items()
        },
        "local_observed_gaps": local_observed_gaps,
        "target_detected_features": target_detected_features,
        "local_finite_rv_counts": local_rv_counts,
        "n_candidates": len(rows),
        "n_better_than_null": sum(row["better_than_null"] for row in rows),
        "runtime_s": elapsed,
        "best_candidate": rows[0],
        "by_scale_radius_factor": summarize_group(rows, "scale_radius_factor"),
        "by_compactness_family": summarize_group(rows, "compactness_family"),
        "by_impact_phi1": summarize_group(rows, "impact_phi1"),
        "all_candidates": rows,
    }


def selection_robustness(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe whether the compactness preference survives selection changes."""
    best = {result["selection"]: result["best_candidate"] for result in results}
    families = {name: row["compactness_family"] for name, row in best.items()}
    factors = {name: row["scale_radius_factor"] for name, row in best.items()}
    best_has_detected_feature = {
        result["selection"]: bool(
            result["target_detected_features"].get(
                str(float(result["best_candidate"]["impact_phi1"])), []
            )
        )
        for result in results
    }
    all_beat_null = all(row["better_than_null"] for row in best.values())
    same_family = len(set(families.values())) == 1
    all_best_target_detected_features = all(best_has_detected_feature.values())
    has_selection_comparison = len(best) >= 2
    robust = (
        has_selection_comparison
        and all_beat_null
        and same_family
        and all_best_target_detected_features
    )
    if robust:
        interpretation = (
            "The best compactness family is stable across the tested selections, "
            "but this remains a fast-mode screening result."
        )
    elif not has_selection_comparison:
        interpretation = (
            "Only one observational selection was evaluated, so selection "
            "robustness cannot be assessed and no physical interpretation is allowed."
        )
    elif not all_best_target_detected_features:
        interpretation = (
            "At least one selection does not contain a significant detected "
            "feature near its winning impact location; the compactness preference is not "
            "eligible for physical interpretation."
        )
    else:
        interpretation = (
            "The profile preference is not selection-robust and must not be "
            "interpreted as a dark-matter density-profile measurement."
        )
    return {
        "robust_screening_preference": robust,
        "has_selection_comparison": has_selection_comparison,
        "n_selections_compared": len(best),
        "all_best_candidates_beat_null": all_beat_null,
        "same_best_compactness_family": same_family,
        "all_best_targets_have_detected_features": all_best_target_detected_features,
        "best_target_has_detected_feature_by_selection": best_has_detected_feature,
        "best_family_by_selection": families,
        "best_scale_factor_by_selection": factors,
        "interpretation": interpretation,
    }


def write_outputs(payload: dict[str, Any], out_json: Path) -> None:
    """Write the full JSON artifact and a flat candidate CSV."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")

    rows = [
        row
        for selection in payload["selection_results"]
        for row in selection["all_candidates"]
    ]
    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(finite(rows))
    LOG.info("Wrote %s and %s", out_json, out_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default="data/processed/GD1_streamfinder.h5")
    parser.add_argument(
        "--selection",
        nargs="+",
        choices=["catalog_native", "trusted", "legacy_narrow"],
        default=["catalog_native", "legacy_narrow"],
    )
    parser.add_argument("--log10-masses", nargs="+", type=float, default=[7.0, 7.5, 8.0, 8.5])
    parser.add_argument("--times", nargs="+", type=float, default=[0.75, 1.25, 1.75, 2.25, 2.75])
    parser.add_argument("--phi1", nargs="+", type=float, default=[30.7, 50.7])
    parser.add_argument("--impact-params", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument("--flyby-velocities", nargs="+", type=float, default=[150.0, 200.0, 300.0])
    parser.add_argument("--scale-factors", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--n-stars", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density-bin-width", type=float, default=1.0)
    parser.add_argument("--kinematic-bin-width", type=float, default=2.0)
    parser.add_argument("--gap-min-depth", type=float, default=0.3)
    parser.add_argument("--gap-min-significance", type=float, default=2.0)
    parser.add_argument(
        "--rv-weight",
        type=float,
        default=0.0,
        help="Radial-velocity score weight; default 0 until sparse-RV calibration is complete",
    )
    parser.add_argument(
        "--score-half-window",
        type=float,
        default=12.0,
        help="Local scoring half-width around each candidate impact phi1 [deg]",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=5.0,
        help="Maximum separation between a grid target and a significant global detected feature [deg]",
    )
    parser.add_argument("--full", action="store_true", help="Use full orbit integration (slow)")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny validation grid")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.log10_masses = [7.5, 8.0]
        args.times = [1.5, 2.0]
        args.phi1 = [30.7]
        args.impact_params = [0.1]
        args.flyby_velocities = [200.0]
        args.scale_factors = [0.5, 1.0, 2.0, 3.0]
        args.n_stars = min(args.n_stars, 1000)
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    mode = "full-orbit" if args.full else "fast-impulse"
    n_grid = (
        len(args.log10_masses)
        * len(args.times)
        * len(args.phi1)
        * len(args.impact_params)
        * len(args.flyby_velocities)
        * len(args.scale_factors)
    )
    LOG.info(
        "GD-1 density-profile grid: %d candidates x %d selections (%s)",
        n_grid,
        len(args.selection),
        mode,
    )

    t0 = time.perf_counter()
    results = [run_selection(args, selection) for selection in args.selection]
    tag = args.tag or ("smoke" if args.smoke else "screening")
    out_json = Path("outputs/profile_grid/GD1") / f"density_profile_grid_{tag}.json"
    payload = {
        "result_label": "Synthetic/Observed Screening",
        "scientific_status": (
            "Fast-mode profile-grid screening only. Not a calibrated posterior or "
            "dark-matter density-profile constraint. Requires selection robustness, "
            "multi-seed stability, injection recovery, and full-orbit confirmation."
        ),
        "mode": mode,
        "input_catalog": args.h5,
        "grid": {
            "log10_masses": args.log10_masses,
            "times_gyr": args.times,
            "impact_phi1_deg": args.phi1,
            "impact_params_kpc": args.impact_params,
            "flyby_velocities_kms": args.flyby_velocities,
            "scale_radius_factors_relative_to_nfw": args.scale_factors,
            "local_score_half_window_deg": args.score_half_window,
            "target_detection_tolerance_deg": args.target_tolerance,
            "radial_velocity_weight": args.rv_weight,
            "radial_velocity_error_weighting": (
                "Per-star e_vrad with a 2 km/s intrinsic-dispersion floor when available."
            ),
            "radial_velocity_membership_weighting": (
                "DESI p_thin when present; otherwise catalog membership_prob."
            ),
            "n_stars_sim": args.n_stars,
            "base_seed": args.seed,
        },
        "selection_robustness": selection_robustness(results),
        "selection_results": results,
        "total_runtime_s": time.perf_counter() - t0,
    }
    write_outputs(payload, out_json)

    LOG.info("Selection robustness: %s", payload["selection_robustness"]["interpretation"])
    for result in results:
        best = result["best_candidate"]
        LOG.info(
            "%s best: family=%s factor=%.2fx logM=%.2f t=%.2f phi1=%.2f "
            "b=%.2f v=%.0f score=%.4f delta=%+.4f",
            result["selection"],
            best["compactness_family"],
            best["scale_radius_factor"],
            best["log10_mass"],
            best["t_since_gyr"],
            best["impact_phi1"],
            best["impact_param_kpc"],
            best["flyby_vel_kms"],
            best["combined_score"],
            best["delta_vs_null"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
