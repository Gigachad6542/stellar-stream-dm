#!/usr/bin/env python
"""Matched two-arm streamgapdf density-profile injection/recovery challenge.

The fixed mode isolates profile identifiability at known encounter geometry.
The nuisance mode searches nearby impact angle, time, and profile scale using
the same predeclared scorer and decision threshold. Candidate impacts are always
paired with a same-seed smooth two-arm control.

This script writes validation evidence only. It does not fit real GD-1 data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_training_data import _fix_galpy_dll_path

_fix_galpy_dll_path()

import numpy as np

from src.data.galstreams_compat import make_mwstreams
from src.forward_model.profile_validation import (
    PROFILE_SCALE_FACTORS,
    detection_passes,
    expand_input_paths,
    finite,
    footprint_select,
    load_gd1_selection_model,
    localize_observed_feature,
    make_synthetic_observation,
    profile_family,
    score_candidate_against_synthetic_observation,
    shard_output_path,
    write_json,
)
from src.simulation.potentials import get_mw_potential
from src.simulation.stream_gen import generate_stream_df_two_arm
from src.simulation.subhalo import scale_radius_from_mass


def impact_params(
    log10_mass: float,
    time_gyr: float,
    impact_angle_rad: float,
    impact_param_kpc: float,
    scale_factor: float,
    flyby_velocity_kms: float,
) -> dict[str, float]:
    mass = 10.0 ** float(log10_mass)
    return {
        "log10_mass": float(log10_mass),
        "mass": mass,
        "timpact_gyr": float(time_gyr),
        "impact_angle_rad": float(impact_angle_rad),
        "impactb_kpc": float(impact_param_kpc),
        "vsub_kms": float(flyby_velocity_kms),
        "scale_radius_factor": float(scale_factor),
        "scale_radius_kpc": float(scale_radius_from_mass(mass) * scale_factor),
        "profile_family": profile_family(scale_factor),
    }


def fixed_truth_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build the predeclared 3 x 2 x 2 x 4 = 48 fixed-geometry truth grid."""
    output = []
    for index, values in enumerate(
        itertools.product(
            args.log10_masses,
            args.times,
            args.impact_params,
            args.scale_factors,
        )
    ):
        log_mass, time_gyr, impact_param, scale_factor = values
        output.append(
            {
                "case_id": f"fixed-{index:03d}",
                **impact_params(
                    log_mass,
                    time_gyr,
                    args.fixed_impact_angle,
                    impact_param,
                    scale_factor,
                    args.flyby_velocity,
                ),
            }
        )
    return output


def nuisance_truth_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Draw deterministic planted cases from the predeclared discrete prior."""
    rng = np.random.default_rng(args.challenge_seed)
    output = []
    for index in range(args.n_cases):
        output.append(
            {
                "case_id": f"nuisance-{index:03d}",
                **impact_params(
                    rng.choice(args.log10_masses),
                    rng.choice(args.times),
                    rng.choice(args.impact_angles),
                    rng.choice(args.impact_params),
                    rng.choice(args.scale_factors),
                    args.flyby_velocity,
                ),
            }
        )
    return output


def candidate_grid_for_truth(
    truth: dict[str, Any],
    mode: str,
    scale_factors: list[float],
    angle_offsets: list[float],
    time_offsets: list[float],
) -> list[dict[str, Any]]:
    """Build the fixed or nuisance candidate search around one planted truth."""
    if mode == "fixed":
        angles = [truth["impact_angle_rad"]]
        times = [truth["timpact_gyr"]]
    else:
        angles = sorted(
            {
                max(0.05, float(truth["impact_angle_rad"]) + float(offset))
                for offset in angle_offsets
            }
        )
        times = sorted(
            {
                max(0.05, float(truth["timpact_gyr"]) + float(offset))
                for offset in time_offsets
            }
        )
    output = []
    for time_gyr, angle, scale_factor in itertools.product(
        times,
        angles,
        scale_factors,
    ):
        output.append(
            impact_params(
                truth["log10_mass"],
                time_gyr,
                angle,
                truth["impactb_kpc"],
                scale_factor,
                truth["vsub_kms"],
            )
        )
    return output


def task_grid(cases: list[dict], truth_seeds: list[int]) -> list[dict[str, Any]]:
    output = []
    for index, (case, seed) in enumerate(itertools.product(cases, truth_seeds)):
        output.append(
            {
                "task_id": index,
                "truth_seed": int(seed),
                "truth": case,
            }
        )
    return output


def _truth_neighborhood(
    candidate: dict,
    truth: dict,
    angle_tolerance: float,
    time_tolerance: float,
) -> bool:
    return bool(
        candidate["profile_family"] == truth["profile_family"]
        and abs(candidate["impact_angle_rad"] - truth["impact_angle_rad"])
        <= angle_tolerance + 1e-12
        and abs(candidate["timpact_gyr"] - truth["timpact_gyr"])
        <= time_tolerance + 1e-12
    )


def _neighborhood_tolerance(offsets: list[float]) -> float:
    """Use half the smallest non-zero grid step around the planted geometry."""
    nonzero = [abs(float(value)) for value in offsets if abs(float(value)) > 1e-12]
    return 0.0 if not nonzero else 0.5 * min(nonzero)


def summarize_rows(
    rows: list[dict[str, Any]],
    mode: str,
    profile_accuracy_gate: float,
    top10_gate: float,
    recall_gate: float,
    seed_consistency_gate: float,
) -> dict[str, Any]:
    """Summarize recovery performance and evaluate the frozen gates."""
    if not rows:
        return {"n_tasks": 0, "gates": {"all_applicable_pass": False}}
    profile_accuracy = float(np.mean([row["profile_family_recovered"] for row in rows]))
    recall = float(np.mean([row["detected"] for row in rows]))
    top10_rate = float(np.mean([row["truth_neighborhood_top10"] for row in rows]))
    scale_biases = np.asarray(
        [row["profile_scale_log10_bias"] for row in rows],
        dtype=float,
    )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["truth"]["case_id"]].append(row)
    case_consistency = []
    for group in grouped.values():
        detection_counts = Counter(row["detected"] for row in group)
        family_counts = Counter(row["recovered_profile_family"] for row in group)
        detection_agreement = max(detection_counts.values()) / len(group)
        family_agreement = max(family_counts.values()) / len(group)
        case_consistency.append(min(detection_agreement, family_agreement))
    consistency_pass_rate = float(
        np.mean(np.asarray(case_consistency) >= seed_consistency_gate)
    )

    gates = {
        "profile_family_accuracy": {
            "value": profile_accuracy,
            "threshold": profile_accuracy_gate,
            "pass": profile_accuracy >= profile_accuracy_gate,
        },
        "detection_recall": {
            "value": recall,
            "threshold": recall_gate,
            "pass": recall >= recall_gate,
        },
        "seed_consistency_case_pass_rate": {
            "value": consistency_pass_rate,
            "threshold": seed_consistency_gate,
            "pass": consistency_pass_rate >= seed_consistency_gate,
        },
    }
    if mode == "nuisance":
        gates["truth_neighborhood_top10_rate"] = {
            "value": top10_rate,
            "threshold": top10_gate,
            "pass": top10_rate >= top10_gate,
        }
    gates["all_applicable_pass"] = bool(
        all(item["pass"] for item in gates.values() if isinstance(item, dict))
    )
    return {
        "n_tasks": len(rows),
        "n_cases": len(grouped),
        "profile_family_accuracy": profile_accuracy,
        "detection_recall": recall,
        "truth_neighborhood_top10_rate": top10_rate,
        "profile_scale_log10_bias_mean": float(np.mean(scale_biases)),
        "profile_scale_log10_rmse": float(np.sqrt(np.mean(scale_biases**2))),
        "seed_consistency_case_pass_rate": consistency_pass_rate,
        "gates": gates,
    }


def merge_shards(args: argparse.Namespace) -> int:
    paths = expand_input_paths(args.merge_shards)
    if not paths:
        raise FileNotFoundError("No shard files matched --merge-shards")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    mode = payloads[0]["mode"]
    if any(payload["mode"] != mode for payload in payloads):
        raise ValueError("Cannot merge injection shards with different modes")
    rows = sorted(
        [row for payload in payloads for row in payload["all_tasks"]],
        key=lambda row: row["task_id"],
    )
    summary = summarize_rows(
        rows,
        mode,
        args.profile_accuracy_gate,
        args.top10_gate,
        args.recall_gate,
        args.seed_consistency_gate,
    )
    payload = {
        "result_label": "Synthetic Validation - Matched streamgapdf Injection Recovery",
        "scientific_status": (
            "Predeclared matched two-arm streamgapdf injection/recovery challenge. "
            "This validates the search and scorer; it is not a real-data profile constraint."
        ),
        "mode": mode,
        "merged_shards": [str(path) for path in paths],
        "validation_contract": payloads[0]["validation_contract"],
        "summary": summary,
        "all_tasks": rows,
    }
    write_json(args.out, payload)
    print(json.dumps(finite(summary), indent=2))
    print(f"Wrote {args.out}")
    return 0 if summary["gates"]["all_applicable_pass"] or not args.fail_on_gate else 2


def run_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    potential,
    mws,
    selection_model,
) -> dict[str, Any]:
    truth = task["truth"]
    truth_seed = task["truth_seed"]
    truth_stream = generate_stream_df_two_arm(
        "GD1",
        potential,
        n_stars=args.n_stars,
        seed=truth_seed,
        impact=True,
        impact_arm=args.impact_arm,
        impact_params=truth,
        config_path=args.config,
        mws=mws,
    )
    observed_primary, observed_morphology, observed_arrays = make_synthetic_observation(
        truth_stream,
        selection_model,
        footprint_seed=truth_seed + args.observation_seed_offset,
        targeting_seed=truth_seed + args.targeting_seed_offset,
    )
    target_center = localize_observed_feature(
        observed_primary.phi1,
        tuple(args.target_search_range),
        args.target_bin_width,
    )

    candidate_seed = truth_seed + args.candidate_seed_offset
    selection_seed = candidate_seed + args.observation_seed_offset
    raw_control = generate_stream_df_two_arm(
        "GD1",
        potential,
        n_stars=args.n_stars,
        seed=candidate_seed,
        impact=False,
        config_path=args.config,
        mws=mws,
    )
    control = footprint_select(raw_control, selection_model, selection_seed)
    candidates = []
    for params in candidate_grid_for_truth(
        truth,
        args.mode,
        args.scale_factors,
        args.angle_offsets,
        args.time_offsets,
    ):
        raw_candidate = generate_stream_df_two_arm(
            "GD1",
            potential,
            n_stars=args.n_stars,
            seed=candidate_seed,
            impact=True,
            impact_arm=args.impact_arm,
            impact_params=params,
            config_path=args.config,
            mws=mws,
        )
        candidate = footprint_select(raw_candidate, selection_model, selection_seed)
        metrics = score_candidate_against_synthetic_observation(
            candidate,
            control,
            observed_primary,
            observed_morphology,
            observed_arrays,
            target_center,
            score_half_window_deg=args.score_half_window,
            density_bin_width_deg=args.density_bin_width,
            kinematic_bin_width_deg=args.kinematic_bin_width,
            morphology_phi1_bin_width_deg=args.morphology_phi1_bin_width,
            morphology_phi2_bin_width_deg=args.morphology_phi2_bin_width,
            morphology_phi2_max_deg=args.morphology_phi2_max,
            track_bin_width_deg=args.track_bin_width,
        )
        candidates.append({**params, "metrics": metrics})
    candidates.sort(
        key=lambda row: (
            np.isfinite(row["metrics"]["balanced_relative_improvement"]),
            row["metrics"]["balanced_relative_improvement"],
        ),
        reverse=True,
    )
    best = candidates[0]
    angle_tolerance = _neighborhood_tolerance(args.angle_offsets)
    time_tolerance = _neighborhood_tolerance(args.time_offsets)
    neighborhood_rank = min(
        index
        for index, row in enumerate(candidates)
        if _truth_neighborhood(row, truth, angle_tolerance, time_tolerance)
    )
    top10_count = max(1, int(np.ceil(0.10 * len(candidates))))
    return {
        "task_id": task["task_id"],
        "truth_seed": truth_seed,
        "candidate_seed": candidate_seed,
        "truth": truth,
        "target_center_phi1": target_center,
        "n_observed_primary": len(observed_primary.phi1),
        "n_observed_morphology": len(observed_morphology.phi1),
        "n_candidates": len(candidates),
        "best_candidate": best,
        "recovered_profile_family": best["profile_family"],
        "profile_family_recovered": best["profile_family"] == truth["profile_family"],
        "profile_scale_log10_bias": float(
            np.log10(best["scale_radius_factor"] / truth["scale_radius_factor"])
        ),
        "truth_neighborhood_rank": neighborhood_rank,
        "truth_neighborhood_rank_fraction": neighborhood_rank / len(candidates),
        "truth_neighborhood_top10": neighborhood_rank < top10_count,
        "detected": detection_passes(best["metrics"], args.detection_threshold),
        "all_candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed", "nuisance"), default="fixed")
    parser.add_argument("--config", default="config/streams.yaml")
    parser.add_argument("--fused-h5", default="data/processed/GD1_pwb18_desi_v3_fused.h5")
    parser.add_argument("--desi-h5", default="data/processed/GD1_desi_dr2_v3_member.h5")
    parser.add_argument("--no-real-selection", action="store_true")
    parser.add_argument("--n-stars", type=int, default=1000)
    parser.add_argument("--truth-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--candidate-seed-offset", type=int, default=100_000)
    parser.add_argument("--observation-seed-offset", type=int, default=200_000)
    parser.add_argument("--targeting-seed-offset", type=int, default=300_000)
    parser.add_argument("--challenge-seed", type=int, default=20260604)
    parser.add_argument("--n-cases", type=int, default=30)
    parser.add_argument("--log10-masses", nargs="+", type=float, default=[7.5, 8.0, 8.5])
    parser.add_argument("--times", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--impact-params", nargs="+", type=float, default=[0.05, 0.2])
    parser.add_argument(
        "--scale-factors",
        nargs="+",
        type=float,
        default=list(PROFILE_SCALE_FACTORS),
    )
    parser.add_argument("--impact-angles", nargs="+", type=float, default=[0.35, 0.65])
    parser.add_argument("--fixed-impact-angle", type=float, default=0.65)
    parser.add_argument("--angle-offsets", nargs="+", type=float, default=[-0.15, 0.0, 0.15])
    parser.add_argument("--time-offsets", nargs="+", type=float, default=[-0.5, 0.0, 0.5])
    parser.add_argument("--impact-arm", choices=("leading", "trailing"), default="trailing")
    parser.add_argument("--flyby-velocity", type=float, default=150.0)
    parser.add_argument("--target-search-range", nargs=2, type=float, default=[20.0, 70.0])
    parser.add_argument("--target-bin-width", type=float, default=2.0)
    parser.add_argument("--score-half-window", type=float, default=12.0)
    parser.add_argument("--density-bin-width", type=float, default=2.0)
    parser.add_argument("--kinematic-bin-width", type=float, default=2.0)
    parser.add_argument("--morphology-phi1-bin-width", type=float, default=4.0)
    parser.add_argument("--morphology-phi2-bin-width", type=float, default=0.2)
    parser.add_argument("--morphology-phi2-max", type=float, default=3.0)
    parser.add_argument("--track-bin-width", type=float, default=2.0)
    parser.add_argument("--detection-threshold", type=float, default=0.05)
    parser.add_argument("--profile-accuracy-gate", type=float, default=0.70)
    parser.add_argument("--top10-gate", type=float, default=0.80)
    parser.add_argument("--recall-gate", type=float, default=0.60)
    parser.add_argument("--seed-consistency-gate", type=float, default=0.80)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--merge-shards", nargs="+", default=None)
    parser.add_argument(
        "--out",
        default="outputs/profile_validation/gd1_streamgapdf_injection_recovery.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge_shards:
        return merge_shards(args)
    if args.n_shards <= 0 or not 0 <= args.shard_index < args.n_shards:
        raise ValueError("Require 0 <= shard-index < n-shards")
    cases = fixed_truth_cases(args) if args.mode == "fixed" else nuisance_truth_cases(args)
    tasks = [
        task
        for task in task_grid(cases, args.truth_seeds)
        if task["task_id"] % args.n_shards == args.shard_index
    ]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "n_cases_total": len(cases),
                    "n_tasks_in_shard": len(tasks),
                    "n_candidates_first_task": len(
                        candidate_grid_for_truth(
                            tasks[0]["truth"],
                            args.mode,
                            args.scale_factors,
                            args.angle_offsets,
                            args.time_offsets,
                        )
                    )
                    if tasks
                    else 0,
                },
                indent=2,
            )
        )
        return 0

    potential = get_mw_potential(args.config)
    mws = make_mwstreams(verbose=False)
    selection_model = None
    if not args.no_real_selection:
        selection_model = load_gd1_selection_model(args.fused_h5, args.desi_h5, mws=mws)

    rows = []
    for index, task in enumerate(tasks, start=1):
        rows.append(run_task(task, args, potential, mws, selection_model))
        print(f"{index}/{len(tasks)} tasks complete", flush=True)
    summary = summarize_rows(
        rows,
        args.mode,
        args.profile_accuracy_gate,
        args.top10_gate,
        args.recall_gate,
        args.seed_consistency_gate,
    )
    payload = {
        "result_label": "Synthetic Validation - Matched streamgapdf Injection Recovery",
        "scientific_status": (
            "Predeclared matched two-arm streamgapdf injection/recovery challenge. "
            "The PWB18 footprint ratio and DESI targeting pattern are approximated "
            "without copying real GD-1 density or morphology. Not a real-data constraint."
        ),
        "mode": args.mode,
        "inputs": vars(args),
        "shard": {"index": args.shard_index, "count": args.n_shards},
        "validation_contract": {
            "detection_rule": (
                "Best candidate must improve both local primary and conditional "
                "morphology scores by at least the frozen threshold."
            ),
            "detection_threshold": args.detection_threshold,
            "truth_neighborhood": (
                "The planted angle/time cell, with a half-step tolerance to avoid "
                "crediting the full nuisance grid."
            ),
            "profile_family_accuracy_gate": args.profile_accuracy_gate,
            "truth_neighborhood_top10_gate": args.top10_gate,
            "detection_recall_gate": args.recall_gate,
            "seed_consistency_gate": args.seed_consistency_gate,
        },
        "summary": summary,
        "all_tasks": rows,
    }
    out = shard_output_path(args.out, args.shard_index, args.n_shards)
    write_json(out, payload)
    print(json.dumps(finite(summary), indent=2))
    print(f"Wrote {out}")
    return 0 if summary["gates"]["all_applicable_pass"] or not args.fail_on_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
