#!/usr/bin/env python
"""No-impact false-positive challenge for the matched two-arm streamgapdf search.

Each synthetic observation is a smooth two-arm stream. The script localizes its
strongest density fluctuation, runs the same nuisance candidate search and
frozen decision threshold as injection/recovery, and reports the false-positive
rate. Candidate impacts retain same-seed matched smooth controls.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_training_data import _fix_galpy_dll_path

_fix_galpy_dll_path()

import numpy as np

from scripts.run_gd1_streamgapdf_injection_recovery import (
    candidate_grid_for_truth,
    nuisance_truth_cases,
    task_grid,
)
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
    score_candidate_against_synthetic_observation,
    shard_output_path,
    write_json,
)
from src.simulation.potentials import get_mw_potential
from src.simulation.stream_gen import generate_stream_df_two_arm


def summarize_rows(
    rows: list[dict[str, Any]],
    fpr_gate: float,
    seed_consistency_gate: float,
) -> dict[str, Any]:
    if not rows:
        return {"n_tasks": 0, "gates": {"all_applicable_pass": False}}
    fpr = float(np.mean([row["false_positive"] for row in rows]))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["anchor"]["case_id"]].append(row)
    consistency = []
    for group in grouped.values():
        counts = Counter(row["false_positive"] for row in group)
        consistency.append(max(counts.values()) / len(group))
    consistency_pass_rate = float(
        np.mean(np.asarray(consistency) >= seed_consistency_gate)
    )
    gates = {
        "null_false_positive_rate": {
            "value": fpr,
            "threshold": fpr_gate,
            "pass": fpr <= fpr_gate,
        },
        "seed_consistency_case_pass_rate": {
            "value": consistency_pass_rate,
            "threshold": seed_consistency_gate,
            "pass": consistency_pass_rate >= seed_consistency_gate,
        },
    }
    gates["all_applicable_pass"] = bool(
        all(item["pass"] for item in gates.values() if isinstance(item, dict))
    )
    return {
        "n_tasks": len(rows),
        "n_cases": len(grouped),
        "false_positive_rate": fpr,
        "seed_consistency_case_pass_rate": consistency_pass_rate,
        "gates": gates,
    }


def merge_shards(args: argparse.Namespace) -> int:
    paths = expand_input_paths(args.merge_shards)
    if not paths:
        raise FileNotFoundError("No shard files matched --merge-shards")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = sorted(
        [row for payload in payloads for row in payload["all_tasks"]],
        key=lambda row: row["task_id"],
    )
    summary = summarize_rows(rows, args.fpr_gate, args.seed_consistency_gate)
    payload = {
        "result_label": "Synthetic Validation - Matched streamgapdf Null FPR",
        "scientific_status": (
            "Predeclared no-impact challenge using the same candidate search and "
            "decision threshold as injection/recovery. Not a real-data constraint."
        ),
        "merged_shards": [str(path) for path in paths],
        "validation_contract": payloads[0]["validation_contract"],
        "summary": summary,
        "all_tasks": rows,
    }
    write_json(args.out, payload)
    print(json.dumps(finite(summary), indent=2))
    print(f"Wrote {args.out}")
    return 0 if summary["gates"]["all_applicable_pass"] or not args.fail_on_gate else 2


def run_task(task, args, potential, mws, selection_model) -> dict[str, Any]:
    anchor = task["truth"]
    truth_seed = task["truth_seed"]
    truth_stream = generate_stream_df_two_arm(
        "GD1",
        potential,
        n_stars=args.n_stars,
        seed=truth_seed,
        impact=False,
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
        anchor,
        "nuisance",
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
    return {
        "task_id": task["task_id"],
        "truth_seed": truth_seed,
        "candidate_seed": candidate_seed,
        "anchor": anchor,
        "target_center_phi1": target_center,
        "n_observed_primary": len(observed_primary.phi1),
        "n_observed_morphology": len(observed_morphology.phi1),
        "n_candidates": len(candidates),
        "best_candidate": best,
        "false_positive": detection_passes(best["metrics"], args.detection_threshold),
        "all_candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--fpr-gate", type=float, default=0.05)
    parser.add_argument("--seed-consistency-gate", type=float, default=0.80)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--merge-shards", nargs="+", default=None)
    parser.add_argument(
        "--out",
        default="outputs/profile_validation/gd1_streamgapdf_null_fpr.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.merge_shards:
        return merge_shards(args)
    if args.n_shards <= 0 or not 0 <= args.shard_index < args.n_shards:
        raise ValueError("Require 0 <= shard-index < n-shards")
    anchors = nuisance_truth_cases(args)
    tasks = [
        task
        for task in task_grid(anchors, args.truth_seeds)
        if task["task_id"] % args.n_shards == args.shard_index
    ]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "n_cases_total": len(anchors),
                    "n_tasks_in_shard": len(tasks),
                    "n_candidates_first_task": len(
                        candidate_grid_for_truth(
                            tasks[0]["truth"],
                            "nuisance",
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
    summary = summarize_rows(rows, args.fpr_gate, args.seed_consistency_gate)
    payload = {
        "result_label": "Synthetic Validation - Matched streamgapdf Null FPR",
        "scientific_status": (
            "Predeclared no-impact false-positive challenge using the same nuisance "
            "search and frozen decision threshold as injection/recovery. The PWB18 "
            "footprint ratio and DESI targeting pattern are approximated without "
            "copying real GD-1 features. Not a real-data constraint."
        ),
        "inputs": vars(args),
        "shard": {"index": args.shard_index, "count": args.n_shards},
        "validation_contract": {
            "detection_rule": (
                "Best candidate must improve both local primary and conditional "
                "morphology scores by at least the frozen threshold."
            ),
            "detection_threshold": args.detection_threshold,
            "null_false_positive_rate_gate": args.fpr_gate,
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
