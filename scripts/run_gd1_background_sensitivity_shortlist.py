#!/usr/bin/env python
"""Re-test GD-1 single-encounter candidates across smooth-background models.

The candidate encounters are fixed representatives from the existing fast
diagnostic. The experiment changes only the smooth particle-spray background
and random seed. Every candidate is compared to a matched no-kick control with
the same spray parameters, realization, and impact-time integration split.

This is a robustness diagnostic, not a calibrated likelihood or posterior.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_gd1_two_perturbation_shortlist import (
    TARGETS,
    evaluate_stream,
    load_desi_member,
    select_target_representatives,
    to_encounter,
)
from src.forward_model.evolve import generate_perturbed_stream_evolved
from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
from src.forward_model.scoring import ScoreWeights


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def background_parameters(name: str, row: dict) -> dict:
    return {
        "name": name,
        "stream_age_gyr": float(row["stream_age_gyr"]),
        "progenitor_mass_msun": float(row["progenitor_mass_msun"]),
        "velocity_dispersion_factor": float(row["velocity_dispersion_factor"]),
    }


def deduplicate_backgrounds(backgrounds: list[dict]) -> list[dict]:
    output = []
    seen = set()
    for background in backgrounds:
        key = (
            background["stream_age_gyr"],
            background["progenitor_mass_msun"],
            background["velocity_dispersion_factor"],
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(background)
    return output


def aggregate_seed_metrics(seed_results: list[dict]) -> dict:
    """Average target-specific improvements and retain seed-level stability."""
    targets = {}
    relative = []
    for target in TARGETS:
        key = str(target)
        primary = np.asarray(
            [
                result["metrics"]["targets"][key]["primary_relative_improvement"]
                for result in seed_results
            ],
            dtype=np.float64,
        )
        morphology = np.asarray(
            [
                result["metrics"]["targets"][key]["morphology_relative_improvement"]
                for result in seed_results
            ],
            dtype=np.float64,
        )
        targets[key] = {
            "primary_relative_improvement_mean": float(np.mean(primary)),
            "primary_relative_improvement_std": float(np.std(primary, ddof=1))
            if len(primary) > 1
            else 0.0,
            "primary_positive_fraction": float(np.mean(primary > 0)),
            "morphology_relative_improvement_mean": float(np.mean(morphology)),
            "morphology_relative_improvement_std": float(np.std(morphology, ddof=1))
            if len(morphology) > 1
            else 0.0,
            "morphology_positive_fraction": float(np.mean(morphology > 0)),
        }
        relative.extend([float(np.mean(primary)), float(np.mean(morphology))])
    return {
        "targets": targets,
        "balanced_mean_relative_improvement": float(min(relative)),
        "mean_relative_improvement": float(np.mean(relative)),
        "all_four_mean_improve": bool(all(value > 0 for value in relative)),
        "all_four_mean_material_5pct": bool(all(value >= 0.05 for value in relative)),
        "all_seeds_all_four_improve": bool(
            all(result["metrics"]["all_four_improve"] for result in seed_results)
        ),
    }


def add_event_target_summary(aggregate: dict, event: dict) -> dict:
    """Add local-target support without requiring an unrelated region to improve."""
    own_target = "30.7" if event["target_label"] == "canonical_gap_1" else "50.7"
    other_target = "50.7" if own_target == "30.7" else "30.7"
    own = aggregate["targets"][own_target]
    other = aggregate["targets"][other_target]
    own_values = [
        own["primary_relative_improvement_mean"],
        own["morphology_relative_improvement_mean"],
    ]
    other_values = [
        other["primary_relative_improvement_mean"],
        other["morphology_relative_improvement_mean"],
    ]
    aggregate["event_target"] = own_target
    aggregate["event_target_both_mean_improve"] = bool(
        all(value > 0 for value in own_values)
    )
    aggregate["event_target_both_mean_material_5pct"] = bool(
        all(value >= 0.05 for value in own_values)
    )
    aggregate["event_target_both_positive_all_seeds"] = bool(
        own["primary_positive_fraction"] == 1.0
        and own["morphology_positive_fraction"] == 1.0
    )
    aggregate["other_target_worst_mean_improvement"] = float(min(other_values))
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-fast",
        default="outputs/profile_grid/GD1/gd1_joint_profile_diagnostic.json",
    )
    parser.add_argument(
        "--calibration-json",
        default=(
            "outputs/profile_grid/GD1/"
            "gd1_background_calibration_expanded_multiseed.json"
        ),
    )
    parser.add_argument(
        "--fused-h5",
        default="data/processed/GD1_pwb18_desi_v3_fused.h5",
    )
    parser.add_argument(
        "--desi-member-h5",
        default="data/processed/GD1_desi_dr2_v3_member.h5",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n-stars", type=int, default=1000)
    parser.add_argument("--score-half-window", type=float, default=12.0)
    parser.add_argument("--phi1-bin-width", type=float, default=4.0)
    parser.add_argument("--phi2-bin-width", type=float, default=0.2)
    parser.add_argument("--phi2-max", type=float, default=3.0)
    parser.add_argument("--track-bin-width", type=float, default=2.0)
    parser.add_argument(
        "--out",
        default=(
            "outputs/profile_grid/GD1/"
            "gd1_background_sensitivity_single_shortlist.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    joint = json.loads(Path(args.joint_fast).read_text(encoding="utf-8"))
    calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    with open("config/streams.yaml", encoding="utf-8") as handle:
        gd1 = yaml.safe_load(handle)["streams"]["GD1"]

    representatives = {
        str(target): select_target_representatives(joint["all_candidates"], target)
        for target in TARGETS
    }
    events = [
        event
        for target_events in representatives.values()
        for event in target_events
    ]
    backgrounds = deduplicate_backgrounds(
        [
            {
                "name": "legacy_physical_age",
                "stream_age_gyr": float(gd1["disruption_age_gyr"]),
                "progenitor_mass_msun": float(gd1["prog_mass_solar"]),
                "velocity_dispersion_factor": 0.3,
            },
            {
                "name": "length_calibrated_default",
                "stream_age_gyr": float(gd1["spray_age_gyr"]),
                "progenitor_mass_msun": float(gd1["prog_mass_solar"]),
                "velocity_dispersion_factor": 0.3,
            },
            background_parameters(
                "best_control_primary",
                calibration["best_control_primary"],
            ),
            background_parameters(
                "best_control_morphology",
                calibration["best_control_morphology"],
            ),
        ]
    )

    desi = load_desi_member(Path(args.desi_member_h5))
    model = TimelineForwardModel(
        ForwardModelConfig(
            stream_name="GD1",
            config_path="config/streams.yaml",
            processed_h5_path=args.fused_h5,
            n_stars_sim=args.n_stars,
            base_seed=args.seeds[0],
            phi2_cut_deg=None,
            apply_pm_cuts=False,
            score_weights=ScoreWeights(
                density=1.0,
                gap=1.5,
                kinematic=0.8,
                radial_velocity=0.2,
                profile=0.0,
            ),
            use_gnn_scorer=False,
            use_fast_mode=False,
            n_workers=1,
        )
    )
    model.prepare()

    results = []
    total = len(backgrounds) * len(events)
    completed = 0
    for background in backgrounds:
        control_cache = {}
        for event in events:
            encounter = to_encounter(event)
            seed_results = []
            for seed in args.seeds:
                common = {
                    "stream_name": "GD1",
                    "potential": model.potential,
                    "encounter": encounter,
                    "n_stars": args.n_stars,
                    "seed": seed,
                    "config_path": model.cfg.config_path,
                    "mws": model._mws,
                    "stream_age_gyr": background["stream_age_gyr"],
                    "progenitor_mass_msun": background["progenitor_mass_msun"],
                    "velocity_dispersion_factor": background[
                        "velocity_dispersion_factor"
                    ],
                }
                stream = generate_perturbed_stream_evolved(**common)
                control_key = (seed, float(event["t_since_gyr"]))
                if control_key not in control_cache:
                    control_cache[control_key] = generate_perturbed_stream_evolved(
                        **common,
                        apply_kick=False,
                    )
                metrics = evaluate_stream(
                    model,
                    stream,
                    control_cache[control_key],
                    desi,
                    args.score_half_window,
                    args.phi1_bin_width,
                    args.phi2_bin_width,
                    args.phi2_max,
                    args.track_bin_width,
                )
                seed_results.append({"seed": seed, "metrics": metrics})
            aggregate = add_event_target_summary(
                aggregate_seed_metrics(seed_results),
                event,
            )
            results.append(
                {
                    "background": background,
                    "event": event,
                    "aggregate": aggregate,
                    "seed_results": seed_results,
                }
            )
            completed += 1
            print(f"{completed}/{total}", flush=True)

    by_event = {}
    for result in results:
        event_key = f"{result['event']['target_label']}::{result['event']['role']}"
        by_event.setdefault(event_key, []).append(result)
    survival = []
    for event_key, rows in by_event.items():
        survival.append(
            {
                "event": event_key,
                "n_backgrounds": len(rows),
                "n_backgrounds_all_four_mean_improve": sum(
                    row["aggregate"]["all_four_mean_improve"] for row in rows
                ),
                "n_backgrounds_all_four_mean_material_5pct": sum(
                    row["aggregate"]["all_four_mean_material_5pct"] for row in rows
                ),
                "n_backgrounds_event_target_both_mean_improve": sum(
                    row["aggregate"]["event_target_both_mean_improve"] for row in rows
                ),
                "n_backgrounds_event_target_both_mean_material_5pct": sum(
                    row["aggregate"]["event_target_both_mean_material_5pct"]
                    for row in rows
                ),
                "n_backgrounds_event_target_both_positive_all_seeds": sum(
                    row["aggregate"]["event_target_both_positive_all_seeds"]
                    for row in rows
                ),
                "best_background_result": max(
                    rows,
                    key=lambda row: row["aggregate"][
                        "balanced_mean_relative_improvement"
                    ],
                ),
            }
        )

    payload = {
        "result_label": "Observed/Synthetic Smooth-Background Sensitivity Diagnostic",
        "scientific_status": (
            "Fixed single-encounter representatives tested across smooth-stream "
            "background models and three seeds with matched no-kick controls. "
            "Scores are not calibrated likelihoods or a density-profile posterior."
        ),
        "inputs": vars(args),
        "backgrounds": backgrounds,
        "representatives": representatives,
        "n_results": len(results),
        "n_results_all_four_mean_improve": sum(
            result["aggregate"]["all_four_mean_improve"] for result in results
        ),
        "n_results_all_four_mean_material_5pct": sum(
            result["aggregate"]["all_four_mean_material_5pct"] for result in results
        ),
        "n_results_event_target_both_mean_improve": sum(
            result["aggregate"]["event_target_both_mean_improve"] for result in results
        ),
        "n_results_event_target_both_mean_material_5pct": sum(
            result["aggregate"]["event_target_both_mean_material_5pct"]
            for result in results
        ),
        "event_survival": survival,
        "all_results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
    print(
        f"{payload['n_results_all_four_mean_improve']}/{len(results)} results "
        "improve all four diagnostics on average."
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
