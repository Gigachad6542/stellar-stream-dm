#!/usr/bin/env python
"""Test a constrained two-perturbation GD-1 shortlist with matched controls.

One encounter is placed near the canonical density feature and one near the
spur/cocoon region. Candidate events are not freely optimized: each region gets
three representatives from the existing fast diagnostic (best primary,
best conditional morphology, and best balanced), producing a 3x3 full-orbit
shortlist.

Every single- and two-event stream is compared to a no-kick control following
the identical spray realization and sequential integration epochs. The output
uses target-specific relative improvements and an explicit complexity guard;
it is not a calibrated likelihood or posterior.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forward_model.evolve import (
    generate_perturbed_stream_evolved,
    generate_perturbed_stream_multi_evolved,
)
from src.forward_model.morphology import (
    conditional_cross_track_js_score,
    reference_track_residuals,
)
from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
from src.forward_model.scoring import ScoreWeights
from src.simulation.subhalo import EncounterParams


TARGETS = (30.7, 50.7)
EVENT_FIELDS = (
    "impact_phi1",
    "scale_radius_factor",
    "scale_radius_kpc",
    "log10_mass",
    "t_since_gyr",
    "impact_param_kpc",
    "flyby_vel_kms",
    "target_label",
    "compactness_family",
)


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


def event_key(row: dict) -> tuple:
    return tuple(float(row[name]) for name in EVENT_FIELDS[:7])


def select_target_representatives(rows: list[dict], target: float) -> list[dict]:
    """Select primary, morphology, and balanced representatives for one target."""
    target_rows = [row for row in rows if float(row["impact_phi1"]) == float(target)]
    if not target_rows:
        raise ValueError(f"No diagnostic candidates for target {target}")
    selected = [
        ("best_fast_primary", max(target_rows, key=lambda row: row["primary_relative_improvement"])),
        (
            "best_fast_morphology",
            max(target_rows, key=lambda row: row["morphology_relative_improvement"]),
        ),
        (
            "best_fast_balanced",
            max(target_rows, key=lambda row: row["balanced_relative_improvement"]),
        ),
    ]
    output = []
    seen = set()
    for role, row in selected:
        key = event_key(row)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "role": role,
                **{field: row[field] for field in EVENT_FIELDS},
                "fast_primary_relative_improvement": row["primary_relative_improvement"],
                "fast_morphology_relative_improvement": row[
                    "morphology_relative_improvement"
                ],
            }
        )
    return output


def to_encounter(event: dict) -> EncounterParams:
    mass = 10.0 ** float(event["log10_mass"])
    return EncounterParams(
        mass_solar=mass,
        scale_radius_kpc=float(event["scale_radius_kpc"]),
        impact_param_kpc=float(event["impact_param_kpc"]),
        flyby_vel_kms=float(event["flyby_vel_kms"]),
        encounter_phi1=float(event["impact_phi1"]),
        t_since_impact_gyr=float(event["t_since_gyr"]),
        is_valid=True,
        is_massive=mass > 1e8,
    )


def load_desi_member(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["streams/GD1/members"]
        return {
            name: group[name][:].astype(np.float64)
            for name in ("phi1", "delta_phi2", "p_thin", "p_cocoon")
        }


def evaluate_stream(
    model: TimelineForwardModel,
    stream,
    control,
    desi: dict[str, np.ndarray],
    half_width: float,
    phi1_bin_width: float,
    phi2_bin_width: float,
    phi2_max: float,
    track_bin_width: float,
) -> dict:
    """Evaluate local primary and conditional morphology scores at both targets."""
    p_member = np.clip(desi["p_thin"] + desi["p_cocoon"], 0.0, 1.0)
    stream_delta = reference_track_residuals(
        stream.phi1,
        stream.phi2,
        control.phi1,
        control.phi2,
        bin_width_deg=track_bin_width,
    )
    control_delta = reference_track_residuals(
        control.phi1,
        control.phi2,
        control.phi1,
        control.phi2,
        bin_width_deg=track_bin_width,
    )
    target_metrics = {}
    relative_values = []
    for target in TARGETS:
        score_range = (target - half_width, target + half_width)
        primary = model._score_stream_vs_obs(stream, phi1_range=score_range)
        primary_null = model._score_stream_vs_obs(control, phi1_range=score_range)
        morphology = conditional_cross_track_js_score(
            stream.phi1,
            stream_delta,
            desi["phi1"],
            desi["delta_phi2"],
            score_range,
            obs_weight=p_member,
            phi1_bin_width_deg=phi1_bin_width,
            phi2_range=(-phi2_max, phi2_max),
            phi2_bin_width_deg=phi2_bin_width,
        )
        morphology_null = conditional_cross_track_js_score(
            control.phi1,
            control_delta,
            desi["phi1"],
            desi["delta_phi2"],
            score_range,
            obs_weight=p_member,
            phi1_bin_width_deg=phi1_bin_width,
            phi2_range=(-phi2_max, phi2_max),
            phi2_bin_width_deg=phi2_bin_width,
        )
        primary_relative = (
            float(primary_null.combined - primary.combined)
            / max(abs(float(primary_null.combined)), 1e-12)
        )
        if morphology["active"] and morphology_null["active"]:
            morphology_relative = (
                float(morphology_null["score"] - morphology["score"])
                / max(abs(float(morphology_null["score"])), 1e-12)
            )
        else:
            morphology_relative = np.nan
        relative_values.extend([primary_relative, morphology_relative])
        target_metrics[str(target)] = {
            "primary_score": primary.combined,
            "primary_null_score": primary_null.combined,
            "primary_delta_vs_null": primary_null.combined - primary.combined,
            "primary_relative_improvement": primary_relative,
            "density_residual": primary.density_residual,
            "kinematic_perturbation": primary.kinematic_perturbation,
            "radial_velocity": primary.radial_velocity,
            "morphology_score": morphology["score"],
            "morphology_null_score": morphology_null["score"],
            "morphology_delta_vs_null": (
                morphology_null["score"] - morphology["score"]
                if morphology["active"] and morphology_null["active"]
                else np.nan
            ),
            "morphology_relative_improvement": morphology_relative,
            "morphology_active": morphology["active"],
            "morphology_n_bins": morphology["n_bins"],
        }

    relative = np.asarray(relative_values, dtype=np.float64)
    finite_relative = relative[np.isfinite(relative)]
    return {
        "targets": target_metrics,
        "balanced_relative_improvement": (
            float(np.min(finite_relative)) if len(finite_relative) == len(relative) else np.nan
        ),
        "mean_relative_improvement": (
            float(np.mean(finite_relative)) if finite_relative.size else np.nan
        ),
        "all_four_improve": bool(
            finite_relative.size == 4 and np.all(finite_relative > 0)
        ),
        "all_four_material_5pct": bool(
            finite_relative.size == 4 and np.all(finite_relative >= 0.05)
        ),
    }


def assess_complexity(pair_result: dict, single_results: list[dict]) -> dict:
    """Apply a conservative non-likelihood complexity guard."""
    best_single_balanced = max(
        result["metrics"]["balanced_relative_improvement"] for result in single_results
    )
    best_single_mean = max(
        result["metrics"]["mean_relative_improvement"] for result in single_results
    )
    pair_metrics = pair_result["metrics"]
    return {
        "best_single_balanced_relative_improvement": best_single_balanced,
        "best_single_mean_relative_improvement": best_single_mean,
        "balanced_gain_over_best_single": (
            pair_metrics["balanced_relative_improvement"] - best_single_balanced
        ),
        "mean_gain_over_best_single": pair_metrics["mean_relative_improvement"]
        - best_single_mean,
        "passes_complexity_guard": bool(
            pair_metrics["all_four_material_5pct"]
            and pair_metrics["balanced_relative_improvement"]
            >= best_single_balanced + 0.02
        ),
    }


def write_checkpoint(
    path: Path,
    status: str,
    representatives: dict,
    single_results: list[dict],
    pairs: list[dict],
    n_stars: int,
) -> None:
    """Persist incremental progress so a long full-orbit run is auditable."""
    payload = {
        "status": status,
        "n_stars": n_stars,
        "representatives": representatives,
        "n_single_complete": len(single_results),
        "n_pairs_complete": len(pairs),
        "single_results": single_results,
        "pair_results": pairs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-fast",
        default="outputs/profile_grid/GD1/gd1_joint_profile_diagnostic.json",
    )
    parser.add_argument(
        "--fused-h5",
        default="data/processed/GD1_pwb18_desi_v3_fused.h5",
    )
    parser.add_argument(
        "--desi-member-h5",
        default="data/processed/GD1_desi_dr2_v3_member.h5",
    )
    parser.add_argument(
        "--out",
        default="outputs/profile_grid/GD1/gd1_two_perturbation_full_orbit_shortlist.json",
    )
    parser.add_argument("--n-stars", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-half-window", type=float, default=12.0)
    parser.add_argument("--phi1-bin-width", type=float, default=4.0)
    parser.add_argument("--phi2-bin-width", type=float, default=0.2)
    parser.add_argument("--phi2-max", type=float, default=3.0)
    parser.add_argument("--track-bin-width", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    checkpoint = out.with_name(f"{out.stem}.partial.json")
    joint = json.loads(Path(args.joint_fast).read_text(encoding="utf-8"))
    representatives = {
        str(target): select_target_representatives(joint["all_candidates"], target)
        for target in TARGETS
    }
    events = [
        event
        for target_events in representatives.values()
        for event in target_events
    ]
    desi = load_desi_member(Path(args.desi_member_h5))

    model = TimelineForwardModel(
        ForwardModelConfig(
            stream_name="GD1",
            config_path="config/streams.yaml",
            processed_h5_path=args.fused_h5,
            n_stars_sim=args.n_stars,
            base_seed=args.seed,
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

    single_results = []
    single_control_cache = {}
    for index, event in enumerate(events, start=1):
        encounter = to_encounter(event)
        stream = generate_perturbed_stream_evolved(
            stream_name="GD1",
            potential=model.potential,
            encounter=encounter,
            n_stars=args.n_stars,
            seed=args.seed,
            config_path=model.cfg.config_path,
            mws=model._mws,
        )
        control_key = float(event["t_since_gyr"])
        if control_key not in single_control_cache:
            single_control_cache[control_key] = generate_perturbed_stream_evolved(
                stream_name="GD1",
                potential=model.potential,
                encounter=encounter,
                n_stars=args.n_stars,
                seed=args.seed,
                config_path=model.cfg.config_path,
                mws=model._mws,
                apply_kick=False,
            )
        control = single_control_cache[control_key]
        single_results.append(
            {
                "event": event,
                "metrics": evaluate_stream(
                    model,
                    stream,
                    control,
                    desi,
                    args.score_half_window,
                    args.phi1_bin_width,
                    args.phi2_bin_width,
                    args.phi2_max,
                    args.track_bin_width,
                ),
            }
        )
        write_checkpoint(
            checkpoint,
            "single_events",
            representatives,
            single_results,
            [],
            args.n_stars,
        )
        print(f"single {index}/{len(events)}", flush=True)

    pairs = []
    pair_control_cache = {}
    canonical_events = representatives[str(TARGETS[0])]
    spur_events = representatives[str(TARGETS[1])]
    for index, (canonical, spur) in enumerate(
        ((canonical, spur) for canonical in canonical_events for spur in spur_events),
        start=1,
    ):
        encounters = [to_encounter(canonical), to_encounter(spur)]
        stream = generate_perturbed_stream_multi_evolved(
            stream_name="GD1",
            potential=model.potential,
            encounters=encounters,
            n_stars=args.n_stars,
            seed=args.seed,
            config_path=model.cfg.config_path,
            mws=model._mws,
        )
        control_key = tuple(
            sorted(
                (float(encounter.t_since_impact_gyr) for encounter in encounters),
                reverse=True,
            )
        )
        if control_key not in pair_control_cache:
            pair_control_cache[control_key] = generate_perturbed_stream_multi_evolved(
                stream_name="GD1",
                potential=model.potential,
                encounters=encounters,
                n_stars=args.n_stars,
                seed=args.seed,
                config_path=model.cfg.config_path,
                mws=model._mws,
                apply_kicks=False,
            )
        control = pair_control_cache[control_key]
        pair = {
            "canonical_event": canonical,
            "spur_event": spur,
            "metrics": evaluate_stream(
                model,
                stream,
                control,
                desi,
                args.score_half_window,
                args.phi1_bin_width,
                args.phi2_bin_width,
                args.phi2_max,
                args.track_bin_width,
            ),
        }
        pair["complexity_guard"] = assess_complexity(pair, single_results)
        pairs.append(pair)
        write_checkpoint(
            checkpoint,
            "pair_events",
            representatives,
            single_results,
            pairs,
            args.n_stars,
        )
        print(
            f"pair {index}/{len(canonical_events) * len(spur_events)}",
            flush=True,
        )

    pairs.sort(
        key=lambda pair: pair["metrics"]["balanced_relative_improvement"],
        reverse=True,
    )
    best = pairs[0]
    payload = {
        "result_label": "Observed/Synthetic Full-Orbit Diagnostic",
        "scientific_status": (
            "Constrained 3x3 two-perturbation full-orbit shortlist with matched "
            "no-kick controls. Candidate events originate from a fast diagnostic; "
            "scores are not calibrated likelihoods or a density-profile posterior."
        ),
        "inputs": {
            "joint_fast": args.joint_fast,
            "fused_h5": args.fused_h5,
            "desi_member_h5": args.desi_member_h5,
        },
        "settings": {
            "n_stars": args.n_stars,
            "seed": args.seed,
            "score_half_window_deg": args.score_half_window,
        },
        "complexity_policy": (
            "A two-event candidate passes only if all four target-specific "
            "diagnostics improve by at least 5% and its weakest improvement is "
            "at least 2 percentage points better than the best single event."
        ),
        "representatives": representatives,
        "n_single_events": len(single_results),
        "n_pairs": len(pairs),
        "n_pairs_all_four_improve": sum(
            pair["metrics"]["all_four_improve"] for pair in pairs
        ),
        "n_pairs_all_four_material_5pct": sum(
            pair["metrics"]["all_four_material_5pct"] for pair in pairs
        ),
        "n_pairs_pass_complexity_guard": sum(
            pair["complexity_guard"]["passes_complexity_guard"] for pair in pairs
        ),
        "best_pair": best,
        "interpretation": (
            "At least one two-perturbation candidate passes the predeclared "
            "complexity guard; it is eligible for a finer full-orbit study."
            if best["complexity_guard"]["passes_complexity_guard"]
            else "No two-perturbation candidate passes the predeclared complexity "
            "guard. This shortlist does not justify adding a second subhalo encounter."
        ),
        "single_results": single_results,
        "pair_results": pairs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    print(payload["interpretation"])
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
