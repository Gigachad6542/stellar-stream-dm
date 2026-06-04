#!/usr/bin/env python
"""Screen encounter candidates against DESI conditional cross-stream morphology.

This is deliberately separate from the primary PWB18 along-stream density
score. DESI spectroscopy is non-uniformly targeted along GD-1, so this script
compares only p(delta_phi2 | phi1), normalizing every populated phi1 bin
independently. The output is a diagnostic, not a calibrated likelihood or a
density-profile constraint.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forward_model.morphology import (
    component_morphology_summary,
    conditional_cross_track_js_score,
    reference_track_residuals,
)
from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
from src.forward_model.evolve import generate_perturbed_stream_evolved
from src.simulation.subhalo import EncounterParams, apply_impulse_approximation


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def load_desi_member(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["streams/GD1/members"]
        required = ("phi1", "delta_phi2", "p_thin", "p_cocoon")
        missing = [name for name in required if name not in group]
        if missing:
            raise KeyError(f"DESI member catalog is missing {missing}")
        return {name: group[name][:].astype(np.float64) for name in required}


def summarize(rows: list[dict], key: str) -> list[dict]:
    groups: dict[Any, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = []
    for value, group in groups.items():
        active = [row for row in group if row["active"]]
        best = min(active, key=lambda row: row["score"]) if active else None
        output.append(
            {
                key: value,
                "n_candidates": len(group),
                "n_active": len(active),
                "best_candidate": best,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["best_candidate"] is None,
            row["best_candidate"]["score"] if row["best_candidate"] else np.inf,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-json",
        default="outputs/profile_grid/GD1/density_profile_grid_fused_rv02_errorweighted_screening.json",
    )
    parser.add_argument(
        "--desi-member-h5",
        default="data/processed/GD1_desi_dr2_v3_member.h5",
    )
    parser.add_argument(
        "--simulation-reference-h5",
        default="data/processed/GD1_pwb18_desi_v3_fused.h5",
    )
    parser.add_argument(
        "--out",
        default=None,
    )
    parser.add_argument("--full", action="store_true", help="Orbit-evolve a small shortlist")
    parser.add_argument(
        "--shortlist-json",
        default="outputs/profile_grid/GD1/gd1_joint_profile_diagnostic.json",
        help="Pareto diagnostic containing the three named shortlist candidates",
    )
    parser.add_argument(
        "--shortlist-fields",
        nargs="+",
        default=[
            "best_primary_candidate",
            "best_morphology_candidate",
            "best_balanced_candidate",
        ],
    )
    parser.add_argument("--phi1-bin-width", type=float, default=4.0)
    parser.add_argument("--phi2-bin-width", type=float, default=0.2)
    parser.add_argument("--phi2-max", type=float, default=3.0)
    parser.add_argument("--track-bin-width", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid_payload = json.loads(Path(args.grid_json).read_text(encoding="utf-8"))
    selection = grid_payload["selection_results"][0]
    candidates = selection["all_candidates"]
    if args.full:
        shortlist = json.loads(Path(args.shortlist_json).read_text(encoding="utf-8"))
        candidates = [shortlist[field] for field in args.shortlist_fields]
    targets = sorted({float(row["impact_phi1"]) for row in candidates})
    half_width = float(grid_payload["grid"]["local_score_half_window_deg"])

    model = TimelineForwardModel(
        ForwardModelConfig(
            stream_name="GD1",
            config_path="config/streams.yaml",
            processed_h5_path=args.simulation_reference_h5,
            n_stars_sim=int(grid_payload["grid"]["n_stars_sim"]),
            base_seed=int(grid_payload["grid"]["base_seed"]),
            phi2_cut_deg=None,
            apply_pm_cuts=False,
            use_gnn_scorer=False,
            use_fast_mode=not args.full,
            n_workers=1,
        )
    )
    model.prepare()
    desi = load_desi_member(Path(args.desi_member_h5))
    p_member = np.clip(desi["p_thin"] + desi["p_cocoon"], 0.0, 1.0)

    target_observations = {}
    null_scores = {}
    base_delta = reference_track_residuals(
        model.base_stream.phi1,
        model.base_stream.phi2,
        model.base_stream.phi1,
        model.base_stream.phi2,
        bin_width_deg=args.track_bin_width,
    )
    for target in targets:
        score_range = (target - half_width, target + half_width)
        obs_mask = (
            (desi["phi1"] >= score_range[0])
            & (desi["phi1"] <= score_range[1])
            & np.isfinite(desi["delta_phi2"])
        )
        target_observations[str(target)] = {
            "score_range": list(score_range),
            "n_observed": int(obs_mask.sum()),
            "component_summary": component_morphology_summary(
                desi["delta_phi2"][obs_mask],
                desi["p_thin"][obs_mask],
                desi["p_cocoon"][obs_mask],
            ),
        }
        null_scores[target] = conditional_cross_track_js_score(
            model.base_stream.phi1,
            base_delta,
            desi["phi1"],
            desi["delta_phi2"],
            score_range,
            obs_weight=p_member,
            phi1_bin_width_deg=args.phi1_bin_width,
            phi2_range=(-args.phi2_max, args.phi2_max),
            phi2_bin_width_deg=args.phi2_bin_width,
        )

    rows = []
    matched_control_scores = {}
    for index, candidate in enumerate(candidates, start=1):
        encounter = EncounterParams(
            mass_solar=10.0 ** float(candidate["log10_mass"]),
            scale_radius_kpc=float(candidate["scale_radius_kpc"]),
            impact_param_kpc=float(candidate["impact_param_kpc"]),
            flyby_vel_kms=float(candidate["flyby_vel_kms"]),
            encounter_phi1=float(candidate["impact_phi1"]),
            t_since_impact_gyr=float(candidate["t_since_gyr"]),
            is_valid=True,
        )
        if args.full:
            perturbed = generate_perturbed_stream_evolved(
                stream_name="GD1",
                potential=model.potential,
                encounter=encounter,
                n_stars=model.cfg.n_stars_sim,
                seed=model.cfg.base_seed,
                config_path=model.cfg.config_path,
                mws=model._mws,
            )
            control = generate_perturbed_stream_evolved(
                stream_name="GD1",
                potential=model.potential,
                encounter=encounter,
                n_stars=model.cfg.n_stars_sim,
                seed=model.cfg.base_seed,
                config_path=model.cfg.config_path,
                mws=model._mws,
                apply_kick=False,
            )
            delta_phi2 = reference_track_residuals(
                perturbed.phi1,
                perturbed.phi2,
                control.phi1,
                control.phi2,
                bin_width_deg=args.track_bin_width,
            )
            control_delta_phi2 = reference_track_residuals(
                control.phi1,
                control.phi2,
                control.phi1,
                control.phi2,
                bin_width_deg=args.track_bin_width,
            )
        else:
            perturbed = apply_impulse_approximation(model.base_stream, encounter)
            delta_phi2 = reference_track_residuals(
                perturbed.phi1,
                perturbed.phi2,
                model.base_stream.phi1,
                model.base_stream.phi2,
                bin_width_deg=args.track_bin_width,
            )
        target = float(candidate["impact_phi1"])
        score_range = (target - half_width, target + half_width)
        score = conditional_cross_track_js_score(
            perturbed.phi1,
            delta_phi2,
            desi["phi1"],
            desi["delta_phi2"],
            score_range,
            obs_weight=p_member,
            phi1_bin_width_deg=args.phi1_bin_width,
            phi2_range=(-args.phi2_max, args.phi2_max),
            phi2_bin_width_deg=args.phi2_bin_width,
        )
        if args.full:
            control_score = conditional_cross_track_js_score(
                control.phi1,
                control_delta_phi2,
                desi["phi1"],
                desi["delta_phi2"],
                score_range,
                obs_weight=p_member,
                phi1_bin_width_deg=args.phi1_bin_width,
                phi2_range=(-args.phi2_max, args.phi2_max),
                phi2_bin_width_deg=args.phi2_bin_width,
            )
            null_score = control_score["score"]
            matched_control_scores[
                f"phi1={target:g},t={float(candidate['t_since_gyr']):g}"
            ] = control_score
        else:
            null_score = null_scores[target]["score"]
        rows.append(
            {
                "target_label": candidate["target_label"],
                "impact_phi1": target,
                "compactness_family": candidate["compactness_family"],
                "scale_radius_factor": float(candidate["scale_radius_factor"]),
                "scale_radius_kpc": float(candidate["scale_radius_kpc"]),
                "log10_mass": float(candidate["log10_mass"]),
                "t_since_gyr": float(candidate["t_since_gyr"]),
                "impact_param_kpc": float(candidate["impact_param_kpc"]),
                "flyby_vel_kms": float(candidate["flyby_vel_kms"]),
                "score": score["score"],
                "active": bool(score["active"]),
                "n_conditional_bins": int(score["n_bins"]),
                "null_score": null_score,
                "delta_vs_null": (
                    float(null_score - score["score"])
                    if score["active"] and np.isfinite(null_score)
                    else np.nan
                ),
                "primary_density_rv_delta_vs_null": float(
                    candidate["delta_vs_null"]
                    if "delta_vs_null" in candidate
                    else candidate["primary_delta_vs_null"]
                ),
            }
        )
        if index % 500 == 0:
            print(f"evaluated {index}/{len(candidates)}")

    payload = {
        "result_label": "Observed/Synthetic Diagnostic",
        "scientific_status": (
            "Conditional cross-stream morphology screening only. DESI targeting "
            "is normalized out along phi1 but has not been proven independent of "
            "cross-stream position. Component probabilities are model-derived. "
            "This is not a calibrated likelihood or density-profile constraint."
        ),
        "mode": "full-orbit-shortlist" if args.full else "fast-impulse-grid",
        "input_grid": args.grid_json,
        "shortlist_input": args.shortlist_json if args.full else None,
        "desi_member_catalog": args.desi_member_h5,
        "simulation_reference_catalog": args.simulation_reference_h5,
        "settings": {
            "phi1_bin_width_deg": args.phi1_bin_width,
            "phi2_bin_width_deg": args.phi2_bin_width,
            "phi2_range_deg": [-args.phi2_max, args.phi2_max],
            "track_bin_width_deg": args.track_bin_width,
            "score_half_window_deg": half_width,
        },
        "target_observations": target_observations,
        "null_scores": {str(key): value for key, value in null_scores.items()},
        "null_policy": (
            "Candidate-specific matched no-kick full-orbit control; candidate "
            "cross-track residuals are measured relative to that control track."
            if args.full
            else "Ordinary unperturbed base stream."
        ),
        "matched_full_orbit_control_scores": matched_control_scores,
        "best_overall": min(
            (row for row in rows if row["active"]),
            key=lambda row: row["score"],
        ),
        "by_impact_phi1": summarize(rows, "impact_phi1"),
        "by_scale_radius_factor": summarize(rows, "scale_radius_factor"),
        "all_candidates": rows,
    }
    out = Path(
        args.out
        or (
            "outputs/profile_grid/GD1/gd1_cross_track_morphology_full_orbit_shortlist.json"
            if args.full
            else "outputs/profile_grid/GD1/gd1_cross_track_morphology_screening.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
