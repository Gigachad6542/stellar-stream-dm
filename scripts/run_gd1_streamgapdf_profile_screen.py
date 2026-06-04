#!/usr/bin/env python
"""Screen continuous perturber profiles with the matched two-arm streamgapdf model.

The smooth null and localized-impact candidate share the same action-angle
stream model. The non-impacted arm remains smooth while streamgapdf perturbs
the selected arm. This first-stage screen maps streamgapdf impact angle onto
the observed GD-1 coordinate and tests continuous Plummer scale radius.

Results are diagnostic scores, not a calibrated likelihood or posterior.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_training_data import _fix_galpy_dll_path

_fix_galpy_dll_path()

import numpy as np

from scripts.run_gd1_two_perturbation_shortlist import (
    evaluate_stream,
    load_desi_member,
)
from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
from src.forward_model.scoring import ScoreWeights, compute_density_profile
from src.simulation.stream_gen import generate_stream_df_two_arm
from src.simulation.subhalo import scale_radius_from_mass


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


def candidate_grid(args: argparse.Namespace) -> list[dict]:
    output = []
    for log_mass, time_gyr, angle_rad, factor in itertools.product(
        args.log10_masses,
        args.times,
        args.impact_angles,
        args.scale_factors,
    ):
        mass = 10.0 ** float(log_mass)
        output.append(
            {
                "log10_mass": float(log_mass),
                "mass": mass,
                "timpact_gyr": float(time_gyr),
                "impact_angle_rad": float(angle_rad),
                "impactb_kpc": float(args.impact_param),
                "vsub_kms": float(args.flyby_velocity),
                "scale_radius_factor": float(factor),
                "scale_radius_kpc": scale_radius_from_mass(mass) * float(factor),
            }
        )
    return output


def strongest_density_deficit(
    candidate_phi1: np.ndarray,
    control_phi1: np.ndarray,
    phi1_range: tuple[float, float],
    bin_width_deg: float,
) -> dict:
    candidate = compute_density_profile(
        candidate_phi1,
        phi1_range,
        bin_width_deg=bin_width_deg,
    )
    control = compute_density_profile(
        control_phi1,
        phi1_range,
        bin_width_deg=bin_width_deg,
    )
    deficit = control.density - candidate.density
    index = int(np.argmax(deficit))
    return {
        "phi1": float(control.bin_centers[index]),
        "fractional_density_deficit": float(
            deficit[index] / max(control.density[index], 1e-12)
        ),
        "absolute_normalized_deficit": float(deficit[index]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fused-h5",
        default="data/processed/GD1_pwb18_desi_v3_fused.h5",
    )
    parser.add_argument(
        "--desi-member-h5",
        default="data/processed/GD1_desi_dr2_v3_member.h5",
    )
    parser.add_argument("--log10-masses", nargs="+", type=float, default=[8.0])
    parser.add_argument("--times", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument(
        "--impact-angles",
        nargs="+",
        type=float,
        default=[0.2, 0.35, 0.5, 0.65, 0.8],
        help="Absolute streamgapdf angle; trailing-arm sign is applied internally.",
    )
    parser.add_argument("--scale-factors", nargs="+", type=float, default=[1.0])
    parser.add_argument("--impact-param", type=float, default=0.05)
    parser.add_argument("--flyby-velocity", type=float, default=150.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--n-stars", type=int, default=1000)
    parser.add_argument("--score-half-window", type=float, default=12.0)
    parser.add_argument("--phi1-bin-width", type=float, default=4.0)
    parser.add_argument("--phi2-bin-width", type=float, default=0.2)
    parser.add_argument("--phi2-max", type=float, default=3.0)
    parser.add_argument("--track-bin-width", type=float, default=2.0)
    parser.add_argument("--deficit-bin-width", type=float, default=2.0)
    parser.add_argument(
        "--out",
        default="outputs/profile_grid/GD1/gd1_streamgapdf_profile_screen.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    grid = candidate_grid(args)
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

    controls = {
        seed: generate_stream_df_two_arm(
            "GD1",
            model.potential,
            n_stars=args.n_stars,
            seed=seed,
            impact=False,
            config_path=model.cfg.config_path,
            mws=model._mws,
        )
        for seed in args.seeds
    }
    rows = []
    total = len(grid) * len(args.seeds)
    completed = 0
    for params in grid:
        for seed in args.seeds:
            candidate = generate_stream_df_two_arm(
                "GD1",
                model.potential,
                n_stars=args.n_stars,
                seed=seed,
                impact=True,
                impact_arm="trailing",
                impact_params=params,
                config_path=model.cfg.config_path,
                mws=model._mws,
            )
            metrics = evaluate_stream(
                model,
                candidate,
                controls[seed],
                desi,
                args.score_half_window,
                args.phi1_bin_width,
                args.phi2_bin_width,
                args.phi2_max,
                args.track_bin_width,
            )
            rows.append(
                {
                    "seed": seed,
                    **params,
                    "realized_strongest_density_deficit": strongest_density_deficit(
                        candidate.phi1,
                        controls[seed].phi1,
                        (20.0, 70.0),
                        args.deficit_bin_width,
                    ),
                    "metrics": metrics,
                }
            )
            completed += 1
            print(f"{completed}/{total}", flush=True)

    rows.sort(
        key=lambda row: row["metrics"]["balanced_relative_improvement"],
        reverse=True,
    )
    payload = {
        "result_label": "Observed/Synthetic Matched streamgapdf Profile Screen",
        "scientific_status": (
            "Matched two-arm action-angle diagnostic with a smooth streamdf null "
            "and a localized trailing-arm streamgapdf impact. Not a calibrated "
            "likelihood or density-profile posterior."
        ),
        "inputs": vars(args),
        "n_candidates": len(rows),
        "n_all_four_improve": sum(
            row["metrics"]["all_four_improve"] for row in rows
        ),
        "n_all_four_material_5pct": sum(
            row["metrics"]["all_four_material_5pct"] for row in rows
        ),
        "best_candidate": rows[0],
        "all_candidates": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
    print(
        f"{payload['n_all_four_improve']}/{len(rows)} candidates improve all four "
        "diagnostics."
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
