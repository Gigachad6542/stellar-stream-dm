#!/usr/bin/env python
"""Consolidate fast and full-orbit GD-1 Pareto-shortlist diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = (
    "impact_phi1",
    "scale_radius_factor",
    "log10_mass",
    "t_since_gyr",
    "impact_param_kpc",
    "flyby_vel_kms",
)


def key(row: dict) -> tuple:
    return tuple(float(row[name]) for name in KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast-joint",
        default="outputs/profile_grid/GD1/gd1_joint_profile_diagnostic.json",
    )
    parser.add_argument(
        "--full-primary",
        nargs="+",
        default=[
            "outputs/profile_grid/GD1/density_profile_grid_full_orbit_pareto_primary.json",
            "outputs/profile_grid/GD1/density_profile_grid_full_orbit_pareto_morphology.json",
            "outputs/profile_grid/GD1/density_profile_grid_full_orbit_pareto_balanced.json",
        ],
    )
    parser.add_argument(
        "--full-morphology",
        default="outputs/profile_grid/GD1/gd1_cross_track_morphology_full_orbit_shortlist.json",
    )
    parser.add_argument(
        "--out",
        default="outputs/profile_grid/GD1/gd1_full_orbit_pareto_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fast = json.loads(Path(args.fast_joint).read_text(encoding="utf-8"))
    roles = {
        key(fast["best_primary_candidate"]): "best_fast_primary",
        key(fast["best_morphology_candidate"]): "best_fast_morphology",
        key(fast["best_balanced_candidate"]): "best_fast_balanced",
    }
    fast_rows = {key(row): row for row in fast["all_candidates"]}
    full_morphology = json.loads(Path(args.full_morphology).read_text(encoding="utf-8"))
    morphology_rows = {key(row): row for row in full_morphology["all_candidates"]}

    candidates = []
    for path_string in args.full_primary:
        path = Path(path_string)
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = payload["selection_results"][0]["best_candidate"]
        candidate_key = key(row)
        fast_row = fast_rows[candidate_key]
        morphology_row = morphology_rows[candidate_key]
        primary_relative = float(row["delta_vs_null"]) / max(
            abs(float(row["local_null_score"])), 1e-12
        )
        morphology_relative = float(morphology_row["delta_vs_null"]) / max(
            abs(float(morphology_row["null_score"])), 1e-12
        )
        candidates.append(
            {
                "role": roles[candidate_key],
                **{name: row[name] for name in KEYS},
                "target_label": row["target_label"],
                "compactness_family": row["compactness_family"],
                "fast_primary_relative_improvement": fast_row[
                    "primary_relative_improvement"
                ],
                "fast_morphology_relative_improvement": fast_row[
                    "morphology_relative_improvement"
                ],
                "full_primary_delta_vs_null": row["delta_vs_null"],
                "full_primary_relative_improvement": primary_relative,
                "full_morphology_delta_vs_null": morphology_row["delta_vs_null"],
                "full_morphology_relative_improvement": morphology_relative,
                "full_primary_improves_null": primary_relative > 0,
                "full_morphology_improves_null": morphology_relative > 0,
                "full_improves_both": primary_relative > 0 and morphology_relative > 0,
                "full_material_both_5pct": primary_relative >= 0.05
                and morphology_relative >= 0.05,
                "full_primary_artifact": str(path),
            }
        )

    material = any(row["full_material_both_5pct"] for row in candidates)
    payload = {
        "result_label": "Observed/Synthetic Full-Orbit Diagnostic",
        "scientific_status": (
            "Three-candidate full-orbit Pareto-shortlist check only. This rejects "
            "the fast-impulse shortlist but is not an exhaustive full-orbit grid "
            "or calibrated density-profile posterior."
        ),
        "fast_joint_input": args.fast_joint,
        "full_morphology_input": args.full_morphology,
        "material_joint_improvement_threshold": 0.05,
        "has_material_full_orbit_joint_improvement": material,
        "interpretation": (
            "At least one shortlisted full-orbit encounter materially improves "
            "both observables."
            if material
            else "None of the three fast-impulse Pareto representatives materially "
            "improves both observables after full orbit evolution. The fast-mode "
            "compactness preference does not survive this physical check."
        ),
        "candidates": candidates,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(payload["interpretation"])
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
