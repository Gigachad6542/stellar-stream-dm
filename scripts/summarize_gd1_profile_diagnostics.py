#!/usr/bin/env python
"""Join GD-1 density+RV and cross-stream morphology screening diagnostics.

The two screens use different observables and should not be combined with an
arbitrary weighted sum. This script instead reports relative null improvement,
the Pareto front, and the candidate that maximizes the weaker of the two
relative improvements. It is a shortlist diagnostic, not a likelihood.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


KEYS = (
    "impact_phi1",
    "scale_radius_factor",
    "log10_mass",
    "t_since_gyr",
    "impact_param_kpc",
    "flyby_vel_kms",
)


def candidate_key(row: dict) -> tuple:
    return tuple(float(row[key]) for key in KEYS)


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


def pareto_front(rows: list[dict]) -> list[dict]:
    """Return rows not dominated in both relative-improvement coordinates."""
    ordered = sorted(
        rows,
        key=lambda row: (
            row["primary_relative_improvement"],
            row["morphology_relative_improvement"],
        ),
        reverse=True,
    )
    front = []
    best_morphology = -np.inf
    for row in ordered:
        morphology = row["morphology_relative_improvement"]
        if morphology > best_morphology:
            front.append(row)
            best_morphology = morphology
    return front


def summarize_by(rows: list[dict], key: str) -> list[dict]:
    groups: dict[Any, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = []
    for value, group in groups.items():
        best = max(group, key=lambda row: row["balanced_relative_improvement"])
        output.append(
            {
                key: value,
                "n_candidates": len(group),
                "n_improve_both": sum(
                    row["primary_relative_improvement"] > 0
                    and row["morphology_relative_improvement"] > 0
                    for row in group
                ),
                "best_balanced_candidate": best,
            }
        )
    return sorted(
        output,
        key=lambda row: row["best_balanced_candidate"]["balanced_relative_improvement"],
        reverse=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        default="outputs/profile_grid/GD1/density_profile_grid_fused_rv02_errorweighted_screening.json",
    )
    parser.add_argument(
        "--morphology",
        default="outputs/profile_grid/GD1/gd1_cross_track_morphology_screening.json",
    )
    parser.add_argument(
        "--out",
        default="outputs/profile_grid/GD1/gd1_joint_profile_diagnostic.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    primary = json.loads(Path(args.primary).read_text(encoding="utf-8"))
    morphology = json.loads(Path(args.morphology).read_text(encoding="utf-8"))
    primary_rows = primary["selection_results"][0]["all_candidates"]
    morphology_rows = morphology["all_candidates"]
    morphology_by_key = {candidate_key(row): row for row in morphology_rows}

    rows = []
    n_excluded_inactive = 0
    for primary_row in primary_rows:
        key = candidate_key(primary_row)
        if key not in morphology_by_key:
            raise KeyError(f"Morphology screen is missing candidate {key}")
        morphology_row = morphology_by_key[key]
        if (
            not morphology_row.get("active", False)
            or morphology_row.get("delta_vs_null") is None
        ):
            n_excluded_inactive += 1
            continue
        primary_null = float(primary_row["local_null_score"])
        morphology_null = float(morphology_row["null_score"])
        primary_relative = float(primary_row["delta_vs_null"]) / max(abs(primary_null), 1e-12)
        morphology_relative = float(morphology_row["delta_vs_null"]) / max(
            abs(morphology_null), 1e-12
        )
        rows.append(
            {
                **{name: primary_row[name] for name in KEYS},
                "target_label": primary_row["target_label"],
                "compactness_family": primary_row["compactness_family"],
                "scale_radius_kpc": primary_row["scale_radius_kpc"],
                "primary_delta_vs_null": primary_row["delta_vs_null"],
                "primary_relative_improvement": primary_relative,
                "morphology_delta_vs_null": morphology_row["delta_vs_null"],
                "morphology_relative_improvement": morphology_relative,
                "balanced_relative_improvement": min(primary_relative, morphology_relative),
                "mean_relative_improvement": 0.5 * (primary_relative + morphology_relative),
                "improves_both": primary_relative > 0 and morphology_relative > 0,
            }
        )

    best_primary = max(rows, key=lambda row: row["primary_relative_improvement"])
    best_morphology = max(rows, key=lambda row: row["morphology_relative_improvement"])
    best_balanced = max(rows, key=lambda row: row["balanced_relative_improvement"])
    front = pareto_front(rows)
    material_joint_improvement = best_balanced["balanced_relative_improvement"] >= 0.05
    payload = {
        "result_label": "Observed/Synthetic Diagnostic",
        "scientific_status": (
            "Pareto shortlist diagnostic only. The density+RV and conditional "
            "cross-stream morphology screens are not calibrated likelihoods and "
            "are not combined with an arbitrary weighted score."
        ),
        "primary_input": args.primary,
        "morphology_input": args.morphology,
        "n_candidates": len(rows),
        "n_excluded_inactive_morphology": n_excluded_inactive,
        "n_improve_both": sum(row["improves_both"] for row in rows),
        "material_joint_improvement_threshold": 0.05,
        "has_material_joint_improvement": material_joint_improvement,
        "interpretation": (
            "At least one candidate improves both diagnostics by at least 5% "
            "relative to their local nulls; prioritize it for full-orbit checks."
            if material_joint_improvement
            else "No candidate improves both diagnostics by at least 5% relative "
            "to their local nulls. A single fast-impulse encounter does not "
            "currently explain both the along-stream density+RV and conditional "
            "cross-stream morphology."
        ),
        "best_primary_candidate": best_primary,
        "best_morphology_candidate": best_morphology,
        "best_balanced_candidate": best_balanced,
        "pareto_front": front,
        "by_impact_phi1": summarize_by(rows, "impact_phi1"),
        "by_scale_radius_factor": summarize_by(rows, "scale_radius_factor"),
        "all_candidates": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
    print(payload["interpretation"])
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
