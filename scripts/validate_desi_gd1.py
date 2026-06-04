#!/usr/bin/env python
"""Validate DESI DR2 GD-1 v3 ingestion before profile inference."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forward_model.scoring import compute_density_profile, detect_gaps, find_density_minima


TARGETS = {
    "canonical_gap_1": 30.7,
    "spur_region": 50.7,
}


def finite(value):
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return finite(value.item())
    return value


def load_catalog(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        group = handle["streams/GD1"]
        return {
            "path": str(path),
            "attrs": dict(group.attrs),
            "members": {name: group["members"][name][:] for name in group["members"]},
        }


def feature_dict(feature, include_significance: bool = True) -> dict:
    return {
        "phi1_center": feature.phi1_center,
        "phi1_width": feature.phi1_width,
        "depth": feature.depth,
        "significance": feature.significance if include_significance else None,
    }


def nearest(features: list[dict], target: float) -> dict | None:
    if not features:
        return None
    return min(features, key=lambda feature: abs(feature["phi1_center"] - target))


def density_diagnostics(members: dict) -> dict:
    membership = np.asarray(members["membership_prob"], dtype=float)
    selected = membership >= 0.5
    phi1 = np.asarray(members["phi1"], dtype=float)[selected]
    thin_weight = np.asarray(members["p_thin"], dtype=float)[selected]
    extent = tuple(np.percentile(phi1, [2, 98]).astype(float))
    result = {}
    for label, weights in (
        ("thresholded_raw_counts", None),
        ("p_thin_weighted_shape", thin_weight),
    ):
        cases = []
        for bin_width in (1.0, 2.0, 4.0):
            profile = compute_density_profile(phi1, extent, bin_width, weights=weights)
            gaps = []
            if weights is None:
                gaps = [
                    feature_dict(feature)
                    for feature in detect_gaps(profile, min_depth=0.2, min_significance=1.5)
                ]
            minima = [
                feature_dict(feature, include_significance=weights is None)
                for feature in find_density_minima(profile, top_k=8)
            ]
            cases.append(
                {
                    "bin_width_deg": bin_width,
                    "profile_total": float(profile.counts.sum()),
                    "gap_detection_status": (
                        "raw-count Poisson approximation"
                        if weights is None
                        else "not run on fractional membership weights"
                    ),
                    "detected_gaps": gaps,
                    "prominent_minima": minima,
                    "targets": {
                        name: {
                            "nearest_detected_gap": nearest(gaps, target),
                            "nearest_prominent_minimum": nearest(minima, target),
                        }
                        for name, target in TARGETS.items()
                    },
                }
            )
        result[label] = cases
    return result


def target_kinematics(members: dict) -> dict:
    result = {}
    phi1 = np.asarray(members["phi1"], dtype=float)
    membership = np.asarray(members["membership_prob"], dtype=float)
    for name, target in TARGETS.items():
        mask = (membership >= 0.5) & (np.abs(phi1 - target) <= 3.0)
        result[name] = {
            "n_selected": int(np.sum(mask)),
            "thin_probability_sum": float(np.sum(members["p_thin"][mask])),
            "cocoon_probability_sum": float(np.sum(members["p_cocoon"][mask])),
            "n_finite_vlos": int(np.sum(np.isfinite(members["vrad"][mask]))),
            "median_delta_vgsr_kms": float(np.nanmedian(members["delta_vgsr"][mask])),
            "std_delta_vgsr_kms": float(np.nanstd(members["delta_vgsr"][mask])),
            "median_delta_phi2_deg": float(np.nanmedian(members["delta_phi2"][mask])),
        }
    return result


def validate(thin_path: Path, member_path: Path) -> dict:
    thin = load_catalog(thin_path)
    member = load_catalog(member_path)
    tm = thin["members"]
    mm = member["members"]
    p_sum = np.clip(mm["p_thin"] + mm["p_cocoon"], 0.0, 1.0)
    x = np.asarray(mm["phi1_desi"], dtype=float)
    y = np.asarray(mm["phi1"], dtype=float)
    coefficients = np.polyfit(x, y, deg=2)
    residual = y - np.polyval(coefficients, x)
    checks = {
        "same_source_ids": bool(np.array_equal(tm["source_id"], mm["source_id"])),
        "all_vlos_finite": bool(np.all(np.isfinite(mm["vrad"]))),
        "all_vlos_errors_positive": bool(np.all(mm["e_vrad"] > 0)),
        "probability_sum_bounded": bool(np.all(mm["p_thin"] + mm["p_cocoon"] <= 1.00001)),
        "thin_membership_matches_p_thin": bool(
            np.allclose(tm["membership_prob"], tm["p_thin"])
        ),
        "member_membership_matches_sum": bool(
            np.allclose(mm["membership_prob"], p_sum)
        ),
    }
    return {
        "result_label": "Observed Catalog Validation",
        "scientific_status": (
            "DESI v3 catalog/probability/kinematic validation. Passing checks permits "
            "diagnostic profile screening, not a calibrated perturber constraint."
        ),
        "thin_catalog": {
            "path": str(thin_path),
            "n_rows": len(tm["phi1"]),
            "n_membership_ge_0_5": int(np.sum(tm["membership_prob"] >= 0.5)),
            "membership_weight_sum": float(np.sum(tm["membership_prob"])),
            "attrs": thin["attrs"],
        },
        "member_catalog": {
            "path": str(member_path),
            "n_rows": len(mm["phi1"]),
            "n_membership_ge_0_5": int(np.sum(mm["membership_prob"] >= 0.5)),
            "membership_weight_sum": float(np.sum(mm["membership_prob"])),
            "attrs": member["attrs"],
        },
        "coordinate_crosscheck": {
            "quadratic_desi_to_i21_coefficients": coefficients,
            "rms_residual_deg": float(np.sqrt(np.mean(residual**2))),
            "max_abs_residual_deg": float(np.max(np.abs(residual))),
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "target_kinematics_member_selection": target_kinematics(mm),
        "thin_density_profile_diagnostics": density_diagnostics(tm),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thin", default="data/processed/GD1_desi_dr2_v3_thin.h5")
    parser.add_argument("--member", default="data/processed/GD1_desi_dr2_v3_member.h5")
    parser.add_argument(
        "--out", default="outputs/diagnostics/desi_gd1_v3_catalog_validation.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = validate(Path(args.thin), Path(args.member))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
    print(json.dumps(finite({
        "thin_catalog": payload["thin_catalog"],
        "member_catalog": payload["member_catalog"],
        "coordinate_crosscheck": payload["coordinate_crosscheck"],
        "checks": payload["checks"],
        "all_checks_pass": payload["all_checks_pass"],
        "target_kinematics_member_selection": payload["target_kinematics_member_selection"],
    }), indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
