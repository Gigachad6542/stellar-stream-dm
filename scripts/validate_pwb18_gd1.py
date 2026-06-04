#!/usr/bin/env python
"""Validate the mask-aware PWB18 GD-1 ingestion before physical inference."""
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
    "canonical_gap_1": {"phi1_pwb18": -40.0, "phi1_i21": 30.7},
    "spur_region": {"phi1_pwb18": -20.0, "phi1_i21": 50.7},
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


def feature_dict(feature, include_significance: bool = True) -> dict:
    return {
        "phi1_center": feature.phi1_center,
        "phi1_width": feature.phi1_width,
        "depth": feature.depth,
        "significance": feature.significance if include_significance else None,
    }


def load_catalog(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        group = handle["streams/GD1"]
        members = group["members"]
        result = {
            "path": str(path),
            "attrs": {key: value for key, value in group.attrs.items()},
            "members": {key: members[key][:] for key in members.keys()},
        }
        if "selection_profile" in group:
            profile = group["selection_profile"]
            result["selection_profile"] = {
                "attrs": {key: value for key, value in profile.attrs.items()},
                **{key: profile[key][:] for key in profile.keys()},
            }
    return result


def nearest_feature(features: list[dict], target: float) -> dict | None:
    if not features:
        return None
    return min(features, key=lambda feature: abs(feature["phi1_center"] - target))


def profile_diagnostics(track: dict) -> dict:
    members = track["members"]
    phi1 = np.asarray(members["phi1"], dtype=float)
    extent = tuple(np.percentile(phi1, [2, 98]).astype(float))
    weight_cases = {"raw_track_counts": None}
    if "selection_weight" in members:
        weight_cases["selection_corrected_track_excess"] = np.asarray(
            members["selection_weight"], dtype=float
        )

    diagnostics = {}
    for label, weights in weight_cases.items():
        cases = []
        for bin_width in (0.5, 1.0, 2.0):
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
                    "weighted_total": float(profile.counts.sum()),
                    "gap_detection_status": (
                        "raw-count Poisson approximation"
                        if weights is None
                        else "not run on fractional selection weights"
                    ),
                    "detected_gaps": gaps,
                    "prominent_minima": minima,
                    "targets": {
                        name: {
                            "nearest_detected_gap": nearest_feature(gaps, target["phi1_i21"]),
                            "nearest_prominent_minimum": nearest_feature(
                                minima, target["phi1_i21"]
                            ),
                        }
                        for name, target in TARGETS.items()
                    },
                }
            )
        diagnostics[label] = cases
    return diagnostics


def selection_profile_diagnostics(track: dict) -> dict:
    profile = track.get("selection_profile")
    if not profile:
        return {"available": False}
    centers = np.asarray(profile["bin_centers_pwb18"], dtype=float)
    result = {
        "available": True,
        "attrs": profile["attrs"],
        "total_pmcmd_track": float(np.sum(profile["pmcmd_track_counts"])),
        "total_expected_background_track": float(
            np.sum(profile["expected_background_track"])
        ),
        "total_main_track_excess": float(np.sum(profile["main_track_excess"])),
        "targets": {},
    }
    for name, target in TARGETS.items():
        index = int(np.argmin(np.abs(centers - target["phi1_pwb18"])))
        result["targets"][name] = {
            "bin_center_pwb18": centers[index],
            "pmcmd_track_count": profile["pmcmd_track_counts"][index],
            "expected_background_track": profile["expected_background_track"][index],
            "main_track_excess": profile["main_track_excess"][index],
            "main_track_excess_significance": profile[
                "main_track_excess_significance"
            ][index],
            "main_track_signal_fraction": profile["main_track_signal_fraction"][index],
        }
    return result


def fusion_diagnostics(track: dict) -> dict:
    members = track["members"]
    if "desi_match" not in members:
        return {"available": False}
    matched = np.asarray(members["desi_match"], dtype=bool)
    phi1_pwb18 = np.asarray(members["phi1_pwb18"], dtype=float)
    result = {
        "available": True,
        "n_desi_matches": int(np.sum(matched)),
        "n_finite_rv": int(np.sum(np.isfinite(members["vrad"]))),
        "targets": {},
    }
    for name, target in TARGETS.items():
        mask = matched & (np.abs(phi1_pwb18 - target["phi1_pwb18"]) <= 3.0)
        result["targets"][name] = {
            "n_matches": int(np.sum(mask)),
            "n_finite_rv": int(np.sum(np.isfinite(members["vrad"][mask]))),
            "median_vlos_kms": float(np.nanmedian(members["vrad"][mask])),
            "median_vlos_error_kms": float(np.nanmedian(members["e_vrad"][mask])),
            "desi_thin_probability_sum": float(np.nansum(members["desi_p_thin"][mask])),
            "desi_cocoon_probability_sum": float(
                np.nansum(members["desi_p_cocoon"][mask])
            ),
        }
    return result


def validate(track_path: Path, pmcmd_path: Path) -> dict:
    track = load_catalog(track_path)
    pmcmd = load_catalog(pmcmd_path)
    track_members = track["members"]
    pmcmd_members = pmcmd["members"]

    x = np.asarray(track_members["phi1_pwb18"], dtype=float)
    y = np.asarray(track_members["phi1"], dtype=float)
    coefficients = np.polyfit(x, y, deg=2)
    residual = y - np.polyval(coefficients, x)

    track_source_ids = set(np.asarray(track_members["source_id"], dtype=np.int64).tolist())
    pmcmd_track_ids = set(
        np.asarray(pmcmd_members["source_id"], dtype=np.int64)[
            np.asarray(pmcmd_members["stream_track_mask"], dtype=bool)
        ].tolist()
    )
    checks = {
        "track_mask_all_true": bool(np.all(track_members["stream_track_mask"])),
        "track_ids_match_pmcmd_track_subset": track_source_ids == pmcmd_track_ids,
        "track_is_subset_of_pmcmd": track_source_ids.issubset(
            set(np.asarray(pmcmd_members["source_id"], dtype=np.int64).tolist())
        ),
        "selection_weights_finite": bool(
            np.all(np.isfinite(track_members.get("selection_weight", np.array([1.0]))))
        ),
    }

    return {
        "result_label": "Observed Catalog Validation",
        "scientific_status": (
            "Catalog/selection diagnostic only. Passing checks permits profile-grid "
            "screening but does not validate a dark-matter density-profile inference."
        ),
        "track_catalog": {
            "path": str(track_path),
            "n_rows": len(track_members["phi1"]),
            "attrs": track["attrs"],
            "phi1_i21_range": [float(np.min(y)), float(np.max(y))],
            "phi1_pwb18_range": [float(np.min(x)), float(np.max(x))],
            "selection_weight_sum": float(
                np.sum(track_members.get("selection_weight", np.ones(len(y))))
            ),
        },
        "pmcmd_catalog": {
            "path": str(pmcmd_path),
            "n_rows": len(pmcmd_members["phi1"]),
            "n_stream_track_mask": int(np.sum(pmcmd_members["stream_track_mask"])),
            "attrs": pmcmd["attrs"],
        },
        "coordinate_crosscheck": {
            "quadratic_pwb18_to_i21_coefficients": coefficients,
            "rms_residual_deg": float(np.sqrt(np.mean(residual**2))),
            "max_abs_residual_deg": float(np.max(np.abs(residual))),
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "selection_profile": selection_profile_diagnostics(track),
        "fusion_diagnostics": fusion_diagnostics(track),
        "density_profile_diagnostics": profile_diagnostics(track),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="data/processed/GD1_pwb18_track.h5")
    parser.add_argument("--pmcmd", default="data/processed/GD1_pwb18_pmcmd.h5")
    parser.add_argument(
        "--out", default="outputs/diagnostics/pwb18_gd1_catalog_validation.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = validate(Path(args.track), Path(args.pmcmd))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
    print(json.dumps(finite({key: payload[key] for key in (
        "track_catalog", "pmcmd_catalog", "coordinate_crosscheck", "checks",
        "all_checks_pass", "selection_profile",
    )}), indent=2))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
