#!/usr/bin/env python
"""Calibrate the smooth GD-1 particle-spray model on held-out control regions.

The two candidate-impact windows are excluded from model selection. Smooth
stream nuisance parameters are ranked using only the remaining along-stream
density/kinematic data and conditional cross-stream morphology. Scores in the
candidate windows are reported as held-out diagnostics and never select the
background model.

This is a bounded diagnostic calibration, not a posterior over stream-formation
parameters. Its purpose is to determine whether the fixed smooth-stream model is
the main bottleneck before adding more subhalo encounters.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forward_model.evolve import generate_perturbed_stream_evolved
from src.forward_model.morphology import (
    conditional_cross_track_js_score,
    reference_track_residuals,
)
from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
from src.forward_model.scoring import ScoreWeights
from src.simulation.subhalo import EncounterParams, scale_radius_from_mass


TARGETS = (30.7, 50.7)


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


def subtract_windows(
    full_range: tuple[float, float],
    excluded: list[tuple[float, float]],
    min_width: float = 2.0,
) -> list[tuple[float, float]]:
    """Return disjoint control windows after removing excluded intervals."""
    lo, hi = map(float, full_range)
    if hi <= lo:
        raise ValueError("full_range must have positive width")
    windows = [(lo, hi)]
    for cut_lo, cut_hi in sorted(excluded):
        if cut_hi <= cut_lo:
            raise ValueError("excluded windows must have positive width")
        updated = []
        for left, right in windows:
            if cut_hi <= left or cut_lo >= right:
                updated.append((left, right))
                continue
            if cut_lo - left >= min_width:
                updated.append((left, min(cut_lo, right)))
            if right - cut_hi >= min_width:
                updated.append((max(cut_hi, left), right))
        windows = updated
    return windows


def load_desi_member(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["streams/GD1/members"]
        return {
            name: group[name][:].astype(np.float64)
            for name in ("phi1", "delta_phi2", "p_thin", "p_cocoon")
        }


def weighted_mean(values: list[float], weights: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(array) & np.isfinite(weight) & (weight > 0)
    if not np.any(valid):
        return np.nan
    return float(np.average(array[valid], weights=weight[valid]))


def score_primary_windows(
    model: TimelineForwardModel,
    stream,
    windows: list[tuple[float, float]],
) -> dict:
    rows = []
    for window in windows:
        score = model._score_stream_vs_obs(stream, phi1_range=window)
        obs_count = int(
            np.sum(
                (model.obs_particles["phi1"] >= window[0])
                & (model.obs_particles["phi1"] <= window[1])
            )
        )
        rows.append(
            {
                "window": list(window),
                "n_observed": obs_count,
                **asdict(score),
            }
        )
    weights = [row["n_observed"] for row in rows]
    return {
        "windows": rows,
        "combined": weighted_mean([row["combined"] for row in rows], weights),
        "density_residual": weighted_mean(
            [row["density_residual"] for row in rows], weights
        ),
        "gap_agreement": weighted_mean([row["gap_agreement"] for row in rows], weights),
        "kinematic_perturbation": weighted_mean(
            [row["kinematic_perturbation"] for row in rows], weights
        ),
        "radial_velocity": weighted_mean(
            [row["radial_velocity"] for row in rows], weights
        ),
    }


def score_morphology_windows(
    stream,
    desi: dict[str, np.ndarray],
    windows: list[tuple[float, float]],
    phi1_bin_width: float,
    phi2_bin_width: float,
    phi2_max: float,
    track_bin_width: float,
) -> dict:
    delta_phi2 = reference_track_residuals(
        stream.phi1,
        stream.phi2,
        stream.phi1,
        stream.phi2,
        bin_width_deg=track_bin_width,
    )
    p_member = np.clip(desi["p_thin"] + desi["p_cocoon"], 0.0, 1.0)
    rows = []
    for window in windows:
        score = conditional_cross_track_js_score(
            stream.phi1,
            delta_phi2,
            desi["phi1"],
            desi["delta_phi2"],
            window,
            obs_weight=p_member,
            phi1_bin_width_deg=phi1_bin_width,
            phi2_range=(-phi2_max, phi2_max),
            phi2_bin_width_deg=phi2_bin_width,
        )
        rows.append({"window": list(window), **score})
    return {
        "windows": rows,
        "score": weighted_mean(
            [row["score"] for row in rows],
            [row["n_bins"] for row in rows],
        ),
        "n_active_bins": int(sum(row["n_bins"] for row in rows if row["active"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        default="data/processed/GD1_pwb18_desi_v3_fused.h5",
    )
    parser.add_argument(
        "--desi-member-h5",
        default="data/processed/GD1_desi_dr2_v3_member.h5",
    )
    parser.add_argument("--ages", nargs="+", type=float, default=[3.0, 4.5, 6.0, 9.0])
    parser.add_argument(
        "--progenitor-masses",
        nargs="+",
        type=float,
        default=[2500.0, 5000.0, 10000.0, 20000.0],
    )
    parser.add_argument(
        "--velocity-factors",
        nargs="+",
        type=float,
        default=[0.15, 0.3, 0.6],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--n-stars", type=int, default=1000)
    parser.add_argument("--control-epoch-gyr", type=float, default=2.75)
    parser.add_argument("--reference-age", type=float, default=3.0)
    parser.add_argument("--reference-progenitor-mass", type=float, default=5000.0)
    parser.add_argument("--reference-velocity-factor", type=float, default=0.3)
    parser.add_argument("--exclude-half-width", type=float, default=8.0)
    parser.add_argument("--target-half-width", type=float, default=8.0)
    parser.add_argument("--phi1-bin-width", type=float, default=4.0)
    parser.add_argument("--phi2-bin-width", type=float, default=0.2)
    parser.add_argument("--phi2-max", type=float, default=3.0)
    parser.add_argument("--track-bin-width", type=float, default=2.0)
    parser.add_argument(
        "--out",
        default="outputs/profile_grid/GD1/gd1_background_calibration.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.ages) <= args.control_epoch_gyr:
        raise ValueError("Every tested age must be older than the no-kick split epoch")

    model = TimelineForwardModel(
        ForwardModelConfig(
            stream_name="GD1",
            config_path="config/streams.yaml",
            processed_h5_path=args.h5,
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
    desi = load_desi_member(Path(args.desi_member_h5))

    excluded = [
        (target - args.exclude_half_width, target + args.exclude_half_width)
        for target in TARGETS
    ]
    control_windows = subtract_windows(model.phi1_range, excluded)
    target_windows = [
        (target - args.target_half_width, target + args.target_half_width)
        for target in TARGETS
    ]

    dummy_mass = 1e7
    dummy = EncounterParams(
        mass_solar=dummy_mass,
        scale_radius_kpc=scale_radius_from_mass(dummy_mass),
        impact_param_kpc=0.1,
        flyby_vel_kms=200.0,
        encounter_phi1=TARGETS[0],
        t_since_impact_gyr=args.control_epoch_gyr,
        is_valid=True,
    )

    combinations = list(
        itertools.product(
            args.ages,
            args.progenitor_masses,
            args.velocity_factors,
        )
    )
    reference_parameters = (
        float(args.reference_age),
        float(args.reference_progenitor_mass),
        float(args.reference_velocity_factor),
    )
    if reference_parameters not in combinations:
        combinations.insert(0, reference_parameters)
    rows = []
    started = time.perf_counter()
    for index, (age, progenitor_mass, velocity_factor) in enumerate(combinations, start=1):
        seed_rows = []
        for seed in args.seeds:
            stream = generate_perturbed_stream_evolved(
                stream_name="GD1",
                potential=model.potential,
                encounter=dummy,
                n_stars=args.n_stars,
                seed=seed,
                config_path=model.cfg.config_path,
                mws=model._mws,
                apply_kick=False,
                stream_age_gyr=age,
                progenitor_mass_msun=progenitor_mass,
                velocity_dispersion_factor=velocity_factor,
            )
            control_primary = score_primary_windows(model, stream, control_windows)
            control_morphology = score_morphology_windows(
                stream,
                desi,
                control_windows,
                args.phi1_bin_width,
                args.phi2_bin_width,
                args.phi2_max,
                args.track_bin_width,
            )
            target_primary = score_primary_windows(model, stream, target_windows)
            target_morphology = score_morphology_windows(
                stream,
                desi,
                target_windows,
                args.phi1_bin_width,
                args.phi2_bin_width,
                args.phi2_max,
                args.track_bin_width,
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "n_selected_stars": len(stream.phi1),
                    "control_primary": control_primary,
                    "control_morphology": control_morphology,
                    "held_out_target_primary": target_primary,
                    "held_out_target_morphology": target_morphology,
                }
            )
        rows.append(
            {
                "stream_age_gyr": age,
                "progenitor_mass_msun": progenitor_mass,
                "velocity_dispersion_factor": velocity_factor,
                "control_primary_score": float(
                    np.mean([row["control_primary"]["combined"] for row in seed_rows])
                ),
                "control_morphology_score": float(
                    np.mean([row["control_morphology"]["score"] for row in seed_rows])
                ),
                "held_out_target_primary_score": float(
                    np.mean(
                        [row["held_out_target_primary"]["combined"] for row in seed_rows]
                    )
                ),
                "held_out_target_morphology_score": float(
                    np.mean(
                        [row["held_out_target_morphology"]["score"] for row in seed_rows]
                    )
                ),
                "mean_selected_stars": float(
                    np.mean([row["n_selected_stars"] for row in seed_rows])
                ),
                "seed_results": seed_rows,
            }
        )
        print(f"{index}/{len(combinations)}", flush=True)

    default = next(
        row
        for row in rows
        if (
            row["stream_age_gyr"],
            row["progenitor_mass_msun"],
            row["velocity_dispersion_factor"],
        )
        == reference_parameters
    )
    for row in rows:
        row["control_primary_relative_improvement"] = (
            default["control_primary_score"] - row["control_primary_score"]
        ) / max(abs(default["control_primary_score"]), 1e-12)
        row["control_morphology_relative_improvement"] = (
            default["control_morphology_score"] - row["control_morphology_score"]
        ) / max(abs(default["control_morphology_score"]), 1e-12)
        row["balanced_control_relative_improvement"] = min(
            row["control_primary_relative_improvement"],
            row["control_morphology_relative_improvement"],
        )

    best_primary = min(rows, key=lambda row: row["control_primary_score"])
    best_morphology = min(rows, key=lambda row: row["control_morphology_score"])
    best_balanced = max(rows, key=lambda row: row["balanced_control_relative_improvement"])
    payload = {
        "result_label": "Synthetic smooth-stream background calibration",
        "scientific_status": (
            "Diagnostic nuisance-parameter calibration on control regions only; "
            "candidate-impact windows are held out and do not select the model."
        ),
        "settings": vars(args),
        "control_windows": control_windows,
        "held_out_target_windows": target_windows,
        "reference_parameters": {
            "stream_age_gyr": reference_parameters[0],
            "progenitor_mass_msun": reference_parameters[1],
            "velocity_dispersion_factor": reference_parameters[2],
        },
        "default_model": default,
        "best_control_primary": best_primary,
        "best_control_morphology": best_morphology,
        "best_balanced_control": best_balanced,
        "n_grid_candidates": len(args.ages)
        * len(args.progenitor_masses)
        * len(args.velocity_factors),
        "n_candidates_including_reference": len(rows),
        "runtime_s": time.perf_counter() - started,
        "all_candidates": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(finite(payload), indent=2), encoding="utf-8")
    print(
        "Best balanced control model:",
        f"age={best_balanced['stream_age_gyr']},",
        f"mass={best_balanced['progenitor_mass_msun']:.0f},",
        f"k_v={best_balanced['velocity_dispersion_factor']},",
        f"primary={best_balanced['control_primary_relative_improvement']:+.3f},",
        f"morphology={best_balanced['control_morphology_relative_improvement']:+.3f}",
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
