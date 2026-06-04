#!/usr/bin/env python
"""
SIDM discrimination via gap shape, the only handle SIDM has in this setup.

SIDM and CDM share the same abundance and mass-spectrum assumptions in
dm_discrimination_forecast.py. At fixed subhalo mass, however, a cored or
low-concentration SIDM perturber delivers a softer impulse than a cuspy NFW halo,
which should produce a shallower and broader gap.

This script runs matched-mass CDM/SIDM streamgapdf impacts and reports gap-depth
separability. Defaults reproduce the small exploratory report grid; command-line
arguments make the experiment scalable for a publication-grade follow-up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BACKEND: dict[str, object] | None = None


def load_backend() -> dict[str, object]:
    """Load the heavy astro backend only when the experiment is actually run."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    from scripts.generate_training_data import _fix_galpy_dll_path

    _fix_galpy_dll_path()

    import astropy.units as u
    from galpy.actionAngle import actionAngleIsochroneApprox, estimateBIsochrone
    from galpy.df import streamdf, streamgapdf  # noqa: F401 - validates backend availability.

    from scripts.validate_generator import auc, max_gap_depth
    from src.simulation.potentials import _RO, _VO, get_mw_potential
    from src.simulation.stream_gen import (
        _galactocentric_to_stream_coords,
        _pos_vel_to_orbit,
        set_progenitor_ic_track6d,
    )
    from src.simulation.subhalo import scale_radius_from_mass

    _BACKEND = {
        "u": u,
        "actionAngleIsochroneApprox": actionAngleIsochroneApprox,
        "estimateBIsochrone": estimateBIsochrone,
        "streamgapdf": streamgapdf,
        "auc": auc,
        "max_gap_depth": max_gap_depth,
        "_RO": _RO,
        "_VO": _VO,
        "get_mw_potential": get_mw_potential,
        "_galactocentric_to_stream_coords": _galactocentric_to_stream_coords,
        "_pos_vel_to_orbit": _pos_vel_to_orbit,
        "set_progenitor_ic_track6d": set_progenitor_ic_track6d,
        "scale_radius_from_mass": scale_radius_from_mass,
    }
    return _BACKEND


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def bootstrap_auc_ci(
    cdm_depths: np.ndarray,
    sidm_depths: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    auc_fn = load_backend()["auc"]
    if n_boot <= 0:
        return float("nan"), float("nan")
    draws = []
    for _ in range(n_boot):
        cdm = rng.choice(cdm_depths, size=len(cdm_depths), replace=True)
        sidm = rng.choice(sidm_depths, size=len(sidm_depths), replace=True)
        draws.append(auc_fn(cdm, sidm))
    lo, hi = np.percentile(draws, [16, 84])
    return float(lo), float(hi)


def load_galstreams():
    import galstreams

    return galstreams


def build_stream_context(args: argparse.Namespace):
    backend = load_backend()
    u = backend["u"]
    _RO = backend["_RO"]
    _VO = backend["_VO"]
    pot = backend["get_mw_potential"](args.stream_config)
    frame = None
    mws = None
    if args.frame_mode in {"auto", "galstreams"}:
        try:
            galstreams = load_galstreams()
            mws = galstreams.MWStreams(verbose=False)
            frame = mws[args.galstreams_key].stream_frame
            print(f"Using galstreams frame: {args.galstreams_key}")
        except Exception as exc:
            if args.frame_mode == "galstreams":
                raise
            if args.config_stream != "GD1":
                raise RuntimeError(
                    "galstreams frame failed and the built-in fallback is GD1-only. "
                    "Install gala/galstreams or run --config-stream GD1."
                ) from exc
            print(f"galstreams frame unavailable ({exc}); using GD1 Koposov fallback frame")

    ic = backend["set_progenitor_ic_track6d"](args.config_stream, args.stream_config, mws=mws)
    prog = backend["_pos_vel_to_orbit"](np.array(ic["pos_kpc"]), np.array(ic["vel_kms"]))
    b_iso = float(
        backend["estimateBIsochrone"](
            pot,
            prog.R(use_physical=True) / _RO,
            prog.z(use_physical=True) / _RO,
        )
    )
    aA = backend["actionAngleIsochroneApprox"](pot=pot, b=b_iso)
    common = dict(
        progenitor=prog,
        pot=pot,
        aA=aA,
        leading=True,
        nTrackChunks=args.track_chunks,
        tdisrupt=args.tdisrupt_gyr * u.Gyr,
        ro=_RO,
        vo=_VO,
    )
    if frame is not None:
        project_phi1 = galstreams_phi1_projector(frame)
    else:
        project_phi1 = gd1_koposov_phi1_projector()
    return project_phi1, common


def galstreams_phi1_projector(frame):
    backend = load_backend()
    _RO = backend["_RO"]
    to_stream_coords = backend["_galactocentric_to_stream_coords"]

    def project(xv):
        xv = np.asarray(xv, float)
        R, _vR, _vT, z, _vz, phi = xv
        if np.median(np.abs(R)) < 3.0:
            R = R * _RO
            z = z * _RO
        pos = np.array([R * np.cos(phi), R * np.sin(phi), z])
        vel = np.zeros((3, len(R)))
        return to_stream_coords(pos, vel, frame)[0]

    return project


def gd1_koposov_phi1_projector():
    """Fallback stream longitude using the published GD-1 Koposov rotation matrix."""
    backend = load_backend()
    u = backend["u"]
    _RO = backend["_RO"]
    from astropy.coordinates import Galactocentric, SkyCoord
    from src.data.gd1_frames import R_ICRS_TO_GD1

    galcen = Galactocentric(galcen_distance=_RO * u.kpc, z_sun=0.0208 * u.kpc)

    def project(xv):
        xv = np.asarray(xv, float)
        R, _vR, _vT, z, _vz, phi = xv
        if np.median(np.abs(R)) < 3.0:
            R = R * _RO
            z = z * _RO
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        sc = SkyCoord(x=x * u.kpc, y=y * u.kpc, z=z * u.kpc, frame=galcen, representation_type="cartesian")
        icrs = sc.transform_to("icrs")
        ra = icrs.ra.to_value(u.rad)
        dec = icrs.dec.to_value(u.rad)
        v_icrs = np.stack(
            [np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)],
            axis=-1,
        )
        v_gd1 = v_icrs @ R_ICRS_TO_GD1.T
        phi1 = np.degrees(np.arctan2(v_gd1[:, 1], v_gd1[:, 0]))
        return (phi1 + 180.0) % 360.0 - 180.0

    return project


def gap_profile_metrics(phi1: np.ndarray, bin_deg: float = 2.0, smooth: int = 7) -> dict[str, float]:
    """Return simple morphology metrics from a 1D stream-density profile."""
    from scipy.ndimage import uniform_filter1d

    phi1 = np.asarray(phi1, float)
    phi1 = phi1[np.isfinite(phi1)]
    if len(phi1) < 60:
        return {
            "depth": 0.0,
            "width_deg": 0.0,
            "center_phi1": float("nan"),
            "asymmetry": 0.0,
            "depth_width_ratio": 0.0,
        }
    lo, hi = np.percentile(phi1, [2, 98])
    if hi - lo < 4.0 * bin_deg:
        return {
            "depth": 0.0,
            "width_deg": 0.0,
            "center_phi1": float("nan"),
            "asymmetry": 0.0,
            "depth_width_ratio": 0.0,
        }
    edges = np.arange(lo, hi + bin_deg, bin_deg)
    hist, _ = np.histogram(phi1, bins=edges)
    if len(hist) < 5 or hist.sum() < 40:
        return {
            "depth": 0.0,
            "width_deg": 0.0,
            "center_phi1": float("nan"),
            "asymmetry": 0.0,
            "depth_width_ratio": 0.0,
        }

    base = np.maximum(uniform_filter1d(hist.astype(float), size=smooth, mode="nearest"), 1.0)
    gap = 1.0 - hist / base
    idx = int(np.argmax(gap))
    depth = float(max(gap[idx], 0.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if depth <= 0.0:
        return {
            "depth": 0.0,
            "width_deg": 0.0,
            "center_phi1": float(centers[idx]),
            "asymmetry": 0.0,
            "depth_width_ratio": 0.0,
        }

    half_depth = 0.5 * depth
    left = idx
    while left > 0 and gap[left - 1] >= half_depth:
        left -= 1
    right = idx
    while right < len(gap) - 1 and gap[right + 1] >= half_depth:
        right += 1
    width = float((right - left + 1) * bin_deg)
    left_area = float(np.sum(np.clip(gap[left:idx], 0.0, None)))
    right_area = float(np.sum(np.clip(gap[idx + 1 : right + 1], 0.0, None)))
    denom = left_area + right_area + depth
    asymmetry = float((right_area - left_area) / denom) if denom > 0.0 else 0.0
    return {
        "depth": depth,
        "width_deg": width,
        "center_phi1": float(centers[idx]),
        "asymmetry": asymmetry,
        "depth_width_ratio": float(depth / max(width, bin_deg)),
    }


def sample_gap_metrics(
    common: dict,
    project_phi1,
    logm: float,
    scale_radius_kpc: float,
    impact_angle_rad: float,
    impact_b_kpc: float,
    impact_time_gyr: float,
    n_stars: int,
) -> dict[str, float]:
    backend = load_backend()
    u = backend["u"]
    mass = 10.0**logm
    sg = backend["streamgapdf"](
        0.5 * u.km / u.s,
        **common,
        impactb=impact_b_kpc * u.kpc,
        subhalovel=np.array([0.0, 150.0, 0.0]) * u.km / u.s,
        timpact=impact_time_gyr * u.Gyr,
        impact_angle=impact_angle_rad * u.rad,
        GM=mass * u.Msun,
        rs=scale_radius_kpc * u.kpc,
    )
    return gap_profile_metrics(project_phi1(sg.sample(n=n_stars)))


def run_grid(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    boot_rng = np.random.default_rng(args.seed + 1000)
    backend = load_backend()
    auc_fn = backend["auc"]
    scale_radius_from_mass = backend["scale_radius_from_mass"]
    project_phi1, common = build_stream_context(args)
    logm_grid = parse_float_list(args.logm)
    core_factors = parse_float_list(args.core_factors)
    details = []

    print(
        "SIDM morphology grid: "
        f"core_factors={core_factors}, logM={logm_grid}, trials={args.trials}"
    )
    print(
        f"{'core':>6} {'logM':>5} {'r_s NFW':>8} {'gapdepth CDM':>13} "
        f"{'gapdepth SIDM':>14} {'AUC(CDM>SIDM)':>14} {'CI16-84':>15}"
    )

    for core_factor in core_factors:
        for logm in logm_grid:
            rs_nfw = scale_radius_from_mass(10.0**logm)
            cdm_metrics = []
            sidm_metrics = []
            for _ in range(args.trials):
                impact_angle = rng.uniform(args.angle_min, args.angle_max)
                impact_b = rng.uniform(args.impact_b_min, args.impact_b_max)
                impact_time = rng.uniform(args.timpact_min, args.timpact_max)
                cdm_metrics.append(
                    sample_gap_metrics(
                        common,
                        project_phi1,
                        logm,
                        rs_nfw,
                        impact_angle,
                        impact_b,
                        impact_time,
                        args.n_stars,
                    )
                )
                sidm_metrics.append(
                    sample_gap_metrics(
                        common,
                        project_phi1,
                        logm,
                        rs_nfw * core_factor,
                        impact_angle,
                        impact_b,
                        impact_time,
                        args.n_stars,
                    )
                )

            cdm_depths_arr = np.asarray([m["depth"] for m in cdm_metrics])
            sidm_depths_arr = np.asarray([m["depth"] for m in sidm_metrics])
            cdm_widths_arr = np.asarray([m["width_deg"] for m in cdm_metrics])
            sidm_widths_arr = np.asarray([m["width_deg"] for m in sidm_metrics])
            cdm_ratios_arr = np.asarray([m["depth_width_ratio"] for m in cdm_metrics])
            sidm_ratios_arr = np.asarray([m["depth_width_ratio"] for m in sidm_metrics])
            cdm_asym_arr = np.asarray([m["asymmetry"] for m in cdm_metrics])
            sidm_asym_arr = np.asarray([m["asymmetry"] for m in sidm_metrics])
            auc_value = float(auc_fn(cdm_depths_arr, sidm_depths_arr))
            auc_width_value = float(auc_fn(sidm_widths_arr, cdm_widths_arr))
            auc_ratio_value = float(auc_fn(cdm_ratios_arr, sidm_ratios_arr))
            ci_lo, ci_hi = bootstrap_auc_ci(
                cdm_depths_arr,
                sidm_depths_arr,
                boot_rng,
                args.bootstrap,
            )
            row = {
                "core_factor": float(core_factor),
                "logM": float(logm),
                "rs_nfw_kpc": float(rs_nfw),
                "median_gapdepth_cdm": float(np.median(cdm_depths_arr)),
                "median_gapdepth_sidm": float(np.median(sidm_depths_arr)),
                "median_gapwidth_cdm_deg": float(np.median(cdm_widths_arr)),
                "median_gapwidth_sidm_deg": float(np.median(sidm_widths_arr)),
                "median_depth_width_ratio_cdm": float(np.median(cdm_ratios_arr)),
                "median_depth_width_ratio_sidm": float(np.median(sidm_ratios_arr)),
                "median_asymmetry_cdm": float(np.median(cdm_asym_arr)),
                "median_asymmetry_sidm": float(np.median(sidm_asym_arr)),
                "auc_cdm_deeper_than_sidm": auc_value,
                "auc_sidm_broader_than_cdm": auc_width_value,
                "auc_cdm_larger_depth_width_ratio": auc_ratio_value,
                "auc_ci16": ci_lo,
                "auc_ci84": ci_hi,
                "n_trials": int(args.trials),
                "n_stars": int(args.n_stars),
            }
            details.append(row)
            print(
                f"{core_factor:6.2f} {logm:5.1f} {rs_nfw:8.2f} "
                f"{row['median_gapdepth_cdm']:13.2f} {row['median_gapdepth_sidm']:14.2f} "
                f"{auc_value:14.2f} [{ci_lo:.2f},{ci_hi:.2f}]"
            )

    primary_core = args.primary_core_factor
    if primary_core is None:
        primary_core = 3.0 if 3.0 in core_factors else core_factors[0]
    if primary_core not in core_factors:
        raise ValueError(
            f"--primary-core-factor {primary_core} is not in --core-factors {core_factors}"
        )
    primary = [r for r in details if r["core_factor"] == primary_core]
    mean_auc = float(np.mean([r["auc_cdm_deeper_than_sidm"] for r in primary]))
    legacy_rows = [
        (
            r["logM"],
            r["median_gapdepth_cdm"],
            r["median_gapdepth_sidm"],
            r["auc_cdm_deeper_than_sidm"],
        )
        for r in primary
    ]
    print(f"\nPrimary core-factor ({primary_core:g}x) mean AUC = {mean_auc:.2f}")
    print("AUC ~0.5 is indistinguishable; >0.7 means cored gaps are systematically shallower.")
    return {
        "core_factor": float(primary_core),
        "core_factors": core_factors,
        "logm_grid": logm_grid,
        "trials": int(args.trials),
        "n_stars": int(args.n_stars),
        "bootstrap": int(args.bootstrap),
        "rows": legacy_rows,
        "mean_auc": mean_auc,
        "details": details,
        "note": (
            "Rows preserves the historical figure format for the primary core factor; "
            "details contains the full scalable grid and bootstrap interval."
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-config", default="config/streams.yaml")
    parser.add_argument("--config-stream", default="GD1")
    parser.add_argument("--galstreams-key", default="GD-1-I21")
    parser.add_argument(
        "--frame-mode",
        choices=["auto", "galstreams", "gd1-koposov"],
        default="auto",
        help="Projection frame for gap-depth measurement. auto falls back to GD1 Koposov if gala/galstreams is unavailable.",
    )
    parser.add_argument("--core-factors", default="3.0")
    parser.add_argument(
        "--primary-core-factor",
        type=float,
        default=None,
        help="Core factor used for legacy rows/mean_auc figure compatibility. Defaults to 3.0 when present, otherwise the first core factor.",
    )
    parser.add_argument("--logm", default="8.0,8.3,8.6")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--n-stars", type=int, default=1200)
    parser.add_argument("--track-chunks", type=int, default=5)
    parser.add_argument("--tdisrupt-gyr", type=float, default=3.0)
    parser.add_argument("--angle-min", type=float, default=0.3)
    parser.add_argument("--angle-max", type=float, default=0.6)
    parser.add_argument("--impact-b-min", type=float, default=0.0)
    parser.add_argument("--impact-b-max", type=float, default=0.2)
    parser.add_argument("--timpact-min", type=float, default=0.4)
    parser.add_argument("--timpact-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/dm/sidm_morphology.json")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_grid(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
