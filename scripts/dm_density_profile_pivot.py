#!/usr/bin/env python
"""Build the density-profile pivot artifact.

The population-count route is underpowered for the current stream sample.  This
artifact reframes the near-term science target around the density profile of an
individual perturber: compact/cuspy, broadened/cored, solitonic, or baryonic.

Outputs:
  outputs/dm/density_profile_pivot.json
  outputs/dm/density_profile_pivot.md
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.subhalo import G_KPC_KMS, scale_radius_from_mass, subhalo_profile_for_model


OUT_JSON = Path("outputs/dm/density_profile_pivot.json")
OUT_MD = Path("outputs/dm/density_profile_pivot.md")
STREAM_CFG = Path("config/streams.yaml")
PROCESSED = Path("data/processed")
SIDM_MORPH = Path("outputs/dm/sidm_morphology.json")
MASS_GRID = (1e6, 1e7, 1e8, 1e9)
IMPACT_B_KPC = 0.1
FLYBY_V_KMS = 200.0


@dataclass(frozen=True)
class ProfileCase:
    name: str
    family: str
    description: str
    scale_radius_kpc: float
    core_radius_kpc: float
    concentration: float | None
    kick_suppression_proxy: float


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 1000 or (abs(value) < 0.01 and value != 0):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    if isinstance(value, tuple):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def h5_member_count(path: Path) -> tuple[int, str | None]:
    try:
        with h5py.File(path, "r") as f:
            for stream in f.get("streams", {}).keys():
                grp = f[f"streams/{stream}/members"]
                if not grp:
                    return 0, stream
                first = next(iter(grp.values()))
                return int(len(first)), stream
    except OSError:
        return 0, None
    return 0, None


def stream_catalogs() -> list[dict[str, Any]]:
    cfg = yaml.safe_load(STREAM_CFG.read_text(encoding="utf-8"))["streams"]
    rows = []
    for stream, sc in cfg.items():
        candidates = []
        for kind in ("clean", "streamfinder", "multiepoch"):
            path = PROCESSED / f"{stream}_{kind}.h5"
            if path.exists():
                n, _ = h5_member_count(path)
                candidates.append({"kind": kind, "path": str(path), "n_members": n})

        best_clean = max(
            [c for c in candidates if c["kind"] in {"clean", "streamfinder"}],
            key=lambda c: c["n_members"],
            default=None,
        )
        has_known_gap = bool(sc.get("known_gaps"))
        tier = sc.get("analysis_tier", "")
        if best_clean is None:
            verdict = "no local clean catalog"
        elif best_clean["n_members"] >= 500 and has_known_gap:
            verdict = "priority profile target"
        elif best_clean["n_members"] >= 500:
            verdict = "high-N control / null target"
        elif best_clean["n_members"] >= 150:
            verdict = "sparse profile target; needs membership cleaning"
        else:
            verdict = "too sparse for profile inference today"

        rows.append(
            {
                "stream": stream,
                "display_name": sc.get("display_name", stream),
                "analysis_tier": tier,
                "has_known_gap": has_known_gap,
                "best_clean_catalog": best_clean,
                "all_local_catalogs": candidates,
                "profile_pivot_verdict": verdict,
            }
        )
    return rows


def plummer_kick_metrics(mass: float, scale_radius: float) -> dict[str, float]:
    """Impulse-shape metrics for a Plummer/Hernquist-equivalent softened perturber."""
    a = max(float(scale_radius), 1.0e-8)
    b = IMPACT_B_KPC
    v = FLYBY_V_KMS
    dv_at_b = 2.0 * G_KPC_KMS * mass / v * b / (b * b + a * a)
    dv_peak = G_KPC_KMS * mass / (v * a)
    halfmax_width_kpc = 2.0 * math.sqrt(3.0) * a
    return {
        "impact_b_kpc": b,
        "flyby_v_kms": v,
        "dv_at_b_kms": float(dv_at_b),
        "dv_peak_kms": float(dv_peak),
        "halfmax_width_kpc": float(halfmax_width_kpc),
        "softening_to_impact_ratio": float(a / b),
    }


def profile_cases(mass: float) -> list[ProfileCase]:
    cdm = subhalo_profile_for_model("CDM", mass)
    wdm = subhalo_profile_for_model("WDM", mass, m_wdm_kev=3.0)
    fdm = subhalo_profile_for_model("FDM", mass, m_axion_ev=1e-22)
    sidm1 = subhalo_profile_for_model("SIDM", mass, sigma_sidm_cm2g=1.0)
    sidm10 = subhalo_profile_for_model("SIDM", mass, sigma_sidm_cm2g=10.0)
    compact = dict(cdm)
    compact["scale_radius_kpc"] = max(0.25 * cdm["scale_radius_kpc"], 1.0e-5)
    compact["profile_type"] = "compact/core-collapsed proxy"
    compact["kick_suppression"] = 1.0
    return [
        ProfileCase("CDM_NFW", "cuspy", "reference NFW/cuspy subhalo", **case_kwargs(cdm)),
        ProfileCase("WDM_3keV_reduced_c", "less-concentrated", "WDM-like lower concentration at low mass", **case_kwargs(wdm)),
        ProfileCase("FDM_1e-22_soliton", "solitonic", "FDM-like soliton plus NFW envelope", **case_kwargs(fdm)),
        ProfileCase("SIDM_sigma1_core", "cored", "SIDM-like moderate isothermal core", **case_kwargs(sidm1)),
        ProfileCase("SIDM_sigma10_core", "cored", "SIDM-like larger isothermal core", **case_kwargs(sidm10)),
        ProfileCase("compact_collapsed_proxy", "compact", "compact/core-collapsed or baryonic-like perturber proxy", **case_kwargs(compact)),
    ]


def case_kwargs(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale_radius_kpc": float(profile["scale_radius_kpc"]),
        "core_radius_kpc": float(profile.get("core_radius_kpc", 0.0)),
        "concentration": float(profile["concentration"]) if "concentration" in profile else None,
        "kick_suppression_proxy": float(profile.get("kick_suppression", 1.0)),
    }


def profile_ladder() -> list[dict[str, Any]]:
    rows = []
    for mass in MASS_GRID:
        cdm_a = subhalo_profile_for_model("CDM", mass)["scale_radius_kpc"]
        cdm_dv_peak = plummer_kick_metrics(mass, cdm_a)["dv_peak_kms"]
        for case in profile_cases(mass):
            metrics = plummer_kick_metrics(mass, case.scale_radius_kpc)
            rows.append(
                {
                    "log10_mass": float(np.log10(mass)),
                    "profile": case.name,
                    "family": case.family,
                    "description": case.description,
                    "scale_radius_kpc": case.scale_radius_kpc,
                    "core_radius_kpc": case.core_radius_kpc,
                    "concentration": case.concentration,
                    "kick_suppression_proxy": case.kick_suppression_proxy,
                    "dv_peak_ratio_to_cdm": metrics["dv_peak_kms"] / max(cdm_dv_peak, 1.0e-12),
                    **metrics,
                }
            )
    return rows


def sidm_morphology_summary() -> dict[str, Any] | None:
    if not SIDM_MORPH.exists():
        return None
    data = json.loads(SIDM_MORPH.read_text(encoding="utf-8"))
    details = data.get("details", [])
    by_core: dict[str, list[float]] = {}
    for row in details:
        by_core.setdefault(str(row["core_factor"]), []).append(float(row["auc_cdm_deeper_than_sidm"]))
    return {
        "artifact": str(SIDM_MORPH),
        "primary_core_factor": data.get("core_factor"),
        "primary_mean_auc": data.get("mean_auc"),
        "mean_depth_auc_by_core_factor": {
            core: float(np.mean(values)) for core, values in sorted(by_core.items(), key=lambda kv: float(kv[0]))
        },
        "interpretation": (
            "AUC near 0.5 means profile shape is not separable; values above about 0.7 "
            "mean cuspy and cored profiles produce systematically different gap depths."
        ),
    }


def strategy() -> list[dict[str, str]]:
    return [
        {
            "step": "GD-1 density-profile grid",
            "why": "GD-1 is the only current high-N stream with a known gap/spur.",
            "implementation": "Fit mass, time, impact location, impact parameter, velocity, and scale-radius/core-factor; compare compact, NFW, cored, and soliton profiles against the same observed gap.",
        },
        {
            "step": "Morphology likelihood",
            "why": "Counts are underpowered, but gap depth, width, asymmetry, spur offset, and track kink encode perturber compactness.",
            "implementation": "Replace hard model labels with continuous profile parameters such as log10(M), scale radius, core radius, compactness, and density within the impact radius.",
        },
        {
            "step": "Power-spectrum / correlation baseline",
            "why": "It uses all density fluctuations rather than only detected gaps.",
            "implementation": "Run a Bovy/Banik-style 1D density power-spectrum baseline on GD-1 and ATLAS controls, then compare profile-grid residuals.",
        },
        {
            "step": "Kinematic sharpening",
            "why": "Proper-motion and radial-velocity kicks can separate compact from broad perturbers when density alone is degenerate.",
            "implementation": "Use multi-epoch/RV catalogs where available; score differential tracks after removing global velocity zero points.",
        },
        {
            "step": "Membership cleaning expansion",
            "why": "More streams help only if their member catalogs are clean enough for profile morphology.",
            "implementation": "Prioritize converting sparse STREAMFINDER/S5 candidates into clean high-probability catalogs; use ATLAS as the null/control calibration target.",
        },
        {
            "step": "External-prior comparison",
            "why": "A single stream constrains the perturber, not the whole DM model by itself.",
            "implementation": "Report profile constraints as likelihoods over compact/cuspy/cored/solitonic perturbers and compare to satellite/lensing/literature priors separately.",
        },
    ]


def build_output() -> dict[str, Any]:
    streams = stream_catalogs()
    ladder = profile_ladder()
    return {
        "meta": {
            "pivot": "density profile / perturber compactness",
            "reason": "Current stream counts are underpowered for CDM/WDM/FDM population model selection.",
            "impact_b_kpc_for_ladder": IMPACT_B_KPC,
            "flyby_v_kms_for_ladder": FLYBY_V_KMS,
        },
        "real_stream_targets": streams,
        "profile_ladder": ladder,
        "sidm_morphology_summary": sidm_morphology_summary(),
        "strategy": strategy(),
    }


def markdown(output: dict[str, Any]) -> str:
    lines = [
        "# Density-Profile Pivot",
        "",
        "The near-term science target is no longer global CDM/WDM/FDM model selection from impact counts. It is perturber-density-profile inference for the best individual gaps.",
        "",
        "## Real Targets",
        "",
        "| Stream | Best clean catalog | N | Known gap? | Profile-pivot verdict |",
        "|---|---|---:|---|---|",
    ]
    for row in output["real_stream_targets"]:
        best = row["best_clean_catalog"] or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    row["display_name"],
                    best.get("kind", "none"),
                    str(best.get("n_members", 0)),
                    "yes" if row["has_known_gap"] else "no",
                    row["profile_pivot_verdict"],
                ]
            )
            + " |"
        )

    sidm = output.get("sidm_morphology_summary")
    if sidm:
        pieces = ", ".join(
            f"{float(core):g}x:{auc:.2f}"
            for core, auc in sidm["mean_depth_auc_by_core_factor"].items()
        )
        lines.extend(
            [
                "",
                "## Existing Profile Evidence",
                "",
                (
                    f"The SIDM morphology grid already shows core-strength sensitivity. "
                    f"Mean depth-AUC by core factor: {pieces}. "
                    f"The primary plotted core factor is {sidm['primary_core_factor']}x "
                    f"with mean AUC {sidm['primary_mean_auc']:.2f}."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Analytic Impulse Profile Ladder",
            "",
            f"Fixed geometry for this ladder: impact parameter {IMPACT_B_KPC:.2f} kpc, flyby velocity {FLYBY_V_KMS:.0f} km/s.",
            "",
            "| log10 M | Profile | Family | scale radius [kpc] | core radius [kpc] | peak kick / CDM | kick at b [km/s] | half-max width [kpc] |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in output["profile_ladder"]:
        if row["log10_mass"] not in {7.0, 8.0, 9.0}:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["log10_mass"], 1),
                    row["profile"],
                    row["family"],
                    fmt(row["scale_radius_kpc"], 3),
                    fmt(row["core_radius_kpc"], 3),
                    fmt(row["dv_peak_ratio_to_cdm"], 2),
                    fmt(row["dv_at_b_kms"], 2),
                    fmt(row["halfmax_width_kpc"], 2),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Next Experiments", ""])
    for item in output["strategy"]:
        lines.append(f"- **{item['step']}**: {item['why']} {item['implementation']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    output = build_output()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(finite(output), indent=2), encoding="utf-8")
    OUT_MD.write_text(markdown(output), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
