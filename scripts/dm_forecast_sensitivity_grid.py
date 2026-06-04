#!/usr/bin/env python
"""Stress-test the DM abundance/mass discrimination forecast.

The baseline forecast in ``dm_discrimination_forecast.py`` is intentionally compact.
This script expands it into a reviewer-facing sensitivity grid over the assumptions
most likely to be challenged:

* encounter-rate normalization and the historical low-rate floor,
* completeness-curve uncertainty,
* detector-threshold proxy effects,
* the log-mass sensitivity band, and
* the number of clean streams available.

The detector-threshold axis is a proxy: it rescales completeness to mimic lower or
higher operating thresholds, but it does not model false-positive contamination.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.dm_discrimination_forecast import (  # noqa: E402
    ALPHA,
    CL_LOGM,
    CL_P,
    MODELS,
    RHO_SUB_MSUN_KPC3,
    STREAM_AGE_GYR,
    STREAM_LEN_KPC,
    STREAM_WIDTH_PC,
    finite_json,
    kl_divergence,
    raw_encounter_rate,
    suppression,
)
from src.simulation.mass_functions import compute_encounter_rate  # noqa: E402


OUT_JSON = Path("outputs/dm/dm_forecast_sensitivity_grid.json")
OUT_MD = Path("outputs/dm/dm_forecast_sensitivity_grid.md")
STREAM_COUNTS = (7, 50, 100, 300, 500, 1000, 1500)
MODEL_ORDER = ("WDM 6 keV", "WDM 4 keV", "WDM 3 keV", "FDM 1e-21", "FDM 1e-22", "SIDM")


@dataclass(frozen=True)
class RateVariant:
    name: str
    use_floor: bool
    scale: float
    note: str


@dataclass(frozen=True)
class CompletenessVariant:
    name: str
    scale: float
    note: str


@dataclass(frozen=True)
class ThresholdProxy:
    name: str
    threshold: float
    completeness_scale: float
    note: str


@dataclass(frozen=True)
class MassBand:
    name: str
    lo: float
    hi: float
    note: str


RATE_VARIANTS = (
    RateVariant("raw_no_floor", False, 1.0, "no historical low-rate floor"),
    RateVariant("floor_half", True, 0.5, "floored encounter rate scaled down by 2"),
    RateVariant("floor_baseline", True, 1.0, "baseline floored encounter rate"),
    RateVariant("floor_double", True, 2.0, "floored encounter rate scaled up by 2"),
)

COMPLETENESS_VARIANTS = (
    CompletenessVariant("low_completeness", 0.8, "measured completeness curve scaled down by 20%"),
    CompletenessVariant("nominal_completeness", 1.0, "measured completeness curve"),
    CompletenessVariant("high_completeness", 1.2, "measured completeness curve scaled up by 20%"),
)

THRESHOLD_PROXIES = (
    ThresholdProxy(
        "loose_threshold_proxy",
        0.35,
        1.15,
        "proxy for a lower detector threshold; false positives not modeled",
    ),
    ThresholdProxy("nominal_threshold", 0.50, 1.0, "reported zero-FP operating point"),
    ThresholdProxy(
        "strict_threshold_proxy",
        0.65,
        0.85,
        "proxy for a stricter detector threshold; false positives remain zero by construction",
    ),
)

MASS_BANDS = (
    MassBand("low_extended_7p3_8p7", 7.3, 8.7, "lower minimum mass sensitivity"),
    MassBand("nominal_7p5_8p7", 7.5, 8.7, "reported sensitivity band"),
    MassBand("narrow_7p7_8p5", 7.7, 8.5, "conservative narrow band"),
    MassBand("high_extended_7p5_9p0", 7.5, 9.0, "extends high-mass tail"),
    MassBand("wide_7p3_9p0", 7.3, 9.0, "optimistic wider band"),
)


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "not separable"
    value = float(value)
    if abs(value) >= 10000:
        return f"{value:.2e}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def completeness(logm: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.interp(logm, CL_LOGM, CL_P) * scale, 0.0, 1.0)


def cdm_band_rate(rate: RateVariant, band: MassBand) -> dict[str, float]:
    raw = raw_encounter_rate(
        STREAM_AGE_GYR,
        STREAM_LEN_KPC,
        STREAM_WIDTH_PC,
        band.lo,
        band.hi,
        alpha=ALPHA,
        rho_sub_msun_kpc3=RHO_SUB_MSUN_KPC3,
    )
    floored = compute_encounter_rate(
        STREAM_AGE_GYR,
        STREAM_LEN_KPC,
        STREAM_WIDTH_PC,
        band.lo,
        band.hi,
        alpha=ALPHA,
        rho_sub_msun_kpc3=RHO_SUB_MSUN_KPC3,
    )
    chosen = floored if rate.use_floor else raw
    return {
        "raw_pre_completeness": float(raw),
        "floored_pre_completeness": float(floored),
        "chosen_pre_completeness": float(chosen * rate.scale),
    }


def model_rate_pdf(
    model: str,
    params: dict[str, float],
    logm: np.ndarray,
    band_rate: float,
    completeness_scale: float,
) -> tuple[float, np.ndarray]:
    mass = 10.0**logm
    shape = mass ** (ALPHA + 1.0)
    cdm_rate_density = shape / np.trapezoid(shape, logm) * band_rate
    integrand = (
        cdm_rate_density
        * suppression(model, params, mass)
        * completeness(logm, completeness_scale)
    )
    lam_det = float(np.trapezoid(integrand, logm))
    pdf = integrand / max(lam_det, 1.0e-30)
    return lam_det, pdf


def evaluate_case(
    rate: RateVariant,
    comp: CompletenessVariant,
    threshold: ThresholdProxy,
    band: MassBand,
    z_target: float,
) -> dict[str, Any]:
    logm = np.linspace(band.lo, band.hi, 900)
    rates = cdm_band_rate(rate, band)
    total_comp_scale = comp.scale * threshold.completeness_scale
    lam_cdm, pdf_cdm = model_rate_pdf("CDM", {}, logm, rates["chosen_pre_completeness"], total_comp_scale)

    models: dict[str, Any] = {
        "CDM": {
            "lambda_det_per_stream": lam_cdm,
            "expected_detections": {str(n): n * lam_cdm for n in STREAM_COUNTS},
            "p_zero_detections": {str(n): math.exp(-n * lam_cdm) for n in STREAM_COUNTS},
        }
    }

    for name, (family, params) in MODELS.items():
        lam_model, pdf_model = model_rate_pdf(
            family, params, logm, rates["chosen_pre_completeness"], total_comp_scale
        )
        kl_mass = kl_divergence(pdf_model, pdf_cdm, logm)
        llr_rate = (lam_cdm - lam_model) + (
            lam_model * math.log(lam_model / lam_cdm)
            if lam_model > 0.0 and lam_cdm > 0.0
            else 0.0
        )
        llr_mass = lam_model * kl_mass
        llr_total = llr_rate + llr_mass
        finite = llr_total > 1.0e-12
        n_streams = z_target**2 / (2.0 * llr_total) if finite else math.inf
        models[name] = {
            "lambda_det_per_stream": lam_model,
            "rate_ratio_to_cdm": lam_model / lam_cdm if lam_cdm > 0.0 else math.inf,
            "kl_mass": kl_mass,
            "llr_rate_per_stream": llr_rate,
            "llr_mass_per_stream": llr_mass,
            "llr_total_per_stream": llr_total,
            "n_streams_3sigma": n_streams,
            "n_detections_3sigma": n_streams * lam_model if finite else math.inf,
            "expected_detections": {str(n): n * lam_model for n in STREAM_COUNTS},
            "p_zero_detections": {str(n): math.exp(-n * lam_model) for n in STREAM_COUNTS},
            "z_by_stream_count": {
                str(n): math.sqrt(max(0.0, 2.0 * n * llr_total)) for n in STREAM_COUNTS
            },
        }

    return {
        "case": {
            "rate_variant": rate.name,
            "rate_floor": rate.use_floor,
            "rate_scale": rate.scale,
            "completeness_variant": comp.name,
            "completeness_scale": comp.scale,
            "threshold_proxy": threshold.name,
            "detector_threshold": threshold.threshold,
            "threshold_completeness_scale": threshold.completeness_scale,
            "total_completeness_scale": total_comp_scale,
            "mass_band": band.name,
            "mass_band_log10_msun": [band.lo, band.hi],
            "raw_cdm_band_rate_pre_completeness": rates["raw_pre_completeness"],
            "floored_cdm_band_rate_pre_completeness": rates["floored_pre_completeness"],
            "chosen_cdm_band_rate_pre_completeness": rates["chosen_pre_completeness"],
        },
        "models": models,
    }


def finite_values(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def quantile(values: list[float], q: float) -> float | None:
    vals = finite_values(values)
    if not vals:
        return None
    return float(np.quantile(vals, q))


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(
        c
        for c in cases
        if c["case"]["rate_variant"] == "floor_baseline"
        and c["case"]["completeness_variant"] == "nominal_completeness"
        and c["case"]["threshold_proxy"] == "nominal_threshold"
        and c["case"]["mass_band"] == "nominal_7p5_8p7"
    )
    cdm_p0_7 = [c["models"]["CDM"]["p_zero_detections"]["7"] for c in cases]
    cdm_lambda = [c["models"]["CDM"]["lambda_det_per_stream"] for c in cases]
    summary: dict[str, Any] = {}
    for model in MODEL_ORDER:
        n_streams = [c["models"][model]["n_streams_3sigma"] for c in cases]
        z7 = [c["models"][model]["z_by_stream_count"]["7"] for c in cases]
        p0_7 = [c["models"][model]["p_zero_detections"]["7"] for c in cases]
        n_det = [c["models"][model]["n_detections_3sigma"] for c in cases]
        summary[model] = {
            "baseline_n_streams_3sigma": baseline["models"][model]["n_streams_3sigma"],
            "baseline_n_detections_3sigma": baseline["models"][model]["n_detections_3sigma"],
            "n_streams_3sigma_min": min(finite_values(n_streams)) if finite_values(n_streams) else None,
            "n_streams_3sigma_p16": quantile(n_streams, 0.16),
            "n_streams_3sigma_median": quantile(n_streams, 0.50),
            "n_streams_3sigma_p84": quantile(n_streams, 0.84),
            "n_streams_3sigma_max": max(finite_values(n_streams)) if finite_values(n_streams) else None,
            "n_detections_3sigma_median": quantile(n_det, 0.50),
            "seven_stream_z_min": min(z7),
            "seven_stream_z_max": max(z7),
            "seven_stream_p0_min": min(p0_7),
            "seven_stream_p0_max": max(p0_7),
        }
    return {
        "baseline_case": baseline["case"],
        "cdm_null_summary": {
            "baseline_lambda_det_per_stream": baseline["models"]["CDM"]["lambda_det_per_stream"],
            "lambda_det_per_stream_min": min(cdm_lambda),
            "lambda_det_per_stream_max": max(cdm_lambda),
            "baseline_p0_7_streams": baseline["models"]["CDM"]["p_zero_detections"]["7"],
            "p0_7_streams_min": min(cdm_p0_7),
            "p0_7_streams_max": max(cdm_p0_7),
        },
        "models": summary,
        "conclusions": build_conclusions(summary),
    }


def build_conclusions(summary: dict[str, Any]) -> list[str]:
    wdm3 = summary["WDM 3 keV"]
    fdm = summary["FDM 1e-22"]
    wdm6 = summary["WDM 6 keV"]
    return [
        (
            "The seven-stream sample remains underpowered across the grid: even the largest "
            f"seven-stream Z values are {wdm3['seven_stream_z_max']:.2f} for WDM 3 keV and "
            f"{fdm['seven_stream_z_max']:.2f} for FDM 1e-22."
        ),
        (
            "Favorable suppressed models remain detectable in principle, but the stream "
            "requirement is assumption-sensitive: median 3-sigma stream counts are "
            f"{wdm3['n_streams_3sigma_median']:.0f} for WDM 3 keV and "
            f"{fdm['n_streams_3sigma_median']:.0f} for FDM 1e-22 across this grid."
        ),
        (
            "Near-CDM alternatives remain effectively unreachable with this statistic: "
            f"WDM 6 keV has median N_streams={wdm6['n_streams_3sigma_median']:.0f}."
        ),
        (
            "The detection-threshold axis is only a completeness proxy; a publication "
            "forecast should replace it with measured ROC/completeness curves if the "
            "operating threshold is changed."
        ),
    ]


def fmt_range(lo: float | None, hi: float | None, digits: int = 0) -> str:
    if lo is None or hi is None:
        return "not separable"
    return f"{fmt_num(lo, digits)}-{fmt_num(hi, digits)}"


def markdown(output: dict[str, Any]) -> str:
    summary = output["summary"]["models"]
    cdm = output["summary"]["cdm_null_summary"]
    lines = [
        "# DM Forecast Sensitivity Grid",
        "",
        "Grid axes: encounter-rate normalization/floor, completeness scale, detector-threshold proxy, mass band, and stream count.",
        "The threshold axis rescales completeness only; false-positive contamination is not modeled.",
        "",
        "## CDM Null Check",
        "",
        (
            f"Baseline CDM lambda_det={cdm['baseline_lambda_det_per_stream']:.4f}/stream "
            f"and P0(7 streams)={cdm['baseline_p0_7_streams']:.2f}. Across the grid, "
            f"lambda_det spans {cdm['lambda_det_per_stream_min']:.4f}-"
            f"{cdm['lambda_det_per_stream_max']:.4f}/stream and P0(7 streams) spans "
            f"{cdm['p0_7_streams_min']:.2f}-{cdm['p0_7_streams_max']:.2f}."
        ),
        "",
        "## Model Summary",
        "",
        "| Model | Baseline N_streams(3-sigma) | Grid median | Grid 16-84% | Grid min-max | Max Z at 7 streams | P0 range at 7 streams |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        row = summary[model]
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    fmt_num(row["baseline_n_streams_3sigma"], 0),
                    fmt_num(row["n_streams_3sigma_median"], 0),
                    fmt_range(row["n_streams_3sigma_p16"], row["n_streams_3sigma_p84"], 0),
                    fmt_range(row["n_streams_3sigma_min"], row["n_streams_3sigma_max"], 0),
                    fmt_num(row["seven_stream_z_max"], 2),
                    f"{fmt_num(row['seven_stream_p0_min'], 2)}-{fmt_num(row['seven_stream_p0_max'], 2)}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Conclusions", ""])
    for item in output["summary"]["conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--z", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = [
        evaluate_case(rate, comp, threshold, band, args.z)
        for rate in RATE_VARIANTS
        for comp in COMPLETENESS_VARIANTS
        for threshold in THRESHOLD_PROXIES
        for band in MASS_BANDS
    ]
    output = {
        "meta": {
            "forecast_type": "DM abundance/mass Asimov sensitivity grid",
            "z_target": args.z,
            "n_cases": len(cases),
            "stream_counts": list(STREAM_COUNTS),
            "rate_variants": [r.__dict__ for r in RATE_VARIANTS],
            "completeness_variants": [c.__dict__ for c in COMPLETENESS_VARIANTS],
            "threshold_proxies": [t.__dict__ for t in THRESHOLD_PROXIES],
            "mass_bands": [m.__dict__ for m in MASS_BANDS],
            "caveat": (
                "Detector-threshold variants are completeness proxies only; false-positive "
                "contamination is not included."
            ),
        },
        "summary": summarize_cases(cases),
        "cases": cases,
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(finite_json(output), indent=2), encoding="utf-8")
    out_md.write_text(markdown(output), encoding="utf-8")
    print(f"Wrote {out_json} ({len(cases)} cases)")
    print(f"Wrote {out_md}")
    for item in output["summary"]["conclusions"]:
        print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
