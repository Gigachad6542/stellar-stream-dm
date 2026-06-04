#!/usr/bin/env python
"""
DM-model discrimination forecast: abundance (Poisson rate) + detected-mass shape.

This script answers a population question:

    How many clean streams / detected impacts are needed to distinguish CDM from
    WDM/FDM/SIDM, given the detector's measured completeness?

The earlier mass-shape-only estimate discarded the strongest signal: suppressed
models produce fewer detectable impacts.  Here we combine:

    1. a Poisson abundance term for the number of detectable impacts, and
    2. a KL term for the detected-impact mass distribution.

The headline result is an Asimov forecast, not a measured constraint.  To make that
status explicit, the output includes sensitivity scenarios for the encounter-rate
normalization and for the completeness curve.  The default "baseline_floor" scenario
matches the historical report value; "raw_no_floor" exposes how strongly the stream
count depends on the low-rate floor inside compute_encounter_rate().
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

from src.simulation.mass_functions import (  # noqa: E402
    compute_encounter_rate,
    fdm_jeans_mass,
    fdm_suppression,
    wdm_half_mode_mass,
    wdm_suppression,
)


# Measured detector completeness vs log10(M), from detector_completeness.py on the
# corrected detector.  Edge values are extrapolation anchors used only to keep the
# interpolation well behaved.
CL_LOGM = np.array([7.0, 7.65, 7.95, 8.25, 8.55, 9.0])
CL_P = np.array([0.25, 0.478, 0.502, 0.604, 0.723, 0.85])

# Representative GD-1-like stream used for per-stream rate normalization.
STREAM_AGE_GYR = 3.0
STREAM_LEN_KPC = 14.0
STREAM_WIDTH_PC = 50.0
BAND = (7.5, 8.7)
ALPHA = -1.9
RHO_SUB_MSUN_KPC3 = 5.0e3

MODELS: dict[str, tuple[str, dict[str, float]]] = {
    "SIDM": ("SIDM", {}),
    "WDM 6 keV": ("WDM", {"m_wdm_kev": 6.0}),
    "WDM 4 keV": ("WDM", {"m_wdm_kev": 4.0}),
    "WDM 3 keV": ("WDM", {"m_wdm_kev": 3.0}),
    "FDM 1e-21": ("FDM", {"m_axion_ev": 1.0e-21}),
    "FDM 1e-22": ("FDM", {"m_axion_ev": 1.0e-22}),
}


@dataclass(frozen=True)
class Scenario:
    name: str
    rate_floor: bool
    rate_scale: float = 1.0
    completeness_scale: float = 1.0
    note: str = ""


SCENARIOS = [
    Scenario(
        name="baseline_floor",
        rate_floor=True,
        note="Matches the report: compute_encounter_rate floor retained.",
    ),
    Scenario(
        name="raw_no_floor",
        rate_floor=False,
        note="No low-rate floor; exposes normalization sensitivity.",
    ),
    Scenario(
        name="half_rate_floor",
        rate_floor=True,
        rate_scale=0.5,
        note="Same shape, half the encounter normalization.",
    ),
    Scenario(
        name="double_rate_floor",
        rate_floor=True,
        rate_scale=2.0,
        note="Same shape, double the encounter normalization.",
    ),
    Scenario(
        name="low_completeness_floor",
        rate_floor=True,
        completeness_scale=0.8,
        note="Completeness curve scaled down by 20%.",
    ),
    Scenario(
        name="high_completeness_floor",
        rate_floor=True,
        completeness_scale=1.2,
        note="Completeness curve scaled up by 20%, clipped at 1.",
    ),
]


def raw_encounter_rate(
    stream_age_gyr: float,
    stream_length_kpc: float,
    stream_width_pc: float,
    log10_m_min: float,
    log10_m_max: float,
    alpha: float = ALPHA,
    rho_sub_msun_kpc3: float = RHO_SUB_MSUN_KPC3,
) -> float:
    """Same normalization as compute_encounter_rate(), but without its 0.05 floor."""
    m_min = 10.0 ** log10_m_min
    m_max = 10.0 ** log10_m_max
    v_sub_kms = 150.0
    m_ref = math.sqrt(m_min * m_max)
    beta = alpha + 2.0
    if abs(beta) < 1.0e-6:
        n_sub = rho_sub_msun_kpc3 / m_ref * math.log(m_max / m_min)
    else:
        n_sub = rho_sub_msun_kpc3 / (m_ref * beta) * (
            (m_max / m_ref) ** beta - (m_min / m_ref) ** beta
        )

    g_kpc_kms2 = 4.3009e-6
    m_sub = 1.0e7
    r_sub_kpc = (3.0 * m_sub / (4.0 * math.pi * 1.0e8)) ** (1.0 / 3.0)
    v_esc_kms = math.sqrt(2.0 * g_kpc_kms2 * m_sub / r_sub_kpc)
    v_rel = math.sqrt(v_sub_kms**2 + v_esc_kms**2)
    sigma_geo_kpc2 = math.pi * r_sub_kpc**2 * (v_rel / v_sub_kms) ** 2
    v_sub_kpc_per_gyr = v_sub_kms * 1.022
    return float(n_sub * sigma_geo_kpc2 * v_sub_kpc_per_gyr * stream_age_gyr)


def finite_json(value: Any) -> Any:
    """Convert numpy values and infinities to strict JSON-safe objects."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def completeness(logm: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.clip(np.interp(logm, CL_LOGM, CL_P) * scale, 0.0, 1.0)


def suppression(model: str, params: dict[str, float], mass: np.ndarray) -> np.ndarray:
    if model in {"CDM", "SIDM"}:
        return np.ones_like(mass)
    if model == "WDM":
        return wdm_suppression(mass, wdm_half_mode_mass(params["m_wdm_kev"]))
    if model == "FDM":
        return fdm_suppression(mass, fdm_jeans_mass(params["m_axion_ev"]))
    raise ValueError(f"Unknown model: {model}")


def cdm_band_rate(scenario: Scenario) -> tuple[float, float]:
    raw = raw_encounter_rate(
        STREAM_AGE_GYR,
        STREAM_LEN_KPC,
        STREAM_WIDTH_PC,
        BAND[0],
        BAND[1],
        alpha=ALPHA,
        rho_sub_msun_kpc3=RHO_SUB_MSUN_KPC3,
    )
    floored = compute_encounter_rate(
        STREAM_AGE_GYR,
        STREAM_LEN_KPC,
        STREAM_WIDTH_PC,
        BAND[0],
        BAND[1],
        alpha=ALPHA,
        rho_sub_msun_kpc3=RHO_SUB_MSUN_KPC3,
    )
    chosen = floored if scenario.rate_floor else raw
    return raw, float(chosen * scenario.rate_scale)


def model_rate_and_pdf(
    model: str,
    params: dict[str, float],
    logm: np.ndarray,
    scenario: Scenario,
) -> tuple[float, np.ndarray, float]:
    """Return lambda_det, detected-mass PDF over logm, and raw CDM band rate."""
    mass = 10.0**logm
    shape = mass ** (ALPHA + 1.0)  # dN/dlogM
    raw_band_rate, lam_band_cdm = cdm_band_rate(scenario)
    r_cdm = shape / np.trapezoid(shape, logm) * lam_band_cdm
    integrand = r_cdm * suppression(model, params, mass) * completeness(
        logm, scenario.completeness_scale
    )
    lam_det = float(np.trapezoid(integrand, logm))
    pdf = integrand / max(lam_det, 1.0e-30)
    return lam_det, pdf, raw_band_rate


def kl_divergence(p: np.ndarray, q: np.ndarray, logm: np.ndarray) -> float:
    m = (p > 0.0) & (q > 0.0)
    out = np.zeros_like(p, dtype=float)
    out[m] = p[m] * np.log(p[m] / q[m])
    return float(np.trapezoid(out, logm))


def evaluate_scenario(scenario: Scenario, z_target: float = 3.0) -> dict[str, Any]:
    logm = np.linspace(BAND[0], BAND[1], 800)
    lam_cdm, p_cdm, raw_band_rate = model_rate_and_pdf("CDM", {}, logm, scenario)
    rows: dict[str, Any] = {
        "CDM": {
            "lam_det": lam_cdm,
            "expected_detections_7_streams": 7.0 * lam_cdm,
            "p_zero_detections_7_streams": float(np.exp(-7.0 * lam_cdm)),
        }
    }
    for name, (model, params) in MODELS.items():
        lam_t, p_t, _ = model_rate_and_pdf(model, params, logm, scenario)
        kl_mass = kl_divergence(p_t, p_cdm, logm)
        llr_rate = (lam_cdm - lam_t) + (
            lam_t * math.log(lam_t / lam_cdm) if lam_t > 0.0 and lam_cdm > 0.0 else 0.0
        )
        llr_mass = lam_t * kl_mass
        llr_total = llr_rate + llr_mass
        discriminable = llr_total > 1.0e-12
        if discriminable:
            n_streams = z_target**2 / (2.0 * llr_total)
            n_det = n_streams * lam_t
        else:
            n_streams = math.inf
            n_det = math.inf
        rows[name] = {
            "lam_det": lam_t,
            "rate_ratio": lam_t / lam_cdm if lam_cdm > 0.0 else math.inf,
            "kl_mass": kl_mass,
            "llr_rate_per_stream": llr_rate,
            "llr_mass_per_stream": llr_mass,
            "llr_total_per_stream": llr_total,
            "discriminable_from_cdm": discriminable,
            "N_streams_3sig": n_streams,
            "N_det_3sig": n_det,
            "expected_detections_7_streams": 7.0 * lam_t,
            "p_zero_detections_7_streams": float(np.exp(-7.0 * lam_t)),
        }
    return {
        "_scenario": {
            "name": scenario.name,
            "rate_floor": scenario.rate_floor,
            "rate_scale": scenario.rate_scale,
            "completeness_scale": scenario.completeness_scale,
            "raw_cdm_band_rate_pre_completeness": raw_band_rate,
            "chosen_cdm_band_rate_pre_completeness": cdm_band_rate(scenario)[1],
            "note": scenario.note,
        },
        **rows,
    }


def print_summary(primary: dict[str, Any], scenarios: dict[str, dict[str, Any]]) -> None:
    meta = primary["_scenario"]
    lam_cdm = primary["CDM"]["lam_det"]
    print(f"Scenario: {meta['name']} ({meta['note']})")
    print(
        "CDM detectable-impact rate per GD-1-like stream: "
        f"lambda_det = {lam_cdm:.4f}"
    )
    print(
        "Raw/floored pre-completeness CDM band rate: "
        f"{meta['raw_cdm_band_rate_pre_completeness']:.4g} / "
        f"{meta['chosen_cdm_band_rate_pre_completeness']:.4g}"
    )
    print("=" * 96)
    print(
        f"  {'model':10} {'lam_det':>8} {'rate/CDM':>9} {'KL':>8} "
        f"{'LLR_rate':>9} {'LLR_mass':>9} {'N_str(3s)':>10} {'N_det(3s)':>10}"
    )
    for name in MODELS:
        row = primary[name]
        nstr = row["N_streams_3sig"]
        ndet = row["N_det_3sig"]
        print(
            f"  {name:10} {row['lam_det']:8.4f} {row['rate_ratio']:9.3f} "
            f"{row['kl_mass']:8.3f} {row['llr_rate_per_stream']:9.5f} "
            f"{row['llr_mass_per_stream']:9.5f} "
            f"{nstr:10.0f} {ndet:10.1f}"
            if math.isfinite(nstr)
            else f"  {name:10} {row['lam_det']:8.4f} {row['rate_ratio']:9.3f} "
            f"{row['kl_mass']:8.3f} {row['llr_rate_per_stream']:9.5f} "
            f"{row['llr_mass_per_stream']:9.5f} {'inf':>10} {'inf':>10}"
        )
    print("=" * 96)
    print("Sensitivity: N_streams for WDM 3 keV / FDM 1e-22")
    for sname, scen in scenarios.items():
        wdm = scen["WDM 3 keV"]["N_streams_3sig"]
        fdm = scen["FDM 1e-22"]["N_streams_3sig"]
        wdm_s = f"{wdm:.0f}" if math.isfinite(wdm) else "inf"
        fdm_s = f"{fdm:.0f}" if math.isfinite(fdm) else "inf"
        print(f"  {sname:24} WDM3={wdm_s:>8}   FDM1e-22={fdm_s:>8}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/dm/discrimination_forecast.json")
    parser.add_argument("--z", type=float, default=3.0, help="Target sigma level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = {s.name: evaluate_scenario(s, z_target=args.z) for s in SCENARIOS}
    primary = scenarios["baseline_floor"]

    # Backward-compatible top-level keys for figure scripts, plus full scenario audit.
    output = {
        "_meta": {
            "forecast_type": "Asimov Poisson abundance + detected-mass KL",
            "z_target": args.z,
            "mass_band_log10_msun": list(BAND),
            "completeness_log10_msun": CL_LOGM.tolist(),
            "completeness_p_detect": CL_P.tolist(),
            "stream_age_gyr": STREAM_AGE_GYR,
            "stream_length_kpc": STREAM_LEN_KPC,
            "stream_width_pc": STREAM_WIDTH_PC,
            "alpha": ALPHA,
            "strict_json": True,
        },
        **{k: v for k, v in primary.items() if not k.startswith("_")},
        "_scenarios": scenarios,
    }

    print_summary(primary, scenarios)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(finite_json(output), indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
