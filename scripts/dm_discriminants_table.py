#!/usr/bin/env python
"""Build a unified DM-model discriminants table from forecast artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FORECAST = Path("outputs/dm/discrimination_forecast.json")
SIDM = Path("outputs/dm/sidm_morphology.json")
OUT_JSON = Path("outputs/dm/dm_discriminants_table.json")
OUT_MD = Path("outputs/dm/dm_discriminants_table.md")


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "not separable"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any) -> str:
    if value is None:
        return "not separable"
    try:
        return f"{round(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def scenario_streams(forecast: dict[str, Any], model: str, scenario: str) -> str:
    try:
        return fmt_int(forecast["_scenarios"][scenario][model]["N_streams_3sig"])
    except KeyError:
        return "n/a"


def summarize_sidm(sidm: dict[str, Any]) -> tuple[str, str]:
    details = sidm.get("details", [])
    if not details:
        return f"AUC {fmt_float(sidm.get('mean_auc'))}", "single core-factor grid"

    by_core: dict[float, list[float]] = {}
    for row in details:
        by_core.setdefault(float(row["core_factor"]), []).append(
            float(row["auc_cdm_deeper_than_sidm"])
        )
    pieces = []
    for core in sorted(by_core):
        vals = by_core[core]
        pieces.append(f"{core:g}x:{sum(vals) / len(vals):.2f}")
    primary = sidm.get("core_factor", 3.0)
    return (
        "mean depth-AUC by core " + ", ".join(pieces),
        f"primary plotted core={primary:g}x; low-trial morphology grid",
    )


def build_rows(forecast: dict[str, Any], sidm: dict[str, Any]) -> list[dict[str, str]]:
    rows = [
        {
            "model": "CDM",
            "discriminant": "reference abundance and NFW/cuspy gap morphology",
            "forecast_metric": (
                "lambda_det="
                f"{forecast['CDM']['lam_det']:.4f}/stream; "
                f"E[N_det,7 streams]={forecast['CDM']['expected_detections_7_streams']:.2f}; "
                f"P(0 detections)={forecast['CDM']['p_zero_detections_7_streams']:.2f}"
            ),
            "current_data_status": "seven-stream null is unsurprising under CDM",
            "main_caveat": "rate normalization remains a first-order systematic",
        }
    ]

    for model in ["WDM 6 keV", "WDM 4 keV", "WDM 3 keV", "FDM 1e-21", "FDM 1e-22"]:
        row = forecast[model]
        rows.append(
            {
                "model": model,
                "discriminant": "suppressed detectable-impact abundance + detected-mass distribution",
                "forecast_metric": (
                    f"rate/CDM={row['rate_ratio']:.2f}; "
                    f"Ndet_3sigma={fmt_float(row['N_det_3sig'], 1)}; "
                    f"Nstreams_3sigma={fmt_int(row['N_streams_3sig'])} baseline, "
                    f"{scenario_streams(forecast, model, 'raw_no_floor')} raw-no-floor"
                ),
                "current_data_status": "current seven-stream sample is underpowered",
                "main_caveat": "stream-count forecast is normalization-sensitive",
            }
        )

    sidm_metric, sidm_caveat = summarize_sidm(sidm)
    rows.append(
        {
            "model": "SIDM",
            "discriminant": "no abundance/mass-spectrum signal here; uses shallower/broader cored-gap morphology",
            "forecast_metric": sidm_metric,
            "current_data_status": "requires detected impacts and morphology calibration",
            "main_caveat": sidm_caveat,
        }
    )
    return rows


def markdown(rows: list[dict[str, str]]) -> str:
    headers = ["Model", "Discriminant vs CDM", "Forecast / validation metric", "Current-data status", "Main caveat"]
    keys = ["model", "discriminant", "forecast_metric", "current_data_status", "main_caveat"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        vals = [row[k].replace("|", "/") for k in keys]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    forecast = json.loads(FORECAST.read_text(encoding="utf-8"))
    sidm = json.loads(SIDM.read_text(encoding="utf-8"))
    rows = build_rows(forecast, sidm)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    OUT_MD.write_text(markdown(rows), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
