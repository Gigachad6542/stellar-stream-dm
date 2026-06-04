#!/usr/bin/env python
"""Summarize what the current seven-stream null result can and cannot say."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


FORECAST = Path("outputs/dm/discrimination_forecast.json")
OUT_JSON = Path("outputs/dm/dm_null_power_table.json")
OUT_MD = Path("outputs/dm/dm_null_power_table.md")
N_STREAMS_CURRENT = 7
ORDER = ["CDM", "WDM 6 keV", "WDM 4 keV", "WDM 3 keV", "FDM 1e-21", "FDM 1e-22", "SIDM"]


def z7(row: dict[str, Any]) -> float | None:
    llr = row.get("llr_total_per_stream")
    if llr is None:
        return None
    return math.sqrt(max(0.0, 2.0 * N_STREAMS_CURRENT * float(llr)))


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_rows(forecast: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = forecast["_scenarios"]
    rows = []
    for model in ORDER:
        if model not in forecast:
            continue
        base = forecast[model]
        raw = scenarios["raw_no_floor"][model]
        expected = float(base["expected_detections_7_streams"])
        p_zero = float(base["p_zero_detections_7_streams"])
        raw_expected = float(raw["expected_detections_7_streams"])
        raw_p_zero = float(raw["p_zero_detections_7_streams"])
        rows.append(
            {
                "model": model,
                "lambda_det_per_stream_baseline": float(base["lam_det"]),
                "expected_detections_7_streams_baseline": expected,
                "p_zero_detections_7_streams_baseline": p_zero,
                "p_at_least_one_detection_7_streams_baseline": 1.0 - p_zero,
                "expected_detections_7_streams_raw_no_floor": raw_expected,
                "p_zero_detections_7_streams_raw_no_floor": raw_p_zero,
                "z_7_streams_vs_cdm": z7(base),
                "interpretation": interpretation(model, expected, p_zero, z7(base)),
            }
        )
    return rows


def interpretation(model: str, expected: float, p_zero: float, z: float | None) -> str:
    if model == "CDM":
        return "Null is expected; seven streams have little detection power."
    if model == "SIDM":
        return "Same abundance as CDM here; zero-count result does not test SIDM morphology."
    if z is not None and z < 1.0:
        return "Current seven-stream count/mass sample is far below discriminating power."
    if expected < 1.0 and p_zero > 0.5:
        return "Zero detections remain probable; sample is underpowered."
    return "Some separation forecast exists, but not with seven streams."


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | E[N_det] baseline | P(0 det.) baseline | P(>=1 det.) baseline | P(0 det.) raw no-floor | Seven-stream Z vs CDM | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    fmt(row["expected_detections_7_streams_baseline"], 2),
                    fmt(row["p_zero_detections_7_streams_baseline"], 2),
                    fmt(row["p_at_least_one_detection_7_streams_baseline"], 2),
                    fmt(row["p_zero_detections_7_streams_raw_no_floor"], 2),
                    fmt(row["z_7_streams_vs_cdm"], 2),
                    row["interpretation"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    forecast = json.loads(FORECAST.read_text(encoding="utf-8"))
    rows = build_rows(forecast)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    OUT_MD.write_text(markdown(rows), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
