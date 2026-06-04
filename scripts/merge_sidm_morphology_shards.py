#!/usr/bin/env python
"""Merge SIDM morphology shard JSON files into the standard report artifact."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Shard JSON files or glob patterns.")
    parser.add_argument("--primary-core-factor", type=float, default=3.0)
    parser.add_argument("--out", default="outputs/dm/sidm_morphology.json")
    return parser.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def main() -> int:
    args = parse_args()
    paths = expand_inputs(args.inputs)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard files: {missing}")

    details: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        details.extend(obj.get("details", []))
        metadata.append(
            {
                "path": str(path),
                "core_factors": obj.get("core_factors", []),
                "logm_grid": obj.get("logm_grid", []),
                "trials": obj.get("trials"),
                "n_stars": obj.get("n_stars"),
            }
        )

    if not details:
        raise ValueError("No detail rows found in shard inputs.")

    details.sort(key=lambda r: (float(r["core_factor"]), float(r["logM"])))
    core_factors = sorted({float(r["core_factor"]) for r in details})
    logm_grid = sorted({float(r["logM"]) for r in details})
    primary = args.primary_core_factor
    if primary not in core_factors:
        raise ValueError(f"Primary core {primary} not in merged core factors {core_factors}")
    primary_rows = [r for r in details if float(r["core_factor"]) == primary]
    mean_auc = sum(float(r["auc_cdm_deeper_than_sidm"]) for r in primary_rows) / len(primary_rows)
    rows = [
        [
            float(r["logM"]),
            float(r["median_gapdepth_cdm"]),
            float(r["median_gapdepth_sidm"]),
            float(r["auc_cdm_deeper_than_sidm"]),
        ]
        for r in primary_rows
    ]
    out = {
        "core_factor": primary,
        "core_factors": core_factors,
        "logm_grid": logm_grid,
        "trials": sorted({int(r.get("n_trials", 0)) for r in details}),
        "n_stars": sorted({int(r.get("n_stars", 0)) for r in details}),
        "rows": rows,
        "mean_auc": mean_auc,
        "details": details,
        "shards": metadata,
        "note": "Merged SIDM morphology shard grid; rows/mean_auc use primary core factor for report compatibility.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Merged {len(paths)} shards / {len(details)} rows -> {out_path}")
    print(f"Primary core {primary:g}x mean depth AUC = {mean_auc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
