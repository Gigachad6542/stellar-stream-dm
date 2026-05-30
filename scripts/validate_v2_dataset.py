"""Validate a V2 simulation chunk dataset before GNN training.

Checks the HDF5 handoff contract expected by scripts/train_v2.py:
  - expected chunk and simulation counts
  - required stream_data and label fields
  - DM model balance and M_hm label sanity
  - malformed chunks/groups

Usage:
    python scripts/validate_v2_dataset.py
    python scripts/validate_v2_dataset.py --sim-dir data/simulations_v2_detector_balanced --expected-chunks 200 --expected-sims 20000 --chunk-size 100
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

DM_NAMES = ["CDM", "WDM", "FDM", "SIDM"]
REQUIRED_STREAM_DATA = [
    "phi1", "phi2", "dist", "pm1", "pm2", "vrad",
    "e_dist", "e_pm1", "e_pm2", "e_vrad", "membership_prob",
    "R_cyl", "z_cyl",
]
REQUIRED_LABELS = [
    "dm_model_idx", "log_m_sub_mean", "log10_M_hm",
    "n_subhalos", "log_t_since_last_impact_gyr",
]


def _to_float(value) -> float:
    try:
        return float(value[()])
    except Exception:
        return float(value)


def validate_dataset(
    sim_dir: Path,
    expected_chunks: int,
    expected_sims: int,
    chunk_size: int,
) -> dict:
    files = sorted(sim_dir.glob("chunk_*.h5"))
    chunk_ids = []
    errors: list[str] = []
    warnings: list[str] = []
    dm_counts: Counter[str] = Counter()
    stream_counts: Counter[str] = Counter()
    mhm_by_dm: dict[str, list[float]] = defaultdict(list)
    nstars_min = None
    nstars_max = 0
    total_sims = 0

    for path in files:
        try:
            chunk_ids.append(int(path.stem.split("_")[1]))
        except Exception:
            warnings.append(f"Unexpected chunk filename: {path.name}")

        try:
            with h5py.File(path, "r") as f:
                if "simulations" not in f:
                    errors.append(f"{path.name}: missing /simulations group")
                    continue
                sims = f["simulations"]
                n_in_file = len(sims)
                total_sims += n_in_file
                if n_in_file != chunk_size and path.name != f"chunk_{expected_chunks - 1:05d}.h5":
                    warnings.append(f"{path.name}: expected {chunk_size} sims, found {n_in_file}")

                for run_id in sims:
                    grp = sims[run_id]
                    if "stream_data" not in grp or "labels" not in grp:
                        errors.append(f"{path.name}/{run_id}: missing stream_data or labels")
                        continue

                    sd = grp["stream_data"]
                    labels = grp["labels"]
                    missing_sd = [name for name in REQUIRED_STREAM_DATA if name not in sd]
                    missing_labels = [name for name in REQUIRED_LABELS if name not in labels]
                    if missing_sd:
                        errors.append(f"{path.name}/{run_id}: missing stream_data {missing_sd}")
                    if missing_labels:
                        errors.append(f"{path.name}/{run_id}: missing labels {missing_labels}")
                    if missing_sd or missing_labels:
                        continue

                    nstars = len(sd["phi1"])
                    nstars_min = nstars if nstars_min is None else min(nstars_min, nstars)
                    nstars_max = max(nstars_max, nstars)
                    for name in ["phi2", "membership_prob"]:
                        if len(sd[name]) != nstars:
                            errors.append(f"{path.name}/{run_id}: {name} length mismatch")

                    dm_idx = int(labels["dm_model_idx"][()])
                    dm_name = DM_NAMES[dm_idx] if 0 <= dm_idx < len(DM_NAMES) else f"bad_{dm_idx}"
                    dm_counts[dm_name] += 1
                    stream_name = grp.attrs.get("stream_name", "unknown")
                    if isinstance(stream_name, bytes):
                        stream_name = stream_name.decode()
                    stream_counts[str(stream_name)] += 1

                    mhm = _to_float(labels["log10_M_hm"])
                    if not np.isfinite(mhm):
                        errors.append(f"{path.name}/{run_id}: non-finite log10_M_hm")
                    mhm_by_dm[dm_name].append(mhm)
        except OSError as exc:
            errors.append(f"{path.name}: failed to open ({exc})")

    missing_chunks = sorted(set(range(expected_chunks)) - set(chunk_ids))
    duplicate_chunks = sorted([cid for cid, count in Counter(chunk_ids).items() if count > 1])
    if missing_chunks:
        errors.append(f"Missing chunk ids: {missing_chunks[:20]}{'...' if len(missing_chunks) > 20 else ''}")
    if duplicate_chunks:
        errors.append(f"Duplicate chunk ids: {duplicate_chunks}")
    if len(files) != expected_chunks:
        errors.append(f"Expected {expected_chunks} chunks, found {len(files)}")
    if total_sims != expected_sims:
        errors.append(f"Expected {expected_sims} simulations, found {total_sims}")

    mhm_summary = {}
    for dm_name, values in mhm_by_dm.items():
        arr = np.asarray(values, dtype=float)
        mhm_summary[dm_name] = {
            "min": float(np.min(arr)) if len(arr) else None,
            "median": float(np.median(arr)) if len(arr) else None,
            "max": float(np.max(arr)) if len(arr) else None,
        }

    # Sanity expectations for Plan 2 labels.
    for unsuppressed in ["CDM", "SIDM"]:
        values = np.asarray(mhm_by_dm.get(unsuppressed, []), dtype=float)
        if len(values) and np.nanmedian(values) > 5.0:
            warnings.append(
                f"{unsuppressed}: unsuppressed log10_M_hm labels look like a high-M_hm sentinel "
                f"rather than a CDM-like lower-edge value; range {values.min():.3f}-{values.max():.3f}"
            )
    for suppressed in ["WDM", "FDM"]:
        values = np.asarray(mhm_by_dm.get(suppressed, []), dtype=float)
        if len(values) and np.nanstd(values) < 0.05:
            warnings.append(f"{suppressed}: log10_M_hm appears nearly constant")

    report = {
        "sim_dir": str(sim_dir),
        "chunks": len(files),
        "simulations": total_sims,
        "expected_chunks": expected_chunks,
        "expected_simulations": expected_sims,
        "dm_counts": dict(dm_counts),
        "stream_counts": dict(stream_counts),
        "nstars_min": nstars_min,
        "nstars_max": nstars_max,
        "log10_M_hm": mhm_summary,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate V2 simulation chunks.")
    parser.add_argument("--sim-dir", default="data/simulations_v2_plan2")
    parser.add_argument("--expected-chunks", type=int, default=400)
    parser.add_argument("--expected-sims", type=int, default=100000)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--output", default="outputs/validation/v2_dataset_integrity.json")
    args = parser.parse_args()

    report = validate_dataset(
        Path(args.sim_dir),
        expected_chunks=args.expected_chunks,
        expected_sims=args.expected_sims,
        chunk_size=args.chunk_size,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
