"""Derive timeline-aware impact labels from existing V2 HDF5 simulations.

The V2 simulation chunks already store per-encounter subhalo parameters,
including ``t_since_impact_gyr``. This script turns those raw encounter
timelines into training targets without rerunning the simulations.

By default the script is read-only and writes a JSON audit. Pass
``--write-in-place`` only after inspecting the audit.

New labels are stored as HDF5 label attributes:

* impact_strength: maximum per-encounter detectability score, kept for
  compatibility with existing V2 training code.
* n_detectable_impacts: count of encounters above the detectability threshold.
* effective_n_impacts: soft count, sum of clipped strength / threshold.
* strongest_impact_t_since_gyr: age of the strongest encounter.
* log_t_since_strongest_impact_gyr: log10 age of the strongest encounter.
* timeline_* arrays: binned counts/strengths over impact age bins.

Example:
    python scripts/compute_timeline_impact_labels.py \
        --sim-dir data/simulations_v2_detector_balanced \
        --out outputs/diagnostics/timeline_impact_label_audit.json

    python scripts/compute_timeline_impact_labels.py \
        --sim-dir data/simulations_v2_detector_balanced \
        --write-in-place
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DM_NAMES = {0: "CDM", 1: "WDM", 2: "FDM", 3: "SIDM"}
DEFAULT_BINS = (0.0, 1.0, 3.0, 6.0, 8.0, math.inf)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode(errors="replace")
    return str(value)


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and nonfinite floats for strict JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {}
    qs = np.quantile(arr, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "min": float(qs[0]),
        "q01": float(qs[1]),
        "q05": float(qs[2]),
        "q25": float(qs[3]),
        "q50": float(qs[4]),
        "q75": float(qs[5]),
        "q95": float(qs[6]),
        "q99": float(qs[7]),
        "max": float(qs[8]),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 5:
        return None
    if float(np.std(x_arr[mask])) == 0.0 or float(np.std(y_arr[mask])) == 0.0:
        return None
    return float(np.corrcoef(x_arr[mask], y_arr[mask])[0, 1])


def per_encounter_strength(
    mass_solar: np.ndarray,
    impact_param_kpc: np.ndarray,
    flyby_vel_kms: np.ndarray,
    t_since_gyr: np.ndarray,
) -> np.ndarray:
    """Compute per-encounter detectability scores.

    This mirrors ``scripts/compute_impact_strength.py`` but returns one score
    per encounter instead of only the maximum. The time term grows as sqrt(t)
    and saturates at old ages where phase mixing makes gaps less informative.
    """
    mass_solar = np.asarray(mass_solar, dtype=np.float64)
    impact_param_kpc = np.asarray(impact_param_kpc, dtype=np.float64)
    flyby_vel_kms = np.asarray(flyby_vel_kms, dtype=np.float64)
    t_since_gyr = np.asarray(t_since_gyr, dtype=np.float64)

    if len(mass_solar) == 0:
        return np.asarray([], dtype=np.float64)

    n = min(len(mass_solar), len(impact_param_kpc), len(flyby_vel_kms), len(t_since_gyr))
    if n == 0:
        return np.asarray([], dtype=np.float64)

    mass_solar = np.clip(mass_solar[:n], 1.0, None)
    impact_param_kpc = np.clip(impact_param_kpc[:n], 0.01, None)
    flyby_vel_kms = np.clip(flyby_vel_kms[:n], 1.0, None)
    t_since_gyr = np.clip(t_since_gyr[:n], 0.1, 8.0)

    m_norm = mass_solar / 1e7
    b_norm = impact_param_kpc / 0.15
    v_norm = flyby_vel_kms / 200.0
    t_factor = np.sqrt(t_since_gyr / 5.0)

    scores = (m_norm ** (1.0 / 3.0)) * t_factor / (b_norm * v_norm)
    scores[~np.isfinite(scores)] = 0.0
    return scores.astype(np.float64)


def _bin_name(lo: float, hi: float) -> str:
    def fmt(v: float) -> str:
        if not math.isfinite(v):
            return "inf"
        if float(v).is_integer():
            return str(int(v))
        return str(v).replace(".", "p")

    return f"t_{fmt(lo)}_{fmt(hi)}_gyr"


def timeline_features(
    masses: np.ndarray,
    impact_params: np.ndarray,
    flyby_vels: np.ndarray,
    t_since: np.ndarray,
    threshold: float,
    bins: tuple[float, ...],
) -> dict[str, Any]:
    """Return scalar and binned timeline features for one simulation."""
    scores = per_encounter_strength(masses, impact_params, flyby_vels, t_since)
    t_since = np.asarray(t_since, dtype=np.float64)[: len(scores)]
    finite = np.isfinite(scores) & np.isfinite(t_since)
    scores = scores[finite]
    t_since = t_since[finite]

    n_encounters = int(len(scores))
    detectable = scores > threshold if threshold > 0 else scores > 0
    n_detectable = int(np.sum(detectable))
    if threshold > 0:
        effective = float(np.sum(np.clip(scores / threshold, 0.0, 1.0)))
    else:
        effective = float(np.sum(scores > 0))

    if n_encounters:
        strongest_idx = int(np.argmax(scores))
        max_strength = float(scores[strongest_idx])
        strongest_t = float(t_since[strongest_idx])
        most_recent_t = float(np.min(t_since))
        oldest_t = float(np.max(t_since))
        mean_t = float(np.mean(t_since))
    else:
        max_strength = 0.0
        strongest_t = math.nan
        most_recent_t = math.nan
        oldest_t = math.nan
        mean_t = math.nan

    n_bins = len(bins) - 1
    timeline_n = np.zeros(n_bins, dtype=np.float32)
    timeline_detectable = np.zeros(n_bins, dtype=np.float32)
    timeline_effective = np.zeros(n_bins, dtype=np.float32)
    timeline_max_strength = np.zeros(n_bins, dtype=np.float32)

    scalar_bins: dict[str, float] = {}
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if math.isinf(hi):
            in_bin = t_since >= lo
        else:
            in_bin = (t_since >= lo) & (t_since < hi)
        bin_scores = scores[in_bin]
        timeline_n[i] = float(np.sum(in_bin))
        timeline_detectable[i] = float(np.sum(bin_scores > threshold)) if threshold > 0 else float(np.sum(bin_scores > 0))
        if threshold > 0:
            timeline_effective[i] = float(np.sum(np.clip(bin_scores / threshold, 0.0, 1.0)))
        else:
            timeline_effective[i] = float(np.sum(bin_scores > 0))
        timeline_max_strength[i] = float(np.max(bin_scores)) if len(bin_scores) else 0.0

        name = _bin_name(lo, hi)
        scalar_bins[f"n_impacts_{name}"] = float(timeline_n[i])
        scalar_bins[f"n_detectable_{name}"] = float(timeline_detectable[i])
        scalar_bins[f"effective_impacts_{name}"] = float(timeline_effective[i])
        scalar_bins[f"max_strength_{name}"] = float(timeline_max_strength[i])

    out: dict[str, Any] = {
        "impact_timeline_version": "v1",
        "impact_strength": max_strength,
        "impact_strength_max": max_strength,
        "impact_strong": int(max_strength > threshold),
        "impact_strength_threshold": float(threshold),
        "n_encounters_timeline": n_encounters,
        "n_detectable_impacts": n_detectable,
        "effective_n_impacts": effective,
        "strongest_impact_t_since_gyr": strongest_t,
        "log_t_since_strongest_impact_gyr": float(np.log10(strongest_t)) if math.isfinite(strongest_t) and strongest_t > 0 else math.nan,
        "most_recent_impact_t_since_gyr": most_recent_t,
        "oldest_impact_t_since_gyr": oldest_t,
        "mean_impact_t_since_gyr": mean_t,
        "timeline_bins_gyr": np.asarray(bins, dtype=np.float32),
        "timeline_n_impacts": timeline_n,
        "timeline_n_detectable_impacts": timeline_detectable,
        "timeline_effective_impacts": timeline_effective,
        "timeline_max_strength": timeline_max_strength,
    }
    out.update(scalar_bins)
    return out


def _read_encounters(grp: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "subhalos" not in grp:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty
    sub = grp["subhalos"]
    required = ("mass", "impact_param", "flyby_vel", "t_since_impact_gyr")
    if any(name not in sub for name in required):
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty
    return (
        np.asarray(sub["mass"], dtype=np.float64),
        np.asarray(sub["impact_param"], dtype=np.float64),
        np.asarray(sub["flyby_vel"], dtype=np.float64),
        np.asarray(sub["t_since_impact_gyr"], dtype=np.float64),
    )


def _write_attrs(labels: h5py.Group, features: dict[str, Any]) -> None:
    for key, value in features.items():
        if isinstance(value, str):
            labels.attrs[key] = np.bytes_(value)
        elif isinstance(value, np.ndarray):
            labels.attrs[key] = value
        else:
            labels.attrs[key] = value


def process_chunk(path: Path, threshold: float, bins: tuple[float, ...], dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mode = "r" if dry_run else "r+"
    with h5py.File(path, mode) as f:
        if "simulations" not in f:
            return rows
        for run_id in f["simulations"].keys():
            grp = f["simulations"][run_id]
            labels = grp["labels"]
            masses, bkpc, vkms, t_since = _read_encounters(grp)
            features = timeline_features(masses, bkpc, vkms, t_since, threshold, bins)

            dm_idx = int(labels["dm_model_idx"][()])
            stream_name = _decode(grp.attrs.get("stream_name", "unknown"))
            raw_n = _safe_float(labels.attrs.get("n_subhalos", labels["n_subhalos"][()]), 0.0)
            stored_strength = _safe_float(labels.attrs.get("impact_strength", math.nan), math.nan)
            stored_threshold = _safe_float(labels.attrs.get("impact_strength_threshold", math.nan), math.nan)

            row = {
                "path": str(path),
                "run_id": run_id,
                "dm_idx": dm_idx,
                "dm_model": DM_NAMES.get(dm_idx, str(dm_idx)),
                "stream": stream_name,
                "n_subhalos": raw_n,
                "stored_impact_strength": stored_strength,
                "stored_impact_strength_threshold": stored_threshold,
                "impact_strength": float(features["impact_strength"]),
                "impact_strong": int(features["impact_strong"]),
                "n_detectable_impacts": int(features["n_detectable_impacts"]),
                "effective_n_impacts": float(features["effective_n_impacts"]),
                "strongest_impact_t_since_gyr": _safe_float(features["strongest_impact_t_since_gyr"]),
                "most_recent_impact_t_since_gyr": _safe_float(features["most_recent_impact_t_since_gyr"]),
                "oldest_impact_t_since_gyr": _safe_float(features["oldest_impact_t_since_gyr"]),
            }
            for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
                name = _bin_name(lo, hi)
                row[f"timeline_n_{name}"] = float(features["timeline_n_impacts"][i])
                row[f"timeline_detectable_{name}"] = float(features["timeline_n_detectable_impacts"][i])
                row[f"timeline_effective_{name}"] = float(features["timeline_effective_impacts"][i])
                row[f"timeline_max_strength_{name}"] = float(features["timeline_max_strength"][i])
            rows.append(row)

            if not dry_run:
                _write_attrs(labels, features)
    return rows


def _counts_by(rows: list[dict[str, Any]], key: str, mask: np.ndarray | None = None) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if mask is None:
        for row in rows:
            counts[str(row[key])] += 1
    else:
        for row, keep in zip(rows, mask):
            if bool(keep):
                counts[str(row[key])] += 1
    return dict(sorted(counts.items()))


def build_summary(rows: list[dict[str, Any]], threshold: float, bins: tuple[float, ...], dry_run: bool, sim_dir: Path) -> dict[str, Any]:
    raw_n = np.asarray([r["n_subhalos"] for r in rows], dtype=np.float64)
    strength = np.asarray([r["impact_strength"] for r in rows], dtype=np.float64)
    stored_strength = np.asarray([r["stored_impact_strength"] for r in rows], dtype=np.float64)
    n_detectable = np.asarray([r["n_detectable_impacts"] for r in rows], dtype=np.float64)
    effective_n = np.asarray([r["effective_n_impacts"] for r in rows], dtype=np.float64)
    strongest_t = np.asarray([r["strongest_impact_t_since_gyr"] for r in rows], dtype=np.float64)
    most_recent_t = np.asarray([r["most_recent_impact_t_since_gyr"] for r in rows], dtype=np.float64)
    impacted = raw_n > 0
    detectable = n_detectable > 0

    bin_summary: dict[str, Any] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        name = _bin_name(lo, hi)
        bin_summary[name] = {
            "total_impacts": float(np.sum([r[f"timeline_n_{name}"] for r in rows])),
            "total_detectable_impacts": float(np.sum([r[f"timeline_detectable_{name}"] for r in rows])),
            "total_effective_impacts": float(np.sum([r[f"timeline_effective_{name}"] for r in rows])),
            "max_strength_quantiles": _quantiles([r[f"timeline_max_strength_{name}"] for r in rows]),
        }

    by_model: dict[str, Any] = {}
    for model in sorted({r["dm_model"] for r in rows}):
        model_mask = np.asarray([r["dm_model"] == model for r in rows], dtype=bool)
        by_model[model] = {
            "n": int(np.sum(model_mask)),
            "raw_impact_fraction": float(np.mean(raw_n[model_mask] > 0)) if np.any(model_mask) else None,
            "detectable_fraction": float(np.mean(n_detectable[model_mask] > 0)) if np.any(model_mask) else None,
            "n_subhalos": _quantiles(raw_n[model_mask]),
            "n_detectable_impacts": _quantiles(n_detectable[model_mask]),
            "effective_n_impacts": _quantiles(effective_n[model_mask]),
            "impact_strength": _quantiles(strength[model_mask]),
        }

    strength_delta = stored_strength - strength
    finite_stored = np.isfinite(stored_strength)
    return {
        "sim_dir": str(sim_dir),
        "dry_run": bool(dry_run),
        "threshold": float(threshold),
        "bins_gyr": list(bins),
        "n_total": len(rows),
        "counts_by_dm": _counts_by(rows, "dm_model"),
        "counts_by_stream": _counts_by(rows, "stream"),
        "raw_impacted_count": int(np.sum(impacted)),
        "raw_impacted_fraction": float(np.mean(impacted)) if len(rows) else 0.0,
        "detectable_count": int(np.sum(detectable)),
        "detectable_fraction": float(np.mean(detectable)) if len(rows) else 0.0,
        "n_subhalos": _quantiles(raw_n),
        "n_detectable_impacts": _quantiles(n_detectable),
        "effective_n_impacts": _quantiles(effective_n),
        "impact_strength": _quantiles(strength),
        "strongest_impact_t_since_gyr_impacted_only": _quantiles(strongest_t[np.isfinite(strongest_t)]),
        "most_recent_impact_t_since_gyr_impacted_only": _quantiles(most_recent_t[np.isfinite(most_recent_t)]),
        "correlations": {
            "raw_n_vs_n_detectable": _corr(raw_n, n_detectable),
            "raw_n_vs_effective_n": _corr(raw_n, effective_n),
            "raw_n_vs_impact_strength": _corr(raw_n, strength),
            "n_detectable_vs_effective_n": _corr(n_detectable, effective_n),
            "strongest_t_vs_impact_strength": _corr(strongest_t, strength),
            "most_recent_t_vs_impact_strength": _corr(most_recent_t, strength),
        },
        "stored_impact_strength_comparison": {
            "n_with_stored_strength": int(np.sum(finite_stored)),
            "max_abs_delta": float(np.nanmax(np.abs(strength_delta[finite_stored]))) if np.any(finite_stored) else None,
            "delta_quantiles": _quantiles(strength_delta[finite_stored]) if np.any(finite_stored) else {},
        },
        "timeline_bins": bin_summary,
        "by_model": by_model,
        "recommended_next_targets": {
            "binary": "n_detectable_impacts > 0",
            "count_regression": "effective_n_impacts or n_detectable_impacts, not raw n_subhalos",
            "time_regression": "log_t_since_strongest_impact_gyr for sims with detectable impacts",
            "timeline_auxiliary": "timeline_effective_impacts over age bins",
        },
    }


def parse_bins(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for part in raw.split(","):
        token = part.strip().lower()
        if token in {"inf", "infinity", "np.inf"}:
            values.append(math.inf)
        else:
            values.append(float(token))
    if len(values) < 2:
        raise ValueError("Need at least two bin edges")
    if any(b <= a for a, b in zip(values[:-1], values[1:])):
        raise ValueError(f"Bins must be strictly increasing: {values}")
    return tuple(values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    sim_dir = Path(args.sim_dir)
    chunks = sorted(sim_dir.glob("**/chunk_*.h5"))
    if args.sample_chunks > 0:
        chunks = chunks[: args.sample_chunks]
    if not chunks:
        raise RuntimeError(f"No chunk_*.h5 files found under {sim_dir}")

    dry_run = not args.write_in_place
    bins = parse_bins(args.bins)

    if dry_run:
        log.info("Read-only mode. Use --write-in-place to store timeline labels.")
    else:
        log.warning("Writing timeline impact labels in place under %s", sim_dir)
    log.info("Processing %d chunks, threshold=%.4f, bins=%s", len(chunks), args.threshold, bins)

    rows: list[dict[str, Any]] = []
    for idx, path in enumerate(chunks, start=1):
        rows.extend(process_chunk(path, args.threshold, bins, dry_run=dry_run))
        if idx % args.log_every == 0 or idx == len(chunks):
            log.info("Processed %d/%d chunks (%d sims)", idx, len(chunks), len(rows))

    return build_summary(rows, args.threshold, bins, dry_run, sim_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute timeline-aware impact labels for V2 HDF5 simulations.")
    parser.add_argument("--sim-dir", required=True, help="Directory containing chunk_*.h5 files.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detectable-impact strength threshold.")
    parser.add_argument("--bins", default="0,1,3,6,8,inf", help="Comma-separated impact-age bin edges in Gyr.")
    parser.add_argument("--write-in-place", action="store_true", help="Write labels into HDF5 attrs. Default is audit-only.")
    parser.add_argument("--sample-chunks", type=int, default=0, help="Only process the first N chunks.")
    parser.add_argument("--log-every", type=int, default=50, help="Progress logging cadence in chunks.")
    parser.add_argument("--out", default="outputs/diagnostics/timeline_impact_label_audit.json")
    args = parser.parse_args()

    summary = run(args)
    text = json.dumps(_jsonable(summary), indent=2, allow_nan=False)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
