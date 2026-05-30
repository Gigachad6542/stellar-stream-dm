"""Audit whether V2 impact labels are learnable before another expensive GNN run.

This script separates three questions that were previously blurred together:

1. Are the labels balanced enough for training?
2. Do impact-strength thresholds create physically sensible positive classes?
3. Can cheap summary-feature baselines see the target at all?

If simple profile baselines cannot reach useful ROC-AUC on a target, a larger
GNN run is unlikely to fix the problem by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.diagnose_v2_signal import _summary_features

DM_NAMES = {0: "CDM", 1: "WDM", 2: "FDM", 3: "SIDM"}


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _quantiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    qs = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "min": float(values.min()),
        "q01": float(qs[0]),
        "q05": float(qs[1]),
        "q25": float(qs[2]),
        "q50": float(qs[3]),
        "q75": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _scan_metadata(sim_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(sim_dir.glob("**/chunk_*.h5")):
        with h5py.File(path, "r") as f:
            if "simulations" not in f:
                continue
            for run_id in f["simulations"].keys():
                grp = f["simulations"][run_id]
                labels = grp["labels"]
                dm_idx = int(labels["dm_model_idx"][()])
                stream_name = grp.attrs.get("stream_name", b"unknown")
                if isinstance(stream_name, bytes):
                    stream_name = stream_name.decode()
                n_sub = _safe_float(labels.attrs.get("n_subhalos", labels["n_subhalos"][()]))
                impact_strength = _safe_float(labels.attrs.get("impact_strength", 0.0), 0.0)

                max_log_mass = np.nan
                min_impact_param = np.nan
                min_flyby_vel = np.nan
                max_t_since = np.nan
                if n_sub > 0 and "subhalos" in grp:
                    sub = grp["subhalos"]
                    if "mass" in sub and len(sub["mass"]):
                        masses = np.asarray(sub["mass"], dtype=np.float64)
                        max_log_mass = float(np.log10(np.nanmax(masses)))
                    if "impact_param" in sub and len(sub["impact_param"]):
                        min_impact_param = float(np.nanmin(np.asarray(sub["impact_param"], dtype=np.float64)))
                    if "flyby_vel" in sub and len(sub["flyby_vel"]):
                        min_flyby_vel = float(np.nanmin(np.asarray(sub["flyby_vel"], dtype=np.float64)))
                    if "t_since_impact_gyr" in sub and len(sub["t_since_impact_gyr"]):
                        max_t_since = float(np.nanmax(np.asarray(sub["t_since_impact_gyr"], dtype=np.float64)))

                rows.append({
                    "path": str(path),
                    "run_id": run_id,
                    "dm_idx": dm_idx,
                    "dm_model": DM_NAMES.get(dm_idx, str(dm_idx)),
                    "stream": str(stream_name),
                    "n_subhalos": float(n_sub),
                    "impact_strength": float(impact_strength),
                    "log10_M_hm": _safe_float(labels.attrs.get("log10_M_hm", labels["log10_M_hm"][()] if "log10_M_hm" in labels else np.nan)),
                    "baryonic_applied": int(labels.attrs.get("baryonic_applied", 0)),
                    "contamination_fraction": _safe_float(labels.attrs.get("contamination_fraction", np.nan)),
                    "noise_scale_factor": _safe_float(labels.attrs.get("noise_scale_factor", np.nan)),
                    "membership_corruption_fraction": _safe_float(labels.attrs.get("membership_corruption_fraction", np.nan)),
                    "max_log_mass": max_log_mass,
                    "min_impact_param": min_impact_param,
                    "min_flyby_vel": min_flyby_vel,
                    "max_t_since": max_t_since,
                })
    return rows


def _counts_by(rows: list[dict], key: str, mask: np.ndarray | None = None) -> dict:
    out: dict[str, int] = defaultdict(int)
    if mask is None:
        iterable = rows
    else:
        iterable = [row for row, keep in zip(rows, mask) if keep]
    for row in iterable:
        out[str(row[key])] += 1
    return dict(sorted(out.items()))


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _load_feature_sample(rows: list[dict], sample_size: int, seed: int) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    if sample_size <= 0 or sample_size >= len(rows):
        selected = list(rows)
    else:
        idx = rng.choice(len(rows), size=sample_size, replace=False)
        selected = [rows[int(i)] for i in idx]

    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_file[row["path"]].append(row)

    features: list[np.ndarray] = []
    ordered_rows: list[dict] = []
    for path, items in by_file.items():
        with h5py.File(path, "r") as f:
            for row in items:
                grp = f["simulations"][row["run_id"]]
                features.append(_summary_features(grp["stream_data"]))
                ordered_rows.append(row)

    return np.vstack(features), ordered_rows


def _baseline_metrics(X: np.ndarray, rows: list[dict], threshold: float, seed: int) -> dict:
    y = np.asarray([row["impact_strength"] > threshold for row in rows], dtype=np.int64)
    counts = {"0": int((y == 0).sum()), "1": int((y == 1).sum())}
    if min(counts.values()) < 50:
        return {"skipped": True, "reason": "too_few_examples", "counts": counts}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    models = {
        "logreg": make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "extratrees": make_pipeline(
            SimpleImputer(),
            ExtraTreesClassifier(
                n_estimators=180,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=seed,
                class_weight="balanced",
            ),
        ),
    }

    out = {"skipped": False, "counts": counts, "models": {}}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out["models"][name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "confusion_matrix": cm.tolist(),
        }
    return out


def run(args: argparse.Namespace) -> dict:
    sim_dir = Path(args.sim_dir)
    rows = _scan_metadata(sim_dir)
    if not rows:
        raise RuntimeError(f"No chunk_*.h5 simulations found under {sim_dir}")

    strength = np.asarray([row["impact_strength"] for row in rows], dtype=np.float64)
    impacted = np.asarray([row["n_subhalos"] > 0 for row in rows], dtype=bool)
    n_sub = np.asarray([row["n_subhalos"] for row in rows], dtype=np.float64)

    thresholds = list(args.thresholds)
    impacted_strength = strength[impacted & np.isfinite(strength) & (strength > 0)]
    if len(impacted_strength):
        thresholds.extend(np.quantile(impacted_strength, [0.25, 0.33, 0.5, 0.66, 0.75]).tolist())
    thresholds = sorted({round(float(t), 6) for t in thresholds if np.isfinite(t)})

    summary = {
        "sim_dir": str(sim_dir),
        "n_total": len(rows),
        "counts_by_dm": _counts_by(rows, "dm_model"),
        "counts_by_stream": _counts_by(rows, "stream"),
        "n_impacted": int(impacted.sum()),
        "impact_fraction": float(impacted.mean()),
        "impact_strength_all": _quantiles(strength),
        "impact_strength_impacted": _quantiles(impacted_strength),
        "n_subhalos": _quantiles(n_sub),
        "metadata_correlations_with_strength": {
            "n_subhalos": _corr(n_sub, strength),
            "max_log_mass": _corr(np.asarray([r["max_log_mass"] for r in rows], dtype=float), strength),
            "min_impact_param": _corr(np.asarray([r["min_impact_param"] for r in rows], dtype=float), strength),
            "min_flyby_vel": _corr(np.asarray([r["min_flyby_vel"] for r in rows], dtype=float), strength),
            "max_t_since": _corr(np.asarray([r["max_t_since"] for r in rows], dtype=float), strength),
            "noise_scale_factor": _corr(np.asarray([r["noise_scale_factor"] for r in rows], dtype=float), strength),
            "contamination_fraction": _corr(np.asarray([r["contamination_fraction"] for r in rows], dtype=float), strength),
        },
        "thresholds": {},
    }

    X = None
    feature_rows: list[dict] = []
    if args.feature_sample_size:
        X, feature_rows = _load_feature_sample(rows, args.feature_sample_size, args.seed)
        summary["feature_sample_size"] = len(feature_rows)

    for threshold in thresholds:
        y = strength > threshold
        pos_rows = [row for row, keep in zip(rows, y) if keep]
        neg_rows = [row for row, keep in zip(rows, y) if not keep]
        entry = {
            "positive_count": int(y.sum()),
            "positive_fraction": float(y.mean()),
            "positive_by_dm": _counts_by(rows, "dm_model", y),
            "positive_by_stream": _counts_by(rows, "stream", y),
            "mean_n_sub_positive": float(np.mean([r["n_subhalos"] for r in pos_rows])) if pos_rows else None,
            "mean_n_sub_negative": float(np.mean([r["n_subhalos"] for r in neg_rows])) if neg_rows else None,
        }
        if X is not None:
            entry["feature_baselines"] = _baseline_metrics(X, feature_rows, threshold, args.seed)
        summary["thresholds"][str(threshold)] = entry

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep V2 impact-strength thresholds and learnability.")
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--feature-sample-size", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    summary = run(args)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
