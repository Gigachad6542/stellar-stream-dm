"""Diagnose whether a V2 binary target is visible in simulated stream features.

This is the gatekeeper before expensive GNN training. It reads HDF5 chunks,
builds cheap density/kinematic summary features, and reports whether simple
baselines can generalize. If these baselines are at chance, a GNN run should be
treated as a data/target problem rather than a training-speed problem.

Examples:
    python scripts/diagnose_v2_signal.py --sim-dir data/simulations_v2_plan2 --target model_family
    python scripts/diagnose_v2_signal.py --sim-dir data/simulations_signal_ladder/clean_impact --target impact_detectable
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values",
    category=UserWarning,
)


DM_NAMES = {0: "CDM", 1: "WDM", 2: "FDM", 3: "SIDM"}


def _target_from_labels(dm_idx: int, n_subhalos: float, target: str, impact_threshold: float,
                        impact_strength: float = 0.0, strength_threshold: float = 0.5) -> int:
    if target == "model_family":
        return int(dm_idx in (1, 2))
    if target == "impact_detectable":
        return int(n_subhalos > impact_threshold)
    if target == "impact_strong":
        return int(impact_strength > strength_threshold)
    raise ValueError(f"Unknown target: {target}")


def _finite(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _summary_features(stream_data: h5py.Group) -> np.ndarray:
    cols = [
        "phi1", "phi2", "pm1", "pm2", "dist", "vrad",
        "membership_prob", "e_pm1", "e_pm2", "e_dist", "e_vrad",
    ]
    phi1_raw = np.asarray(stream_data["phi1"], dtype=np.float64)
    mem_raw = np.asarray(stream_data.get("membership_prob", np.ones_like(phi1_raw)), dtype=np.float64)
    mask = np.isfinite(phi1_raw)
    phi1 = phi1_raw[mask]
    mem = mem_raw[mask]

    feats: list[float] = [
        float(len(phi1)),
        float(np.nanmean(mem)) if len(mem) else np.nan,
        float(np.nanstd(mem)) if len(mem) else np.nan,
    ]

    for col in cols:
        if col not in stream_data:
            feats.extend([np.nan] * 7)
            continue
        values = _finite(stream_data[col])
        if len(values) == 0:
            feats.extend([np.nan] * 7)
            continue
        q = np.nanquantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        feats.extend([float(np.nanmean(values)), float(np.nanstd(values)), *map(float, q)])

    if len(phi1) >= 10:
        lo, hi = np.nanpercentile(phi1, [1, 99])
        hi = max(hi, lo + 1e-3)
        hist, _ = np.histogram(phi1, bins=48, range=(lo, hi), weights=np.clip(mem, 0, 1))
        h = hist.astype(np.float64)
        hn = h / (h.mean() + 1e-6)
        feats.extend(map(float, hn))
        feats.extend([
            float(hn.std()),
            float(hn.min()),
            float(np.percentile(hn, 5)),
            float(np.sum(hn < 0.5)),
            float(np.sum(hn < 0.25)),
            float(np.max(np.abs(np.diff(hn)))),
        ])
    else:
        feats.extend([np.nan] * (48 + 6))

    for col in ["phi2", "pm1", "pm2", "dist"]:
        if col not in stream_data or len(phi1) < 20:
            feats.extend([np.nan] * 4)
            continue
        values_all = np.asarray(stream_data[col], dtype=np.float64)
        values = values_all[mask]
        ok = np.isfinite(values)
        if ok.sum() < 20:
            feats.extend([np.nan] * 4)
            continue
        x = phi1[ok]
        y = values[ok]
        lo, hi = np.nanpercentile(x, [1, 99])
        bins = np.linspace(lo, max(hi, lo + 1e-3), 17)
        medians = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            in_bin = (x >= b0) & (x < b1)
            medians.append(np.nanmedian(y[in_bin]) if in_bin.any() else np.nan)
        medians = np.asarray(medians, dtype=np.float64)
        diffs = np.diff(medians)
        feats.extend([
            float(np.nanstd(medians)),
            float(np.nanstd(diffs)),
            float(np.nanmax(medians) - np.nanmin(medians)),
            float(np.sum(~np.isfinite(medians))),
        ])

    return np.asarray(feats, dtype=np.float32)


def _collect_pointers(sim_dir: Path, target: str, impact_threshold: float,
                      strength_threshold: float = 0.5) -> list[dict]:
    pointers: list[dict] = []
    for path in sorted(sim_dir.glob("**/chunk_*.h5")):
        with h5py.File(path, "r") as f:
            if "simulations" not in f:
                continue
            for run_id in f["simulations"].keys():
                grp = f["simulations"][run_id]
                labels = grp["labels"]
                dm_idx = int(labels["dm_model_idx"][()])
                n_sub = float(labels["n_subhalos"][()])
                log_mhm = float(labels["log10_M_hm"][()]) if "log10_M_hm" in labels else float(labels.attrs.get("log10_M_hm", np.nan))
                impact_strength = float(labels.attrs.get("impact_strength", 0.0))
                y = _target_from_labels(dm_idx, n_sub, target, impact_threshold,
                                        impact_strength=impact_strength,
                                        strength_threshold=strength_threshold)
                pointers.append({
                    "path": str(path),
                    "run_id": run_id,
                    "target": y,
                    "dm_idx": dm_idx,
                    "n_subhalos": n_sub,
                    "log10_M_hm": log_mhm,
                    "impact_strength": impact_strength,
                })
    return pointers


def _balanced_sample(pointers: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_target: dict[int, list[dict]] = defaultdict(list)
    for ptr in pointers:
        by_target[int(ptr["target"])].append(ptr)
    if len(by_target) < 2:
        return pointers[: min(n, len(pointers))]
    per_class = max(1, n // len(by_target))
    selected: list[dict] = []
    for items in by_target.values():
        take = min(per_class, len(items))
        idx = rng.choice(len(items), size=take, replace=False)
        selected.extend(items[int(i)] for i in idx)
    rng.shuffle(selected)
    return selected


def run(args: argparse.Namespace) -> dict:
    sim_dir = Path(args.sim_dir)
    strength_thresh = getattr(args, "strength_threshold", 0.5)
    pointers = _collect_pointers(sim_dir, args.target, args.impact_threshold,
                                 strength_threshold=strength_thresh)
    if not pointers:
        raise RuntimeError(f"No chunk_*.h5 simulations found under {sim_dir}")

    selected = _balanced_sample(pointers, args.sample_size, args.seed)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for ptr in selected:
        by_file[ptr["path"]].append(ptr)

    X: list[np.ndarray] = []
    y: list[int] = []
    dm: list[int] = []
    n_sub: list[float] = []
    mhm: list[float] = []

    for path, items in by_file.items():
        with h5py.File(path, "r") as f:
            for ptr in items:
                grp = f["simulations"][ptr["run_id"]]
                X.append(_summary_features(grp["stream_data"]))
                y.append(int(ptr["target"]))
                dm.append(int(ptr["dm_idx"]))
                n_sub.append(float(ptr["n_subhalos"]))
                mhm.append(float(ptr["log10_M_hm"]))

    X_arr = np.vstack(X)
    y_arr = np.asarray(y, dtype=np.int64)
    dm_arr = np.asarray(dm, dtype=np.int64)
    n_sub_arr = np.asarray(n_sub, dtype=np.float64)
    mhm_arr = np.asarray(mhm, dtype=np.float64)

    summary = {
        "sim_dir": str(sim_dir),
        "target": args.target,
        "n_total": len(pointers),
        "n_sample": int(len(y_arr)),
        "target_counts_total": {
            "0": int(sum(ptr["target"] == 0 for ptr in pointers)),
            "1": int(sum(ptr["target"] == 1 for ptr in pointers)),
        },
        "target_counts_sample": {
            "0": int((y_arr == 0).sum()),
            "1": int((y_arr == 1).sum()),
        },
        "dm_counts_sample": {
            DM_NAMES.get(int(k), str(k)): int((dm_arr == k).sum())
            for k in sorted(np.unique(dm_arr))
        },
        "n_subhalos_by_target": {},
        "log10_M_hm_by_target": {},
        "models": {},
    }

    for cls in [0, 1]:
        mask = y_arr == cls
        if mask.any():
            summary["n_subhalos_by_target"][str(cls)] = {
                "mean": float(np.mean(n_sub_arr[mask])),
                "zero_fraction": float(np.mean(n_sub_arr[mask] == 0)),
                "q05_q50_q95": [float(v) for v in np.quantile(n_sub_arr[mask], [0.05, 0.5, 0.95])],
            }
            finite_mhm = mhm_arr[mask][np.isfinite(mhm_arr[mask])]
            if len(finite_mhm):
                summary["log10_M_hm_by_target"][str(cls)] = {
                    "q05_q50_q95": [float(v) for v in np.quantile(finite_mhm, [0.05, 0.5, 0.95])],
                }

    if len(np.unique(y_arr)) < 2:
        summary["status"] = "one_class_only"
        return summary

    X_train, X_test, y_train, y_test, n_train, n_test, m_train, m_test = train_test_split(
        X_arr,
        y_arr,
        n_sub_arr,
        mhm_arr,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_arr,
    )

    models = {
        "logreg_summary": make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "rf_summary": make_pipeline(
            SimpleImputer(),
            RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=args.seed,
                class_weight="balanced",
            ),
        ),
        "extratrees_summary": make_pipeline(
            SimpleImputer(),
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=args.seed + 1,
                class_weight="balanced",
            ),
        ),
    }

    best_auc = 0.0
    best_acc = 0.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        acc = float(accuracy_score(y_test, pred))
        auc = float(roc_auc_score(y_test, prob))
        best_auc = max(best_auc, auc)
        best_acc = max(best_acc, acc)
        summary["models"][name] = {
            "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
            "test_accuracy": acc,
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }

    if args.target == "model_family":
        label_pred = (m_test < 9.999).astype(np.int64)
        summary["label_oracle_log10_M_hm_lt_10"] = {
            "test_accuracy": float(accuracy_score(y_test, label_pred)),
            "confusion_matrix": confusion_matrix(y_test, label_pred).tolist(),
        }
    if args.target == "impact_detectable":
        label_pred = (n_test > args.impact_threshold).astype(np.int64)
        summary["label_oracle_n_subhalos"] = {
            "test_accuracy": float(accuracy_score(y_test, label_pred)),
            "confusion_matrix": confusion_matrix(y_test, label_pred).tolist(),
        }

    summary["recommendation"] = (
        "proceed_to_gnn"
        if best_auc >= args.min_auc and best_acc >= args.min_accuracy
        else "fix_simulation_signal_before_gnn"
    )
    summary["thresholds"] = {
        "min_auc": args.min_auc,
        "min_accuracy": args.min_accuracy,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose V2 simulation signal visibility.")
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--target", choices=["model_family", "impact_detectable", "impact_strong"],
                        default="model_family")
    parser.add_argument("--strength-threshold", type=float, default=0.5,
                        help="Impact strength threshold for impact_strong target.")
    parser.add_argument("--impact-threshold", type=float, default=0.5)
    parser.add_argument("--sample-size", type=int, default=6000)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-auc", type=float, default=0.70)
    parser.add_argument("--min-accuracy", type=float, default=0.65)
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
