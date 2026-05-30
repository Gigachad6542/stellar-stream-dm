"""Evaluate classical summary-feature baselines on the exact V2 train/val/test split.

This is the fair comparison companion to ``evaluate_v2_classifier.py``.  The
older signal diagnostic intentionally used a balanced random sample as a cheap
gatekeeper, which is useful for deciding whether a GNN run is worth attempting
but is not an apples-to-apples held-out benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import site
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [p for p in sys.path if not p or Path(p).resolve() != user_site]
except Exception:
    pass

import h5py
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset import StreamSimDataset, stream_one_hot
from src.data.splits import load_split_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _threshold_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _split_tag_for_target(target: str, impact_threshold: float, strength_threshold: float) -> str:
    if target == "model_family":
        return "dm"
    if target == "impact_detectable":
        return f"impact_detectable_t{_threshold_tag(impact_threshold)}"
    if target == "impact_strong":
        return f"impact_strong_s{_threshold_tag(strength_threshold)}"
    return target


def _resolve_split_path(
    ckpt_dir: Path,
    n_dataset: int,
    split_seed: int,
    target: str,
    impact_threshold: float,
    strength_threshold: float,
) -> Path:
    split_tag = _split_tag_for_target(target, impact_threshold, strength_threshold)
    split_path = ckpt_dir / f"split_n{n_dataset}_seed{split_seed}_{split_tag}.npz"
    if split_path.exists():
        return split_path
    legacy_tag = target if target != "model_family" else "dm"
    legacy_path = ckpt_dir / f"split_n{n_dataset}_seed{split_seed}_{legacy_tag}.npz"
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(f"No split file found at {split_path} or {legacy_path}")


def _target_from_labels(
    labels: np.ndarray,
    target: str,
    impact_threshold: float,
    strength_threshold: float,
) -> np.ndarray:
    if target == "model_family":
        return np.isin(labels[:, 0].astype(np.int64), [1, 2]).astype(np.int64)
    if target == "impact_detectable":
        return (labels[:, 2] > impact_threshold).astype(np.int64)
    if target == "impact_strong":
        return (labels[:, 5] > strength_threshold).astype(np.int64)
    raise ValueError(f"Unknown target {target!r}")


def _load_node_features(sd: h5py.Group) -> np.ndarray:
    def _g(name: str, fill: float = 0.0) -> np.ndarray:
        if name in sd:
            arr = sd[name][:].astype(np.float32)
            return np.where(np.isfinite(arr), arr, fill)
        return np.full(sd["phi1"].shape, fill, dtype=np.float32)

    return np.column_stack(
        [
            _g("phi1"),
            _g("phi2"),
            _g("dist"),
            _g("pm1"),
            _g("pm2"),
            _g("vrad", fill=0.0),
            _g("e_dist", fill=1.0),
            _g("e_pm1", fill=0.1),
            _g("e_pm2", fill=0.1),
            _g("e_vrad", fill=1.0),
            _g("membership_prob", fill=1.0),
        ]
    )


def _deterministic_downsample(x: np.ndarray, max_stars: int, seed: int) -> np.ndarray:
    if max_stars <= 0 or len(x) <= max_stars:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_stars, replace=False)
    return x[idx]


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _summary_from_x(x: np.ndarray) -> np.ndarray:
    """Build RF/ExtraTrees features from the same node columns used by the GNN."""
    phi1_raw = x[:, 0]
    mem_raw = x[:, 10] if x.shape[1] > 10 else np.ones(len(x), dtype=np.float32)
    mask = np.isfinite(phi1_raw)
    phi1 = phi1_raw[mask]
    mem = mem_raw[mask]

    feats: list[float] = [
        float(len(phi1)),
        float(np.nanmean(mem)) if len(mem) else np.nan,
        float(np.nanstd(mem)) if len(mem) else np.nan,
    ]

    for col in range(11):
        values = _finite(x[:, col])
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
        feats.extend(
            [
                float(hn.std()),
                float(hn.min()),
                float(np.percentile(hn, 5)),
                float(np.mean(hn < 0.5)),
                float(np.mean(hn < 0.25)),
                float(np.max(np.abs(np.diff(hn)))),
            ]
        )
    else:
        feats.extend([np.nan] * (48 + 6))

    for col in [1, 3, 4, 2]:  # phi2, pm1, pm2, dist
        values = x[:, col][mask]
        ok = np.isfinite(values)
        if ok.sum() < 20:
            feats.extend([np.nan] * 4)
            continue
        xpos = phi1[ok]
        ypos = values[ok]
        lo, hi = np.nanpercentile(xpos, [1, 99])
        bins = np.linspace(lo, max(hi, lo + 1e-3), 17)
        medians = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            in_bin = (xpos >= b0) & (xpos < b1)
            medians.append(np.nanmedian(ypos[in_bin]) if in_bin.any() else np.nan)
        medians_arr = np.asarray(medians, dtype=np.float64)
        diffs = np.diff(medians_arr)
        feats.extend(
            [
                float(np.nanstd(medians_arr)),
                float(np.nanstd(diffs)),
                float(np.nanmax(medians_arr) - np.nanmin(medians_arr)),
                float(np.sum(~np.isfinite(medians_arr))),
            ]
        )

    if x.shape[1] > 11:
        feats.extend(map(float, x[0, 11:].tolist()))

    return np.asarray(feats, dtype=np.float32)


def _collect_features(
    dataset: StreamSimDataset,
    max_stars: int,
    include_stream_onehot: bool,
    seed: int,
) -> np.ndarray:
    rows: list[np.ndarray] = [None] * len(dataset)  # type: ignore[list-item]
    file_groups: dict[Path, list[tuple[int, str, str]]] = defaultdict(list)
    for idx, (h5_path, run_id, stream_name) in enumerate(dataset._index):  # noqa: SLF001
        file_groups[h5_path].append((idx, run_id, stream_name))

    for h5_path, items in file_groups.items():
        log.info("Reading %s (%d sims)", h5_path.name, len(items))
        with h5py.File(str(h5_path), "r") as f:
            for idx, run_id, stream_name in items:
                x = _load_node_features(f[f"simulations/{run_id}/stream_data"])
                x = _deterministic_downsample(x, max_stars=max_stars, seed=seed + idx)
                if include_stream_onehot:
                    x = np.column_stack([x, stream_one_hot(stream_name, len(x))])
                rows[idx] = _summary_from_x(x)
    return np.vstack(rows)


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "confusion_matrix": cm.tolist(),
    }


def _best_threshold(y_true: np.ndarray, prob: np.ndarray, metric: str) -> tuple[float, float]:
    thresholds = np.unique(np.concatenate(([0.0], prob, [1.0])))
    best_t = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        pred = (prob >= threshold).astype(np.int64)
        if metric == "accuracy":
            score = float((pred == y_true).mean())
        elif metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, pred))
        elif metric == "f1":
            score = float(f1_score(y_true, pred, zero_division=0))
        else:
            raise ValueError(metric)
        if score > best_score:
            best_t = float(threshold)
            best_score = score
    return best_t, best_score


def _evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    p_val = model.predict_proba(x_val)[:, 1]
    p_test = model.predict_proba(x_test)[:, 1]
    thresholds = {
        "default_0p5": 0.5,
        "val_best_accuracy": _best_threshold(y_val, p_val, "accuracy")[0],
        "val_best_balanced_accuracy": _best_threshold(y_val, p_val, "balanced_accuracy")[0],
        "val_best_f1": _best_threshold(y_val, p_val, "f1")[0],
    }
    return {
        "val_auc": float(roc_auc_score(y_val, p_val)),
        "val_average_precision": float(average_precision_score(y_val, p_val)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "test_average_precision": float(average_precision_score(y_test, p_test)),
        "val_metrics": {name: _binary_metrics(y_val, p_val, t) for name, t in thresholds.items()},
        "test_metrics": {name: _binary_metrics(y_test, p_test, t) for name, t in thresholds.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--target", choices=["model_family", "impact_detectable", "impact_strong"], default="impact_strong")
    parser.add_argument("--impact-threshold", type=float, default=0.5)
    parser.add_argument("--strength-threshold", type=float, default=1.0)
    parser.add_argument("--max-stars", type=int, default=1200)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--feature-seed", type=int, default=12345)
    parser.add_argument("--include-stream-onehot", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sim_dir = Path(args.sim_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    dataset = StreamSimDataset(sim_dir, max_stars=args.max_stars, preload_ram=False)
    labels = dataset.get_label_matrix()
    y = _target_from_labels(
        labels,
        args.target,
        impact_threshold=args.impact_threshold,
        strength_threshold=args.strength_threshold,
    )

    split_path = _resolve_split_path(
        ckpt_dir,
        len(dataset),
        args.split_seed,
        args.target,
        args.impact_threshold,
        args.strength_threshold,
    )
    split_indices = load_split_indices(split_path)
    x = _collect_features(
        dataset,
        max_stars=args.max_stars,
        include_stream_onehot=args.include_stream_onehot,
        seed=args.feature_seed,
    )

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    models = {
        "logreg_summary": make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "rf_summary": make_pipeline(
            SimpleImputer(),
            RandomForestClassifier(
                n_estimators=350,
                max_depth=14,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=args.feature_seed,
                class_weight="balanced",
            ),
        ),
        "extratrees_summary": make_pipeline(
            SimpleImputer(),
            ExtraTreesClassifier(
                n_estimators=350,
                max_depth=14,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=args.feature_seed + 1,
                class_weight="balanced",
            ),
        ),
    }

    result = {
        "sim_dir": str(sim_dir),
        "target": args.target,
        "impact_threshold": float(args.impact_threshold),
        "strength_threshold": float(args.strength_threshold),
        "max_stars": int(args.max_stars),
        "include_stream_onehot": bool(args.include_stream_onehot),
        "split_path": str(split_path),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "target_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "n_features": int(x.shape[1]),
        "models": {},
    }

    for name, model in models.items():
        log.info("Training %s", name)
        result["models"][name] = _evaluate_model(
            model,
            x[train_idx],
            y[train_idx],
            x[val_idx],
            y[val_idx],
            x[test_idx],
            y[test_idx],
        )

    text = json.dumps(result, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        log.info("Wrote %s", out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
