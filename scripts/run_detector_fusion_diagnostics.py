"""Compare and fuse detector-balanced GNN and summary-feature baselines.

This is intentionally cheap: it uses a precomputed profile-feature cache for
the RF/ExtraTrees/logistic baselines and for the GNN profile branch. Fusion
weights/calibration are selected only on the validation split, then reported on
the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import site
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [p for p in sys.path if not p or Path(p).resolve() != user_site]
except Exception:
    pass

import numpy as np
import torch
from joblib import dump
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader as PyGDataLoader

from scripts.evaluate_summary_baseline import (
    _binary_metrics,
    _best_threshold,
    _target_from_labels,
)
from scripts.evaluate_v2_classifier import (
    _build_model,
    _collect_probs,
    _resolve_split_path,
)
from src.data.dataset import FeatureNormalizer, StreamSimDataset
from src.data.splits import load_split_indices
from src.models.utils import load_normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_feature_cache(path: Path) -> np.ndarray:
    payload = np.load(str(path), allow_pickle=False)
    if "features" not in payload:
        raise KeyError(f"{path} does not contain a 'features' array")
    return payload["features"].astype(np.float32, copy=False)


def _ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        conf = float(prob[mask].mean())
        acc = float(y_true[mask].mean())
        value += (mask.sum() / total) * abs(acc - conf)
    return float(value)


def _metrics(y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray) -> dict:
    thresholds = {
        "default_0p5": 0.5,
        "val_best_accuracy": _best_threshold(y_val, p_val, "accuracy")[0],
        "val_best_balanced_accuracy": _best_threshold(y_val, p_val, "balanced_accuracy")[0],
        "val_best_f1": _best_threshold(y_val, p_val, "f1")[0],
    }
    return {
        "val_auc": float(roc_auc_score(y_val, p_val)),
        "val_average_precision": float(average_precision_score(y_val, p_val)),
        "val_brier": float(brier_score_loss(y_val, p_val)),
        "val_ece_10": _ece(y_val, p_val),
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "test_average_precision": float(average_precision_score(y_test, p_test)),
        "test_brier": float(brier_score_loss(y_test, p_test)),
        "test_ece_10": _ece(y_test, p_test),
        "val_metrics": {name: _binary_metrics(y_val, p_val, t) for name, t in thresholds.items()},
        "test_metrics": {name: _binary_metrics(y_test, p_test, t) for name, t in thresholds.items()},
    }


def _reliability_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        rows.append({
            "lo": float(lo),
            "hi": float(hi),
            "n": int(mask.sum()),
            "mean_probability": float(prob[mask].mean()) if np.any(mask) else None,
            "empirical_positive_rate": float(y_true[mask].mean()) if np.any(mask) else None,
        })
    return rows


def _bootstrap_metric_ci(
    y_test: np.ndarray,
    prob: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict:
    """Bootstrap test-set uncertainty for rank/calibration metrics."""
    if n_bootstrap <= 0:
        return {}
    aucs: list[float] = []
    aps: list[float] = []
    briers: list[float] = []
    n = len(y_test)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_test[idx]
        p_b = prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_b, p_b)))
        aps.append(float(average_precision_score(y_b, p_b)))
        briers.append(float(brier_score_loss(y_b, p_b)))

    def _ci(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "p16": float(np.percentile(arr, 16)),
            "p50": float(np.percentile(arr, 50)),
            "p84": float(np.percentile(arr, 84)),
            "p2p5": float(np.percentile(arr, 2.5)),
            "p97p5": float(np.percentile(arr, 97.5)),
        }

    return {
        "n_bootstrap": int(len(aucs)),
        "test_auc": _ci(aucs),
        "test_average_precision": _ci(aps),
        "test_brier": _ci(briers),
    }


def _logit_clip(prob: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(prob, eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


def _fit_platt(y_val: np.ndarray, p_val: np.ndarray):
    model = LogisticRegression(max_iter=1000)
    model.fit(_logit_clip(p_val), y_val)
    return model


def _apply_platt(model, prob: np.ndarray) -> np.ndarray:
    return model.predict_proba(_logit_clip(prob))[:, 1]


def _fit_isotonic(y_val: np.ndarray, p_val: np.ndarray):
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(p_val, y_val)
    return model


def _fit_predict_baseline(name: str, seed: int, x_train, y_train, x_val, x_test):
    if name == "logreg_summary":
        model = make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    elif name == "rf_summary":
        model = make_pipeline(
            SimpleImputer(),
            RandomForestClassifier(
                n_estimators=350,
                max_depth=14,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=seed,
                class_weight="balanced",
            ),
        )
    elif name == "extratrees_summary":
        model = make_pipeline(
            SimpleImputer(),
            ExtraTreesClassifier(
                n_estimators=350,
                max_depth=14,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=seed + 1,
                class_weight="balanced",
            ),
        )
    else:
        raise ValueError(name)
    model.fit(x_train, y_train)
    return model, model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]


def _best_weighted_mean(
    y_val: np.ndarray,
    p_val_a: np.ndarray,
    p_val_b: np.ndarray,
    p_test_a: np.ndarray,
    p_test_b: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    best_weight = 0.5
    best_score = -np.inf
    for weight in np.linspace(0.0, 1.0, 101):
        p_val = weight * p_val_a + (1.0 - weight) * p_val_b
        score = float(average_precision_score(y_val, p_val))
        if score > best_score:
            best_score = score
            best_weight = float(weight)
    return (
        best_weight,
        best_weight * p_val_a + (1.0 - best_weight) * p_val_b,
        best_weight * p_test_a + (1.0 - best_weight) * p_test_b,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--target", choices=["model_family", "impact_detectable", "impact_strong"],
                        default="impact_strong")
    parser.add_argument("--impact-threshold", type=float, default=0.5)
    parser.add_argument("--strength-threshold", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--downsample-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--artifact-out", default=None,
                        help="Optional joblib artifact containing RF and fusion calibrators.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else checkpoint.parent
    feature_cache = Path(args.feature_cache)

    x = _load_feature_cache(feature_cache)
    dataset_labels = StreamSimDataset(args.sim_dir, max_stars=1200, preload_ram=False)
    labels = dataset_labels.get_label_matrix()
    y = _target_from_labels(
        labels,
        args.target,
        impact_threshold=args.impact_threshold,
        strength_threshold=args.strength_threshold,
    )

    split_path = _resolve_split_path(
        ckpt_dir,
        len(y),
        args.split_seed,
        args.target,
        args.impact_threshold,
        args.strength_threshold,
    )
    splits = load_split_indices(split_path)
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    if len(x) != len(y):
        raise RuntimeError(f"Feature cache length {len(x)} != label length {len(y)}")

    results: dict[str, object] = {
        "sim_dir": args.sim_dir,
        "checkpoint": str(checkpoint),
        "feature_cache": str(feature_cache),
        "split_path": str(split_path),
        "target": args.target,
        "strength_threshold": float(args.strength_threshold),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "models": {},
        "fusion": {},
        "reliability_bins": {},
        "bootstrap_ci": {},
    }

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    baseline_models: dict[str, object] = {}
    baseline_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ["logreg_summary", "rf_summary", "extratrees_summary"]:
        log.info("Training %s", name)
        baseline_model, p_val, p_test = _fit_predict_baseline(
            name,
            args.seed,
            x_train,
            y_train,
            x_val,
            x_test,
        )
        baseline_models[name] = baseline_model
        baseline_probs[name] = (p_val, p_test)
        results["models"][name] = _metrics(y_val, p_val, y_test, p_test)

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_normalizer(ckpt_dir / "normalizer_v2.npz")
    normalizer = FeatureNormalizer(mean, std)
    gnn_dataset = StreamSimDataset(
        args.sim_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        normalizer=normalizer,
        preload_ram=True,
        use_orbital_features=bool(cfg["graph"].get("orbital_features", {}).get("enabled", False)),
        profile_features_path=feature_cache,
        downsample_seed=args.downsample_seed,
    )
    val_loader = PyGDataLoader(
        torch.utils.data.Subset(gnn_dataset, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = PyGDataLoader(
        torch.utils.data.Subset(gnn_dataset, test_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = _build_model(cfg, checkpoint, device)
    y_val_gnn, p_val_gnn = _collect_probs(
        model,
        val_loader,
        cfg,
        normalizer,
        device,
        args.target,
        args.impact_threshold,
        args.strength_threshold,
    )
    y_test_gnn, p_test_gnn = _collect_probs(
        model,
        test_loader,
        cfg,
        normalizer,
        device,
        args.target,
        args.impact_threshold,
        args.strength_threshold,
    )
    if not np.array_equal(y_val, y_val_gnn) or not np.array_equal(y_test, y_test_gnn):
        raise RuntimeError("GNN and baseline split labels disagree")
    results["models"]["gnn_s2_best_loss"] = _metrics(y_val, p_val_gnn, y_test, p_test_gnn)

    p_val_rf, p_test_rf = baseline_probs["rf_summary"]
    platt_gnn = _fit_platt(y_val, p_val_gnn)
    platt_rf = _fit_platt(y_val, p_val_rf)
    iso_gnn = _fit_isotonic(y_val, p_val_gnn)
    iso_rf = _fit_isotonic(y_val, p_val_rf)

    p_val_gnn_platt = _apply_platt(platt_gnn, p_val_gnn)
    p_test_gnn_platt = _apply_platt(platt_gnn, p_test_gnn)
    p_val_rf_platt = _apply_platt(platt_rf, p_val_rf)
    p_test_rf_platt = _apply_platt(platt_rf, p_test_rf)
    results["calibrated"] = {
        "gnn_platt": _metrics(y_val, p_val_gnn_platt, y_test, p_test_gnn_platt),
        "rf_platt": _metrics(y_val, p_val_rf_platt, y_test, p_test_rf_platt),
        "gnn_isotonic": _metrics(
            y_val,
            iso_gnn.predict(p_val_gnn),
            y_test,
            iso_gnn.predict(p_test_gnn),
        ),
        "rf_isotonic": _metrics(
            y_val,
            iso_rf.predict(p_val_rf),
            y_test,
            iso_rf.predict(p_test_rf),
        ),
    }

    weight, p_val_mean, p_test_mean = _best_weighted_mean(
        y_val,
        p_val_gnn,
        p_val_rf,
        p_test_gnn,
        p_test_rf,
    )
    results["fusion"]["weighted_mean_gnn_rf"] = {
        "weight_on_gnn_selected_by_val_ap": float(weight),
        **_metrics(y_val, p_val_mean, y_test, p_test_mean),
    }

    weight_platt, p_val_mean_platt, p_test_mean_platt = _best_weighted_mean(
        y_val,
        p_val_gnn_platt,
        p_val_rf_platt,
        p_test_gnn_platt,
        p_test_rf_platt,
    )
    results["fusion"]["weighted_mean_platt_gnn_rf"] = {
        "weight_on_gnn_selected_by_val_ap": float(weight_platt),
        **_metrics(y_val, p_val_mean_platt, y_test, p_test_mean_platt),
    }

    stacker = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    stack_x_val = np.column_stack([p_val_gnn, p_val_rf])
    stack_x_test = np.column_stack([p_test_gnn, p_test_rf])
    stacker.fit(stack_x_val, y_val)
    p_val_stack = stacker.predict_proba(stack_x_val)[:, 1]
    p_test_stack = stacker.predict_proba(stack_x_test)[:, 1]
    results["fusion"]["stacked_logreg_on_val_gnn_rf"] = _metrics(
        y_val,
        p_val_stack,
        y_test,
        p_test_stack,
    )

    prob_bank = {
        "gnn_s2_best_loss": (p_val_gnn, p_test_gnn),
        "rf_summary": (p_val_rf, p_test_rf),
        "weighted_mean_gnn_rf": (p_val_mean, p_test_mean),
        "weighted_mean_platt_gnn_rf": (p_val_mean_platt, p_test_mean_platt),
        "stacked_logreg_on_val_gnn_rf": (p_val_stack, p_test_stack),
    }
    rng = np.random.default_rng(args.seed)
    for name, (p_val, p_test) in prob_bank.items():
        results["reliability_bins"][name] = {
            "val": _reliability_bins(y_val, p_val),
            "test": _reliability_bins(y_test, p_test),
        }
        results["bootstrap_ci"][name] = _bootstrap_metric_ci(
            y_test,
            p_test,
            rng,
            args.bootstrap_samples,
        )

    probs_out = Path(args.out).with_suffix(".probs.npz")
    np.savez_compressed(
        probs_out,
        y_val=y_val,
        y_test=y_test,
        val_idx=val_idx,
        test_idx=test_idx,
        p_val_gnn=p_val_gnn,
        p_test_gnn=p_test_gnn,
        p_val_rf=p_val_rf,
        p_test_rf=p_test_rf,
        p_val_weighted=p_val_mean,
        p_test_weighted=p_test_mean,
        p_val_stacked=p_val_stack,
        p_test_stacked=p_test_stack,
    )
    results["probabilities_npz"] = str(probs_out)

    artifact_out = Path(args.artifact_out) if args.artifact_out else Path(args.out).with_suffix(".joblib")
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "metadata": {
                "sim_dir": args.sim_dir,
                "checkpoint": str(checkpoint),
                "feature_cache": str(feature_cache),
                "target": args.target,
                "strength_threshold": float(args.strength_threshold),
                "selected_weight_on_gnn": float(weight),
                "selected_weight_on_gnn_platt": float(weight_platt),
            },
            "rf_summary_model": baseline_models["rf_summary"],
            "stacked_logreg_on_val_gnn_rf": stacker,
            "platt_gnn": platt_gnn,
            "platt_rf": platt_rf,
            "isotonic_gnn": iso_gnn,
            "isotonic_rf": iso_rf,
        },
        artifact_out,
    )
    results["artifact_out"] = str(artifact_out)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
