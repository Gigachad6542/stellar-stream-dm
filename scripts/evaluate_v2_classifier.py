"""
Evaluate a trained V2 binary GNN checkpoint on validation and test splits.

The key guardrail is threshold calibration: thresholds are selected on the
validation split only, then applied once to the held-out test split.
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
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.dataset import (
    FeatureNormalizer,
    StreamSimDataset,
    build_knn_graph_batched,
    build_profile_features_batched,
    build_segment_graph,
    profile_feature_dim,
)
from src.data.splits import load_split_indices
from src.models.gnn import StreamGNNMultiTaskV2
from src.models.utils import load_normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _target_from_labels(
    labels: torch.Tensor,
    target: str,
    impact_threshold: float,
    strength_threshold: float = 0.5,
) -> torch.Tensor:
    if target == "impact_detectable":
        return (labels[:, 2] > impact_threshold).float()
    if target == "impact_strong":
        if labels.shape[1] >= 6:
            return (labels[:, 5] > strength_threshold).float()
        return (labels[:, 2] > impact_threshold).float()
    if target == "model_family":
        raw = labels[:, 0].long()
        remap = torch.tensor([0, 1, 1, 0], dtype=torch.float32, device=labels.device)
        return remap[raw]
    raise ValueError(f"Unknown target {target!r}")


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
    for t in thresholds:
        pred = (prob >= t).astype(np.int64)
        if metric == "accuracy":
            score = float((pred == y_true).mean())
        elif metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, pred))
        elif metric == "f1":
            score = float(f1_score(y_true, pred, zero_division=0))
        else:
            raise ValueError(metric)
        if score > best_score:
            best_t = float(t)
            best_score = score
    return best_t, best_score


def _collect_probs(
    model: torch.nn.Module,
    loader: PyGDataLoader,
    cfg: dict,
    normalizer: FeatureNormalizer,
    device: torch.device,
    target: str,
    impact_threshold: float,
    strength_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    truths: list[np.ndarray] = []

    profile_cfg = cfg["graph"].get("profile_branch", {})
    use_profile = bool(profile_cfg.get("enabled", False))
    profile_n_bins = int(profile_cfg.get("n_bins", 48))
    profile_feature_set = str(profile_cfg.get("feature_set", "compact")).lower()
    profile_include_stream_onehot = bool(profile_cfg.get("include_stream_onehot", True))
    ms_cfg = cfg["graph"].get("multi_scale", {})
    use_multi_scale = bool(ms_cfg.get("enabled", False))
    use_orbital = bool(cfg["graph"].get("orbital_features", {}).get("enabled", False))

    norm_mean = normalizer.mean.to(device)
    norm_std = normalizer.std.to(device)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            build_knn_graph_batched(
                batch,
                cfg["graph"]["k_neighbors"],
                normalizer,
                orbital_features=use_orbital,
            )
            if use_profile:
                build_profile_features_batched(
                    batch,
                    n_bins=profile_n_bins,
                    feature_set=profile_feature_set,
                    include_stream_onehot=profile_include_stream_onehot,
                )
            if use_multi_scale:
                build_segment_graph(
                    batch,
                    n_segments=ms_cfg.get("n_segments", 20),
                    k_seg=ms_cfg.get("k_segment_neighbors", 4),
                )
            batch.x = (batch.x - norm_mean) / norm_std
            _, binary_logit, _, _ = model(batch)
            labels = batch.y.view(batch.num_graphs, -1)
            y = _target_from_labels(labels, target, impact_threshold, strength_threshold)
            prob = torch.sigmoid(binary_logit.squeeze(-1))
            probs.append(prob.detach().cpu().numpy())
            truths.append(y.detach().cpu().numpy().astype(np.int64))

    return np.concatenate(truths), np.concatenate(probs)


def _build_model(cfg: dict, checkpoint: Path, device: torch.device) -> StreamGNNMultiTaskV2:
    gcfg = cfg["model"]["gnn"]
    v2_cfg = cfg.get("training_v2", {})
    ms_cfg = cfg["graph"].get("multi_scale", {})
    profile_cfg = cfg["graph"].get("profile_branch", {})
    profile_dim = 0
    if profile_cfg.get("enabled", False):
        profile_dim = profile_feature_dim(
            int(profile_cfg.get("n_bins", 48)),
            str(profile_cfg.get("feature_set", "compact")).lower(),
            bool(profile_cfg.get("include_stream_onehot", True)),
        )

    model = StreamGNNMultiTaskV2(
        n_reg_targets=v2_cfg.get("n_reg_targets", 2),
        predict_uncertainty=v2_cfg.get("predict_uncertainty", False),
        profile_dim=profile_dim,
        profile_hidden_dim=int(profile_cfg.get("hidden_dim", 64)),
        profile_layer_norm=bool(profile_cfg.get("layer_norm", False)),
        n_node_features=cfg["graph"]["n_node_features"],
        n_edge_features=cfg["graph"]["n_edge_features"],
        hidden_dim=gcfg["hidden_dim"],
        n_layers=gcfg["n_layers"],
        embedding_dim=gcfg["embedding_dim"],
        dropout=gcfg["dropout"],
        use_attention_readout=gcfg.get("use_attention_readout", False),
        use_multi_scale=ms_cfg.get("enabled", False),
        seg_embedding_dim=ms_cfg.get("seg_embedding_dim", 64),
    ).to(device)
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return model


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
    """Resolve threshold-aware split files with legacy fallback for old runs."""
    split_tag = _split_tag_for_target(target, impact_threshold, strength_threshold)
    split_path = ckpt_dir / f"split_n{n_dataset}_seed{split_seed}_{split_tag}.npz"
    if split_path.exists():
        return split_path

    legacy_tag = target if target != "model_family" else "dm"
    legacy_path = ckpt_dir / f"split_n{n_dataset}_seed{split_seed}_{legacy_tag}.npz"
    if legacy_path.exists():
        log.warning(
            "Using legacy split file without threshold in name: %s. "
            "Regenerate training splits for threshold-varied runs.",
            legacy_path,
        )
        return legacy_path
    return split_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--target", choices=["model_family", "impact_detectable", "impact_strong"],
                        default="impact_strong")
    parser.add_argument("--strength-threshold", type=float, default=0.5,
                        help="Strength threshold for impact_strong target.")
    parser.add_argument("--impact-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--profile-features-path", default=None,
                        help="Optional precomputed profile-feature cache from precompute_profile_features.py.")
    parser.add_argument("--downsample-seed", type=int, default=None,
                        help="Deterministic node downsampling seed; defaults to cache metadata/config when available.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else checkpoint.parent
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = payload["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluating %s on %s", checkpoint, device)

    mean, std = load_normalizer(ckpt_dir / "normalizer_v2.npz")
    normalizer = FeatureNormalizer(mean, std)
    profile_cfg = cfg["graph"].get("profile_branch", {})
    profile_features_path = args.profile_features_path or profile_cfg.get("features_path")
    downsample_seed = args.downsample_seed
    if profile_features_path and downsample_seed is None:
        downsample_seed = int(profile_cfg.get("downsample_seed", 42))
    dataset = StreamSimDataset(
        args.sim_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        normalizer=normalizer,
        preload_ram=True,
        use_orbital_features=bool(cfg["graph"].get("orbital_features", {}).get("enabled", False)),
        profile_features_path=profile_features_path,
        downsample_seed=downsample_seed,
    )
    split_seed = cfg["training"].get("split_seed", 42)
    split_path = _resolve_split_path(
        ckpt_dir,
        len(dataset),
        split_seed,
        args.target,
        args.impact_threshold,
        args.strength_threshold,
    )
    split_indices = load_split_indices(split_path)

    val_set = torch.utils.data.Subset(dataset, split_indices["val"].tolist())
    test_set = torch.utils.data.Subset(dataset, split_indices["test"].tolist())
    val_loader = PyGDataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = PyGDataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = _build_model(cfg, checkpoint, device)
    y_val, p_val = _collect_probs(
        model, val_loader, cfg, normalizer, device,
        args.target, args.impact_threshold, args.strength_threshold,
    )
    y_test, p_test = _collect_probs(
        model, test_loader, cfg, normalizer, device,
        args.target, args.impact_threshold, args.strength_threshold,
    )

    thresholds = {
        "default_0p5": 0.5,
        "val_best_accuracy": _best_threshold(y_val, p_val, "accuracy")[0],
        "val_best_balanced_accuracy": _best_threshold(y_val, p_val, "balanced_accuracy")[0],
        "val_best_f1": _best_threshold(y_val, p_val, "f1")[0],
    }

    result = {
        "checkpoint": str(checkpoint),
        "sim_dir": args.sim_dir,
        "profile_features_path": str(profile_features_path) if profile_features_path else None,
        "downsample_seed": int(downsample_seed) if downsample_seed is not None else None,
        "target": args.target,
        "impact_threshold": float(args.impact_threshold),
        "strength_threshold": float(args.strength_threshold),
        "split_path": str(split_path),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "val_auc": float(roc_auc_score(y_val, p_val)),
        "val_average_precision": float(average_precision_score(y_val, p_val)),
        "test_auc": float(roc_auc_score(y_test, p_test)),
        "test_average_precision": float(average_precision_score(y_test, p_test)),
        "val_metrics": {name: _binary_metrics(y_val, p_val, t) for name, t in thresholds.items()},
        "test_metrics": {name: _binary_metrics(y_test, p_test, t) for name, t in thresholds.items()},
    }

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
