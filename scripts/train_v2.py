"""
Train GNN v2 binary/regression heads.

This script supports two V2 stages:

  1. Detector training: impact/no-impact or impact_strong on balanced signal
     datasets. This learns observable subhalo morphology before asking the
     harder dark-matter-family question.
  2. Suppression inference: suppressed (WDM/FDM) vs unsuppressed (CDM/SIDM)
     plus M_hm regression on physics-prior simulations.

The original 3-class scheme is intentionally avoided because WDM and FDM
suppression shapes are degenerate at Gaia-like sensitivity.

Outputs:
  - binary head: target selected by --binary-target
  - M_hm head: active for model_family, disabled for observable impact targets
  - auxiliary regression: n_impacts and log_t_since_last_impact

Curriculum learning:
  The Plan 2 HDF5 files already include domain randomization, so this script
  cannot turn Gaia noise/foregrounds off after generation. The curriculum here
  ramps training augmentation strength and M_hm loss weight, while the dataset
  itself remains the full randomized v2 dataset.

Usage:
    python scripts/train_v2.py --epochs 300
    python scripts/train_v2.py --sim-dir data/simulations_v2_detector_balanced --binary-target impact_strong --strength-threshold 1.0 --use-profile-branch
    python scripts/train_v2.py --epochs 300 --resume checkpoints/gnn_v2_best.pt
    python scripts/train_v2.py --epochs 1 --no-preload-ram --normalizer-samples 64 --max-train-batches 2 --max-val-batches 1
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import math
import site
import time

# Keep user-site packages from shadowing the conda environment. On this Windows
# machine the user site contains a CPU-only torch build, while the project env
# contains the CUDA build needed for full V2 training.
try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [
        p for p in sys.path
        if not p or Path(p).resolve() != user_site
    ]
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.dataset import (
    StreamSimDataset,
    build_knn_graph_batched,
    build_profile_features_batched,
    build_segment_graph,
    compute_dataset_normalizer,
    profile_feature_dim,
)
from src.data.splits import (
    compute_split_indices,
    get_split_hash,
    load_split_indices,
    save_split_indices,
)
from src.models.gnn import StreamGNNMultiTaskV2
from src.models.utils import (
    build_scheduler,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

# Raw DM model indices from HDF5: CDM=0, WDM=1, FDM=2, SIDM=3
# Binary: Unsuppressed (CDM/SIDM) = 0, Suppressed (WDM/FDM) = 1
BINARY_REMAP = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
BINARY_NAMES = ["Unsuppressed (CDM+SIDM)", "Suppressed (WDM+FDM)"]
BINARY_TARGET_NAMES = {
    "model_family": ["Unsuppressed (CDM+SIDM)", "Suppressed (WDM+FDM)"],
    "impact_detectable": ["No visible subhalo impact", "One or more subhalo impacts"],
    "impact_strong": [
        "No strong subhalo impact",
        "One or more strong subhalo impacts",
    ],
    "impact_timeline_detectable": [
        "No timeline-detectable impact",
        "One or more timeline-detectable impacts",
    ],
}

# M_hm fallback for legacy files only.  Current generated data should store a
# lower-edge CDM-like value for unsuppressed models; the M_hm loss is masked to
# WDM/FDM, so this constant should not carry physical meaning.
MHM_UNSUPPRESSED_LEGACY_SENTINEL = 10.0


def binary_targets_from_label_matrix(
    label_matrix: np.ndarray,
    mode: str,
    impact_threshold: float = 0.5,
    strength_threshold: float = 0.5,
) -> np.ndarray:
    """Compute binary targets from compact labels for splitting/balancing.

    Modes:
        model_family: Suppressed (WDM/FDM) vs Unsuppressed (CDM/SIDM).
        impact_detectable: n_subhalos > impact_threshold (any impact).
        impact_strong: impact_strength > strength_threshold (morphologically
            detectable impacts only). Requires label column 5 (impact_strength)
            computed by compute_impact_strength.py. Falls back to impact_detectable
            if column 5 is all zeros (old data without impact_strength).
    """
    mode = mode.lower()
    if mode == "model_family":
        dm = label_matrix[:, 0].astype(np.int64)
        return np.isin(dm, [1, 2]).astype(np.int64)
    if mode == "impact_detectable":
        return (label_matrix[:, 2] > impact_threshold).astype(np.int64)
    if mode == "impact_strong":
        if label_matrix.shape[1] >= 6 and label_matrix[:, 5].max() > 0:
            return (label_matrix[:, 5] > strength_threshold).astype(np.int64)
        else:
            log.warning("impact_strong requested but no impact_strength in labels; "
                        "falling back to impact_detectable")
            return (label_matrix[:, 2] > impact_threshold).astype(np.int64)
    if mode == "impact_timeline_detectable":
        if label_matrix.shape[1] >= 7:
            return (label_matrix[:, 6] > impact_threshold).astype(np.int64)
        log.warning("impact_timeline_detectable requested but timeline labels are absent; "
                    "falling back to impact_strong")
        return binary_targets_from_label_matrix(
            label_matrix,
            "impact_strong",
            impact_threshold=impact_threshold,
            strength_threshold=strength_threshold,
        )
    raise ValueError(f"Unknown binary target mode: {mode}")


def _threshold_tag(value: float) -> str:
    """Stable filename fragment for threshold-dependent binary targets."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def split_tag_for_target(mode: str, impact_threshold: float, strength_threshold: float) -> str:
    """Return a split-cache tag that encodes target-defining thresholds."""
    mode = mode.lower()
    if mode == "model_family":
        return "dm"
    if mode == "impact_detectable":
        return f"impact_detectable_t{_threshold_tag(impact_threshold)}"
    if mode == "impact_strong":
        return f"impact_strong_s{_threshold_tag(strength_threshold)}"
    if mode == "impact_timeline_detectable":
        return f"impact_timeline_detectable_t{_threshold_tag(impact_threshold)}"
    return mode


def binary_target_from_batch(
    labels: torch.Tensor,
    raw_y: torch.Tensor,
    remap: torch.Tensor,
    mode: str,
    impact_threshold: float = 0.5,
    strength_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute per-batch binary target as [B, 1] float tensor."""
    mode = mode.lower()
    if mode == "model_family":
        return remap[raw_y].unsqueeze(-1)
    if mode == "impact_detectable":
        return (labels[:, 2] > impact_threshold).float().unsqueeze(-1)
    if mode == "impact_strong":
        if labels.shape[1] >= 6:
            return (labels[:, 5] > strength_threshold).float().unsqueeze(-1)
        else:
            return (labels[:, 2] > impact_threshold).float().unsqueeze(-1)
    if mode == "impact_timeline_detectable":
        if labels.shape[1] >= 7:
            return (labels[:, 6] > impact_threshold).float().unsqueeze(-1)
        if labels.shape[1] >= 6:
            return (labels[:, 5] > strength_threshold).float().unsqueeze(-1)
        return (labels[:, 2] > impact_threshold).float().unsqueeze(-1)
    raise ValueError(f"Unknown binary target mode: {mode}")


def regression_targets_from_labels(
    labels: torch.Tensor,
    n_reg_targets: int,
    mode: str = "raw",
) -> torch.Tensor:
    """Build auxiliary regression targets from compact or timeline labels.

    Label columns:
        compact: 0 dm, 1 log_m, 2 raw n_subhalos, 3 log_t_last,
                 4 log10_M_hm, 5 impact_strength
        timeline extension: 6 n_detectable, 7 effective_n, 8 log_t_strongest,
                            9 strongest_t, 10+ timeline effective bins
    """
    mode = mode.lower()
    if mode == "raw":
        cols = [labels[:, 2]]
        if n_reg_targets >= 2 and labels.shape[1] >= 4:
            cols.append(labels[:, 3])
        elif n_reg_targets >= 2:
            cols.append(labels[:, 1])
    elif mode == "timeline_effective":
        if labels.shape[1] < 9:
            raise ValueError("timeline_effective regression requires label_schema='timeline'")
        cols = [labels[:, 7], labels[:, 8]]
        if n_reg_targets > 2 and labels.shape[1] > 10:
            for col_idx in range(10, min(labels.shape[1], 10 + n_reg_targets - 2)):
                cols.append(labels[:, col_idx])
    elif mode == "timeline_detectable":
        if labels.shape[1] < 9:
            raise ValueError("timeline_detectable regression requires label_schema='timeline'")
        cols = [labels[:, 6], labels[:, 8]]
        if n_reg_targets > 2 and labels.shape[1] > 10:
            for col_idx in range(10, min(labels.shape[1], 10 + n_reg_targets - 2)):
                cols.append(labels[:, col_idx])
    elif mode == "mass_time":
        # Characterize impact TYPE: subhalo mass (col 1) + time-since-impact
        # (col 3). Use for a detect+characterize multi-task model.
        cols = [labels[:, 1]]
        if n_reg_targets >= 2:
            cols.append(labels[:, 3])
    elif mode == "strength_time":
        if labels.shape[1] < 6:
            raise ValueError("strength_time regression requires impact_strength labels")
        cols = [labels[:, 5]]
        if n_reg_targets >= 2:
            cols.append(labels[:, 8] if labels.shape[1] >= 9 else labels[:, 3])
    else:
        raise ValueError(f"Unknown regression target mode: {mode}")

    while len(cols) < n_reg_targets:
        cols.append(torch.zeros_like(cols[0]))
    y_reg = torch.stack(cols[:n_reg_targets], dim=1)
    return torch.nan_to_num(y_reg, nan=0.0, posinf=0.0, neginf=0.0)


def limit_indices_balanced(indices: np.ndarray, targets: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Return a deterministic, class-balanced subset when possible."""
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    classes = np.unique(targets[indices])
    per_class = max(1, limit // max(len(classes), 1))
    for cls in classes:
        cls_idx = indices[targets[indices] == cls]
        take = min(per_class, len(cls_idx))
        selected.append(rng.choice(cls_idx, size=take, replace=False))
    chosen = np.concatenate(selected) if selected else indices[:0]
    if len(chosen) < limit:
        remaining = np.setdiff1d(indices, chosen, assume_unique=False)
        if len(remaining):
            extra = rng.choice(remaining, size=min(limit - len(chosen), len(remaining)), replace=False)
            chosen = np.concatenate([chosen, extra])
    rng.shuffle(chosen)
    return chosen


class CurriculumAugmentView(torch.utils.data.Dataset):
    """Apply lightweight training augmentation on top of one preloaded dataset."""

    def __init__(self, base_dataset, enabled: bool = True) -> None:
        self.base_dataset = base_dataset
        self.enabled = enabled
        self.strength = 1.0

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        data = self.base_dataset[idx].clone()
        if self.enabled and self.strength > 0.0:
            x = data.x.clone()
            strength = float(self.strength)
            x[:, 0] += torch.empty((), dtype=x.dtype).uniform_(-5.0, 5.0) * strength
            x[:, 1] += torch.randn(len(x), dtype=x.dtype) * (0.05 * strength)
            x[:, 3] += torch.randn(len(x), dtype=x.dtype) * (0.02 * strength)
            x[:, 4] += torch.randn(len(x), dtype=x.dtype) * (0.02 * strength)
            data.x = x
        return data


def get_curriculum_strength(epoch: int, cfg: dict) -> float:
    """Compute augmentation/loss strength for curriculum learning.

    The domain randomization itself is baked into the generated simulations.
    This value controls extra training augmentation and the M_hm loss ramp.

    Returns:
        Float in [0.0, 1.0] controlling noise/augmentation intensity.
    """
    curriculum = cfg.get("curriculum", {})
    if not curriculum.get("enabled", True):
        return 1.0  # No curriculum: full strength always

    stage1_end = curriculum.get("stage1_end", 50)
    stage2_end = curriculum.get("stage2_end", 150)

    if epoch <= stage1_end:
        return 0.0
    elif epoch <= stage2_end:
        progress = (epoch - stage1_end) / (stage2_end - stage1_end)
        return min(progress, 1.0)
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gnn_v2(args, cfg: dict) -> None:
    requested_device = str(cfg["training"].get("device", "cpu")).lower()
    cuda_available = torch.cuda.is_available()
    device = torch.device(cfg["training"]["device"] if cuda_available else "cpu")
    log.info("Training GNN V2 (binary + M_hm regression) on device: %s", device)
    log.info("Python executable: %s", sys.executable)
    log.info("PyTorch: %s (cuda_available=%s)", torch.__version__, cuda_available)
    if requested_device.startswith("cuda") and not cuda_available:
        log.warning(
            "Config requested CUDA, but this Python environment has CPU-only PyTorch. "
            "Large V2 training runs will be extremely slow until the CUDA PyTorch env is restored."
        )

    if cuda_available:
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_fraction = cfg["training"].get("cuda_memory_fraction", 0.75)
        torch.cuda.set_per_process_memory_fraction(vram_fraction)
        log.info("CUDA allocator capped at %.0f%% of VRAM (%.1f / %.1f GB)",
                 vram_fraction * 100, vram_total_gb * vram_fraction, vram_total_gb)

    sim_dir = Path(args.sim_dir or cfg["paths"]["simulations"])
    ckpt_dir = Path(args.checkpoint_dir or cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    v2_cfg = cfg.get("training_v2", {})
    binary_target_mode = (args.binary_target or v2_cfg.get("binary_target", "model_family")).lower()
    label_schema = (args.label_schema or v2_cfg.get("label_schema", "compact")).lower()
    regression_target_mode = (
        args.regression_target
        or v2_cfg.get("regression_target", "raw")
    ).lower()
    if (
        binary_target_mode == "impact_timeline_detectable"
        or regression_target_mode.startswith("timeline")
    ) and label_schema != "timeline":
        log.info(
            "Switching label_schema to 'timeline' because target mode requires timeline labels."
        )
        label_schema = "timeline"
    impact_threshold = float(v2_cfg.get("impact_threshold", 0.5))
    strength_threshold = float(
        args.strength_threshold
        if args.strength_threshold is not None
        else v2_cfg.get("strength_threshold", 0.5)
    )
    v2_cfg["binary_target"] = binary_target_mode
    v2_cfg["label_schema"] = label_schema
    v2_cfg["regression_target"] = regression_target_mode
    v2_cfg["impact_threshold"] = impact_threshold
    v2_cfg["strength_threshold"] = strength_threshold
    if args.pos_weight_multiplier is not None:
        v2_cfg["pos_weight_multiplier"] = float(args.pos_weight_multiplier)
    if args.label_smoothing is not None:
        v2_cfg["label_smoothing"] = float(args.label_smoothing)
    if args.gamma_reg is not None:
        v2_cfg["gamma_reg"] = float(args.gamma_reg)
    if args.no_extra_augmentation:
        cfg["preprocessing"]["augmentation"]["enabled"] = False
    if args.disable_curriculum:
        v2_cfg.setdefault("curriculum", {})["enabled"] = False
    binary_names = BINARY_TARGET_NAMES.get(binary_target_mode, ["Class 0", "Class 1"])
    max_train_batches = args.max_train_batches or v2_cfg.get("max_train_batches")
    max_val_batches = args.max_val_batches or v2_cfg.get("max_val_batches")
    log_every_batches = v2_cfg.get("log_every_batches", 0)
    profile_cfg = cfg["graph"].get("profile_branch", {})
    use_profile_branch = bool(args.use_profile_branch or profile_cfg.get("enabled", False))
    profile_n_bins = int(profile_cfg.get("n_bins", 48))
    profile_feature_set = str(
        args.profile_feature_set or profile_cfg.get("feature_set", "compact")
    ).lower()
    profile_include_stream_onehot = bool(profile_cfg.get("include_stream_onehot", True))
    profile_layer_norm = bool(args.profile_layer_norm or profile_cfg.get("layer_norm", False))
    profile_features_path = (
        args.profile_features_path
        or profile_cfg.get("features_path")
    )
    downsample_seed = args.downsample_seed
    if use_profile_branch and profile_features_path and downsample_seed is None:
        downsample_seed = 42
    profile_dim = (
        profile_feature_dim(profile_n_bins, profile_feature_set, profile_include_stream_onehot)
        if use_profile_branch else 0
    )
    if use_profile_branch:
        profile_cfg["enabled"] = True
        profile_cfg["n_bins"] = profile_n_bins
        profile_cfg["feature_set"] = profile_feature_set
        profile_cfg["include_stream_onehot"] = profile_include_stream_onehot
        profile_cfg["layer_norm"] = profile_layer_norm
        if profile_features_path:
            profile_cfg["features_path"] = str(profile_features_path)
            profile_cfg["downsample_seed"] = int(downsample_seed)

    # ── Dataset and splits ───────────────────────────────────────────────────
    use_orbital = cfg["graph"].get("orbital_features", {}).get("enabled", False)
    log.info("Building raw dataset (orbital_features=%s)...", use_orbital)

    error_dr_cfg = v2_cfg.get("error_domain_randomization") or {}
    if getattr(args, "error_dr", False):
        error_dr_cfg = {**error_dr_cfg, "enabled": True}
    if error_dr_cfg.get("enabled"):
        log.info("Error domain randomization ENABLED: %s", error_dr_cfg)
        # The default config profile-feature cache was built WITHOUT error-DR and
        # would be stale. If the user explicitly passes --profile-features-path
        # (an error-DR cache built by precompute_profile_features.py --error-dr),
        # trust it; otherwise build profile features on the fly.
        if args.profile_features_path:
            log.info("  Using explicitly provided error-DR profile cache: %s",
                     args.profile_features_path)
        elif profile_features_path:
            log.info("  Ignoring stale config profile-feature cache (building on the fly).")
            profile_features_path = None
    else:
        error_dr_cfg = None
    dataset_raw = StreamSimDataset(
        sim_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        preload_ram=False,
        use_orbital_features=use_orbital,
        downsample_seed=downsample_seed,
        label_schema=label_schema,
        error_dr=error_dr_cfg,
    )
    if len(dataset_raw) == 0:
        raise RuntimeError(f"No simulations found in {sim_dir}. Run generate_training_data.py first.")

    split_cfg = cfg["training"].get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    split_seed = cfg["training"].get("split_seed", 42)
    use_stratified = cfg["training"].get("split_stratified", True)

    label_matrix = dataset_raw.get_label_matrix()
    dm_labels = label_matrix[:, 0].astype(np.int64)
    binary_targets_np = binary_targets_from_label_matrix(
        label_matrix,
        binary_target_mode,
        impact_threshold=impact_threshold,
        strength_threshold=strength_threshold,
    )
    log.info("Binary target mode: %s  class0=%d class1=%d",
             binary_target_mode,
             int((binary_targets_np == 0).sum()),
             int((binary_targets_np == 1).sum()))
    if len(np.unique(binary_targets_np)) < 2:
        raise RuntimeError(
            f"Binary target {binary_target_mode!r} has only one class in {sim_dir}."
        )

    split_tag = split_tag_for_target(binary_target_mode, impact_threshold, strength_threshold)
    split_path = ckpt_dir / f"split_n{len(dataset_raw)}_seed{split_seed}_{split_tag}.npz"
    stratify_labels = binary_targets_np if binary_target_mode != "model_family" else dm_labels
    if split_path.exists():
        split_indices = load_split_indices(split_path)
        expected_keys = {"train", "val", "test"}
        max_idx = max(int(v.max()) for v in split_indices.values() if len(v))
        if set(split_indices) != expected_keys or max_idx >= len(dataset_raw):
            log.warning("Cached split %s is incompatible with this dataset; recomputing.", split_path)
            split_path.unlink()
            split_indices = compute_split_indices(
                len(dataset_raw),
                stratify_labels if use_stratified else None,
                split_cfg,
                split_seed,
            )
            save_split_indices(split_indices, split_path)
        log.info("Loaded cached split indices (hash=%s)", get_split_hash(split_indices))
    else:
        split_indices = compute_split_indices(
            len(dataset_raw),
            stratify_labels if use_stratified else None,
            split_cfg,
            split_seed,
        )
        save_split_indices(split_indices, split_path)
        log.info("Split indices saved (hash=%s)", get_split_hash(split_indices))

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    train_idx = limit_indices_balanced(
        train_idx,
        binary_targets_np,
        int(args.limit_train_examples or 0),
        split_seed,
    )
    val_idx = limit_indices_balanced(
        val_idx,
        binary_targets_np,
        int(args.limit_val_examples or 0),
        split_seed + 1,
    )
    n_train, n_val = len(train_idx), len(val_idx)
    log.info("Dataset split: %d train / %d val / %d test (stratified=%s, seed=%d)",
             n_train, n_val, len(split_indices["test"]), use_stratified, split_seed)

    # Compute normalizer on training split
    train_subset = torch.utils.data.Subset(dataset_raw, train_idx.tolist())
    normalizer_samples = min(args.normalizer_samples or 2000, n_train)
    log.info("Computing feature normalizer from training split (%d sims, %d samples) ...",
             n_train, normalizer_samples)
    normalizer = compute_dataset_normalizer(train_subset, n_samples=normalizer_samples)

    from src.models.utils import save_normalizer
    save_normalizer(ckpt_dir / "normalizer_v2.npz", normalizer.mean.numpy(), normalizer.std.numpy())
    log.info("Normalizer saved to %s", ckpt_dir / "normalizer_v2.npz")

    # Build one normalized, preloaded base dataset. The training view applies
    # augmentation on the fly, avoiding two full RAM copies of the 100K dataset.
    preload_ram = bool(v2_cfg.get("preload_ram", True)) and not args.no_preload_ram
    log.info("Building normalized base dataset (preload_ram=%s) ...", preload_ram)
    dataset_base = StreamSimDataset(
        sim_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        normalizer=normalizer,
        preload_ram=preload_ram,
        use_orbital_features=use_orbital,
        profile_features_path=profile_features_path if use_profile_branch else None,
        downsample_seed=downsample_seed,
        label_schema=label_schema,
        error_dr=error_dr_cfg,
    )
    train_view = CurriculumAugmentView(
        dataset_base,
        enabled=cfg["preprocessing"]["augmentation"]["enabled"],
    )

    train_set = torch.utils.data.Subset(train_view, train_idx.tolist())
    val_set = torch.utils.data.Subset(dataset_base, val_idx.tolist())

    batch_size = args.batch_size or cfg["training_v2"].get("batch_size", 64)
    n_workers = cfg["training"]["n_workers_dataloader"] if sys.platform != "win32" else 0
    train_loader = PyGDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = PyGDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # ── Compute class balance for binary BCE ─────────────────────────────────
    log.info("Computing binary class balance ...")
    train_binary_targets = binary_targets_np[train_idx]
    n_suppressed = int((train_binary_targets == 1).sum())
    n_unsuppressed = int((train_binary_targets == 0).sum())
    # pos_weight compensates if classes are imbalanced
    # pos_weight = n_negative / n_positive (for BCEWithLogitsLoss)
    pos_weight_multiplier = float(
        args.pos_weight_multiplier
        if args.pos_weight_multiplier is not None
        else v2_cfg.get("pos_weight_multiplier", 1.0)
    )
    pos_weight = torch.tensor(
        [n_unsuppressed / max(n_suppressed, 1) * pos_weight_multiplier],
        dtype=torch.float32,
    )
    log.info(
        "Binary balance: %d %s / %d %s, pos_weight=%.3f (multiplier=%.3f)",
        n_unsuppressed,
        binary_names[0],
        n_suppressed,
        binary_names[1],
        pos_weight.item(),
        pos_weight_multiplier,
    )

    # ── Build model ──────────────────────────────────────────────────────────
    gcfg = cfg["model"]["gnn"]
    n_reg_targets = v2_cfg.get("n_reg_targets", 2)
    predict_uncertainty = v2_cfg.get("predict_uncertainty", False)
    ms_cfg = cfg["graph"].get("multi_scale", {})
    if getattr(args, "use_multi_scale", False):
        ms_cfg["enabled"] = True
        log.info("Multi-scale segment graph enabled via CLI flag.")

    model = StreamGNNMultiTaskV2(
        n_reg_targets=n_reg_targets,
        predict_uncertainty=predict_uncertainty,
        profile_dim=profile_dim,
        profile_hidden_dim=int(profile_cfg.get("hidden_dim", 64)),
        profile_layer_norm=profile_layer_norm,
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

    total, trainable = count_parameters(model)
    log.info("GNN V2 parameters: %d total, %d trainable", total, trainable)
    if use_profile_branch:
        log.info(
            "Profile branch enabled: feature_set=%s n_bins=%d profile_dim=%d layer_norm=%s cache=%s",
            profile_feature_set,
            profile_n_bins,
            profile_dim,
            profile_layer_norm,
            profile_features_path or "dynamic",
        )

    lr = v2_cfg.get("learning_rate", 1e-3)
    wd = v2_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Cosine annealing with warm restarts
    T_0 = v2_cfg.get("T_0", 100)
    T_mult = v2_cfg.get("T_mult", 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6,
    )

    # ── Resume from checkpoint ───────────────────────────────────────────────
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device=str(device))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        log.info("Resuming from epoch %d (best_val_loss=%.4f)", start_epoch, best_val_loss)
    else:
        start_epoch = 1
        best_val_loss = float("inf")

    # ── Loss functions ───────────────────────────────────────────────────────
    label_smoothing = float(
        args.label_smoothing
        if args.label_smoothing is not None
        else v2_cfg.get("label_smoothing", 0.1)
    )
    # Binary cross entropy (with pos_weight for class balance)
    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    # M_hm regression: Huber loss (robust to outliers)
    mhm_criterion = nn.HuberLoss(delta=1.0)
    # n_impacts regression: Huber loss
    reg_criterion = nn.HuberLoss(delta=1.0)

    # Loss weights
    alpha_binary = v2_cfg.get("alpha_binary", 1.0)
    beta_mhm = v2_cfg.get("beta_mhm", 1.0)
    gamma_reg = float(args.gamma_reg if args.gamma_reg is not None else v2_cfg.get("gamma_reg", 0.3))
    if (
        binary_target_mode != "model_family"
        and v2_cfg.get("disable_mhm_for_observable_targets", True)
    ):
        log.info(
            "Disabling M_hm loss for observable binary target %s; train M_hm in a later stage.",
            binary_target_mode,
        )
        beta_mhm = 0.0

    binary_remap_dev = BINARY_REMAP.to(device)

    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    patience_counter = 0
    early_stopping_patience = v2_cfg.get("early_stopping_patience", 80)
    best_val_acc = 0.0
    best_val_acc_epoch = 0
    norm_mean_dev = normalizer.mean.to(device)
    norm_std_dev = normalizer.std.to(device)
    use_multi_scale = ms_cfg.get("enabled", False)
    cache_flush_interval = 300

    # Training history for logging
    history = {"epoch": [], "train_binary_acc": [], "val_binary_acc": [],
               "train_loss": [], "val_loss": [], "mhm_rmse": [], "curriculum_strength": []}

    log.info("=" * 80)
    log.info("Starting GNN V2 training: %d epochs, batch_size=%d, lr=%.1e",
             args.epochs, batch_size, lr)
    log.info("Binary target: %s", binary_target_mode)
    log.info("Label schema: %s  Regression target: %s", label_schema, regression_target_mode)
    log.info("Fast-run limits: max_train_batches=%s max_val_batches=%s log_every_batches=%s",
             max_train_batches, max_val_batches, log_every_batches)
    log.info("Loss weights: alpha_binary=%.2f, beta_mhm=%.2f, gamma_reg=%.2f",
             alpha_binary, beta_mhm, gamma_reg)
    log.info("Curriculum: stage1_end=%d, stage2_end=%d",
             v2_cfg.get("curriculum", {}).get("stage1_end", 50),
             v2_cfg.get("curriculum", {}).get("stage2_end", 150))
    log.info("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        curriculum_str = get_curriculum_strength(epoch, v2_cfg)
        train_view.strength = curriculum_str

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_bce, train_mhm, train_reg = 0.0, 0.0, 0.0
        train_correct, train_total = 0, 0
        train_mhm_se, train_mhm_n = 0.0, 0  # For RMSE tracking
        per_class = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}}

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            build_knn_graph_batched(batch, cfg["graph"]["k_neighbors"], normalizer,
                                   orbital_features=use_orbital)
            if use_profile_branch:
                build_profile_features_batched(
                    batch,
                    n_bins=profile_n_bins,
                    feature_set=profile_feature_set,
                    include_stream_onehot=profile_include_stream_onehot,
                )
            if use_multi_scale:
                build_segment_graph(batch, n_segments=ms_cfg.get("n_segments", 20),
                                    k_seg=ms_cfg.get("k_segment_neighbors", 4))
            # Normalize features
            batch.x = (batch.x - norm_mean_dev) / norm_std_dev

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, binary_logit, mhm_out, reg_out = model(batch)

                # Extract labels
                labels = batch.y.view(batch.num_graphs, -1)
                raw_y = labels[:, 0].long()

                # Binary target can be model-family suppression or an observable
                # signal-ladder target such as impact/no-impact.
                y_binary = binary_target_from_batch(
                    labels,
                    raw_y,
                    binary_remap_dev,
                    binary_target_mode,
                    impact_threshold=impact_threshold,
                    strength_threshold=strength_threshold,
                )

                # M_hm target: Plan 2 data stores log10(M_hm) in label column 4.
                # Legacy files fall back to an approximate proxy for compatibility.
                if labels.shape[1] >= 5:
                    # New format: [dm_idx, log_m, n_sub, log_t, log_mhm]
                    y_mhm = labels[:, 4].unsqueeze(-1)  # [B, 1]
                else:
                    # Legacy format: use log10_M_sub as proxy for M_hm.
                    # Unsuppressed values are masked out of the M_hm loss.
                    y_mhm_raw = labels[:, 1].clone()  # log10_M_sub
                    is_unsuppressed = (y_binary.squeeze(-1) < 0.5)
                    y_mhm_raw[is_unsuppressed] = MHM_UNSUPPRESSED_LEGACY_SENTINEL
                    y_mhm = y_mhm_raw.unsqueeze(-1)  # [B, 1]

                y_reg = regression_targets_from_labels(
                    labels,
                    n_reg_targets,
                    regression_target_mode,
                )

                # ── Compute losses ───────────────────────────────────────────
                # Binary BCE
                if label_smoothing > 0:
                    y_binary_loss = y_binary * (1.0 - label_smoothing) + 0.5 * label_smoothing
                else:
                    y_binary_loss = y_binary
                loss_bce = bce_criterion(binary_logit, y_binary_loss)

                # M_hm regression is meaningful for WDM/FDM simulations even
                # when the active binary target is impact/no-impact.
                mhm_mask = torch.isin(raw_y, torch.tensor([1, 2], device=device))
                if beta_mhm > 0 and mhm_mask.any():
                    if predict_uncertainty:
                        mhm_mean = mhm_out[mhm_mask, 0:1]
                        mhm_logvar = mhm_out[mhm_mask, 1:2]
                        # Heteroscedastic loss: -log p(y|mean, var)
                        loss_mhm = 0.5 * (torch.exp(-mhm_logvar) *
                                           (y_mhm[mhm_mask] - mhm_mean)**2
                                           + mhm_logvar).mean()
                    else:
                        loss_mhm = mhm_criterion(mhm_out[mhm_mask], y_mhm[mhm_mask])
                    # Track RMSE
                    with torch.no_grad():
                        if predict_uncertainty:
                            mhm_pred_track = mhm_out[mhm_mask, 0:1]
                        else:
                            mhm_pred_track = mhm_out[mhm_mask]
                        se = ((mhm_pred_track - y_mhm[mhm_mask])**2).sum().item()
                        train_mhm_se += se
                        train_mhm_n += mhm_mask.sum().item()
                else:
                    loss_mhm = torch.tensor(0.0, device=device)

                # Regression loss
                loss_reg = reg_criterion(reg_out, y_reg)

                # Curriculum: stage 1 uses low augmentation and lower M_hm weight.
                # Domain randomization is already baked into the HDF5 simulations.
                mhm_weight = beta_mhm if epoch > v2_cfg.get("curriculum", {}).get("stage1_end", 50) else beta_mhm * 0.5
                loss = alpha_binary * loss_bce + mhm_weight * loss_mhm + gamma_reg * loss_reg

            if not torch.isfinite(loss):
                log.warning("NaN/Inf loss at epoch %d batch %d -- skipping", epoch, batch_idx)
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            train_bce += loss_bce.item() * len(y_binary)
            train_mhm += loss_mhm.item() * len(y_binary)
            train_reg += loss_reg.item() * len(y_binary)

            # Binary accuracy
            preds = (binary_logit.squeeze(-1) > 0.0).float()
            y_bin_flat = y_binary.squeeze(-1)
            train_correct += (preds == y_bin_flat).sum().item()
            train_total += len(y_bin_flat)
            for cls_i in [0, 1]:
                mask = (y_bin_flat == cls_i)
                per_class[cls_i]["total"] += mask.sum().item()
                per_class[cls_i]["correct"] += (preds[mask] == cls_i).sum().item()

            if device.type == "cuda" and (batch_idx + 1) % cache_flush_interval == 0:
                torch.cuda.empty_cache()
            if log_every_batches and (batch_idx + 1) % log_every_batches == 0:
                elapsed = time.time() - epoch_t0
                batches_per_sec = (batch_idx + 1) / max(elapsed, 1e-6)
                log.info(
                    "Epoch %d train batch %d/%d  acc=%.3f  mhm_rmse=%.3f  %.2f batches/s",
                    epoch,
                    batch_idx + 1,
                    len(train_loader),
                    train_correct / max(train_total, 1),
                    math.sqrt(train_mhm_se / max(train_mhm_n, 1)),
                    batches_per_sec,
                )

            if max_train_batches and (batch_idx + 1) >= max_train_batches:
                log.info("Stopping training epoch early after %d batches (--max-train-batches).",
                         max_train_batches)
                break

        scheduler.step()
        train_bce /= max(train_total, 1)
        train_mhm /= max(train_total, 1)
        train_reg /= max(train_total, 1)
        train_loss = alpha_binary * train_bce + beta_mhm * train_mhm + gamma_reg * train_reg
        train_acc = train_correct / max(train_total, 1)
        train_mhm_rmse = math.sqrt(train_mhm_se / max(train_mhm_n, 1))

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_bce, val_mhm, val_reg = 0.0, 0.0, 0.0
        val_correct, val_total = 0, 0
        val_mhm_se, val_mhm_n = 0.0, 0
        val_per_class = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}}

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = batch.to(device)
                build_knn_graph_batched(batch, cfg["graph"]["k_neighbors"], normalizer,
                                       orbital_features=use_orbital)
                if use_profile_branch:
                    build_profile_features_batched(
                        batch,
                        n_bins=profile_n_bins,
                        feature_set=profile_feature_set,
                        include_stream_onehot=profile_include_stream_onehot,
                    )
                if use_multi_scale:
                    build_segment_graph(batch, n_segments=ms_cfg.get("n_segments", 20),
                                        k_seg=ms_cfg.get("k_segment_neighbors", 4))
                batch.x = (batch.x - norm_mean_dev) / norm_std_dev

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    _, binary_logit, mhm_out, reg_out = model(batch)
                    labels = batch.y.view(batch.num_graphs, -1)
                    raw_y = labels[:, 0].long()
                    y_binary = binary_target_from_batch(
                        labels,
                        raw_y,
                        binary_remap_dev,
                        binary_target_mode,
                        impact_threshold=impact_threshold,
                        strength_threshold=strength_threshold,
                    )

                    if labels.shape[1] >= 5:
                        y_mhm = labels[:, 4].unsqueeze(-1)
                    else:
                        y_mhm_raw = labels[:, 1].clone()
                        is_unsuppressed = (y_binary.squeeze(-1) < 0.5)
                        y_mhm_raw[is_unsuppressed] = MHM_UNSUPPRESSED_LEGACY_SENTINEL
                        y_mhm = y_mhm_raw.unsqueeze(-1)

                    y_reg = regression_targets_from_labels(
                        labels,
                        n_reg_targets,
                        regression_target_mode,
                    )

                    if label_smoothing > 0:
                        y_binary_loss = y_binary * (1.0 - label_smoothing) + 0.5 * label_smoothing
                    else:
                        y_binary_loss = y_binary
                    loss_bce = bce_criterion(binary_logit, y_binary_loss)

                    mhm_mask = torch.isin(raw_y, torch.tensor([1, 2], device=device))
                    if beta_mhm > 0 and mhm_mask.any():
                        if predict_uncertainty:
                            mhm_mean = mhm_out[mhm_mask, 0:1]
                            mhm_logvar = mhm_out[mhm_mask, 1:2]
                            loss_mhm = 0.5 * (torch.exp(-mhm_logvar) *
                                               (y_mhm[mhm_mask] - mhm_mean)**2
                                               + mhm_logvar).mean()
                        else:
                            loss_mhm = mhm_criterion(mhm_out[mhm_mask], y_mhm[mhm_mask])
                        if predict_uncertainty:
                            mhm_pred_track = mhm_out[mhm_mask, 0:1]
                        else:
                            mhm_pred_track = mhm_out[mhm_mask]
                        se = ((mhm_pred_track - y_mhm[mhm_mask])**2).sum().item()
                        val_mhm_se += se
                        val_mhm_n += mhm_mask.sum().item()
                    else:
                        loss_mhm = torch.tensor(0.0, device=device)

                    loss_reg = reg_criterion(reg_out, y_reg)

                val_bce += loss_bce.item() * len(y_binary)
                val_mhm += loss_mhm.item() * len(y_binary)
                val_reg += loss_reg.item() * len(y_binary)

                preds = (binary_logit.squeeze(-1) > 0.0).float()
                y_bin_flat = y_binary.squeeze(-1)
                val_correct += (preds == y_bin_flat).sum().item()
                val_total += len(y_bin_flat)
                for cls_i in [0, 1]:
                    mask = (y_bin_flat == cls_i)
                    val_per_class[cls_i]["total"] += mask.sum().item()
                    val_per_class[cls_i]["correct"] += (preds[mask] == cls_i).sum().item()
                if max_val_batches and (batch_idx + 1) >= max_val_batches:
                    log.info("Stopping validation epoch early after %d batches (--max-val-batches).",
                             max_val_batches)
                    break

        val_bce /= max(val_total, 1)
        val_mhm /= max(val_total, 1)
        val_reg /= max(val_total, 1)
        val_loss = alpha_binary * val_bce + beta_mhm * val_mhm + gamma_reg * val_reg
        val_acc = val_correct / max(val_total, 1)
        val_mhm_rmse = math.sqrt(val_mhm_se / max(val_mhm_n, 1))
        new_best_acc = val_acc > best_val_acc
        if new_best_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch

        if device.type == "cuda":
            vram_used = torch.cuda.memory_reserved() / 1024**3
            torch.cuda.empty_cache()
        else:
            vram_used = 0.0

        # Determine curriculum stage
        stage1_end = v2_cfg.get("curriculum", {}).get("stage1_end", 50)
        stage2_end = v2_cfg.get("curriculum", {}).get("stage2_end", 150)
        if epoch <= stage1_end:
            stage_str = "STAGE1-low-aug"
        elif epoch <= stage2_end:
            stage_str = f"STAGE2-ramp({curriculum_str:.2f})"
        else:
            stage_str = "STAGE3-full"

        log.info(
            "Epoch %d/%d [%s]  bce=%.4f mhm=%.4f reg=%.4f acc=%.3f mhm_rmse=%.3f | "
            "val_bce=%.4f val_mhm=%.4f acc=%.3f mhm_rmse=%.3f  VRAM=%.1fGB elapsed=%.1fmin",
            epoch, args.epochs, stage_str,
            train_bce, train_mhm, train_reg, train_acc, train_mhm_rmse,
            val_bce, val_mhm, val_acc, val_mhm_rmse, vram_used,
            (time.time() - epoch_t0) / 60,
        )

        # Per-class accuracy every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            for cls_i, name in enumerate(binary_names):
                t = val_per_class[cls_i]["total"]
                c = val_per_class[cls_i]["correct"]
                acc_i = c / max(t, 1)
                log.info("  %s: %d/%d = %.3f", name, c, t, acc_i)

        # Checkpointing
        if epoch % cfg["training"]["checkpoint_every_n_epochs"] == 0:
            save_checkpoint(
                ckpt_dir / f"gnn_v2_epoch_{epoch}.pt",
                model, optimizer, epoch, val_loss, cfg, scheduler=scheduler,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                ckpt_dir / "gnn_v2_best.pt",
                model, optimizer, epoch, val_loss, cfg, scheduler=scheduler,
            )
            log.info("  >> New best val_loss=%.4f (saved gnn_v2_best.pt)", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                log.info("Early stopping at epoch %d", epoch)
                break

        if new_best_acc:
            save_checkpoint(
                ckpt_dir / "gnn_v2_best_acc.pt",
                model, optimizer, epoch, val_loss, cfg, scheduler=scheduler,
            )
            log.info("  >> New best val_binary_acc=%.3f (saved gnn_v2_best_acc.pt)", val_acc)

        # Track history
        history["epoch"].append(epoch)
        history["train_binary_acc"].append(train_acc)
        history["val_binary_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mhm_rmse"].append(val_mhm_rmse)
        history["curriculum_strength"].append(curriculum_str)

    # ── Save training history ────────────────────────────────────────────────
    history_path = ckpt_dir / "gnn_v2_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved to %s", history_path)

    log.info("=" * 80)
    log.info("GNN V2 training complete.")
    log.info("  Best val_loss: %.4f", best_val_loss)
    log.info("  Best val_binary_acc: %.3f (epoch %d)", best_val_acc, best_val_acc_epoch)
    log.info("  Final val_binary_acc: %.3f", val_acc)
    log.info("  Final val_mhm_rmse: %.3f dex", val_mhm_rmse)
    log.info("=" * 80)

    if best_val_acc < 0.80:
        log.warning("WARNING: Best binary accuracy %.3f < 0.80 target.", best_val_acc)
    if val_mhm_rmse > 0.5:
        log.warning("WARNING: M_hm RMSE %.3f dex > 0.5 dex target.", val_mhm_rmse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN V2 (binary + M_hm regression).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--sim-dir", default=None,
                        help="Override cfg paths.simulations, useful for signal-ladder datasets.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint/output directory, useful for smoke tests.")
    parser.add_argument("--binary-target", choices=["model_family", "impact_detectable", "impact_strong", "impact_timeline_detectable"],
                        default=None,
                        help="Binary head target: DM suppression family, any impact, or morphologically strong impact.")
    parser.add_argument("--label-schema", choices=["compact", "timeline"], default=None,
                        help="Use compact labels or append timeline-aware impact labels.")
    parser.add_argument("--regression-target", choices=["raw", "timeline_effective", "timeline_detectable", "strength_time", "mass_time"],
                        default=None,
                        help="Auxiliary regression targets for the reg head.")
    parser.add_argument("--strength-threshold", type=float, default=None,
                        help="Impact strength threshold for impact_strong target (default: 0.5).")
    parser.add_argument("--use-profile-branch", action="store_true",
                        help="Add along-stream density/profile summary features to the V2 heads.")
    parser.add_argument("--profile-feature-set", choices=["compact", "summary"], default=None,
                        help="Profile branch features: compact density roughness or RF-style summary.")
    parser.add_argument("--profile-layer-norm", action="store_true",
                        help="Apply LayerNorm before the profile MLP.")
    parser.add_argument("--profile-features-path", default=None,
                        help="Optional precomputed profile-feature cache from precompute_profile_features.py.")
    parser.add_argument("--downsample-seed", type=int, default=None,
                        help="Deterministic node downsampling seed; defaults to 42 when using a profile cache.")
    parser.add_argument("--use-multi-scale", action="store_true",
                        help="Enable multi-scale segment graph (overrides config graph.multi_scale.enabled).")
    parser.add_argument("--normalizer-samples", type=int, default=None,
                        help="Override the number of training examples used for feature normalization.")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Stop each training epoch after this many batches, for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Stop each validation epoch after this many batches, for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training_v2.batch_size.")
    parser.add_argument("--pos-weight-multiplier", type=float, default=None,
                        help="Multiply BCE positive-class weight; useful for impact-recall probes.")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="Override training_v2.label_smoothing.")
    parser.add_argument("--gamma-reg", type=float, default=None,
                        help="Override training_v2.gamma_reg.")
    parser.add_argument("--no-extra-augmentation", action="store_true",
                        help="Disable lightweight train-time augmentation; generated domain randomization remains.")
    parser.add_argument("--disable-curriculum", action="store_true",
                        help="Disable the V2 curriculum schedule for this run.")
    parser.add_argument("--error-dr", action="store_true",
                        help="Enable error domain randomization (realistic varying per-star errors + "
                             "RV masking) to close the sim-to-real gap. Forces on-the-fly profile "
                             "features since the cache would be stale.")
    parser.add_argument("--limit-train-examples", type=int, default=None,
                        help="Use a balanced subset of the train split for signal-ladder probes.")
    parser.add_argument("--limit-val-examples", type=int, default=None,
                        help="Use a balanced subset of the validation split for signal-ladder probes.")
    parser.add_argument("--no-preload-ram", action="store_true",
                        help="Disable RAM preloading of the normalized dataset.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_gnn_v2(args, cfg)
