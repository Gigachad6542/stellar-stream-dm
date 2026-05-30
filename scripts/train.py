"""
Train the GNN encoder (and optional baseline CNN) on the simulation dataset.

Training stages:
1. Baseline CNN on 1D density profiles (validates data pipeline)
2. GNN encoder on full phase-space graphs with MULTI-TASK training:
   - 3-class classification (CDM+SIDM / WDM / FDM)
   - Regression on log10_M_sub_mean, n_impacts, [log10_t_since_impact]
   CDM and SIDM are merged because their impulse approximation signatures
   are indistinguishable (scale radius tweak is <0.01% at typical b).
   Set model.n_reg_targets=3 in config to include perturbation age.

Usage:
    python scripts/train.py --model gnn --epochs 200
    python scripts/train.py --model baseline --epochs 100
    python scripts/train.py --model gnn --resume checkpoints/gnn_epoch_50.pt
"""

from __future__ import annotations

# sys.path MUST be patched before any src.* imports; doing it here (before even
# importing numpy/torch) prevents a Windows/zipimport circular-import crash in
# Python 3.10 where inserting a path mid-import can trigger recursive zoneinfo loads.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import yaml
# Import PyG DataLoader at module level — this pre-initialises torch_geometric and
# avoids a Python 3.11/Windows zoneinfo import deadlock that occurs when
# torch.utils.data is imported before torch_geometric in the same process.
# NOTE: h5py must NOT be imported at module level before torch_geometric on Windows —
# the combination of their DLLs causes a STATUS_ACCESS_VIOLATION (0xC0000005).
# h5py is imported lazily inside train_baseline() instead.
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.dataset import StreamSimDataset, build_knn_graph_batched, build_segment_graph, compute_dataset_normalizer
from src.data.splits import compute_split_indices, save_split_indices, load_split_indices, get_split_hash
from src.models.baseline import DensityProfileCNN, compute_1d_profile
from src.models.gnn import StreamGNNMultiTask
from src.models.utils import (
    build_scheduler,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Original 4 DM model labels in the HDF5 files
DM_MODELS_RAW = ["CDM", "WDM", "FDM", "SIDM"]

# 3-class scheme: merge CDM (idx=0) and SIDM (idx=3) because their impulse
# approximation signatures are indistinguishable (scale radius tweak is <0.01%).
# Class 0 = CDM+SIDM, Class 1 = WDM, Class 2 = FDM
DM_MODELS_3CLASS = ["CDM+SIDM", "WDM", "FDM"]
N_CLASSES = 3
# Remap raw dm_model_idx → 3-class label: CDM(0)→0, WDM(1)→1, FDM(2)→2, SIDM(3)→0
CLASS_REMAP = torch.tensor([0, 1, 2, 0], dtype=torch.long)


# ---------------------------------------------------------------------------
# GNN training
# ---------------------------------------------------------------------------

def train_gnn(args, cfg: dict) -> None:
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    log.info("Training GNN (multi-task, 3-class) on device: %s", device)

    # Cap PyTorch's CUDA allocator to 75% of total VRAM (~18.4 GB on a 24 GB card).
    if torch.cuda.is_available():
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_fraction = cfg["training"].get("cuda_memory_fraction", 0.75)
        torch.cuda.set_per_process_memory_fraction(vram_fraction)
        log.info("CUDA allocator capped at %.0f%% of VRAM (%.1f / %.1f GB)",
                 vram_fraction * 100, vram_total_gb * vram_fraction, vram_total_gb)

    sim_dir = Path(cfg["paths"]["simulations"])
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: build raw dataset and split BEFORE computing normalizer ──────
    use_orbital_cfg = cfg["graph"].get("orbital_features", {}).get("enabled", False)
    log.info("Building raw dataset for splitting (orbital_features=%s)...", use_orbital_cfg)
    dataset_raw = StreamSimDataset(
        sim_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        preload_ram=False,
        use_orbital_features=use_orbital_cfg,
    )
    if len(dataset_raw) == 0:
        raise RuntimeError(f"No simulations found in {sim_dir}. Run generate_training_data.py first.")

    # ── Stratified splits via src/data/splits.py ─────────────────────────────
    split_cfg = cfg["training"].get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    split_seed = cfg["training"].get("split_seed", 42)
    use_stratified = cfg["training"].get("split_stratified", True)

    # Try to load cached split indices; recompute if missing
    split_path = ckpt_dir / f"split_seed{split_seed}.npz"
    if split_path.exists():
        split_indices = load_split_indices(split_path)
        log.info("Loaded cached split indices (hash=%s)", get_split_hash(split_indices))
    else:
        # Extract DM model labels for stratification
        dm_labels = None
        if use_stratified:
            log.info("Extracting DM model labels for stratified splitting ...")
            dm_labels = np.array([int(dataset_raw[i].y[0].item()) for i in range(len(dataset_raw))])
        split_indices = compute_split_indices(
            len(dataset_raw), dm_labels, split_cfg, split_seed,
        )
        save_split_indices(split_indices, split_path)
        log.info("Split indices saved (hash=%s)", get_split_hash(split_indices))

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
    log.info("Dataset split: %d train / %d val / %d test (stratified=%s, seed=%d)",
             n_train, n_val, n_test, use_stratified, split_seed)

    # Compute normalizer on TRAINING split only (no test leakage)
    train_subset = torch.utils.data.Subset(dataset_raw, train_idx.tolist())
    log.info("Computing feature normalizer from training split (%d sims) ...", n_train)
    normalizer = compute_dataset_normalizer(train_subset, n_samples=min(2000, n_train))
    log.info("Feature means: %s", normalizer.mean.numpy().round(3))
    log.info("Feature stds : %s", normalizer.std.numpy().round(3))

    from src.models.utils import save_normalizer  # noqa: PLC0415
    save_normalizer(ckpt_dir / "normalizer.npz", normalizer.mean.numpy(), normalizer.std.numpy())
    log.info("Normalizer saved to %s", ckpt_dir / "normalizer.npz")

    # ── Step 2: build normalized datasets ────────────────────────────────────
    dataset_aug = StreamSimDataset(
        sim_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=cfg["preprocessing"]["augmentation"]["enabled"],
        normalizer=normalizer,
        cache_dir=cfg["graph"].get("graph_cache_dir"),
        use_orbital_features=use_orbital_cfg,
    )
    dataset_clean = StreamSimDataset(
        sim_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        normalizer=normalizer,
        use_orbital_features=use_orbital_cfg,
    )

    # Use the same deterministic split indices (no re-splitting)
    train_set = torch.utils.data.Subset(dataset_aug, train_idx.tolist())
    val_set = torch.utils.data.Subset(dataset_clean, val_idx.tolist())
    test_set = torch.utils.data.Subset(dataset_clean, test_idx.tolist())

    n_workers = cfg["training"]["n_workers_dataloader"] if sys.platform != "win32" else 0
    train_loader = PyGDataLoader(train_set, batch_size=cfg["training"]["batch_size"],
                                  shuffle=True, num_workers=n_workers)
    val_loader = PyGDataLoader(val_set, batch_size=cfg["training"]["batch_size"],
                                shuffle=False, num_workers=n_workers)

    # ── Step 3: compute class weights for the 3-class scheme ─────────────────
    # After CDM+SIDM merge: class 0 has ~21K samples, classes 1,2 have ~10K each.
    # Inverse-frequency weighting prevents the classifier from being biased toward
    # the over-represented merged class.
    log.info("Computing 3-class weights from training labels ...")
    class_counts = torch.zeros(N_CLASSES, dtype=torch.float32)
    for data_i in train_set:
        raw_idx = int(data_i.y[0].item())
        mapped = CLASS_REMAP[raw_idx].item()
        class_counts[mapped] += 1
    # Inverse-frequency weights, normalized so they sum to N_CLASSES
    class_weights = (class_counts.sum() / (N_CLASSES * class_counts)).clamp(min=0.3, max=3.0)
    log.info("3-class counts: %s  weights: %s",
             {n: int(class_counts[i]) for i, n in enumerate(DM_MODELS_3CLASS)},
             class_weights.numpy().round(3))
    class_weights_dev = class_weights.to(device)

    # ── Step 4: build model ──────────────────────────────────────────────────
    gcfg = cfg["model"]["gnn"]
    n_reg_targets = cfg["model"].get("n_reg_targets", 2)
    ms_model_cfg = cfg["graph"].get("multi_scale", {})
    model = StreamGNNMultiTask(
        n_classes=N_CLASSES,
        n_reg_targets=n_reg_targets,
        n_node_features=cfg["graph"]["n_node_features"],
        n_edge_features=cfg["graph"]["n_edge_features"],
        hidden_dim=gcfg["hidden_dim"],
        n_layers=gcfg["n_layers"],
        embedding_dim=gcfg["embedding_dim"],
        dropout=gcfg["dropout"],
        use_attention_readout=gcfg.get("use_attention_readout", False),
        use_multi_scale=ms_model_cfg.get("enabled", False),
        seg_embedding_dim=ms_model_cfg.get("seg_embedding_dim", 64),
    ).to(device)

    total, trainable = count_parameters(model)
    log.info("GNN MultiTask parameters: %d total, %d trainable", total, trainable)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg["training"], args.epochs)

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device=str(device))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        log.info("Resuming from epoch %d (best_val_loss=%.4f)", start_epoch, best_val_loss)
    else:
        start_epoch = 1
        best_val_loss = float("inf")

    if cfg["training"]["compile_model"] and hasattr(torch, "compile"):
        model = torch.compile(model)
        log.info("Model compiled with torch.compile")

    # ── Loss functions ───────────────────────────────────────────────────────
    # Multi-task loss: α*CE(3-class) + β*Huber(regression)
    # CE with class weights handles CDM+SIDM over-representation.
    # Huber (smooth L1) is robust to outliers from n_impacts=0 sims.
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights_dev, label_smoothing=0.05)
    reg_criterion = nn.HuberLoss(delta=1.0)
    alpha_cls = 1.0   # classification weight
    beta_reg = 0.3    # regression weight

    # Move CLASS_REMAP to device once
    class_remap_dev = CLASS_REMAP.to(device)

    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    patience_counter = 0
    norm_mean_dev = normalizer.mean.to(device) if normalizer else None
    norm_std_dev = normalizer.std.to(device) if normalizer else None
    use_orbital = cfg["graph"].get("orbital_features", {}).get("enabled", False)
    use_multi_scale = cfg["graph"].get("multi_scale", {}).get("enabled", False)
    ms_cfg = cfg["graph"].get("multi_scale", {})
    cache_flush_interval = 300

    for epoch in range(start_epoch, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_cls_loss, train_reg_loss, train_correct, train_total = 0.0, 0.0, 0, 0
        train_per_class = {i: {"correct": 0, "total": 0} for i in range(N_CLASSES)}
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            build_knn_graph_batched(batch, cfg["graph"]["k_neighbors"], normalizer,
                                       orbital_features=use_orbital)
            if use_multi_scale:
                build_segment_graph(batch, n_segments=ms_cfg.get("n_segments", 20),
                                    k_seg=ms_cfg.get("k_segment_neighbors", 4))
            if norm_mean_dev is not None:
                batch.x = (batch.x - norm_mean_dev) / norm_std_dev
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, logits, reg_out = model(batch)
                # Extract labels: [B, 3 or 4] → dm_model_idx, log_m, n_sub, [log_t]
                labels = batch.y.view(batch.num_graphs, -1)
                raw_y = labels[:, 0].long()
                y_cls = class_remap_dev[raw_y]           # 3-class remapped
                # Build regression target tensor matching n_reg_targets
                reg_cols = [labels[:, 1], labels[:, 2]]  # log_m, n_sub
                if n_reg_targets >= 3 and labels.shape[1] >= 4:
                    reg_cols.append(labels[:, 3])         # log_t_since_impact
                y_reg = torch.stack(reg_cols[:n_reg_targets], dim=1)  # [B, n_reg_targets]

                loss_cls = cls_criterion(logits, y_cls)
                loss_reg = reg_criterion(reg_out, y_reg)
                loss = alpha_cls * loss_cls + beta_reg * loss_reg

            if not torch.isfinite(loss):
                log.warning("NaN/Inf loss at epoch %d batch %d — skipping", epoch, batch_idx)
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            train_cls_loss += loss_cls.item() * len(y_cls)
            train_reg_loss += loss_reg.item() * len(y_cls)
            preds = logits.argmax(1)
            train_correct += (preds == y_cls).sum().item()
            train_total += len(y_cls)
            for cls_i in range(N_CLASSES):
                mask = (y_cls == cls_i)
                train_per_class[cls_i]["total"] += mask.sum().item()
                train_per_class[cls_i]["correct"] += (preds[mask] == cls_i).sum().item()

            if device.type == "cuda" and (batch_idx + 1) % cache_flush_interval == 0:
                torch.cuda.empty_cache()

        scheduler.step()
        train_cls_loss /= max(train_total, 1)
        train_reg_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_cls_loss, val_reg_loss, val_correct, val_total = 0.0, 0.0, 0, 0
        val_per_class = {i: {"correct": 0, "total": 0} for i in range(N_CLASSES)}
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                build_knn_graph_batched(batch, cfg["graph"]["k_neighbors"], normalizer,
                                       orbital_features=use_orbital)
                if use_multi_scale:
                    build_segment_graph(batch, n_segments=ms_cfg.get("n_segments", 20),
                                        k_seg=ms_cfg.get("k_segment_neighbors", 4))
                if norm_mean_dev is not None:
                    batch.x = (batch.x - norm_mean_dev) / norm_std_dev
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    _, logits, reg_out = model(batch)
                    labels = batch.y.view(batch.num_graphs, -1)
                    raw_y = labels[:, 0].long()
                    y_cls = class_remap_dev[raw_y]
                    reg_cols = [labels[:, 1], labels[:, 2]]
                    if n_reg_targets >= 3 and labels.shape[1] >= 4:
                        reg_cols.append(labels[:, 3])
                    y_reg = torch.stack(reg_cols[:n_reg_targets], dim=1)

                    loss_cls = cls_criterion(logits, y_cls)
                    loss_reg = reg_criterion(reg_out, y_reg)

                val_cls_loss += loss_cls.item() * len(y_cls)
                val_reg_loss += loss_reg.item() * len(y_cls)
                preds = logits.argmax(1)
                val_correct += (preds == y_cls).sum().item()
                val_total += len(y_cls)
                for cls_i in range(N_CLASSES):
                    mask = (y_cls == cls_i)
                    val_per_class[cls_i]["total"] += mask.sum().item()
                    val_per_class[cls_i]["correct"] += (preds[mask] == cls_i).sum().item()

        val_cls_loss /= max(val_total, 1)
        val_reg_loss /= max(val_total, 1)
        val_loss = alpha_cls * val_cls_loss + beta_reg * val_reg_loss
        val_acc = val_correct / max(val_total, 1)

        if device.type == "cuda":
            vram_used = torch.cuda.memory_reserved() / 1024**3
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_reserved() / 1024**3
        else:
            vram_used = vram_after = 0.0

        log.info("Epoch %d/%d  cls=%.4f reg=%.4f acc=%.3f | val_cls=%.4f val_reg=%.4f acc=%.3f  VRAM=%.1f->%.1fGB",
                 epoch, args.epochs, train_cls_loss, train_reg_loss, train_acc,
                 val_cls_loss, val_reg_loss, val_acc, vram_used, vram_after)

        if epoch % 5 == 0 or epoch == 1:
            cls_accs = []
            for i, name in enumerate(DM_MODELS_3CLASS):
                t = val_per_class[i]["total"]
                c = val_per_class[i]["correct"]
                acc_i = c / max(t, 1)
                cls_accs.append(f"{name}={acc_i:.3f}")
            log.info("  Per-class val_acc: %s", " | ".join(cls_accs))

        if epoch % cfg["training"]["checkpoint_every_n_epochs"] == 0:
            save_checkpoint(ckpt_dir / f"gnn_epoch_{epoch}.pt", model, optimizer, epoch, val_loss, cfg,
                            scheduler=scheduler)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(ckpt_dir / "gnn_best.pt", model, optimizer, epoch, val_loss, cfg,
                            scheduler=scheduler)
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                log.info("Early stopping at epoch %d", epoch)
                break

    log.info("GNN training complete. Best val_loss=%.4f", best_val_loss)
    if val_acc < 0.70:
        log.warning("WARNING: GNN val accuracy %.3f < 0.70 target (3-class).", val_acc)


# ---------------------------------------------------------------------------
# Baseline CNN training
# ---------------------------------------------------------------------------

def train_baseline(args, cfg: dict) -> None:
    """Train the 1D CNN density-profile baseline.

    Uses the same HDF5 simulation files as the GNN but converts each
    simulation to a [3, 200] density profile (density, mean-pm1, mean-pm2).
    Target: >70% 3-class DM model accuracy (CDM+SIDM / WDM / FDM).
    """
    import h5py  # noqa: PLC0415 — lazy import; must come after torch_geometric on Windows
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    log.info("Training 1D CNN baseline on device: %s", device)

    sim_dir = Path(cfg["paths"]["simulations"])
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bcfg = cfg["model"]["baseline_1d"]
    n_bins = bcfg["n_bins"]

    # Build a plain (phi1_min, phi1_max) range from stream configs
    phi1_min, phi1_max = -100.0, 100.0  # conservative range covers all 7 streams

    # ── Build a simple dataset from the HDF5 chunk files ──────────────────
    # Remap table for 3-class baseline: CDM(0)→0, WDM(1)→1, FDM(2)→2, SIDM(3)→0
    _baseline_remap = [0, 1, 2, 0]

    class DensityProfileDataset(torch.utils.data.Dataset):
        """Wraps the HDF5 sims, returning (profile [3, n_bins], dm_idx) pairs.

        Uses 3-class labels (CDM+SIDM merged) for consistency with GNN training.
        """

        def __init__(self, sim_dir: Path) -> None:
            super().__init__()
            self._index = []
            for h5_file in sorted(sim_dir.glob("**/chunk_*.h5")):
                with h5py.File(str(h5_file), "r") as f:
                    if "simulations" not in f:
                        continue
                    for run_id in f["simulations"].keys():
                        self._index.append((h5_file, run_id))
            log.info("DensityProfileDataset: %d sims", len(self._index))

        def __len__(self) -> int:
            return len(self._index)

        def __getitem__(self, idx: int):
            h5_path, run_id = self._index[idx]
            with h5py.File(str(h5_path), "r") as f:
                grp = f[f"simulations/{run_id}"]
                sd = grp["stream_data"]
                phi1 = sd["phi1"][:].astype(np.float32)
                pm1  = sd["pm1"][:].astype(np.float32)
                pm2  = sd["pm2"][:].astype(np.float32)
                dm_idx = int(grp["labels"]["dm_model_idx"][()])
            # 3-class remap: CDM+SIDM → 0
            dm_idx = _baseline_remap[dm_idx]
            profile = compute_1d_profile(phi1, pm1, pm2, n_bins, phi1_min, phi1_max)
            return torch.tensor(profile, dtype=torch.float32), dm_idx

    dataset = DensityProfileDataset(sim_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No simulations found in {sim_dir}. Run generate_training_data.py first.")

    # Use stratified splits consistent with GNN training
    split_cfg = cfg["training"].get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    split_seed = cfg["training"].get("split_seed", 42)

    # Extract 3-class labels for stratification
    baseline_labels = np.array([_baseline_remap[dataset[i][1]] for i in range(len(dataset))])
    split_indices = compute_split_indices(len(dataset), baseline_labels, split_cfg, split_seed)

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    n_train, n_val = len(train_idx), len(val_idx)

    train_set = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_set = torch.utils.data.Subset(dataset, val_idx.tolist())
    log.info("Baseline dataset split: %d train / %d val / %d test (stratified, seed=%d)",
             n_train, n_val, len(split_indices["test"]), split_seed)

    # num_workers=0 on Windows avoids multiprocessing spawn issues
    nw = 0 if sys.platform == "win32" else cfg["training"]["n_workers_dataloader"]
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=nw,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=nw,
    )

    model = DensityProfileCNN(
        n_bins=n_bins, in_channels=len(bcfg["channels"]),
        embedding_dim=128, n_classes=N_CLASSES, dropout=0.25,
    ).to(device)

    total, trainable = count_parameters(model)
    log.info("Baseline CNN parameters: %d total, %d trainable", total, trainable)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-5,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for profiles, labels in train_loader:
            profiles, labels = profiles.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.classify(profiles)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)

        baseline_scheduler.step()
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss_epoch, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for profiles, labels in val_loader:
                profiles, labels = profiles.to(device), labels.to(device)
                logits = model.classify(profiles)
                loss = criterion(logits, labels)
                val_loss_epoch += loss.item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)
        val_loss_epoch /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        log.info("Epoch %d/%d  train_acc=%.3f  val_acc=%.3f  val_loss=%.4f",
                 epoch, args.epochs, train_acc, val_acc, val_loss_epoch)

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_val_acc = max(best_val_acc, val_acc)
            patience_counter = 0
            save_checkpoint(ckpt_dir / "baseline_best.pt", model, optimizer, epoch, val_loss_epoch, cfg)
            log.info("Checkpoint saved to %s (epoch %d, val_loss=%.4f)", ckpt_dir / "baseline_best.pt", epoch, val_loss_epoch)
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                log.info("Early stopping at epoch %d (best val_acc=%.3f)", epoch, best_val_acc)
                break

    log.info("Baseline training complete. Best val_acc=%.3f", best_val_acc)
    if best_val_acc < 0.70:
        log.warning("WARNING: Baseline val_acc %.3f < 0.70 target.", best_val_acc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN encoder or baseline CNN.")
    parser.add_argument("--model", choices=["gnn", "baseline"], default="gnn")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model == "gnn":
        train_gnn(args, cfg)
    else:
        train_baseline(args, cfg)
