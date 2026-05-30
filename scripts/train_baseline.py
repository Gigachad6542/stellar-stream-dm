"""
Train the 1D CNN baseline (DensityProfileCNN) on the simulation dataset.

This script must be run BEFORE concluding the GNN is the right architecture.
Pass/fail criterion: baseline val accuracy > 40% (floor check).
GNN criterion: GNN val accuracy > baseline + 5%.

Usage:
    python -u scripts/train_baseline.py [--epochs 100] [--batch-size 128]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

# --- project root on path ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data.dataset import StreamSimDataset, FeatureNormalizer
from src.data.splits import compute_split_indices
from models.baseline import DensityProfileCNN, compute_1d_profile
from models.utils import save_checkpoint, load_normalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "train_baseline.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1D profile dataset wrapper
# ---------------------------------------------------------------------------

class ProfileDataset(torch.utils.data.Dataset):
    """Wraps StreamSimDataset to return 1D profiles instead of graphs.

    Each item: (profile [3, 200], dm_model_idx int)
    Profiles are computed on-the-fly from the raw node features.
    """

    def __init__(self, sim_dataset: StreamSimDataset, n_bins: int = 200) -> None:
        self.base = sim_dataset
        self.n_bins = n_bins

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        data = self.base[idx]
        x = data.x.numpy()          # [N, 11]
        phi1 = x[:, 0]
        pm1  = x[:, 3]
        pm2  = x[:, 4]
        profile = compute_1d_profile(phi1, pm1, pm2, n_bins=self.n_bins)
        label = int(data.y[0].item())
        return torch.tensor(profile, dtype=torch.float32), label


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_baseline(args: argparse.Namespace) -> None:
    cfg_path = ROOT / "config" / "training.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"].get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    log.info("Training CNN baseline on device: %s", device)

    # VRAM cap (same as GNN training — leaves room for Ollama)
    if device.type == "cuda":
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        frac = cfg["training"].get("cuda_memory_fraction", 0.75)
        torch.cuda.set_per_process_memory_fraction(frac)
        log.info("CUDA allocator capped at %.0f%% of VRAM (%.1f / %.1f GB)",
                 frac * 100, frac * vram_total, vram_total)

    sim_dir = ROOT / cfg["paths"]["simulations"]
    n_epochs = args.epochs
    batch_size = args.batch_size
    split_cfg = cfg["training"].get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    split_seed = cfg["training"].get("split_seed", 42)

    # Build base simulation dataset (no graph, just raw node features)
    log.info("Building raw dataset ...")
    base_ds = StreamSimDataset(
        sim_dir=sim_dir,
        k_neighbors=1,         # k not used; graph not built for baseline
        normalizer=None,
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        preload_ram=True,
    )
    split_indices = compute_split_indices(len(base_ds), split_ratios=split_cfg, seed=split_seed)
    n_train = len(split_indices["train"])
    n_val = len(split_indices["val"])
    n_test = len(split_indices["test"])
    log.info("Dataset split: %d train / %d val / %d test (stratified, seed=%d)",
             n_train, n_val, n_test, split_seed)

    # Use Subset with deterministic split indices
    train_sub = Subset(base_ds, split_indices["train"].tolist())
    val_sub = Subset(base_ds, split_indices["val"].tolist())
    test_sub = Subset(base_ds, split_indices["test"].tolist())

    n_bins = cfg["model"]["baseline_1d"]["n_bins"]
    train_ds = ProfileDataset(train_sub, n_bins=n_bins)
    val_ds   = ProfileDataset(val_sub,   n_bins=n_bins)
    test_ds  = ProfileDataset(test_sub,  n_bins=n_bins)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # Model: 3 classes (CDM+SIDM / WDM / FDM)
    model = DensityProfileCNN(
        n_bins=n_bins,
        embedding_dim=128,
        n_classes=3,
        dropout=0.2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("CNN baseline parameters: %d", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience = 40
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # --- train ---
        model.train()
        t0 = time.time()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for profiles, labels in train_loader:
            profiles = profiles.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify(profiles)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                log.warning("NaN/Inf loss at epoch %d — skipping", epoch)
                optimizer.zero_grad()
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss   += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += len(labels)

        scheduler.step()
        train_loss /= train_total
        train_acc   = train_correct / train_total

        # --- validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for profiles, labels in val_loader:
                profiles = profiles.to(device, non_blocking=True)
                labels   = labels.to(device, non_blocking=True)
                logits = model.classify(profiles)
                loss = criterion(logits, labels)
                val_loss    += loss.item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += len(labels)
        val_loss /= val_total
        val_acc   = val_correct / val_total

        elapsed = time.time() - t0
        log.info("Epoch %d/%d  train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f  (%.1fs)",
                 epoch, n_epochs, train_loss, train_acc, val_loss, val_acc, elapsed)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(
                ckpt_dir / "baseline_best.pt",
                model, optimizer, epoch, val_loss, cfg,
            )
            log.info("Checkpoint saved (epoch %d, val_loss=%.4f)", epoch, val_loss)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    # --- test set evaluation on best checkpoint ---
    log.info("Loading best checkpoint for test evaluation ...")
    ckpt = torch.load(str(ckpt_dir / "baseline_best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for profiles, labels in test_loader:
            profiles = profiles.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)
            preds = model.classify(profiles).argmax(1)
            test_correct += (preds == labels).sum().item()
            test_total   += len(labels)
    test_acc = test_correct / test_total
    log.info("Baseline CNN test accuracy: %.1f%%  (best val_loss=%.4f)", test_acc * 100, best_val_loss)

    gnn_val_acc = 0.498   # recorded from train_gnn_v8 best epoch
    delta = test_acc - gnn_val_acc
    if delta < -0.05:
        log.info("GNN beats baseline by %.1f pp — GNN criterion PASSED", -delta * 100)
    elif delta > 0.05:
        log.warning("Baseline beats GNN by %.1f pp — investigate GNN architecture", delta * 100)
    else:
        log.info("GNN vs baseline gap: %.1f pp (within 5pp — marginal)", delta * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    train_baseline(args)
