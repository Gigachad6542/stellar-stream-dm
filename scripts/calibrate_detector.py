#!/usr/bin/env python
"""
Temperature-calibrate a timeline GNN detector on its simulated validation split.

The binary head is well-discriminating on simulations (high AUC) but
over-confident (many probabilities pinned near 0/1). Temperature scaling fits a
single scalar T that divides the logits to minimise validation NLL, producing
calibrated probabilities without changing the ranking/AUC. The result is written
to ``calibration.json`` next to the checkpoint, where StreamImpactDetector picks
it up automatically.

Usage:
    python scripts/calibrate_detector.py \
        --checkpoint checkpoints/gnn_v2_timeline_s3_.../gnn_v2_best.pt \
        --sim-dir data/simulations_v2_detector_balanced
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    FeatureNormalizer, StreamSimDataset, build_knn_graph_batched,
    build_profile_features_batched, profile_feature_dim,
)
from src.data.splits import load_split_indices
from src.models.gnn import StreamGNNMultiTaskV2
from src.models.utils import load_normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _binary_target(labels: torch.Tensor, mode: str, thr: float) -> torch.Tensor:
    if mode == "impact_timeline_detectable" and labels.shape[1] >= 7:
        return (labels[:, 6] > thr).float()
    if mode == "impact_detectable":
        return (labels[:, 2] > thr).float()
    if mode == "impact_strong" and labels.shape[1] >= 6:
        return (labels[:, 5] > thr).float()
    if mode == "model_family":
        remap = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
        return remap[labels[:, 0].long()]
    return (labels[:, 2] > thr).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--max-val", type=int, default=3000)
    ap.add_argument("--clip-sigma", type=float, default=5.0)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    ckpt_dir = ckpt.parent
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    gcfg = cfg["model"]["gnn"]
    pcfg = cfg["graph"]["profile_branch"]
    tv2 = cfg["training_v2"]
    schema = "timeline" if tv2.get("label_schema") == "timeline" else "compact"
    target = tv2.get("binary_target", "impact_detectable")
    thr = float(tv2.get("impact_threshold", 0.5))

    pdim = profile_feature_dim(int(pcfg["n_bins"]), pcfg["feature_set"],
                              bool(pcfg["include_stream_onehot"])) if pcfg.get("enabled") else 0
    model = StreamGNNMultiTaskV2(
        n_reg_targets=tv2.get("n_reg_targets", 2),
        predict_uncertainty=tv2.get("predict_uncertainty", False),
        profile_dim=pdim, profile_hidden_dim=int(pcfg.get("hidden_dim", 64)),
        profile_layer_norm=bool(pcfg.get("layer_norm", False)),
        n_node_features=cfg["graph"]["n_node_features"], n_edge_features=cfg["graph"]["n_edge_features"],
        hidden_dim=gcfg["hidden_dim"], n_layers=gcfg["n_layers"], embedding_dim=gcfg["embedding_dim"],
        dropout=gcfg["dropout"], use_attention_readout=gcfg.get("use_attention_readout", False),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    mean, std = load_normalizer(ckpt_dir / "normalizer_v2.npz")
    normalizer = FeatureNormalizer(mean, std)
    nm, ns = normalizer.mean, normalizer.std
    k = cfg["graph"]["k_neighbors"]

    ds = StreamSimDataset(args.sim_dir, k_neighbors=k,
        max_stars=cfg["preprocessing"]["max_stars_per_sim"], augment=False,
        normalizer=normalizer, preload_ram=False, label_schema=schema)
    split = load_split_indices(
        ckpt_dir / f"split_n{len(ds)}_seed{cfg['training'].get('split_seed', 42)}_"
                   f"{_tag(target, thr)}.npz")
    val = torch.utils.data.Subset(ds, split["val"].tolist()[: args.max_val])
    loader = PyGDataLoader(val, batch_size=64, shuffle=False)

    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            build_knn_graph_batched(batch, k, normalizer)
            if pcfg.get("enabled"):
                build_profile_features_batched(batch, n_bins=int(pcfg["n_bins"]),
                    feature_set=pcfg["feature_set"], include_stream_onehot=bool(pcfg["include_stream_onehot"]))
            batch.x = torch.clamp((batch.x - nm) / ns, -args.clip_sigma, args.clip_sigma)
            _, bl, _, _ = model(batch)
            logits.append(bl.squeeze(-1))
            labels.append(_binary_target(batch.y.view(batch.num_graphs, -1), target, thr))
    logits = torch.cat(logits)
    labels = torch.cat(labels)

    # Fit temperature by minimising BCE-with-logits(logits / T).
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / torch.exp(logT), labels)
        loss.backward()
        return loss

    nll_before = bce(logits, labels).item()
    opt.step(closure)
    T = float(torch.exp(logT).item())
    nll_after = bce(logits / T, labels).item()

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels.numpy(), torch.sigmoid(logits).numpy())
    frac99_before = float((torch.sigmoid(logits) > 0.99).float().mean())
    frac99_after = float((torch.sigmoid(logits / T) > 0.99).float().mean())

    out = {
        "temperature": T,
        "n_val": int(len(labels)),
        "target": target,
        "auc": float(auc),
        "nll_before": nll_before,
        "nll_after": nll_after,
        "frac_p_gt_0p99_before": frac99_before,
        "frac_p_gt_0p99_after": frac99_after,
        "clip_sigma": args.clip_sigma,
    }
    (ckpt_dir / "calibration.json").write_text(json.dumps(out, indent=2))
    log.info("Calibration: T=%.3f  AUC=%.3f  NLL %.4f -> %.4f  frac(p>0.99) %.2f -> %.2f",
             T, auc, nll_before, nll_after, frac99_before, frac99_after)
    log.info("Wrote %s", ckpt_dir / "calibration.json")
    return 0


def _tag(target: str, thr: float) -> str:
    t = f"{thr:g}".replace("-", "m").replace(".", "p")
    return {"model_family": "dm"}.get(target, f"{target}_t{t}")


if __name__ == "__main__":
    raise SystemExit(main())
