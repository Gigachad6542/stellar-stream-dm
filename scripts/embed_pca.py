"""
PCA/UMAP visualisation of GNN embeddings.

Loads gnn_v2_best.pt by default, encodes the full test split, and saves:
  outputs/figures/embedding_pca.png  — 2D PCA coloured by DM model
  outputs/figures/embedding_pca_log_m.png — 2D PCA coloured by log10(M_sub_mean)
  outputs/embed_pca_results.npz — raw coords + labels for downstream use

Usage:
    python -u scripts/embed_pca.py [--checkpoint checkpoints/gnn_v2_best.pt]
"""
from __future__ import annotations

import argparse
import logging
import site
import sys
from pathlib import Path

import numpy as np

try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [p for p in sys.path if not p or Path(p).resolve() != user_site]
except Exception:
    pass

import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data.dataset import StreamSimDataset, FeatureNormalizer
from data.dataset import build_knn_graph_batched, build_segment_graph
from src.data.splits import compute_split_indices
from models.gnn import StreamGNNMultiTask, StreamGNNMultiTaskV2
from models.utils import load_normalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DM_MODEL_NAMES = ["CDM", "WDM", "FDM", "SIDM"]
DM_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]


def embed_test_set(args: argparse.Namespace) -> None:
    cfg_path = ROOT / "config" / "training.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Running embedding PCA on device: %s", device)

    if device.type == "cuda":
        frac = cfg["training"].get("cuda_memory_fraction", 0.75)
        torch.cuda.set_per_process_memory_fraction(frac)

    sim_dir = ROOT / cfg["paths"]["simulations"]
    split_cfg = cfg["training"].get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    split_seed = cfg["training"].get("split_seed", 42)

    # --- Load normalizer ---
    norm_path = ROOT / args.normalizer
    mean, std = load_normalizer(norm_path)
    normalizer = FeatureNormalizer(mean, std)
    log.info("Normalizer loaded from %s", norm_path)

    # --- Build dataset (same split as training) ---
    log.info("Building dataset ...")
    full_ds = StreamSimDataset(
        sim_dir=sim_dir,
        k_neighbors=cfg["graph"]["k_neighbors"],
        normalizer=normalizer,
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        preload_ram=True,
        use_orbital_features=cfg["graph"].get("orbital_features", {}).get("enabled", False),
    )
    split_indices = compute_split_indices(
        len(full_ds),
        full_ds.get_dm_model_labels(),
        split_ratios=split_cfg,
        seed=split_seed,
    )
    test_idx = split_indices["test"]
    test_sub = torch.utils.data.Subset(full_ds, test_idx.tolist())
    log.info("Test split: %d simulations", len(test_sub))

    loader = DataLoader(test_sub, batch_size=64, shuffle=False, num_workers=0)

    # --- Load GNN model ---
    gnn_cfg = cfg["model"]["gnn"]
    # Checkpoint was saved from StreamGNNMultiTask (encoder + heads);
    # load the full model then extract the encoder for embedding.
    ms_cfg = cfg["graph"].get("multi_scale", {})
    if args.model_version == "v2":
        v2_cfg = cfg.get("training_v2", {})
        classifier_model = StreamGNNMultiTaskV2(
            n_reg_targets=v2_cfg.get("n_reg_targets", 2),
            predict_uncertainty=v2_cfg.get("predict_uncertainty", False),
            n_node_features=cfg["graph"]["n_node_features"],
            n_edge_features=cfg["graph"]["n_edge_features"],
            hidden_dim=gnn_cfg["hidden_dim"],
            embedding_dim=gnn_cfg["embedding_dim"],
            n_layers=gnn_cfg["n_layers"],
            dropout=gnn_cfg["dropout"],
            use_attention_readout=gnn_cfg.get("use_attention_readout", False),
            use_multi_scale=ms_cfg.get("enabled", False),
            seg_embedding_dim=ms_cfg.get("seg_embedding_dim", 64),
        ).to(device)
    else:
        classifier_model = StreamGNNMultiTask(
            n_classes=3,
            n_reg_targets=cfg["model"].get("n_reg_targets", 2),
            n_node_features=cfg["graph"]["n_node_features"],
            n_edge_features=cfg["graph"]["n_edge_features"],
            hidden_dim=gnn_cfg["hidden_dim"],
            embedding_dim=gnn_cfg["embedding_dim"],
            n_layers=gnn_cfg["n_layers"],
            dropout=gnn_cfg["dropout"],
            use_attention_readout=gnn_cfg.get("use_attention_readout", False),
            use_multi_scale=ms_cfg.get("enabled", False),
            seg_embedding_dim=ms_cfg.get("seg_embedding_dim", 64),
        ).to(device)

    ckpt_path = ROOT / args.checkpoint
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    classifier_model.load_state_dict(ckpt["model_state_dict"])
    classifier_model.eval()
    model = classifier_model.encoder   # StreamGNNEncoder — embeddings only
    log.info("Loaded GNN from %s (epoch %d, val_loss=%.4f)",
             ckpt_path, ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan")))

    k = cfg["graph"]["k_neighbors"]
    use_orbital = cfg["graph"].get("orbital_features", {}).get("enabled", False)
    norm_mean_dev = normalizer.mean.to(device)
    norm_std_dev = normalizer.std.to(device)

    # --- Encode ---
    all_embeddings = []
    all_dm_labels  = []
    all_log_m      = []
    all_n_sub      = []

    log.info("Encoding %d test simulations ...", len(test_sub))
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            build_knn_graph_batched(
                batch,
                k=k,
                normalizer=normalizer,
                orbital_features=use_orbital,
            )
            if ms_cfg.get("enabled", False):
                build_segment_graph(
                    batch,
                    n_segments=ms_cfg.get("n_segments", 20),
                    k_seg=ms_cfg.get("k_segment_neighbors", 4),
                )
            batch.x = (batch.x - norm_mean_dev) / norm_std_dev
            # NOTE: Do NOT normalise batch.x here — the GNN was trained on raw
            emb = model(batch)   # [B, 128]
            # batch.y is [B*N_labels] after PyG batching (concatenated 1-D tensors);
            # reshape to [B, N_labels] to access per-sample labels.
            y = batch.y.view(batch.num_graphs, -1)
            all_embeddings.append(emb.cpu().numpy())
            all_dm_labels.append(y[:, 0].long().cpu().numpy())
            all_log_m.append(y[:, 1].cpu().numpy())
            all_n_sub.append(y[:, 2].cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)   # [N_test, 128]
    dm_labels  = np.concatenate(all_dm_labels,  axis=0)
    log_m      = np.concatenate(all_log_m,      axis=0)
    n_sub      = np.concatenate(all_n_sub,       axis=0)
    log.info("Encoded %d embeddings, shape %s", len(embeddings), embeddings.shape)

    # --- PCA ---
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_
    log.info("PCA variance explained: PC1=%.1f%%  PC2=%.1f%%",
             var_explained[0] * 100, var_explained[1] * 100)

    # --- Save raw results ---
    out_dir = ROOT / "outputs"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_dir / "embed_pca_results.npz"),
        embeddings=embeddings,
        pca_coords=coords,
        dm_labels=dm_labels,
        log_m=log_m,
        n_sub=n_sub,
        explained_variance=var_explained,
    )
    log.info("Raw results saved to outputs/embed_pca_results.npz")

    # --- Plot: coloured by DM model ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"GNN Embedding PCA  (epoch {ckpt.get('epoch', '?')}, "
        f"val_acc≈49.8%)\n"
        f"PC1={var_explained[0]*100:.1f}%  PC2={var_explained[1]*100:.1f}% variance",
        fontsize=12,
    )

    # Panel 1: DM model class
    ax = axes[0]
    for i, (name, color) in enumerate(zip(DM_MODEL_NAMES, DM_COLORS)):
        mask = dm_labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=f"{name} (n={mask.sum()})",
                   alpha=0.3, s=4, linewidths=0)
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title("Coloured by DM model")
    ax.legend(markerscale=3, framealpha=0.8, fontsize=9)

    # Panel 2: coloured by log10(M_sub_mean)
    ax = axes[1]
    finite_mask = np.isfinite(log_m) & (log_m != 0)
    sc = ax.scatter(
        coords[finite_mask, 0], coords[finite_mask, 1],
        c=log_m[finite_mask], cmap="plasma",
        alpha=0.4, s=4, linewidths=0,
        vmin=np.percentile(log_m[finite_mask], 5),
        vmax=np.percentile(log_m[finite_mask], 95),
    )
    plt.colorbar(sc, ax=ax, label="log₁₀(M_sub_mean / M☉)")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title("Coloured by subhalo mass")

    plt.tight_layout()
    out_path = str(fig_dir / "embedding_pca.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("PCA plot saved to %s", out_path)

    # --- Per-class purity summary ---
    log.info("--- Embedding cluster summary ---")
    for i, name in enumerate(DM_MODEL_NAMES):
        mask = dm_labels == i
        c = coords[mask]
        log.info("  %s  centroid=(%.2f, %.2f)  spread_pc1=%.2f  spread_pc2=%.2f",
                 name, c[:, 0].mean(), c[:, 1].mean(), c[:, 0].std(), c[:, 1].std())

    # Centroid separation (max pairwise distance between class centroids)
    centroids = np.array([coords[dm_labels == i].mean(axis=0) for i in range(4)])
    dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(centroids[i] - centroids[j])
            dists.append((d, DM_MODEL_NAMES[i], DM_MODEL_NAMES[j]))
    dists.sort(reverse=True)
    log.info("  Most separated pair: %s vs %s  (dist=%.3f)", dists[0][1], dists[0][2], dists[0][0])
    log.info("  Least separated pair: %s vs %s  (dist=%.3f)", dists[-1][1], dists[-1][2], dists[-1][0])

    log.info("Embedding PCA complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/gnn_v2_best.pt")
    parser.add_argument("--normalizer", default="checkpoints/normalizer_v2.npz")
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    embed_test_set(args)
