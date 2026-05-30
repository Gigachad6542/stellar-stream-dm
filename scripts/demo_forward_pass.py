#!/usr/bin/env python
"""Pass real Gaia stream data through the trained GNN and SBI posteriors.

Loads each of the 7 real streams from data/processed/streams.h5, builds k-NN
graphs, runs forward passes through the trained GNN multi-task model, and
displays classification probabilities, regression predictions, and (for GD1
and Pal5) full SBI posterior summaries.

Usage:
    python scripts/demo_forward_pass.py
"""

import os
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

# Set up paths — script is in stellar-stream-dm/scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STREAM_NAMES = ["GD1", "Pal5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]
CLASS_NAMES = ["CDM+SIDM", "WDM", "FDM"]
DM_MODELS_FOR_POSTERIOR = ["CDM", "WDM", "FDM", "SIDM"]
STREAMS_FOR_POSTERIOR = ["GD1", "Pal5"]  # demo subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
STREAMS_H5 = PROJECT_ROOT / "data" / "processed" / "streams.h5"
NORMALIZER_PATH = PROJECT_ROOT / "checkpoints" / "normalizer.npz"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "gnn_best.pt"
TRAINING_CFG_PATH = PROJECT_ROOT / "config" / "training.yaml"
DM_CFG_PATH = PROJECT_ROOT / "config" / "dm_models.yaml"


def load_training_config():
    with open(TRAINING_CFG_PATH) as f:
        return yaml.safe_load(f)


def load_dm_config():
    with open(DM_CFG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Load normalizer
# ---------------------------------------------------------------------------

def load_normalizer():
    from src.data.dataset import FeatureNormalizer
    npz = np.load(str(NORMALIZER_PATH))
    return FeatureNormalizer(npz["mean"], npz["std"])


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(cfg):
    from src.models.gnn import StreamGNNMultiTask

    gnn_cfg = cfg["model"]["gnn"]
    model = StreamGNNMultiTask(
        n_classes=3,
        n_reg_targets=2,
        n_node_features=cfg["graph"]["n_node_features"],
        n_edge_features=cfg["graph"]["n_edge_features"],
        hidden_dim=gnn_cfg["hidden_dim"],
        n_layers=gnn_cfg["n_layers"],
        embedding_dim=gnn_cfg["embedding_dim"],
        dropout=gnn_cfg["dropout"],
    )

    ckpt = torch.load(str(CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
    # Handle both raw state_dict and wrapped checkpoint dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", "?")
        log.info("Loaded checkpoint: epoch=%s, val_loss=%s", epoch, val_loss)
    else:
        log.info("Loaded raw state_dict checkpoint")

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s (%.1f M)", f"{n_params:,}", n_params / 1e6)
    return model


# ---------------------------------------------------------------------------
# Load real stream data
# ---------------------------------------------------------------------------

def load_real_streams(normalizer):
    from src.data.dataset import RealStreamDataset

    dataset = RealStreamDataset(
        processed_path=str(STREAMS_H5),
        stream_names=STREAM_NAMES,
        k_neighbors=16,
        normalizer=normalizer,
        max_stars=3000,
    )
    return dataset


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def run_forward_pass(model, dataset, normalizer):
    """Run each real stream through the GNN and collect outputs."""
    from src.data.dataset import build_knn_graph_batched
    from torch_geometric.data import Batch

    results = {}

    for i, stream_name in enumerate(STREAM_NAMES):
        data = dataset[i]
        n_stars = data.x.shape[0]

        # Create a single-graph batch and move to GPU
        batch = Batch.from_data_list([data])
        batch = batch.to(DEVICE)

        # Rebuild k-NN graph on GPU (faster, matches training pipeline)
        build_knn_graph_batched(batch, k=16, normalizer=normalizer)

        # Normalize node features (matching training pipeline)
        batch.x = normalizer(batch.x)

        with torch.no_grad():
            emb, logits, reg_out = model(batch)

        # Convert to numpy
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        emb_np = emb.cpu().numpy()[0]
        reg_np = reg_out.cpu().numpy()[0]

        results[stream_name] = {
            "n_stars": n_stars,
            "embedding": emb_np,
            "class_probs": probs,
            "log10_M_sub_mean": reg_np[0],
            "n_impacts": max(0, round(reg_np[1])),
            "n_impacts_raw": reg_np[1],
        }

    return results


# ---------------------------------------------------------------------------
# SBI posterior sampling
# ---------------------------------------------------------------------------

def run_posterior_inference(results, dm_cfg):
    """Sample SBI posteriors for selected streams."""
    from src.inference.posteriors import sample_posterior, compute_credible_intervals

    posterior_results = {}

    for stream_name in STREAMS_FOR_POSTERIOR:
        emb = torch.tensor(results[stream_name]["embedding"], dtype=torch.float32)
        stream_posteriors = {}

        for dm_model in DM_MODELS_FOR_POSTERIOR:
            pkl_path = PROJECT_ROOT / "checkpoints" / f"posterior_{dm_model}.pkl"
            if not pkl_path.exists():
                log.warning("Missing posterior: %s", pkl_path)
                continue

            try:
                with open(pkl_path, "rb") as f:
                    posterior = pickle.load(f)

                samples = sample_posterior(
                    posterior, emb,
                    n_samples=5000,
                    dm_model=dm_model,
                    config_path=str(DM_CFG_PATH),
                )
                ci = compute_credible_intervals(samples, ci_level=0.9)
                stream_posteriors[dm_model] = {
                    "samples": samples,
                    "credible_intervals": ci,
                }
                log.info("  %s/%s: %d samples drawn", stream_name, dm_model, len(samples))
            except Exception as e:
                log.error("  %s/%s posterior failed: %s", stream_name, dm_model, e)

        posterior_results[stream_name] = stream_posteriors

    return posterior_results


# ---------------------------------------------------------------------------
# Cosine similarity between stream embeddings
# ---------------------------------------------------------------------------

def compute_embedding_similarity(results):
    """Compute pairwise cosine similarity between stream embeddings."""
    embs = np.stack([results[s]["embedding"] for s in STREAM_NAMES])
    # Normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_normed = embs / (norms + 1e-8)
    sim_matrix = embs_normed @ embs_normed.T
    return sim_matrix


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_separator(title="", width=80):
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "=" * pad + f" {title} " + "=" * pad)
    else:
        print("=" * width)


def print_classification_results(results):
    print_separator("CLASSIFICATION RESULTS (3-class)")
    print(f"{'Stream':<10} {'N_stars':>7} {'CDM+SIDM':>10} {'WDM':>10} {'FDM':>10} {'Predicted':>12}")
    print("-" * 65)
    for name in STREAM_NAMES:
        r = results[name]
        pred_idx = np.argmax(r["class_probs"])
        pred_name = CLASS_NAMES[pred_idx]
        print(f"{name:<10} {r['n_stars']:>7d} "
              f"{r['class_probs'][0]:>10.3f} "
              f"{r['class_probs'][1]:>10.3f} "
              f"{r['class_probs'][2]:>10.3f} "
              f"{pred_name:>12}")


def print_regression_results(results):
    print_separator("REGRESSION PREDICTIONS")
    print(f"{'Stream':<10} {'log10(M_sub)':>14} {'n_impacts':>12} {'n_impacts_raw':>15}")
    print("-" * 55)
    for name in STREAM_NAMES:
        r = results[name]
        print(f"{name:<10} {r['log10_M_sub_mean']:>14.2f} "
              f"{r['n_impacts']:>12d} "
              f"{r['n_impacts_raw']:>15.3f}")


def print_posterior_results(posterior_results):
    print_separator("SBI POSTERIOR INFERENCE")
    for stream_name, models in posterior_results.items():
        print(f"\n--- {stream_name} ---")
        for dm_model, data in models.items():
            ci = data["credible_intervals"]
            print(f"  [{dm_model}]")
            for param in ci.index:
                row = ci.loc[param]
                print(f"    {param:<20s}  median={row['median']:.3f}  "
                      f"90%CI=[{row['lower']:.3f}, {row['upper']:.3f}]  "
                      f"std={row['std']:.3f}")


def print_similarity_matrix(sim_matrix):
    print_separator("EMBEDDING COSINE SIMILARITY")
    header = f"{'':>10}" + "".join(f"{s:>10}" for s in STREAM_NAMES)
    print(header)
    for i, name in enumerate(STREAM_NAMES):
        row = f"{name:>10}" + "".join(f"{sim_matrix[i, j]:>10.3f}" for j in range(len(STREAM_NAMES)))
        print(row)


def print_embedding_stats(results):
    print_separator("EMBEDDING STATISTICS (128-d)")
    print(f"{'Stream':<10} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'L2_norm':>10}")
    print("-" * 55)
    for name in STREAM_NAMES:
        emb = results[name]["embedding"]
        print(f"{name:<10} {emb.mean():>8.4f} {emb.std():>8.4f} "
              f"{emb.min():>8.4f} {emb.max():>8.4f} {np.linalg.norm(emb):>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_separator("STELLAR STREAM DM — REAL DATA FORWARD PASS")
    print(f"Device: {DEVICE}")
    print(f"Streams HDF5: {STREAMS_H5}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # Check all files exist
    for p, label in [(STREAMS_H5, "Streams HDF5"), (NORMALIZER_PATH, "Normalizer"),
                     (CHECKPOINT_PATH, "Checkpoint"), (TRAINING_CFG_PATH, "Training config")]:
        if not p.exists():
            print(f"ERROR: {label} not found at {p}")
            sys.exit(1)
        print(f"  [OK] {label}: {p.name}")

    cfg = load_training_config()

    # Load components
    log.info("Loading normalizer...")
    normalizer = load_normalizer()
    log.info("Normalizer: %s", normalizer)

    log.info("Loading model...")
    model = load_model(cfg)

    log.info("Loading real stream data...")
    dataset = load_real_streams(normalizer)
    log.info("Loaded %d streams", len(dataset))

    # Forward pass
    log.info("Running forward pass on all %d streams...", len(STREAM_NAMES))
    results = run_forward_pass(model, dataset, normalizer)

    # Print results
    print_classification_results(results)
    print_regression_results(results)
    print_embedding_stats(results)

    # Cosine similarity
    sim_matrix = compute_embedding_similarity(results)
    print_similarity_matrix(sim_matrix)

    # SBI posteriors (for GD1 and Pal5)
    dm_cfg = load_dm_config()
    log.info("Running SBI posterior sampling for %s...", STREAMS_FOR_POSTERIOR)
    posterior_results = run_posterior_inference(results, dm_cfg)
    print_posterior_results(posterior_results)

    print_separator("DONE")
    print(f"Successfully processed {len(STREAM_NAMES)} real streams through trained model.")


if __name__ == "__main__":
    main()
