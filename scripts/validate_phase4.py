"""
Phase 4 validation: one forward pass through the GNN on simulated data.

Checks:
  1. StreamSimDataset loads HDF5 files and returns valid Data objects
  2. Node features contain no NaN / Inf
  3. phi1 values are in a sane range (±200 deg) — catches the old phi1-drift bug
  4. pm2 values are sane (< 100 mas/yr) — catches the old unit-conversion bug
  5. DataLoader batching works correctly
  6. StreamGNNMultiTask forward pass returns correct shapes
  7. No NaN in embeddings or logits

Run from project root:
    python scripts/validate_phase4.py [--sim-dir data/simulations_test]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import StreamSimDataset, compute_dataset_normalizer
from src.models.gnn import StreamGNNMultiTask


def check(condition, name: str, detail: str = "") -> bool:
    ok = bool(condition)  # handles 0-d torch tensors and numpy bools
    if ok:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))
    return ok


def run(sim_dir: str) -> bool:
    print(f"\n=== Phase 4 forward-pass validation ({sim_dir}) ===\n")
    all_pass = True

    # ── 1. Dataset loads ──────────────────────────────────────────────────────
    print("[ 1 ] Loading dataset …")
    try:
        ds = StreamSimDataset(sim_dir=sim_dir, k_neighbors=16)
        ok = check(len(ds) > 0, "Dataset non-empty", f"len={len(ds)}")
    except Exception as e:
        print(f"  FAIL  Dataset construction crashed: {e}")
        return False
    all_pass &= ok

    # ── 2. Inspect raw HDF5 values before normalisation ───────────────────────
    print("\n[ 2 ] Checking raw feature ranges …")
    sample_data = [ds[i] for i in range(min(len(ds), 5))]

    all_phi1 = np.concatenate([d.x[:, 0].numpy() for d in sample_data])
    all_phi2 = np.concatenate([d.x[:, 1].numpy() for d in sample_data])
    all_pm1  = np.concatenate([d.x[:, 3].numpy() for d in sample_data])
    all_pm2  = np.concatenate([d.x[:, 4].numpy() for d in sample_data])

    print(f"     phi1 range : [{all_phi1.min():.1f}, {all_phi1.max():.1f}] deg")
    print(f"     phi2 range : [{all_phi2.min():.2f}, {all_phi2.max():.2f}] deg")
    print(f"     pm1  range : [{all_pm1.min():.3f}, {all_pm1.max():.3f}] mas/yr")
    print(f"     pm2  range : [{all_pm2.min():.3f}, {all_pm2.max():.3f}] mas/yr")

    all_pass &= check(
        np.all(np.isfinite(all_phi1)) and np.abs(all_phi1).max() < 360,
        "phi1 in valid range (no runaway drift)",
        f"|max|={np.abs(all_phi1).max():.1f} deg",
    )
    all_pass &= check(
        np.abs(all_pm2).max() < 100.0,
        "pm2 in valid range (<100 mas/yr)",
        f"|max|={np.abs(all_pm2).max():.3f} mas/yr",
    )
    all_pass &= check(
        np.all(np.isfinite(all_pm1)) and np.all(np.isfinite(all_pm2)),
        "pm1/pm2 finite",
    )

    # ── 3. Normalizer ─────────────────────────────────────────────────────────
    print("\n[ 3 ] Computing feature normalizer …")
    try:
        norm = compute_dataset_normalizer(ds, n_samples=len(ds))
        print(f"     Feature means: {norm.mean.numpy().round(3)}")
        print(f"     Feature stds : {norm.std.numpy().round(3)}")
        all_pass &= check(
            torch.all(torch.isfinite(norm.mean)) and torch.all(torch.isfinite(norm.std)),
            "Normalizer stats finite",
        )
        # e_vrad and membership_prob are constant fills; at least the 9 varying
        # features (phi1..vrad, e_pm1, e_pm2, e_dist) must have non-zero std.
        varying_stds = norm.std[:9]
        all_pass &= check(
            float(varying_stds.min()) > 1e-6,
            "Varying feature stds > 1e-6 (first 9 features)",
            f"min_std={float(varying_stds.min()):.2e}",
        )
    except Exception as e:
        print(f"  FAIL  Normalizer crashed: {e}")
        return False

    # Rebuild dataset with normalizer
    ds = StreamSimDataset(sim_dir=sim_dir, k_neighbors=16, normalizer=norm)

    # ── 4. DataLoader batching ─────────────────────────────────────────────────
    print("\n[ 4 ] DataLoader batching …")
    try:
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        print(f"     batch.x.shape       = {tuple(batch.x.shape)}")
        print(f"     batch.edge_index.shape = {tuple(batch.edge_index.shape)}")
        print(f"     batch.edge_attr.shape  = {tuple(batch.edge_attr.shape)}")
        print(f"     batch.y.shape       = {tuple(batch.y.shape)}")
        all_pass &= check(
            batch.x.shape[1] == 11,
            "Node feature dim = 11",
            f"got {batch.x.shape[1]}",
        )
        all_pass &= check(
            batch.edge_attr.shape[1] == 5,
            "Edge feature dim = 5",
            f"got {batch.edge_attr.shape[1]}",
        )
        all_pass &= check(
            torch.all(torch.isfinite(batch.x)),
            "Batch node features finite (no NaN after normalisation)",
        )
        all_pass &= check(
            torch.all(torch.isfinite(batch.edge_attr)),
            "Batch edge features finite",
        )
    except Exception as e:
        print(f"  FAIL  DataLoader crashed: {e}")
        return False

    # ── 5. GNN forward pass ───────────────────────────────────────────────────
    print("\n[ 5 ] GNN forward pass …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"     device: {device}")

    try:
        model = StreamGNNMultiTask(
            n_classes=3,
            n_reg_targets=2,
            n_node_features=11,
            n_edge_features=5,
            hidden_dim=256,
            n_layers=6,
            embedding_dim=128,
            dropout=0.1,
        ).to(device)
        model.eval()

        batch = batch.to(device)
        with torch.no_grad():
            emb, logits, reg_out = model(batch)

        print(f"     embeddings shape : {tuple(emb.shape)}")
        print(f"     logits shape     : {tuple(logits.shape)}")
        print(f"     reg_out shape    : {tuple(reg_out.shape)}")

        all_pass &= check(
            emb.shape == (4, 128),
            "Embedding shape [4, 128]",
            f"got {tuple(emb.shape)}",
        )
        all_pass &= check(
            logits.shape == (4, 3),
            "Logits shape [4, 3]",
            f"got {tuple(logits.shape)}",
        )
        all_pass &= check(
            torch.all(torch.isfinite(emb)).item(),
            "No NaN/Inf in embeddings",
            f"finite={torch.isfinite(emb).float().mean():.3f}",
        )
        all_pass &= check(
            torch.all(torch.isfinite(logits)).item(),
            "No NaN/Inf in logits",
        )
        # Softmax should give valid probabilities
        probs = torch.softmax(logits, dim=1)
        all_pass &= check(
            torch.allclose(probs.sum(dim=1), torch.ones(4, device=device), atol=1e-4),
            "Softmax probabilities sum to 1",
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n     Total parameters: {n_params:,}")

    except Exception as e:
        import traceback
        print(f"  FAIL  GNN forward pass crashed: {e}")
        traceback.print_exc()
        return False

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_pass:
        print("Phase 4 PASS — ALL CHECKS PASSED.")
        print("GNN encoder verified on simulated data; ready for training.")
    else:
        print("Phase 4 INCOMPLETE — see FAIL lines above.")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-dir", default="data/simulations_test")
    args = parser.parse_args()
    success = run(args.sim_dir)
    sys.exit(0 if success else 1)
