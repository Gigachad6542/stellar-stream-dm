#!/usr/bin/env python
"""
Linear-probe characterization of impact TYPE from the detector embedding.

Before committing to any multi-task retrain, ask: how much does the trained
detector's GNN embedding already encode about the impact's physical type
(subhalo mass, impact parameter b, time-since-impact)? We extract the frozen
embedding on the test split, then fit a simple Ridge regressor embedding -> target
and report R^2 + median |error|. Strong R^2 => characterization is easy (a small
head suffices); weak R^2 => a dedicated multi-task retrain is warranted.

We report both over ALL impacts and over DETECTED impacts (p_impact > 0.5), since
characterization only matters where we detect.

Usage:
  python scripts/characterize_probe.py \
      --checkpoint checkpoints/detector_df_20260602/gnn_v2_best_acc.pt \
      --sim-dir data/simulations_detector_df
"""
from __future__ import annotations
import sys, os, argparse, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_training_data import _fix_galpy_dll_path
_fix_galpy_dll_path()

import numpy as np
import torch
import h5py

from src.data.dataset import (
    StreamSimDataset, FeatureNormalizer, profile_feature_dim,
    build_knn_graph_batched, build_profile_features_batched,
)
from src.models.gnn import StreamGNNMultiTaskV2
from src.models.utils import load_normalizer
from src.data.splits import load_split_indices
from torch_geometric.loader import DataLoader as PyGDataLoader


def _tag(target, thr):
    return f"{target}_t{f'{thr:g}'.replace('-', 'm').replace('.', 'p')}"


def _probe(emb, y, name, mask, seed=0):
    """Ridge probe emb->y on `mask` subset, 70/30 split; print R^2 + med|err|."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    e = emb[mask]; t = np.asarray(y)[mask]
    ok = np.isfinite(t)
    e, t = e[ok], t[ok]
    if len(t) < 30:
        print(f"    {name:24s}  (too few: {len(t)})"); return
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(t)); cut = int(0.7 * len(t))
    tr, te = perm[:cut], perm[cut:]
    sc = StandardScaler().fit(e[tr])
    reg = Ridge(alpha=10.0).fit(sc.transform(e[tr]), t[tr])
    pred = reg.predict(sc.transform(e[te]))
    r2 = r2_score(t[te], pred); mae = float(np.median(np.abs(pred - t[te])))
    print(f"    {name:24s}  R2={r2:+.3f}  med|err|={mae:.3f}   (n_test={len(te)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--config", default="config/training.yaml")
    ap.add_argument("--max-sims", type=int, default=4000)
    ap.add_argument("--clip-sigma", type=float, default=5.0)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint); ckpt_dir = ckpt.parent
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    gcfg = cfg["model"]["gnn"]; pcfg = cfg["graph"]["profile_branch"]; tv2 = cfg["training_v2"]
    schema = "timeline" if tv2.get("label_schema") == "timeline" else "compact"
    target = tv2.get("binary_target", "impact_detectable"); thr = float(tv2.get("impact_threshold", 0.5))
    T = 1.0
    cj = ckpt_dir / "calibration.json"
    if cj.exists():
        T = float(json.loads(cj.read_text()).get("temperature", 1.0))

    pdim = profile_feature_dim(int(pcfg["n_bins"]), pcfg["feature_set"],
                              bool(pcfg["include_stream_onehot"])) if pcfg.get("enabled") else 0
    model = StreamGNNMultiTaskV2(
        n_reg_targets=tv2.get("n_reg_targets", 2), predict_uncertainty=tv2.get("predict_uncertainty", False),
        profile_dim=pdim, profile_hidden_dim=int(pcfg.get("hidden_dim", 64)),
        profile_layer_norm=bool(pcfg.get("layer_norm", False)),
        n_node_features=cfg["graph"]["n_node_features"], n_edge_features=cfg["graph"]["n_edge_features"],
        hidden_dim=gcfg["hidden_dim"], n_layers=gcfg["n_layers"], embedding_dim=gcfg["embedding_dim"],
        dropout=gcfg["dropout"], use_attention_readout=gcfg.get("use_attention_readout", False))
    model.load_state_dict(payload["model_state_dict"]); model.eval()

    mean, std = load_normalizer(ckpt_dir / "normalizer_v2.npz")
    normalizer = FeatureNormalizer(mean, std); nm, ns = normalizer.mean, normalizer.std
    k = cfg["graph"]["k_neighbors"]
    ds = StreamSimDataset(args.sim_dir, k_neighbors=k,
        max_stars=cfg["preprocessing"]["max_stars_per_sim"], augment=False,
        normalizer=normalizer, preload_ram=False, label_schema=schema)
    split = load_split_indices(
        ckpt_dir / f"split_n{len(ds)}_seed{cfg['training'].get('split_seed', 42)}_{_tag(target, thr)}.npz")
    idxs = split["test"].tolist()[: args.max_sims]

    def phys_for(i):
        h5_path, run_id, sname = ds._index[i]
        with h5py.File(h5_path, "r") as f:
            sub = f[f"simulations/{run_id}/subhalos"]; lab = f[f"simulations/{run_id}/labels"]
            n = int(lab["n_subhalos"][()]); strength = float(lab.attrs.get("impact_strength", 0.0))
            if n > 0 and sub["mass"].shape[0] > 0:
                return (n, float(np.log10(sub["mass"][0])), float(sub["impact_param"][0]),
                        float(sub["t_since_impact_gyr"][0]), strength)
            return (0, np.nan, np.nan, np.nan, 0.0)

    sub = torch.utils.data.Subset(ds, idxs)
    loader = PyGDataLoader(sub, batch_size=64, shuffle=False)
    embs, probs = [], []
    with torch.no_grad():
        for batch in loader:
            build_knn_graph_batched(batch, k, normalizer)
            if pcfg.get("enabled"):
                build_profile_features_batched(batch, n_bins=int(pcfg["n_bins"]),
                    feature_set=pcfg["feature_set"], include_stream_onehot=bool(pcfg["include_stream_onehot"]))
            batch.x = torch.clamp((batch.x - nm) / ns, -args.clip_sigma, args.clip_sigma)
            emb, bl, _, _ = model(batch)
            embs.append(emb.cpu().numpy()); probs.append(torch.sigmoid(bl.squeeze(-1) / T).numpy())
    emb = np.concatenate(embs); probs = np.concatenate(probs)

    phys = np.array([phys_for(i) for i in idxs])  # [N,5]: n, logM, b, t, strength
    is_imp = phys[:, 0] > 0
    detected = probs > 0.5
    print("=" * 66)
    print(f"EMBEDDING PROBE for impact TYPE  (test split, n={len(idxs)}, dim={emb.shape[1]})")
    print("=" * 66)
    print(f"  impacts={int(is_imp.sum())}  detected-impacts={int((is_imp & detected).sum())}")
    for label, m in (("ALL impacts", is_imp), ("DETECTED impacts (p>0.5)", is_imp & detected)):
        print(f"\n  --- probe over {label} ---")
        _probe(emb, phys[:, 1], "log10 mass [dex]", m)
        _probe(emb, phys[:, 2], "impact param b [kpc]", m)
        _probe(emb, phys[:, 3], "t_since impact [Gyr]", m)
        _probe(emb, phys[:, 4], "realised gap strength", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
