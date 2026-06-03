#!/usr/bin/env python
"""
Evaluate the trained detect+characterize model's regression heads.

Reads the mass_time regression head outputs (reg_out[:,0]=log10 mass,
reg_out[:,1]=log_t_since) on the test split and compares to truth -- overall,
for DETECTED impacts (p>0.5), and binned by realised gap strength. Answers "given
a detection, how well do we recover the impact's mass and recency?"

Usage:
  python scripts/characterize_eval.py \
      --checkpoint checkpoints/detector_char_20260602/gnn_v2_best.pt \
      --sim-dir data/simulations_detector_df
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_training_data import _fix_galpy_dll_path
_fix_galpy_dll_path()

import numpy as np
import torch
import h5py
from sklearn.metrics import r2_score

from src.data.dataset import (
    StreamSimDataset, FeatureNormalizer, profile_feature_dim,
    build_knn_graph_batched, build_profile_features_batched,
)
from src.models.gnn import StreamGNNMultiTaskV2
from src.models.utils import load_normalizer
from src.data.splits import load_split_indices
from torch_geometric.loader import DataLoader as PyGDataLoader


def _tag(t, thr):
    return f"{t}_t{f'{thr:g}'.replace('-', 'm').replace('.', 'p')}"


def _stats(true, pred, mask, name):
    t, p = np.asarray(true)[mask], np.asarray(pred)[mask]
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if len(t) < 10:
        print(f"    {name:22s} (n={len(t)})"); return
    print(f"    {name:22s} R2={r2_score(t, p):+.3f}  med|err|={np.median(np.abs(p - t)):.3f}  "
          f"bias={np.mean(p - t):+.3f}  (n={len(t)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--max-sims", type=int, default=4000)
    ap.add_argument("--clip-sigma", type=float, default=5.0)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint); ckpt_dir = ckpt.parent
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]; gcfg = cfg["model"]["gnn"]; pcfg = cfg["graph"]["profile_branch"]; tv2 = cfg["training_v2"]
    schema = "timeline" if tv2.get("label_schema") == "timeline" else "compact"
    target = tv2.get("binary_target", "impact_strong"); thr = float(tv2.get("impact_threshold", 0.5))
    T = 1.0
    cj = ckpt_dir / "calibration.json"
    if cj.exists():
        T = float(json.loads(cj.read_text()).get("temperature", 1.0))

    pdim = profile_feature_dim(int(pcfg["n_bins"]), pcfg["feature_set"], bool(pcfg["include_stream_onehot"])) if pcfg.get("enabled") else 0
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
    ds = StreamSimDataset(args.sim_dir, k_neighbors=k, max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False, normalizer=normalizer, preload_ram=False, label_schema=schema)
    split = load_split_indices(ckpt_dir / f"split_n{len(ds)}_seed{cfg['training'].get('split_seed', 42)}_{_tag(target, thr)}.npz")
    idxs = split["test"].tolist()[: args.max_sims]

    def phys(i):
        h5p, rid, sname = ds._index[i]
        with h5py.File(h5p, "r") as f:
            sub = f[f"simulations/{rid}/subhalos"]; lab = f[f"simulations/{rid}/labels"]
            n = int(lab["n_subhalos"][()]); strg = float(lab.attrs.get("impact_strength", 0.0))
            if n > 0 and sub["mass"].shape[0] > 0:
                return (1, float(np.log10(sub["mass"][0])),
                        float(np.log10(max(sub["t_since_impact_gyr"][0], 1e-3))), strg)
            return (0, np.nan, np.nan, 0.0)

    loader = PyGDataLoader(torch.utils.data.Subset(ds, idxs), batch_size=64, shuffle=False)
    pm, pt, pp = [], [], []
    with torch.no_grad():
        for batch in loader:
            build_knn_graph_batched(batch, k, normalizer)
            if pcfg.get("enabled"):
                build_profile_features_batched(batch, n_bins=int(pcfg["n_bins"]),
                    feature_set=pcfg["feature_set"], include_stream_onehot=bool(pcfg["include_stream_onehot"]))
            batch.x = torch.clamp((batch.x - nm) / ns, -args.clip_sigma, args.clip_sigma)
            _, bl, _, reg = model(batch)
            pm.append(reg[:, 0].cpu().numpy()); pt.append(reg[:, 1].cpu().numpy())
            pp.append(torch.sigmoid(bl.squeeze(-1) / T).numpy())
    pred_m = np.concatenate(pm); pred_t = np.concatenate(pt); probs = np.concatenate(pp)
    ph = np.array([phys(i) for i in idxs])
    is_imp = ph[:, 0] > 0; det = probs > 0.5
    true_m, true_t, strg = ph[:, 1], ph[:, 2], ph[:, 3]

    print("=" * 64); print(f"CHARACTERIZATION (mass_time heads), test n={len(idxs)}"); print("=" * 64)
    for label, m in (("ALL impacts", is_imp), ("DETECTED impacts (p>0.5)", is_imp & det)):
        print(f"\n  --- {label}  (n={int(m.sum())}) ---")
        _stats(true_m, pred_m, m, "log10 mass [dex]")
        _stats(true_t, pred_t, m, "log10 t_since [dex]")
    print("\n  mass recovery R2 by gap strength (impacts):")
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 1.01)):
        m = is_imp & (strg >= lo) & (strg < hi)
        if m.sum() >= 10:
            r2 = r2_score(true_m[m], pred_m[m])
            print(f"    strength [{lo:.1f},{hi:.1f})  n={int(m.sum()):4d}  mass R2={r2:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
