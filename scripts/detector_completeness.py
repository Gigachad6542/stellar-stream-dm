#!/usr/bin/env python
"""
Completeness / confusion diagnostic for the impact detector across impact TYPES.

Runs the trained binary detector on the held-out test split, joins each
prediction to that sim's physical impact parameters (subhalo mass, impact
parameter b, time-since-impact, realised gap strength) read from the HDF5, and
reports the detection probability P(p_impact > 0.5) as a function of each axis.

This answers "which TYPES of subhalo impact can we detect" and shows exactly
where the detector succeeds vs fails -- the basis for targeted improvement.

Usage:
  python scripts/detector_completeness.py \
      --checkpoint checkpoints/detector_df_20260602/gnn_v2_best_acc.pt \
      --sim-dir data/simulations_detector_df
"""
from __future__ import annotations
import sys, os, argparse, json
from pathlib import Path
from collections import defaultdict

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


def _tag(target: str, thr: float) -> str:
    t = f"{thr:g}".replace("-", "m").replace(".", "p")
    return f"{target}_t{t}"


def _bin_report(name, x, detected, edges):
    """Print P(detect) in bins of x (only over impacts)."""
    x = np.asarray(x); detected = np.asarray(detected)
    print(f"\n  Completeness vs {name}:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() == 0:
            continue
        print(f"    [{lo:6.2f},{hi:6.2f})  n={m.sum():4d}  P(detect)={detected[m].mean():.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--config", default="config/training.yaml")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--max-sims", type=int, default=4000)
    ap.add_argument("--clip-sigma", type=float, default=5.0)
    ap.add_argument("--p-threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint); ckpt_dir = ckpt.parent
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    gcfg = cfg["model"]["gnn"]; pcfg = cfg["graph"]["profile_branch"]; tv2 = cfg["training_v2"]
    schema = "timeline" if tv2.get("label_schema") == "timeline" else "compact"
    target = tv2.get("binary_target", "impact_detectable")
    thr = float(tv2.get("impact_threshold", 0.5))

    # temperature (for calibrated p) if present
    T = 1.0
    cj = ckpt_dir / "calibration.json"
    if cj.exists():
        T = float(json.loads(cj.read_text()).get("temperature", 1.0))

    pdim = profile_feature_dim(int(pcfg["n_bins"]), pcfg["feature_set"],
                              bool(pcfg["include_stream_onehot"])) if pcfg.get("enabled") else 0
    model = StreamGNNMultiTaskV2(
        n_reg_targets=tv2.get("n_reg_targets", 2),
        predict_uncertainty=tv2.get("predict_uncertainty", False),
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
    idxs = split[args.split].tolist()[: args.max_sims]

    # Per-sim physical params from HDF5 (joined by dataset _index order).
    def phys_for(i):
        h5_path, run_id, sname = ds._index[i]
        with h5py.File(h5_path, "r") as f:
            sub = f[f"simulations/{run_id}/subhalos"]
            lab = f[f"simulations/{run_id}/labels"]
            n = int(lab["n_subhalos"][()])
            strength = float(lab.attrs.get("impact_strength", 0.0))
            if n > 0 and sub["mass"].shape[0] > 0:
                return dict(stream=sname, n=n,
                            logM=float(np.log10(sub["mass"][0])),
                            b=float(sub["impact_param"][0]),
                            t=float(sub["t_since_impact_gyr"][0]),
                            strength=strength)
            return dict(stream=sname, n=0, logM=np.nan, b=np.nan, t=np.nan, strength=0.0)

    sub = torch.utils.data.Subset(ds, idxs)
    loader = PyGDataLoader(sub, batch_size=64, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            build_knn_graph_batched(batch, k, normalizer)
            if pcfg.get("enabled"):
                build_profile_features_batched(batch, n_bins=int(pcfg["n_bins"]),
                    feature_set=pcfg["feature_set"], include_stream_onehot=bool(pcfg["include_stream_onehot"]))
            batch.x = torch.clamp((batch.x - nm) / ns, -args.clip_sigma, args.clip_sigma)
            _, bl, _, _ = model(batch)
            probs.append(torch.sigmoid(bl.squeeze(-1) / T))
    probs = torch.cat(probs).numpy()

    phys = [phys_for(i) for i in idxs]
    is_imp = np.array([p["n"] > 0 for p in phys])
    detected = (probs > args.p_threshold)

    # No-impact false-positive rate
    no = ~is_imp
    fpr = float(detected[no].mean()) if no.any() else float("nan")
    overall_compl = float(detected[is_imp].mean()) if is_imp.any() else float("nan")
    print("=" * 66)
    print(f"DETECTOR COMPLETENESS / CONFUSION  ({args.split} split, n={len(idxs)})")
    print("=" * 66)
    print(f"  impacts={int(is_imp.sum())}  no-impacts={int(no.sum())}  T={T:.3f}  p_thr={args.p_threshold}")
    print(f"  overall completeness P(detect | impact) = {overall_compl:.3f}")
    print(f"  false-positive rate  P(detect | no-impact) = {fpr:.3f}")

    impM = np.array([p["logM"] for p in phys])[is_imp]
    impB = np.array([p["b"] for p in phys])[is_imp]
    impT = np.array([p["t"] for p in phys])[is_imp]
    impS = np.array([p["strength"] for p in phys])[is_imp]
    det_imp = detected[is_imp]
    _bin_report("subhalo mass log10(M)", impM, det_imp, np.array([7.5, 7.8, 8.1, 8.4, 8.7]))
    _bin_report("impact parameter b [kpc]", impB, det_imp, np.array([0.0, 0.1, 0.2, 0.3, 0.4]))
    _bin_report("time since impact [Gyr]", impT, det_imp, np.array([0.2, 0.5, 0.8, 1.1, 1.5]))
    _bin_report("realised gap strength", impS, det_imp, np.array([0.0, 0.3, 0.5, 0.7, 1.01]))

    # Per-stream completeness
    streams = np.array([p["stream"] for p in phys])[is_imp]
    print("\n  Completeness by stream:")
    for s in sorted(set(streams)):
        m = streams == s
        print(f"    {s:8s} n={int(m.sum()):4d}  P(detect)={det_imp[m].mean():.3f}")

    if args.out:
        rec = {"overall_completeness": overall_compl, "fpr": fpr, "T": T,
               "n": len(idxs), "impacts": int(is_imp.sum())}
        Path(args.out).write_text(json.dumps(rec, indent=2))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
