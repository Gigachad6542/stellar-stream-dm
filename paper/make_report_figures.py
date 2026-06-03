#!/usr/bin/env python
"""
Extra figures for the full report (current validated pipeline only).

Runs the trained detector (detector_df_20260602) and the detect+characterize model
(detector_char_20260602) over the held-out test split once each, then builds:
  fig5_completeness_2d   P(detect) over mass x time-since-impact
  fig6_score_separation  calibrated p_impact: impact vs no-impact
  fig7_reliability       calibration / reliability diagram
  fig8_characterization  recovered vs true (time weak, mass unrecoverable)
  fig9_embedding_pca     2-D PCA of the GNN embedding
  fig10_transfer         WDM/FDM mass-function suppression vs our sensitive band
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_training_data import _fix_galpy_dll_path
_fix_galpy_dll_path()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import h5py

from src.data.dataset import (StreamSimDataset, FeatureNormalizer, profile_feature_dim,
    build_knn_graph_batched, build_profile_features_batched)
from src.models.gnn import StreamGNNMultiTaskV2
from src.models.utils import load_normalizer
from src.data.splits import load_split_indices
from torch_geometric.loader import DataLoader as PyGDataLoader

FIG = Path("paper/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "figure.facecolor": "white"})
C_S, C_I, C_G, C_B = "#2c7fb8", "#d95f0e", "#31a354", "#de2d26"
SIMDIR = "data/simulations_detector_df"


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {FIG/name}")


def run_model(ckpt_dir: str, want_embed=False, want_reg=False, max_sims=4000, prefer_best=False):
    cd = Path(ckpt_dir)
    ckpt = cd / "gnn_v2_best.pt" if prefer_best else (
        cd / "gnn_v2_best_acc.pt" if (cd/"gnn_v2_best_acc.pt").exists() else cd/"gnn_v2_best.pt")
    pl = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = pl["config"]; g = cfg["model"]["gnn"]; pc = cfg["graph"]["profile_branch"]; tv = cfg["training_v2"]
    T = 1.0
    if (cd/"calibration.json").exists():
        T = float(json.loads((cd/"calibration.json").read_text()).get("temperature", 1.0))
    pdim = profile_feature_dim(int(pc["n_bins"]), pc["feature_set"], bool(pc["include_stream_onehot"])) if pc.get("enabled") else 0
    model = StreamGNNMultiTaskV2(n_reg_targets=tv.get("n_reg_targets", 2), predict_uncertainty=tv.get("predict_uncertainty", False),
        profile_dim=pdim, profile_hidden_dim=int(pc.get("hidden_dim", 64)), profile_layer_norm=bool(pc.get("layer_norm", False)),
        n_node_features=cfg["graph"]["n_node_features"], n_edge_features=cfg["graph"]["n_edge_features"],
        hidden_dim=g["hidden_dim"], n_layers=g["n_layers"], embedding_dim=g["embedding_dim"],
        dropout=g["dropout"], use_attention_readout=g.get("use_attention_readout", False))
    model.load_state_dict(pl["model_state_dict"]); model.eval()
    mean, std = load_normalizer(cd / "normalizer_v2.npz"); nz = FeatureNormalizer(mean, std); nm, ns = nz.mean, nz.std
    k = cfg["graph"]["k_neighbors"]; thr = float(tv.get("impact_threshold", 0.5))
    ds = StreamSimDataset(SIMDIR, k_neighbors=k, max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False, normalizer=nz, preload_ram=False, label_schema="compact")
    tag = f"impact_strong_t{f'{thr:g}'.replace('.', 'p')}"
    split = load_split_indices(cd / f"split_n{len(ds)}_seed{cfg['training'].get('split_seed', 42)}_{tag}.npz")
    idxs = split["test"].tolist()[:max_sims]
    loader = PyGDataLoader(torch.utils.data.Subset(ds, idxs), batch_size=64, shuffle=False)
    probs, embs, regs = [], [], []
    with torch.no_grad():
        for b in loader:
            build_knn_graph_batched(b, k, nz)
            if pc.get("enabled"):
                build_profile_features_batched(b, n_bins=int(pc["n_bins"]), feature_set=pc["feature_set"], include_stream_onehot=bool(pc["include_stream_onehot"]))
            b.x = torch.clamp((b.x - nm) / ns, -5, 5)
            emb, bl, _, reg = model(b)
            probs.append(torch.sigmoid(bl.squeeze(-1) / T).numpy())
            if want_embed: embs.append(emb.cpu().numpy())
            if want_reg: regs.append(reg.cpu().numpy())
    out = {"probs": np.concatenate(probs), "idxs": idxs, "ds": ds}
    if want_embed: out["emb"] = np.concatenate(embs)
    if want_reg: out["reg"] = np.concatenate(regs)
    # join physical params from HDF5
    n, lm, bb, tt, st, sn = [], [], [], [], [], []
    for i in idxs:
        h5p, rid, sname = ds._index[i]
        with h5py.File(h5p, "r") as f:
            sub = f[f"simulations/{rid}/subhalos"]; lab = f[f"simulations/{rid}/labels"]
            ni = int(lab["n_subhalos"][()]); s = float(lab.attrs.get("impact_strength", 0.0))
            if ni > 0 and sub["mass"].shape[0] > 0:
                n.append(1); lm.append(float(np.log10(sub["mass"][0]))); bb.append(float(sub["impact_param"][0]))
                tt.append(float(sub["t_since_impact_gyr"][0])); st.append(s); sn.append(sname)
            else:
                n.append(0); lm.append(np.nan); bb.append(np.nan); tt.append(np.nan); st.append(0.0); sn.append(sname)
    out.update(n=np.array(n), logM=np.array(lm), b=np.array(bb), t=np.array(tt),
               strength=np.array(st), stream=np.array(sn))
    return out


def fig_completeness_2d(d):
    imp = d["n"] > 0; det = (d["probs"] > 0.5)
    mb = np.linspace(7.5, 8.7, 5); tb = np.linspace(0.2, 1.5, 5)
    H = np.full((4, 4), np.nan)
    for i in range(4):
        for j in range(4):
            m = imp & (d["logM"] >= mb[i]) & (d["logM"] < mb[i+1]) & (d["t"] >= tb[j]) & (d["t"] < tb[j+1])
            if m.sum() >= 5: H[j, i] = det[m].mean()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=[7.5, 8.7, 0.2, 1.5])
    for i in range(4):
        for j in range(4):
            if np.isfinite(H[j, i]):
                ax.text(7.5+(i+0.5)*0.3, 0.2+(j+0.5)*0.325, f"{H[j,i]:.2f}", ha="center", va="center",
                        color="white" if H[j,i] < 0.6 else "black", fontsize=9, fontweight="bold")
    ax.set_xlabel("subhalo mass  log$_{10}(M/M_\\odot)$"); ax.set_ylabel("time since impact [Gyr]")
    ax.set_title("Figure 5 — Completeness over mass $\\times$ recency")
    fig.colorbar(im, ax=ax, label="P(detect | impact)")
    save(fig, "fig5_completeness_2d.png")


def fig_score_separation(d):
    imp = d["strength"] > 0.5; no = d["n"] == 0
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bins = np.linspace(0, 1, 31)
    ax.hist(d["probs"][no], bins=bins, color=C_S, alpha=0.7, label="no-impact", density=True)
    ax.hist(d["probs"][imp], bins=bins, color=C_I, alpha=0.7, label="detectable impact", density=True)
    ax.axvline(0.5, color=C_B, ls="--", lw=1.5, label="operating threshold")
    ax.set_xlabel("calibrated  $p_\\mathrm{impact}$"); ax.set_ylabel("normalized density")
    ax.set_title("Figure 6 — Detector score separation (zero false positives)")
    ax.legend()
    save(fig, "fig6_score_separation.png")


def fig_reliability(d):
    lab = (d["strength"] > 0.5).astype(float); p = d["probs"]
    bins = np.linspace(0, 1, 11); idx = np.digitize(p, bins) - 1
    xs, ys = [], []
    for bcell in range(10):
        m = idx == bcell
        if m.sum() >= 10: xs.append(p[m].mean()); ys.append(lab[m].mean())
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot([0, 1], [0, 1], ls=":", color="#999", label="perfect calibration")
    ax.plot(xs, ys, "o-", color=C_G, lw=2, ms=7, label="detector")
    ax.set_xlabel("mean predicted $p_\\mathrm{impact}$"); ax.set_ylabel("observed impact fraction")
    ax.set_title("Figure 7 — Calibration (reliability diagram)")
    ax.legend(loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig7_reliability.png")


def fig_characterization(dc):
    from sklearn.metrics import r2_score
    imp = dc["n"] > 0; reg = dc["reg"]
    tt, mm = dc["t"][imp], dc["logM"][imp]
    pt = reg[imp, 1]; pm = reg[imp, 0]
    ptl = np.log10(np.clip(tt, 1e-3, None))
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.2))
    ax[0].scatter(ptl, pt, s=6, alpha=0.3, color=C_S, edgecolors="none")
    lim0 = [min(ptl.min(), pt.min()), max(ptl.max(), pt.max())]
    ax[0].plot(lim0, lim0, ls=":", color="#666")
    ax[0].set_xlabel("true log$_{10}$ t$_{since}$ [Gyr]"); ax[0].set_ylabel("recovered")
    ax[0].set_title(f"(a) Recency: weakly recoverable (R²={r2_score(ptl,pt):+.2f})")
    ax[1].scatter(mm, pm, s=6, alpha=0.3, color=C_I, edgecolors="none")
    ax[1].plot([7.5, 8.7], [7.5, 8.7], ls=":", color="#666")
    ax[1].set_xlabel("true subhalo log$_{10}$M"); ax[1].set_ylabel("recovered")
    ax[1].set_title(f"(b) Mass: unrecoverable (R²={r2_score(mm,pm):+.2f})")
    fig.suptitle("Figure 8 — Single-gap characterization is degeneracy-limited", y=1.02, fontsize=13)
    save(fig, "fig8_characterization.png")


def fig_embedding_pca(d):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(d["emb"]); Z = PCA(n_components=2).fit_transform(X)
    imp = d["strength"] > 0.5; no = d["n"] == 0
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(Z[no, 0], Z[no, 1], s=7, alpha=0.4, color=C_S, label="no-impact", edgecolors="none")
    ax.scatter(Z[imp, 0], Z[imp, 1], s=7, alpha=0.5, color=C_I, label="detectable impact", edgecolors="none")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
    ax.set_title("Figure 9 — GNN embedding separates impacted streams")
    ax.legend()
    save(fig, "fig9_embedding_pca.png")


def fig_transfer():
    from src.simulation.mass_functions import wdm_suppression, fdm_suppression, wdm_half_mode_mass, fdm_jeans_mass
    M = np.logspace(6.5, 9.0, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(M, np.ones_like(M), color="#444", lw=2, label="CDM / SIDM")
    ax.plot(M, wdm_suppression(M, wdm_half_mode_mass(6.0)), color="#2c7fb8", lw=2, label="WDM 6 keV")
    ax.plot(M, wdm_suppression(M, wdm_half_mode_mass(3.0)), color="#41b6c4", lw=2, label="WDM 3 keV")
    ax.plot(M, fdm_suppression(M, fdm_jeans_mass(1e-22)), color=C_I, lw=2, label="FDM 10$^{-22}$eV")
    ax.axvspan(10**7.5, 10**8.7, color="#fdd", alpha=0.5, label="our sensitive band")
    ax.set_xscale("log"); ax.set_xlabel("subhalo mass  $M/M_\\odot$")
    ax.set_ylabel("mass-function suppression  f(M)")
    ax.set_title("Figure 10 — Why DM models differ only as a population")
    ax.legend(fontsize=8, loc="lower right"); ax.set_ylim(0, 1.05)
    save(fig, "fig10_transfer.png")


if __name__ == "__main__":
    print("loading detector (detector_df)...")
    d = run_model("checkpoints/detector_df_20260602", want_embed=True)
    fig_completeness_2d(d); fig_score_separation(d); fig_reliability(d); fig_embedding_pca(d)
    fig_transfer()
    print("loading characterize model (detector_char)...")
    dc = run_model("checkpoints/detector_char_20260602", want_reg=True, prefer_best=True)
    fig_characterization(dc)
    print("done.")
