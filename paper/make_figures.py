#!/usr/bin/env python
"""
Generate the figure set for the (rebuilt) manuscript.

Each figure is self-contained (try/except) and saved to paper/figures/. Figures
reflect the CURRENT (2026-06-02 corrected-simulator) state of the project.
"""
from __future__ import annotations
import sys, os, json, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_training_data import _fix_galpy_dll_path
_fix_galpy_dll_path()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("paper/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.edgecolor": "#444",
})
C_SMOOTH, C_IMPACT, C_GOOD, C_BAD = "#2c7fb8", "#d95f0e", "#31a354", "#de2d26"


def save(fig, name):
    p = FIG / name
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------------------
def fig_example_streams():
    """What a subhalo impact looks like: smooth vs gapped stream."""
    import galstreams
    from src.simulation.potentials import get_mw_potential
    from src.simulation.stream_gen import generate_stream_df
    mws = galstreams.MWStreams(verbose=False); pot = get_mw_potential("config/streams.yaml")
    smooth = generate_stream_df("GD1", pot, n_stars=1500, seed=3, impact=False, mws=mws)
    ip = dict(mass=10**8.5, impactb_kpc=0.08, timpact_gyr=0.7, impact_angle_rad=0.4, vsub_kms=150.)
    imp = generate_stream_df("GD1", pot, n_stars=1500, seed=3, impact=True, impact_params=ip, mws=mws)

    fig, ax = plt.subplots(3, 1, figsize=(8.2, 7.2), height_ratios=[1, 1, 1.2])
    for a, p, c, t in ((ax[0], smooth, C_SMOOTH, "Smooth stream (no impact)  —  streamdf"),
                       (ax[1], imp, C_IMPACT, "Perturbed stream (10$^{8.5}\\,M_\\odot$ subhalo flyby)  —  streamgapdf")):
        a.scatter(p.phi1, p.phi2, s=4, c=c, alpha=0.5, edgecolors="none")
        a.set_title(t); a.set_ylabel("$\\phi_2$ [deg]"); a.set_ylim(-4, 4)
        a.set_xlim(-30, 35)
    ax[1].set_xlabel("$\\phi_1$ along stream [deg]")
    # density profiles
    bins = np.linspace(-30, 35, 44)
    hs, _ = np.histogram(smooth.phi1, bins=bins)
    hi, _ = np.histogram(imp.phi1, bins=bins)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    scale = hs.sum() / max(hi.sum(), 1)   # equal total counts -> fair density comparison
    hin = hi * scale
    ax[2].step(ctr, hs, where="mid", color=C_SMOOTH, lw=2, label="smooth")
    ax[2].step(ctr, hin, where="mid", color=C_IMPACT, lw=2, label="perturbed")
    # mark the gap: deepest INTERIOR deficit (exclude edge bins so we don't tag the ends)
    ratio = hin / np.maximum(hs, 1.0)
    ratio_m = np.where(np.arange(len(ratio)) >= 3, ratio, np.inf)
    ratio_m = np.where(np.arange(len(ratio)) < len(ratio) - 3, ratio_m, np.inf)
    gi = int(np.argmin(ratio_m)); gap = ctr[gi]
    ax[2].axvspan(gap - 1.5, gap + 1.5, color=C_BAD, alpha=0.12)
    ax[2].annotate("density gap\ncarved by the flyby", xy=(gap, hin[gi]),
                   xytext=(gap - 3, max(hs) * 0.88), color=C_BAD, fontsize=10, ha="right",
                   arrowprops=dict(arrowstyle="->", color=C_BAD))
    ax[2].set_title("Star counts along the stream"); ax[2].set_xlabel("$\\phi_1$ [deg]")
    ax[2].set_ylabel("stars / bin"); ax[2].legend(); ax[2].set_xlim(-30, 35)
    fig.suptitle("Figure 1 — A subhalo flyby imprints a localized density gap", y=1.00, fontsize=13)
    save(fig, "fig1_what_impact_looks_like.png")


# ---------------------------------------------------------------------------
def fig_simulator_fix():
    """Before/after: separability and detector AUC."""
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.0))
    # separability G6
    ax[0].bar(["homemade\ngenerator", "corrected\n(streamgapdf)"], [0.57, 0.94],
              color=[C_BAD, C_GOOD], width=0.6)
    ax[0].axhline(0.85, ls="--", color="#666", lw=1); ax[0].text(1.45, 0.86, "target 0.85", ha="right", fontsize=9, color="#666")
    ax[0].set_ylim(0.5, 1.0); ax[0].set_ylabel("impact vs smooth separability (AUC)")
    ax[0].set_title("(a) Generator separability")
    for i, v in enumerate([0.57, 0.94]):
        ax[0].text(i, v+0.005, f"{v:.2f}", ha="center", fontweight="bold")
    # detector AUC
    ax[1].bar(["v3\n(point-like bug)", "corrected\nsimulator"], [0.62, 0.982],
              color=[C_BAD, C_GOOD], width=0.6)
    ax[1].axhline(0.5, ls=":", color="#999", lw=1); ax[1].text(1.45, 0.51, "chance", ha="right", fontsize=9, color="#999")
    ax[1].set_ylim(0.5, 1.0); ax[1].set_ylabel("detector test AUC")
    ax[1].set_title("(b) Detector performance")
    for i, v in enumerate([0.62, 0.982]):
        ax[1].text(i, v+0.005, f"{v:.3f}", ha="center", fontweight="bold")
    fig.suptitle("Figure 2 — Fixing the simulator (units bug + validated DFs) unlocked the detector",
                 y=1.02, fontsize=13)
    save(fig, "fig2_simulator_fix.png")


# ---------------------------------------------------------------------------
def fig_completeness():
    """Detection completeness vs mass and vs time-since-impact (measured)."""
    massc = [7.65, 7.95, 8.25, 8.55]; massp = [0.478, 0.502, 0.604, 0.723]
    timec = [0.35, 0.65, 0.95, 1.30]; timep = [0.747, 0.636, 0.565, 0.389]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.0))
    ax[0].plot(massc, massp, "o-", color=C_IMPACT, lw=2, ms=8)
    ax[0].set_xlabel("subhalo mass  log$_{10}(M/M_\\odot)$"); ax[0].set_ylabel("P(detect | impact)")
    ax[0].set_title("(a) Heavier = more detectable"); ax[0].set_ylim(0.3, 0.8)
    ax[1].plot(timec, timep, "s-", color=C_SMOOTH, lw=2, ms=8)
    ax[1].set_xlabel("time since impact [Gyr]"); ax[1].set_ylabel("P(detect | impact)")
    ax[1].set_title("(b) Recent = more detectable"); ax[1].set_ylim(0.3, 0.8)
    fig.suptitle("Figure 4 — What the detector can see: completeness across impact types",
                 y=1.02, fontsize=13)
    save(fig, "fig4_completeness.png")


# ---------------------------------------------------------------------------
def fig_dm_family():
    """Detected-mass distributions per DM model + N_det to distinguish from CDM."""
    from src.simulation.mass_functions import sample_cdm_masses, sample_wdm_masses, sample_fdm_masses
    CL_LOGM = np.array([7.0, 7.65, 7.95, 8.25, 8.55, 9.0]); CL_P = np.array([0.25, 0.478, 0.502, 0.604, 0.723, 0.85])
    rng = np.random.default_rng(0); N = 300000; lo, hi, a = 6.5, 9.0, -1.9
    def det(m):
        lm = np.log10(m); keep = rng.uniform(size=len(lm)) < np.clip(np.interp(lm, CL_LOGM, CL_P), 0, 1); return lm[keep]
    series = {
        "CDM / SIDM": (det(sample_cdm_masses(N, lo, hi, a, seed=1)), "#444"),
        "WDM 6 keV": (det(sample_wdm_masses(N, 6.0, lo, hi, a, seed=3)), "#2c7fb8"),
        "WDM 3 keV": (det(sample_wdm_masses(N, 3.0, lo, hi, a, seed=4)), "#41b6c4"),
        "FDM 10$^{-22}$eV": (det(sample_fdm_masses(N, 1e-22, lo, hi, a, seed=5)), C_IMPACT),
    }
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0))
    bins = np.linspace(7.0, 9.0, 30)
    for k, (d, c) in series.items():
        ax[0].hist(d, bins=bins, density=True, histtype="step", lw=2, color=c, label=k)
    ax[0].set_xlabel("detected impact mass  log$_{10}(M/M_\\odot)$"); ax[0].set_ylabel("normalized density")
    ax[0].set_title("(a) Detected-mass distribution by DM model"); ax[0].legend(fontsize=8)
    labels = ["FDM\n10$^{-22}$", "WDM\n3keV", "WDM\n6keV", "FDM\n10$^{-21}$", "SIDM"]
    ndet = [5, 12, 27, 135, 214000]
    cols = [C_GOOD, C_GOOD, "#78c679", "#fec44f", C_BAD]
    ax[1].bar(labels, ndet, color=cols, width=0.65); ax[1].set_yscale("log")
    ax[1].set_ylabel("detections needed to distinguish from CDM (95%)")
    ax[1].set_title("(b) Population test sample size")
    for i, v in enumerate(ndet):
        ax[1].text(i, v*1.3, f"{v:,}" if v < 1000 else f"{v/1000:.0f}k", ha="center", fontsize=8, fontweight="bold")
    fig.suptitle("Figure 5 — Telling dark-matter models apart is a population measurement",
                 y=1.02, fontsize=13)
    save(fig, "fig5_dm_family.png")


# ---------------------------------------------------------------------------
def fig_multistream():
    """Per-stream naive vs look-elsewhere z + joint result."""
    p = Path("outputs/multistream/joint_significance_corrected.json")
    d = json.loads(p.read_text())
    rows = d.get("per_stream", d.get("streams", []))
    names = [r["stream"] for r in rows]
    znaive = [r.get("z", np.nan) for r in rows]
    zle = [r.get("le_z", np.nan) for r in rows]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.bar(x - w/2, znaive, w, label="naive (single best cell)", color="#bdbdbd")
    ax.bar(x + w/2, zle, w, label="look-elsewhere corrected", color=C_SMOOTH)
    ax.axhline(0, color="#444", lw=0.8); ax.axhline(3, ls="--", color=C_BAD, lw=1)
    ax.text(len(names)-0.5, 3.1, "3$\\sigma$", color=C_BAD, fontsize=9, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("significance z")
    sj = d.get("stouffer_z", np.nan); fp = d.get("fisher_p", np.nan)
    ax.set_title(f"Figure 6 — No coherent detection: naive significance collapses under "
                 f"look-elsewhere\njoint Stouffer Z={sj:.2f}, Fisher p={fp:.2f}  (CDM-consistent upper limit)",
                 fontsize=11)
    ax.legend(loc="upper left")
    save(fig, "fig6_multistream.png")


# ---------------------------------------------------------------------------
def fig_roc():
    """ROC of the corrected detector on the test split."""
    import torch, h5py
    from src.data.dataset import (StreamSimDataset, FeatureNormalizer, profile_feature_dim,
        build_knn_graph_batched, build_profile_features_batched)
    from src.models.gnn import StreamGNNMultiTaskV2
    from src.models.utils import load_normalizer
    from src.data.splits import load_split_indices
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sklearn.metrics import roc_curve, roc_auc_score
    ckpt_dir = Path("checkpoints/detector_df_20260602")
    payload = torch.load(str(ckpt_dir / "gnn_v2_best_acc.pt"), map_location="cpu", weights_only=False)
    cfg = payload["config"]; g = cfg["model"]["gnn"]; pc = cfg["graph"]["profile_branch"]; tv = cfg["training_v2"]
    pdim = profile_feature_dim(int(pc["n_bins"]), pc["feature_set"], bool(pc["include_stream_onehot"])) if pc.get("enabled") else 0
    model = StreamGNNMultiTaskV2(n_reg_targets=tv.get("n_reg_targets", 2), predict_uncertainty=tv.get("predict_uncertainty", False),
        profile_dim=pdim, profile_hidden_dim=int(pc.get("hidden_dim", 64)), profile_layer_norm=bool(pc.get("layer_norm", False)),
        n_node_features=cfg["graph"]["n_node_features"], n_edge_features=cfg["graph"]["n_edge_features"],
        hidden_dim=g["hidden_dim"], n_layers=g["n_layers"], embedding_dim=g["embedding_dim"],
        dropout=g["dropout"], use_attention_readout=g.get("use_attention_readout", False))
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    mean, std = load_normalizer(ckpt_dir / "normalizer_v2.npz"); nz = FeatureNormalizer(mean, std); nm, ns = nz.mean, nz.std
    k = cfg["graph"]["k_neighbors"]
    ds = StreamSimDataset("data/simulations_detector_df", k_neighbors=k, max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False, normalizer=nz, preload_ram=False, label_schema="compact")
    thr = float(tv.get("impact_threshold", 0.5))
    tag = f"impact_strong_t{f'{thr:g}'.replace('.', 'p')}"
    split = load_split_indices(ckpt_dir / f"split_n{len(ds)}_seed{cfg['training'].get('split_seed', 42)}_{tag}.npz")
    idxs = split["test"].tolist()[:3000]
    loader = PyGDataLoader(torch.utils.data.Subset(ds, idxs), batch_size=64, shuffle=False)
    probs, labs = [], []
    with torch.no_grad():
        for batch in loader:
            build_knn_graph_batched(batch, k, nz)
            if pc.get("enabled"):
                build_profile_features_batched(batch, n_bins=int(pc["n_bins"]), feature_set=pc["feature_set"], include_stream_onehot=bool(pc["include_stream_onehot"]))
            y = batch.y.view(batch.num_graphs, -1)
            labs.append((y[:, 5] > thr).float().numpy())
            batch.x = torch.clamp((batch.x - nm) / ns, -5, 5)
            _, bl, _, _ = model(batch); probs.append(torch.sigmoid(bl.squeeze(-1)).numpy())
    probs = np.concatenate(probs); labs = np.concatenate(labs)
    fpr, tpr, _ = roc_curve(labs, probs); auc = roc_auc_score(labs, probs)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(fpr, tpr, color=C_GOOD, lw=2.5, label=f"corrected detector (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="#999", label="chance")
    ax.plot([0, 1], [0, 1], alpha=0)  # spacer
    ax.fill_between(fpr, tpr, alpha=0.12, color=C_GOOD)
    ax.text(0.55, 0.20, "v3 (point-like bug)\nAUC = 0.62", color=C_BAD, fontsize=10, fontweight="bold")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("Figure 3 — Detector ROC on faithful simulations")
    ax.legend(loc="lower right"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig3_detector_roc.png")


if __name__ == "__main__":
    for fn in (fig_simulator_fix, fig_completeness, fig_dm_family, fig_multistream,
               fig_example_streams, fig_roc):
        try:
            fn()
        except Exception:
            print(f"  FAILED {fn.__name__}:"); traceback.print_exc()
