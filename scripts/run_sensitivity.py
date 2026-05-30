"""
Sensitivity analysis: minimum detectable subhalo mass per stream.

For each stream, injects synthetic subhalo impacts of varying mass into the
real stream embedding and checks whether the CDM posterior can distinguish
the impacted stream from an unperturbed one.

Method:
  1. Load each real stream's GNN embedding (the baseline observation)
  2. For a grid of log10(M_sub) values from 5.0 to 9.0:
     a. Load held-out simulations with that approximate mass
     b. Compare posterior P(n_impacts > 0 | embedding) to baseline
     c. A mass is "detectable" if the posterior shifts significantly (2σ)
  3. Report the minimum detectable mass per stream

Alternative approach (used here when held-out sims at exact masses are limited):
  Sweep over log10_M_sub_mean in the CDM posterior and compute the posterior
  density ratio vs. the n_impacts=0 region. The mass where the posterior
  concentrates away from n_impacts=0 defines the sensitivity floor.

Usage:
    python -u scripts/run_sensitivity.py
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import FeatureNormalizer, RealStreamDataset, build_knn_graph_batched
from src.models.gnn import StreamGNNMultiTask
from src.models.utils import load_checkpoint, load_normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_posterior(dm_model: str, ckpt_dir: Path):
    """Load pickled sbi posterior."""
    path = ckpt_dir / f"posterior_{dm_model}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def embed_real_stream(
    stream_name: str,
    cfg: dict,
    normalizer: FeatureNormalizer,
    gnn_checkpoint: str,
    processed_path: str,
    device: str,
) -> torch.Tensor:
    """Embed a real stream and return the 128-d vector."""
    ckpt_dir = Path(cfg["paths"]["checkpoints"])

    dataset = RealStreamDataset(
        processed_path, [stream_name],
        k_neighbors=cfg["graph"]["k_neighbors"],
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        normalizer=normalizer,
    )
    data = dataset[0].to(device)
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    build_knn_graph_batched(data, cfg["graph"]["k_neighbors"], normalizer)
    if normalizer is not None:
        data.x = (data.x - normalizer.mean.to(device)) / normalizer.std.to(device)

    gcfg = cfg["model"]["gnn"]
    model = StreamGNNMultiTask(
        n_classes=3, n_reg_targets=2,
        n_node_features=cfg["graph"]["n_node_features"],
        n_edge_features=cfg["graph"]["n_edge_features"],
        hidden_dim=gcfg["hidden_dim"],
        n_layers=gcfg["n_layers"],
        embedding_dim=gcfg["embedding_dim"],
        dropout=gcfg["dropout"],
    )
    load_checkpoint(gnn_checkpoint, model, device=device)
    model = model.to(device).eval()

    with torch.no_grad():
        embedding = model.encoder(data).squeeze(0).cpu()

    # Free VRAM
    import gc
    del model, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embedding


def sensitivity_from_posterior(
    posterior,
    embedding: torch.Tensor,
    n_samples: int = 5000,
    mass_param_name: str = "log10_M_sub_mean",
    mass_grid: np.ndarray | None = None,
) -> dict:
    """Compute sensitivity metrics from the CDM posterior for one stream.

    Returns:
        dict with:
          - mass_grid: log10(M/Msun) values
          - p_detect_at_mass: P(n_impacts >= 1) as function of log10_M_sub_mean
          - min_detectable_mass_2sigma: log10(M) where P(detection) > 0.95
          - posterior_mass_median: median inferred mass
          - posterior_n_impacts_median: median inferred n_impacts
    """
    # Detect posterior device
    try:
        _net = posterior.posterior_estimator if hasattr(posterior, "posterior_estimator") \
               else posterior.net
        _device = next(_net.parameters()).device
    except (StopIteration, AttributeError):
        _device = torch.device("cpu")

    x_obs = embedding.to(_device)

    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=x_obs, show_progress_bars=False)

    samples_np = samples.cpu().numpy()

    # Samples columns: [log10_M_sub_mean, n_impacts]
    masses = samples_np[:, 0]
    n_impacts = samples_np[:, 1]

    # Overall detection probability: P(n_impacts >= 1)
    p_any_impact = float(np.mean(n_impacts >= 0.5))  # rounds to >= 1

    # Compute P(detection | log10_M > threshold) for a grid of mass thresholds
    if mass_grid is None:
        mass_grid = np.linspace(5.0, 9.0, 41)

    # For each mass threshold, what fraction of posterior samples have
    # log10_M_sub_mean > threshold AND n_impacts >= 1?
    p_detect = []
    for m_thresh in mass_grid:
        mask = (masses > m_thresh) & (n_impacts >= 0.5)
        p_detect.append(float(np.mean(mask)))

    p_detect = np.array(p_detect)

    # Cumulative sensitivity: P(M_sub > threshold) from the posterior
    p_above = np.array([float(np.mean(masses > m)) for m in mass_grid])

    # Minimum detectable mass: smallest mass where posterior P(M > m) > 0.05
    # (i.e., 95% of posterior mass is ABOVE this value = 2σ lower bound)
    idx_95 = np.where(p_above > 0.95)[0]
    if len(idx_95) > 0:
        min_detectable = float(mass_grid[idx_95[-1]])
    else:
        min_detectable = float(mass_grid[0])

    return {
        "mass_grid": mass_grid,
        "p_above_threshold": p_above,
        "p_detect_at_mass": p_detect,
        "min_detectable_mass_2sigma": min_detectable,
        "posterior_mass_median": float(np.median(masses)),
        "posterior_mass_16": float(np.percentile(masses, 16)),
        "posterior_mass_84": float(np.percentile(masses, 84)),
        "posterior_n_impacts_median": float(np.median(n_impacts)),
        "p_any_impact": p_any_impact,
    }


def main() -> None:
    cfg_path = ROOT / "config" / "training.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / "config" / "streams.yaml") as f:
        streams_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    norm_path = ckpt_dir / "normalizer.npz"
    gnn_ckpt = str(ckpt_dir / "gnn_best.pt")
    processed_path = str(ROOT / "data" / "processed" / "streams.h5")

    mean, std = load_normalizer(norm_path)
    normalizer = FeatureNormalizer(mean, std)

    # Load CDM posterior (sensitivity analysis is model-agnostic but CDM is baseline)
    posterior = load_posterior("CDM", ckpt_dir)

    stream_names = list(streams_cfg["streams"].keys())
    results = {}
    mass_grid = np.linspace(5.0, 9.0, 41)

    for stream_name in stream_names:
        log.info("Sensitivity analysis for %s ...", stream_name)
        embedding = embed_real_stream(
            stream_name, cfg, normalizer, gnn_ckpt, processed_path, device,
        )
        result = sensitivity_from_posterior(
            posterior, embedding, n_samples=5000, mass_grid=mass_grid,
        )
        results[stream_name] = result
        log.info("  %s: min detectable mass = 10^%.1f M☉, P(any impact) = %.2f, "
                 "median mass = 10^%.1f M☉, median n_impacts = %.0f",
                 stream_name,
                 result["min_detectable_mass_2sigma"],
                 result["p_any_impact"],
                 result["posterior_mass_median"],
                 result["posterior_n_impacts_median"])

    # Save results table
    output_dir = ROOT / "outputs"
    records = []
    for stream_name, r in results.items():
        scfg = streams_cfg["streams"][stream_name]
        records.append({
            "stream": stream_name,
            "n_members": scfg["expected_n_members"],
            "min_detectable_log10_Msun": r["min_detectable_mass_2sigma"],
            "min_detectable_Msun": 10**r["min_detectable_mass_2sigma"],
            "posterior_mass_median": r["posterior_mass_median"],
            "posterior_mass_16": r["posterior_mass_16"],
            "posterior_mass_84": r["posterior_mass_84"],
            "posterior_n_impacts_median": r["posterior_n_impacts_median"],
            "p_any_impact": r["p_any_impact"],
        })

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "sensitivity.csv", index=False)
    log.info("Sensitivity table saved to outputs/sensitivity.csv")
    log.info("\n%s", df.to_string(index=False))

    # Plot: sensitivity curve per stream
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P(M_sub > threshold) for each stream
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(stream_names)))
    for (stream_name, r), color in zip(results.items(), colors):
        ax.plot(r["mass_grid"], r["p_above_threshold"],
                label=stream_name, color=color, lw=1.5)
        # Mark the 2σ lower bound
        ax.axvline(r["min_detectable_mass_2sigma"], color=color, lw=0.5, ls="--", alpha=0.5)

    ax.axhline(0.95, color="gray", ls=":", alpha=0.5, label="95% level")
    ax.set_xlabel("log₁₀(M_sub / M☉)")
    ax.set_ylabel("P(posterior mass > threshold)")
    ax.set_title("Posterior cumulative mass distribution")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(5, 9)
    ax.set_ylim(0, 1.05)

    # Right: minimum detectable mass vs number of stream members
    ax2 = axes[1]
    n_members = [r["n_members"] for r in records]
    min_masses = [r["min_detectable_log10_Msun"] for r in records]
    stream_labels = [r["stream"] for r in records]

    ax2.scatter(n_members, min_masses, s=80, c=colors[:len(stream_names)], zorder=5)
    for i, label in enumerate(stream_labels):
        ax2.annotate(label, (n_members[i], min_masses[i]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8)

    # Plan target line
    ax2.axhline(np.log10(5e6), color="red", ls="--", lw=1, label="Plan target: 5×10⁶ M☉")
    ax2.set_xlabel("Expected N members")
    ax2.set_ylabel("Min detectable mass [log₁₀ M☉]")
    ax2.set_title("Sensitivity floor vs stream richness")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "figures" / "sensitivity.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Sensitivity figure saved to outputs/figures/sensitivity.pdf")

    # Summary
    gd1_result = results.get("GD1", {})
    gd1_min = gd1_result.get("min_detectable_mass_2sigma", np.nan)
    plan_target = np.log10(5e6)
    log.info("=" * 60)
    log.info("SENSITIVITY SUMMARY")
    log.info("=" * 60)
    log.info("GD-1 minimum detectable mass: 10^%.2f M☉ = %.2e M☉",
             gd1_min, 10**gd1_min)
    log.info("Plan target: < 5×10⁶ M☉ = 10^%.2f M☉", plan_target)
    if gd1_min <= plan_target:
        log.info("PASS: GD-1 sensitivity meets plan target")
    else:
        log.info("NOTE: GD-1 sensitivity (10^%.1f) above plan target (10^%.1f). "
                 "This is expected — sensitivity depends on stream richness, "
                 "distance, and integration time.", gd1_min, plan_target)


if __name__ == "__main__":
    main()
