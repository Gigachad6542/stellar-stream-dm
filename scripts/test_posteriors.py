"""
Posterior sanity test + fast SBC on pre-computed test-set embeddings.

Two-stage validation:
  Stage 1 — Sanity check (fast, ~1-3 min):
    For each DM model, take N random test simulations, embed via GNN,
    sample posterior, check that ground-truth theta falls inside the 90% CI
    at roughly the right rate.

  Stage 2 — Fast SBC (slower, ~5-15 min):
    For each DM model, use all available test-set simulations.
    Compute posterior ranks for each (theta, embedding) pair.
    Run K-S test against uniform distribution.
    Plot rank histograms.

Key design: uses pre-computed embeddings (NO new simulations).
Works by loading each DM model's sims, running GNN forward pass, then
conditioning each posterior on the resulting embeddings.

Usage:
    PYTHONNOUSERSITE=1 python -u scripts/test_posteriors.py
    PYTHONNOUSERSITE=1 python -u scripts/test_posteriors.py --n-sanity 20
    PYTHONNOUSERSITE=1 python -u scripts/test_posteriors.py --stage sanity
    PYTHONNOUSERSITE=1 python -u scripts/test_posteriors.py --stage sbc --n-sbc 200
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import StreamSimDataset, FeatureNormalizer, build_knn_graph_batched
from src.inference.sbi_pipeline import extract_theta_from_hdf5
from src.models.gnn import StreamGNNMultiTask
from src.models.utils import load_checkpoint, load_normalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "test_posteriors.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

DM_MODELS = ["CDM", "WDM", "FDM", "SIDM"]


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_encoder(cfg: dict, ckpt_path: Path, device: str) -> torch.nn.Module:
    """Load GNN encoder from StreamGNNMultiTask checkpoint."""
    gcfg = cfg["model"]["gnn"]
    multi_task = StreamGNNMultiTask(
        n_classes=3,
        n_reg_targets=2,
        n_node_features=cfg["graph"]["n_node_features"],
        n_edge_features=cfg["graph"]["n_edge_features"],
        hidden_dim=gcfg["hidden_dim"],
        n_layers=gcfg["n_layers"],
        embedding_dim=gcfg["embedding_dim"],
        dropout=gcfg["dropout"],
    )
    load_checkpoint(str(ckpt_path), multi_task, device=device)
    encoder = multi_task.encoder.to(device).eval()
    log.info("GNN encoder loaded (%d params)", sum(p.numel() for p in encoder.parameters()))
    return encoder


def load_posterior(checkpoint_dir: Path, dm_model: str):
    """Load pickled sbi posterior."""
    pkl_path = checkpoint_dir / f"posterior_{dm_model}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Posterior not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        posterior = pickle.load(f)
    log.info("Posterior loaded for %s from %s", dm_model, pkl_path)
    return posterior


# ---------------------------------------------------------------------------
# Embedding computation for a single DM model
# ---------------------------------------------------------------------------

def compute_test_embeddings_and_theta(
    dm_model: str,
    cfg: dict,
    encoder: torch.nn.Module,
    normalizer: FeatureNormalizer,
    device: str,
    n_max: int | None = None,
    batch_size: int = 64,
    rng_seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GNN embeddings and extract full theta for a DM model's test set.

    Loads sims from data/simulations/{dm_model}/, extracts theta from HDF5 attrs,
    computes GNN embeddings, aligns them, and returns (theta, embeddings).

    Args:
        n_max: Max number of sims to use (randomly subsampled). None = use all.

    Returns:
        theta:      [N, n_params] aligned theta tensor
        embeddings: [N, 128] aligned embedding tensor
    """
    from torch_geometric.loader import DataLoader  # noqa: PLC0415

    sim_dir = ROOT / cfg["paths"]["simulations"]
    k = cfg["graph"]["k_neighbors"]

    # Extract theta (includes valid_indices list, same order as HDF5 iteration)
    log.info("[%s] Extracting theta ...", dm_model)
    theta_all, valid_flat_indices, _theta_run_ids = extract_theta_from_hdf5(
        sim_dir=sim_dir,
        dm_model=dm_model,
        config_path=str(ROOT / "config" / "dm_models.yaml"),
    )
    log.info("[%s] theta shape: %s", dm_model, list(theta_all.shape))

    # Clamp to prior bounds (same as train_sbi.py — handles log_m=0 sentinels)
    try:
        from src.inference.sbi_pipeline import build_prior  # noqa: PLC0415
        prior = build_prior(dm_model, str(ROOT / "config" / "dm_models.yaml"), device="cpu")
        low  = getattr(prior, "low",  None) or getattr(prior, "_low",  None)
        high = getattr(prior, "high", None) or getattr(prior, "_high", None)
        if low is None:
            low  = prior.base_dist.low.cpu()
            high = prior.base_dist.high.cpu()
        else:
            low, high = low.cpu(), high.cpu()
        in_prior = ((theta_all >= low) & (theta_all <= high)).all(dim=1)
        n_out = (~in_prior).sum().item()
        if n_out > 0:
            log.info("[%s] Clamping %d theta rows to prior bounds (same as training)", dm_model, n_out)
            theta_all = theta_all.clamp(min=low, max=high)
    except Exception as e:
        log.warning("[%s] Could not clamp theta to prior: %s", dm_model, e)

    # Build dataset (augment=False, preload_ram=True)
    model_dir = sim_dir / dm_model
    ds = StreamSimDataset(
        sim_dir=model_dir,
        k_neighbors=k,
        normalizer=normalizer,
        max_stars=cfg["preprocessing"]["max_stars_per_sim"],
        augment=False,
        preload_ram=True,
    )

    # Subsample if requested
    rng = np.random.default_rng(rng_seed)
    total_valid = len(valid_flat_indices)
    if n_max is not None and total_valid > n_max:
        subset_rows = rng.choice(total_valid, n_max, replace=False)
        subset_rows_sorted = np.sort(subset_rows)
        theta_use = theta_all[subset_rows_sorted]
        flat_indices_use = [valid_flat_indices[i] for i in subset_rows_sorted]
    else:
        theta_use = theta_all
        flat_indices_use = valid_flat_indices
        subset_rows_sorted = np.arange(total_valid)

    # Compute embeddings for all sims (batch on GPU), then select aligned subset
    log.info("[%s] Computing embeddings for %d sims (batch=%d)...", dm_model, len(ds), batch_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    encoder.eval()
    # Cache normalizer stats on device for node feature normalization
    norm_mean_dev = normalizer.mean.to(device) if normalizer else None
    norm_std_dev = normalizer.std.to(device) if normalizer else None
    all_embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            build_knn_graph_batched(batch, k=k, normalizer=normalizer)
            # Normalize node features — MUST match training pipeline exactly
            if norm_mean_dev is not None:
                batch.x = (batch.x - norm_mean_dev) / norm_std_dev
            emb = encoder(batch)
            all_embs.append(emb.cpu())
    all_embs_t = torch.cat(all_embs, dim=0)  # [N_ds, 128]

    # Align: valid_flat_indices maps row→ds index
    idx_to_emb_row = {fi: row for row, fi in enumerate(range(len(ds)))}
    aligned_emb_rows = [idx_to_emb_row[fi] for fi in flat_indices_use if fi in idx_to_emb_row]

    if len(aligned_emb_rows) != len(theta_use):
        n_use = min(len(theta_use), len(aligned_emb_rows))
        log.warning("[%s] Alignment mismatch — trimming to %d", dm_model, n_use)
        theta_use = theta_use[:n_use]
        aligned_emb_rows = aligned_emb_rows[:n_use]

    embeddings_use = all_embs_t[aligned_emb_rows]

    # Drop any NaN rows (defensive)
    bad = torch.isnan(theta_use).any(dim=1) | torch.isnan(embeddings_use).any(dim=1)
    if bad.any():
        log.warning("[%s] Dropping %d NaN rows", dm_model, bad.sum().item())
        theta_use = theta_use[~bad]
        embeddings_use = embeddings_use[~bad]

    log.info("[%s] Final: theta %s, embeddings %s",
             dm_model, list(theta_use.shape), list(embeddings_use.shape))
    return theta_use, embeddings_use


# ---------------------------------------------------------------------------
# Stage 1: Sanity check
# ---------------------------------------------------------------------------

def run_sanity_check(
    dm_model: str,
    posterior,
    theta: torch.Tensor,
    embeddings: torch.Tensor,
    n_posterior_samples: int = 500,
    ci_level: float = 0.9,
    device: str = "cpu",
) -> dict:
    """Check that ground-truth theta falls inside CI at roughly the right rate.

    For a perfectly calibrated posterior with CI=0.9, expect ~90% coverage.
    Acceptance criterion: empirical coverage in [0.75, 1.00] (loose for small N).
    """
    import yaml  # noqa: PLC0415
    with open(ROOT / "config" / "dm_models.yaml") as f:
        param_names = [p["name"] for p in yaml.safe_load(f)["models"][dm_model]["inferred_parameters"]]

    n = len(theta)
    alpha = (1.0 - ci_level) / 2.0
    in_ci = np.zeros((n, len(param_names)), dtype=bool)

    log.info("[%s] Sanity check: %d sims × %d posterior samples ...", dm_model, n, n_posterior_samples)
    t0 = time.time()

    for i in range(n):
        x_obs = embeddings[i].to(device)  # move to same device as posterior
        with torch.no_grad():
            samples = posterior.sample(
                (n_posterior_samples,),
                x=x_obs,
                show_progress_bars=False,
            )
        theta_np = theta[i].cpu().numpy()
        samples_np = samples.cpu().numpy()
        lo = np.quantile(samples_np, alpha, axis=0)
        hi = np.quantile(samples_np, 1.0 - alpha, axis=0)
        in_ci[i] = (theta_np >= lo) & (theta_np <= hi)

    elapsed = time.time() - t0
    empirical_cov = in_ci.mean(axis=0)

    log.info("[%s] Sanity check complete in %.1fs", dm_model, elapsed)
    for j, pname in enumerate(param_names):
        cov = empirical_cov[j]
        ok = "PASS" if (0.75 <= cov <= 1.0) else "WARN"
        log.info("[%s]   %-25s  %s coverage: empirical=%.3f  (nominal=%.2f)",
                 dm_model, pname, ok, cov, ci_level)

    return {
        "dm_model": dm_model,
        "param_names": param_names,
        "empirical_coverage": empirical_cov.tolist(),
        "nominal_ci": ci_level,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Stage 2: Fast SBC
# ---------------------------------------------------------------------------

def run_sbc_precomputed(
    dm_model: str,
    posterior,
    theta: torch.Tensor,
    embeddings: torch.Tensor,
    n_posterior_samples: int = 500,
    device: str = "cpu",
) -> dict:
    """Fast SBC using pre-computed (theta, embedding) pairs.

    Computes the rank of each ground-truth theta among posterior samples.
    For a well-calibrated posterior, ranks should be uniform over [0, n_posterior_samples].

    Returns:
        dict with ranks [N, n_params], ks_pvalues [n_params], is_calibrated bool.
    """
    from scipy import stats  # noqa: PLC0415
    import yaml  # noqa: PLC0415
    with open(ROOT / "config" / "dm_models.yaml") as f:
        param_names = [p["name"] for p in yaml.safe_load(f)["models"][dm_model]["inferred_parameters"]]

    n = len(theta)
    log.info("[%s] SBC: %d trials × %d posterior samples ...", dm_model, n, n_posterior_samples)
    t0 = time.time()

    all_ranks = []
    for i in range(n):
        x_obs = embeddings[i].to(device)  # move to same device as posterior
        with torch.no_grad():
            samples = posterior.sample(
                (n_posterior_samples,),
                x=x_obs,
                show_progress_bars=False,
            )
        theta_np = theta[i].cpu().numpy()
        samples_np = samples.cpu().numpy()
        # Rank = number of posterior samples less than the true value
        ranks = np.sum(samples_np < theta_np[None, :], axis=0)
        all_ranks.append(ranks)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            log.info("[%s] SBC trial %d/%d  (%.0f/s, ETA %.0fs)", dm_model, i + 1, n, rate, eta)

    elapsed = time.time() - t0
    ranks_arr = np.array(all_ranks)  # [N, n_params]

    # K-S test against uniform over [0, n_posterior_samples]
    ks_pvalues = []
    for j in range(ranks_arr.shape[1]):
        _, p = stats.kstest(ranks_arr[:, j] / n_posterior_samples, "uniform")
        ks_pvalues.append(float(p))

    is_calibrated = all(p > 0.05 for p in ks_pvalues)
    for j, pname in enumerate(param_names):
        status = "PASS" if ks_pvalues[j] > 0.05 else "FAIL"
        log.info("[%s]   %-25s  K-S p=%.4f  [%s]", dm_model, pname, ks_pvalues[j], status)

    if is_calibrated:
        log.info("[%s] SBC PASSED — all K-S p > 0.05 (%.1fs)", dm_model, elapsed)
    else:
        log.warning("[%s] SBC FAILED — some K-S p < 0.05 (posterior miscalibrated)", dm_model)

    return {
        "dm_model": dm_model,
        "param_names": param_names,
        "ranks": ranks_arr,
        "ks_pvalues": np.array(ks_pvalues),
        "is_calibrated": is_calibrated,
        "n_trials": n,
        "n_posterior_samples": n_posterior_samples,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rank_histograms(results: list[dict], output_dir: Path) -> None:
    """Plot SBC rank histograms for all models, one figure per model."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    for res in results:
        dm = res["dm_model"]
        ranks = res["ranks"]
        pnames = res["param_names"]
        n_samp = res["n_posterior_samples"]
        n_params = len(pnames)

        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
        if n_params == 1:
            axes = [axes]
        fig.suptitle(f"SBC Rank Histograms — {dm}\n"
                     f"K-S p-values: {[f'{p:.3f}' for p in res['ks_pvalues']]}  "
                     f"({'CALIBRATED' if res['is_calibrated'] else 'MISCALIBRATED'})",
                     fontsize=11)
        for j, (ax, pname) in enumerate(zip(axes, pnames)):
            ax.hist(ranks[:, j], bins=20, density=True, alpha=0.7, color="steelblue", edgecolor="white")
            uniform_level = 1.0 / n_samp  # ~ 1/n_bins × n_samp in density, simplify
            ax.axhline(20 / (n_samp + 1), color="red", linestyle="--", linewidth=1.5, label="uniform")
            ax.set_title(f"{pname}\nK-S p={res['ks_pvalues'][j]:.3f}", fontsize=10)
            ax.set_xlabel("Rank")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        plt.tight_layout()
        out = output_dir / f"sbc_ranks_{dm}.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Rank histogram saved to %s", out)


def plot_coverage_summary(sanity_results: list[dict], output_dir: Path) -> None:
    """Bar chart of empirical vs nominal coverage per model × parameter."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, res in zip(axes, sanity_results):
        dm = res["dm_model"]
        pnames = res["param_names"]
        emp = res["empirical_coverage"]
        nominal = res["nominal_ci"]
        colors = ["green" if (0.75 <= v <= 1.0) else "red" for v in emp]
        bars = ax.bar(pnames, emp, color=colors, alpha=0.7, width=0.5)
        ax.axhline(nominal, color="black", linestyle="--", linewidth=1.5, label=f"nominal={nominal:.0%}")
        ax.axhline(0.75, color="orange", linestyle=":", linewidth=1, label="lower warn (0.75)")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{dm} — Empirical Coverage @ {nominal:.0%} CI\n(n={res['n']})")
        ax.set_ylabel("Coverage fraction")
        ax.legend(fontsize=8)
        for bar, v in zip(bars, emp):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = output_dir / "coverage_summary.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Coverage summary saved to %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    t_start = time.time()
    cfg_path = ROOT / "config" / "training.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Posterior test on device: %s", device)

    if device == "cuda":
        frac = cfg["training"].get("cuda_memory_fraction", 0.75)
        torch.cuda.set_per_process_memory_fraction(frac)

    checkpoint_dir = ROOT / cfg["paths"]["checkpoints"]
    output_dir = ROOT / "outputs" / "posterior_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shared components
    mean, std = load_normalizer(checkpoint_dir / "normalizer.npz")
    normalizer = FeatureNormalizer(mean, std)
    encoder = load_encoder(cfg, checkpoint_dir / "gnn_best.pt", device)

    dm_models = args.dm_models
    log.info("Testing DM models: %s", dm_models)

    sanity_results = []
    sbc_results = []

    for dm_model in dm_models:
        log.info("=" * 60)
        log.info("DM MODEL: %s", dm_model)
        log.info("=" * 60)
        try:
            posterior = load_posterior(checkpoint_dir, dm_model)

            # Compute embeddings and theta
            theta, embeddings = compute_test_embeddings_and_theta(
                dm_model=dm_model,
                cfg=cfg,
                encoder=encoder,
                normalizer=normalizer,
                device=device,
                n_max=args.n_max,
                batch_size=args.batch_size,
            )

            if args.stage in ("sanity", "both"):
                n_sanity = min(args.n_sanity, len(theta))
                idx = np.random.default_rng(999).choice(len(theta), n_sanity, replace=False)
                theta_s = theta[idx]
                emb_s = embeddings[idx]
                result = run_sanity_check(
                    dm_model=dm_model,
                    posterior=posterior,
                    theta=theta_s,
                    embeddings=emb_s,
                    n_posterior_samples=args.n_posterior_samples,
                    ci_level=0.9,
                    device=device,
                )
                sanity_results.append(result)

            if args.stage in ("sbc", "both"):
                n_sbc = min(args.n_sbc, len(theta))
                idx = np.random.default_rng(1234).choice(len(theta), n_sbc, replace=False)
                theta_s = theta[idx]
                emb_s = embeddings[idx]
                result = run_sbc_precomputed(
                    dm_model=dm_model,
                    posterior=posterior,
                    theta=theta_s,
                    embeddings=emb_s,
                    n_posterior_samples=args.n_posterior_samples,
                    device=device,
                )
                sbc_results.append(result)

        except Exception as e:
            log.error("[%s] FAILED: %s", dm_model, e, exc_info=True)
            log.error("[%s] Continuing ...", dm_model)

    # Save raw results
    np.save(str(output_dir / "sanity_results.npy"), sanity_results, allow_pickle=True)
    if sbc_results:
        np.save(str(output_dir / "sbc_results.npy"), sbc_results, allow_pickle=True)

    # Plots
    if sanity_results:
        plot_coverage_summary(sanity_results, output_dir)
    if sbc_results:
        plot_rank_histograms(sbc_results, output_dir)

    # Summary table
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    if sanity_results:
        log.info("COVERAGE (90%% CI, loose pass >= 0.75):")
        for res in sanity_results:
            for pname, cov in zip(res["param_names"], res["empirical_coverage"]):
                status = "PASS" if cov >= 0.75 else "WARN"
                log.info("  [%s] %-25s %.3f  [%s]", res["dm_model"], pname, cov, status)
    if sbc_results:
        log.info("SBC K-S p-values (pass > 0.05):")
        for res in sbc_results:
            for pname, p in zip(res["param_names"], res["ks_pvalues"]):
                status = "PASS" if p > 0.05 else "FAIL"
                log.info("  [%s] %-25s p=%.4f  [%s]", res["dm_model"], pname, p, status)

    elapsed = time.time() - t_start
    log.info("Total elapsed: %.1f min", elapsed / 60)
    log.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Posterior sanity test + fast SBC.")
    parser.add_argument(
        "--dm-models", nargs="+", default=DM_MODELS,
        choices=DM_MODELS, metavar="MODEL",
    )
    parser.add_argument(
        "--stage", choices=["sanity", "sbc", "both"], default="both",
        help="Which stages to run (default: both)",
    )
    parser.add_argument(
        "--n-sanity", type=int, default=30,
        help="Sims for sanity coverage check per model (default: 30)",
    )
    parser.add_argument(
        "--n-sbc", type=int, default=200,
        help="Sims for SBC rank statistics per model (default: 200)",
    )
    parser.add_argument(
        "--n-posterior-samples", type=int, default=500,
        help="Posterior samples per simulation (default: 500)",
    )
    parser.add_argument(
        "--n-max", type=int, default=None,
        help="Max total sims to load per model (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for GNN embedding (default: 64)",
    )
    args = parser.parse_args()
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    main(args)
