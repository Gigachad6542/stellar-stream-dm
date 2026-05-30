"""
Simulation-based calibration (SBC) and coverage testing.

SBC (Talts+2018): for a well-calibrated posterior, the rank of the true parameter
among posterior samples should be uniformly distributed.

Coverage testing: empirical coverage at multiple CI levels must match nominal coverage.

Both must pass BEFORE applying the pipeline to real data.
Failure criterion: K-S test p < 0.05 OR empirical 90% coverage outside [87%, 93%].

References:
    Talts+2018: arxiv:1804.06788
    Cook+2006: JASA
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import torch
from scipy import stats

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SBC — rank statistics
# ---------------------------------------------------------------------------

def compute_rank(
    theta_true: torch.Tensor,
    posterior_samples: torch.Tensor,
) -> np.ndarray:
    """Compute the rank of theta_true among posterior_samples for each parameter.

    Args:
        theta_true: [n_params] true parameter vector.
        posterior_samples: [n_samples, n_params] posterior samples.

    Returns:
        ranks: [n_params] integer ranks (0 to n_samples).
    """
    true_np = theta_true.cpu().numpy()
    samples_np = posterior_samples.cpu().numpy()
    ranks = np.sum(samples_np < true_np[None, :], axis=0)
    return ranks


def run_sbc(
    posterior,
    simulator_fn: Callable,
    prior,
    gnn_encoder: torch.nn.Module,
    n_trials: int = 500,
    n_posterior_samples: int = 500,
    device: str = "cuda",
    normalizer=None,
) -> dict:
    """Run SBC: draw from prior, simulate, infer posterior, compute ranks.

    Args:
        posterior: Trained sbi posterior.
        simulator_fn: Callable(theta) -> Data (stream graph).
        prior: sbi prior.
        gnn_encoder: Trained GNN encoder.
        n_trials: Number of SBC trials.
        n_posterior_samples: Posterior samples per trial.
        normalizer: FeatureNormalizer for node features (must match training).

    Returns:
        dict with keys:
            ranks: [n_trials, n_params] rank array.
            ks_pvalues: [n_params] K-S test p-values against uniform.
            is_calibrated: bool (all p-values > 0.05).
    """
    gnn_encoder.eval()
    all_ranks = []
    # Cache normalizer stats on device
    norm_mean_dev = normalizer.mean.to(device) if normalizer else None
    norm_std_dev = normalizer.std.to(device) if normalizer else None

    log.info("Running SBC with %d trials...", n_trials)
    for i in range(n_trials):
        # Draw true parameter
        theta_true = prior.sample((1,)).squeeze(0)

        # Simulate observation
        data = simulator_fn(theta_true)
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)

        # Embed (build k-NN graph first — graph construction is deferred in StreamSimDataset)
        with torch.no_grad():
            data = data.to(device)
            if not hasattr(data, "edge_index") or data.edge_index is None:
                from src.data.dataset import build_knn_graph_batched  # noqa: PLC0415
                build_knn_graph_batched(data, k=16, normalizer=normalizer)
            # Normalize node features — MUST match training pipeline
            if norm_mean_dev is not None:
                data.x = (data.x - norm_mean_dev) / norm_std_dev
            x_obs = gnn_encoder(data).squeeze(0).cpu()

        # Sample posterior
        with torch.no_grad():
            samples = posterior.sample(
                (n_posterior_samples,),
                x=x_obs,
                show_progress_bars=False,
            )

        ranks = compute_rank(theta_true, samples)
        all_ranks.append(ranks)

        if (i + 1) % 50 == 0:
            log.info("SBC trial %d/%d", i + 1, n_trials)

    ranks_arr = np.array(all_ranks)  # [n_trials, n_params]

    # K-S test against uniform distribution over [0, n_posterior_samples]
    ks_pvalues = []
    for j in range(ranks_arr.shape[1]):
        _, p = stats.kstest(ranks_arr[:, j] / n_posterior_samples, "uniform")
        ks_pvalues.append(p)

    is_calibrated = all(p > 0.05 for p in ks_pvalues)
    if not is_calibrated:
        log.warning(
            "SBC FAILED: K-S p-values = %s (some < 0.05); posterior is miscalibrated.",
            [f"{p:.3f}" for p in ks_pvalues],
        )
    else:
        log.info("SBC PASSED: all K-S p-values > 0.05: %s", [f"{p:.3f}" for p in ks_pvalues])

    return {
        "ranks": ranks_arr,
        "ks_pvalues": np.array(ks_pvalues),
        "is_calibrated": is_calibrated,
        "n_trials": n_trials,
        "n_posterior_samples": n_posterior_samples,
    }


# ---------------------------------------------------------------------------
# Coverage testing
# ---------------------------------------------------------------------------

def compute_coverage(
    posterior,
    simulator_fn: Callable,
    prior,
    gnn_encoder: torch.nn.Module,
    ci_levels: list[float] | None = None,
    n_trials: int = 500,
    n_posterior_samples: int = 1000,
    device: str = "cuda",
    normalizer=None,
) -> dict:
    """Compute empirical coverage at multiple CI levels.

    For a well-calibrated posterior, empirical coverage should equal nominal CI level.

    Args:
        ci_levels: CI levels to test (default [0.68, 0.90, 0.95]).
        n_trials: Number of test simulations.
        normalizer: FeatureNormalizer for node features (must match training).

    Returns:
        dict with keys: ci_levels, empirical_coverage [n_levels, n_params], is_covered.
    """
    if ci_levels is None:
        ci_levels = [0.68, 0.90, 0.95]

    gnn_encoder.eval()
    # Cache normalizer stats on device
    norm_mean_dev = normalizer.mean.to(device) if normalizer else None
    norm_std_dev = normalizer.std.to(device) if normalizer else None
    # Determine n_params from a trial sample
    n_params = prior.sample((1,)).shape[-1]
    in_ci = np.zeros((n_trials, len(ci_levels), n_params), dtype=bool)

    for i in range(n_trials):
        theta_true = prior.sample((1,)).squeeze(0)
        data = simulator_fn(theta_true)
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)

        with torch.no_grad():
            data_c = data.to(device)
            if not hasattr(data_c, "edge_index") or data_c.edge_index is None:
                from src.data.dataset import build_knn_graph_batched  # noqa: PLC0415
                build_knn_graph_batched(data_c, k=16, normalizer=normalizer)
            # Normalize node features — MUST match training pipeline
            if norm_mean_dev is not None:
                data_c.x = (data_c.x - norm_mean_dev) / norm_std_dev
            x_obs = gnn_encoder(data_c).squeeze(0).cpu()
            samples = posterior.sample((n_posterior_samples,), x=x_obs, show_progress_bars=False)

        theta_np = theta_true.cpu().numpy()
        samples_np = samples.cpu().numpy()

        for li, ci in enumerate(ci_levels):
            alpha = (1.0 - ci) / 2.0
            lo = np.quantile(samples_np, alpha, axis=0)
            hi = np.quantile(samples_np, 1.0 - alpha, axis=0)
            within = (theta_np >= lo) & (theta_np <= hi)  # [n_params] bool
            in_ci[i, li, :] = within  # per-parameter coverage

    empirical_coverage = in_ci.mean(axis=0)  # [n_levels, n_params]

    # Pass: every parameter's empirical coverage is within ±3% of nominal
    is_covered = []
    for li, ci in enumerate(ci_levels):
        for pi in range(n_params):
            emp = float(empirical_coverage[li, pi])
            passed = abs(emp - ci) < 0.03
            is_covered.append(passed)
            status = "PASS" if passed else "FAIL"
            log.info("Coverage %s: param=%d, nominal=%.2f, empirical=%.3f [%s]",
                     status, pi, ci, emp, status)

    return {
        "ci_levels": ci_levels,
        "empirical_coverage": empirical_coverage.tolist(),
        "is_covered": is_covered,
        "all_pass": all(is_covered),
    }


# ---------------------------------------------------------------------------
# Fast SBC on pre-computed embeddings (no simulator required)
# ---------------------------------------------------------------------------

def run_sbc_precomputed(
    posterior,
    theta: "torch.Tensor",
    embeddings: "torch.Tensor",
    n_posterior_samples: int = 500,
) -> dict:
    """Run SBC using pre-computed (theta, embedding) pairs — no simulator needed.

    For a well-calibrated posterior, the rank of the true theta among posterior
    samples should be uniformly distributed over [0, n_posterior_samples].

    Args:
        posterior: Trained sbi posterior object.
        theta: [N, n_params] ground-truth parameters.
        embeddings: [N, 128] pre-computed GNN embeddings.
        n_posterior_samples: Samples drawn per trial.

    Returns:
        dict with keys: ranks [N, n_params], ks_pvalues [n_params], is_calibrated.
    """
    n = len(theta)
    all_ranks = []

    # Detect posterior device so we can move observations to it.
    # sbi posteriors keep the density estimator on the training device (usually CUDA).
    try:
        _net = posterior.posterior_estimator if hasattr(posterior, "posterior_estimator") \
               else posterior.net
        _device = next(_net.parameters()).device
    except (StopIteration, AttributeError):
        _device = torch.device("cpu")

    log.info("Fast SBC: %d trials × %d posterior samples (device=%s) ...",
             n, n_posterior_samples, _device)
    for i in range(n):
        x_obs = embeddings[i].to(_device)
        with torch.no_grad():
            samples = posterior.sample(
                (n_posterior_samples,),
                x=x_obs,
                show_progress_bars=False,
            )
        theta_np = theta[i].cpu().numpy()
        samples_np = samples.cpu().numpy()
        ranks = np.sum(samples_np < theta_np[None, :], axis=0)
        all_ranks.append(ranks)

        if (i + 1) % 50 == 0:
            log.info("  SBC trial %d/%d", i + 1, n)

    ranks_arr = np.array(all_ranks)  # [N, n_params]

    ks_pvalues = []
    for j in range(ranks_arr.shape[1]):
        _, p = stats.kstest(ranks_arr[:, j] / n_posterior_samples, "uniform")
        ks_pvalues.append(float(p))

    is_calibrated = all(p > 0.05 for p in ks_pvalues)
    if not is_calibrated:
        log.warning("SBC FAILED: K-S p-values = %s", [f"{p:.3f}" for p in ks_pvalues])
    else:
        log.info("SBC PASSED: all K-S p > 0.05: %s", [f"{p:.3f}" for p in ks_pvalues])

    return {
        "ranks": ranks_arr,
        "ks_pvalues": np.array(ks_pvalues),
        "is_calibrated": is_calibrated,
        "n_trials": n,
        "n_posterior_samples": n_posterior_samples,
    }


# ---------------------------------------------------------------------------
# Rank histogram plotting utility
# ---------------------------------------------------------------------------

def plot_rank_histograms(
    ranks: np.ndarray,
    param_names: list[str],
    n_posterior_samples: int,
    output_path: str | None = None,
) -> None:
    """Plot SBC rank histograms. Uniform = calibrated; U-shape = overconfident."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n_params = ranks.shape[1]
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 3))
    if n_params == 1:
        axes = [axes]

    for j, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(ranks[:, j], bins=20, density=True, alpha=0.7, color="steelblue")
        ax.axhline(1.0 / (n_posterior_samples + 1) * 20, color="red", linestyle="--", label="uniform")
        ax.set_title(name)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.close(fig)
