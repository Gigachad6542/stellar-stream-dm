"""
Mock Data Challenge: count-rate validation plus V2 artifact sanity checks.

This script is not the full end-to-end GNN+SBI blind challenge yet. It keeps the
existing count/rate inversion challenge and verifies that the selected GNN
checkpoint/normalizer can be loaded with the V2 runtime. The publication-grade
full mock challenge should be run after V2 GNN training and V2 SBI training.

Procedure:
    1. Generate 20 mock streams with known DM parameters:
       - 5 CDM  (no suppression, represented at low log10_M_hm)
       - 5 WDM  (M_hm in {6.5, 7.0, 7.5, 8.0, 8.5})
       - 5 FDM  (same M_hm values)
       - 5 SIDM (unsuppressed reference, like CDM for this mass-function test)
    2. Invert the count/rate model for each mock.
    3. Compare inferred M_hm to ground truth.
    4. Report: coverage at 68%/90%/95%, mean bias, RMSE.

This is intentionally count-only. It is not the publication-grade blind
GNN+SBI challenge.

Usage:
    python scripts/run_mock_challenge.py [--checkpoint checkpoints/gnn_v2_best.pt]
                                         [--n-posterior-samples 5000]
"""
from __future__ import annotations

import argparse
import logging
import site
import sys
from pathlib import Path

try:
    user_site = Path(site.getusersitepackages()).resolve()
    sys.path = [p for p in sys.path if not p or Path(p).resolve() != user_site]
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock stream generation
# ---------------------------------------------------------------------------

# Ground truth M_hm values for each mock (log10 M_sun)
MOCK_TRUTHS = {
    # CDM: no suppression — M_hm below the minimum subhalo mass we consider
    # (10^5 Msun), so the transfer function T(M, M_hm) ≈ 1 for all M in
    # [10^5, 10^9]. Use 4.5 as stand-in for "no suppression".
    "CDM_mock_1": {"dm_model": "CDM", "log10_M_hm": 4.5, "n_impacts_true": 5},
    "CDM_mock_2": {"dm_model": "CDM", "log10_M_hm": 4.5, "n_impacts_true": 4},
    "CDM_mock_3": {"dm_model": "CDM", "log10_M_hm": 4.5, "n_impacts_true": 7},
    "CDM_mock_4": {"dm_model": "CDM", "log10_M_hm": 4.5, "n_impacts_true": 3},
    "CDM_mock_5": {"dm_model": "CDM", "log10_M_hm": 4.5, "n_impacts_true": 6},
    # WDM: moderate suppression
    "WDM_mock_1": {"dm_model": "WDM", "log10_M_hm": 6.5, "n_impacts_true": 4},
    "WDM_mock_2": {"dm_model": "WDM", "log10_M_hm": 7.0, "n_impacts_true": 3},
    "WDM_mock_3": {"dm_model": "WDM", "log10_M_hm": 7.5, "n_impacts_true": 3},
    "WDM_mock_4": {"dm_model": "WDM", "log10_M_hm": 8.0, "n_impacts_true": 2},
    "WDM_mock_5": {"dm_model": "WDM", "log10_M_hm": 8.5, "n_impacts_true": 1},
    # FDM: similar suppression but different particle physics
    "FDM_mock_1": {"dm_model": "FDM", "log10_M_hm": 6.5, "n_impacts_true": 4},
    "FDM_mock_2": {"dm_model": "FDM", "log10_M_hm": 7.0, "n_impacts_true": 3},
    "FDM_mock_3": {"dm_model": "FDM", "log10_M_hm": 7.5, "n_impacts_true": 2},
    "FDM_mock_4": {"dm_model": "FDM", "log10_M_hm": 8.0, "n_impacts_true": 2},
    "FDM_mock_5": {"dm_model": "FDM", "log10_M_hm": 8.5, "n_impacts_true": 1},
    # SIDM: unsuppressed mass function in the current configuration
    "SIDM_mock_1": {"dm_model": "SIDM", "log10_M_hm": 4.5, "n_impacts_true": 5},
    "SIDM_mock_2": {"dm_model": "SIDM", "log10_M_hm": 4.5, "n_impacts_true": 4},
    "SIDM_mock_3": {"dm_model": "SIDM", "log10_M_hm": 4.5, "n_impacts_true": 7},
    "SIDM_mock_4": {"dm_model": "SIDM", "log10_M_hm": 4.5, "n_impacts_true": 3},
    "SIDM_mock_5": {"dm_model": "SIDM", "log10_M_hm": 4.5, "n_impacts_true": 6},
}


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------

def compute_coverage(
    true_values: np.ndarray,
    posterior_samples_list: list[np.ndarray],
    ci_levels: list[float] = None,
) -> dict:
    """Compute empirical coverage at given CI levels.

    For each mock stream, check whether the true parameter lies within
    the [alpha/2, 1 - alpha/2] quantiles of the posterior.

    Args:
        true_values: [n_mocks] true log10_M_hm values.
        posterior_samples_list: List of [n_samples] arrays, one per mock.
        ci_levels: CI levels to evaluate (default [0.68, 0.90, 0.95]).

    Returns:
        Dict with empirical coverage, bias, RMSE, and per-mock results.
    """
    if ci_levels is None:
        ci_levels = [0.68, 0.90, 0.95]

    n_mocks = len(true_values)
    coverage = {ci: 0 for ci in ci_levels}
    biases = []
    squared_errors = []
    per_mock = []

    for i in range(n_mocks):
        truth = true_values[i]
        samples = posterior_samples_list[i]
        median = np.median(samples)
        biases.append(median - truth)
        squared_errors.append((median - truth) ** 2)

        mock_result = {"truth": truth, "median": median, "bias": median - truth}

        for ci in ci_levels:
            alpha = (1.0 - ci) / 2.0
            lo = np.quantile(samples, alpha)
            hi = np.quantile(samples, 1.0 - alpha)
            in_ci = lo <= truth <= hi
            mock_result[f"in_CI_{int(ci*100)}"] = in_ci
            if in_ci:
                coverage[ci] += 1

        per_mock.append(mock_result)

    empirical_coverage = {ci: count / n_mocks for ci, count in coverage.items()}
    mean_bias = float(np.mean(biases))
    rmse = float(np.sqrt(np.mean(squared_errors)))

    return {
        "ci_levels": ci_levels,
        "empirical_coverage": empirical_coverage,
        "mean_bias": mean_bias,
        "rmse": rmse,
        "n_mocks": n_mocks,
        "per_mock": per_mock,
    }


# ---------------------------------------------------------------------------
# Main challenge runner
# ---------------------------------------------------------------------------

def run_mock_challenge(args: argparse.Namespace) -> None:
    """Execute the count-only mock challenge and V2 artifact sanity checks."""
    import yaml  # noqa: PLC0415
    import torch  # noqa: PLC0415

    cfg_path = ROOT / "config" / "training.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Mock challenge on device: %s", device)

    # --- Load GNN model ---
    from src.models.gnn import StreamGNNMultiTask, StreamGNNMultiTaskV2  # noqa: PLC0415
    from src.models.utils import load_normalizer  # noqa: PLC0415
    from src.data.dataset import FeatureNormalizer  # noqa: PLC0415

    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        log.error("Run training first: python scripts/train_v2.py --epochs 300")
        sys.exit(1)

    gcfg = cfg["model"]["gnn"]
    ms_cfg = cfg["graph"].get("multi_scale", {})

    if args.model_version == "v2":
        v2_cfg = cfg.get("training_v2", {})
        model = StreamGNNMultiTaskV2(
            n_reg_targets=v2_cfg.get("n_reg_targets", 2),
            predict_uncertainty=v2_cfg.get("predict_uncertainty", False),
            n_node_features=cfg["graph"]["n_node_features"],
            n_edge_features=cfg["graph"]["n_edge_features"],
            hidden_dim=gcfg["hidden_dim"],
            n_layers=gcfg["n_layers"],
            embedding_dim=gcfg["embedding_dim"],
            dropout=gcfg["dropout"],
            use_attention_readout=gcfg.get("use_attention_readout", False),
            use_multi_scale=ms_cfg.get("enabled", False),
            seg_embedding_dim=ms_cfg.get("seg_embedding_dim", 64),
        ).to(device)
    else:
        model = StreamGNNMultiTask(
            n_classes=3,
            n_reg_targets=cfg["model"].get("n_reg_targets", 2),
            n_node_features=cfg["graph"]["n_node_features"],
            n_edge_features=cfg["graph"]["n_edge_features"],
            hidden_dim=gcfg["hidden_dim"],
            n_layers=gcfg["n_layers"],
            embedding_dim=gcfg["embedding_dim"],
            dropout=gcfg["dropout"],
            use_attention_readout=gcfg.get("use_attention_readout", False),
            use_multi_scale=ms_cfg.get("enabled", False),
            seg_embedding_dim=ms_cfg.get("seg_embedding_dim", 64),
        ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Loaded GNN from %s (epoch %d)", ckpt_path, ckpt.get("epoch", -1))

    # Normalizer
    norm_path = ROOT / args.normalizer
    if norm_path.exists():
        mean, std = load_normalizer(norm_path)
        normalizer = FeatureNormalizer(mean, std)
    else:
        normalizer = None
        log.warning("No normalizer found — using unnormalized features")

    # --- Run challenge ---
    log.info("=" * 60)
    log.info("MOCK DATA CHALLENGE: %d mock streams", len(MOCK_TRUTHS))
    log.info("=" * 60)

    # For now, use simplified inference: the GNN regression head predicts
    # n_impacts directly. We convert to M_hm via the hierarchical rate model.
    from src.inference.hierarchical import expected_subhalo_rate  # noqa: PLC0415

    true_M_hm_values = []
    posterior_samples_list = []

    for mock_name, truth in MOCK_TRUTHS.items():
        true_M_hm = truth["log10_M_hm"]
        true_M_hm_values.append(true_M_hm)

        # Generate a simple posterior by inverting the rate model
        # (In the full pipeline, this would use the NPE posterior)
        n_obs = truth["n_impacts_true"]

        # Sample M_hm posterior via likelihood inversion (grid method)
        log_m_grid = np.linspace(4.0, 10.5, 650)
        log_likelihoods = np.zeros(len(log_m_grid))
        for j, lm in enumerate(log_m_grid):
            rate = expected_subhalo_rate(lm, stream_length_deg=60.0,
                                         stream_age_gyr=5.0, stream_distance_kpc=15.0)
            # Poisson log-likelihood
            from scipy.special import gammaln  # noqa: PLC0415
            log_likelihoods[j] = n_obs * np.log(max(rate, 1e-10)) - rate - gammaln(n_obs + 1)

        # Convert to posterior (flat prior in this range)
        log_likelihoods -= log_likelihoods.max()
        weights = np.exp(log_likelihoods)
        weights /= weights.sum()

        # Sample from posterior
        samples = np.random.choice(log_m_grid, size=args.n_posterior_samples,
                                    replace=True, p=weights)
        # Add small jitter to avoid discrete artifacts
        samples += np.random.uniform(-0.01, 0.01, len(samples))
        posterior_samples_list.append(samples)

        median = np.median(samples)
        log.info("  %s: truth=%.1f, inferred_median=%.2f, bias=%+.2f",
                 mock_name, true_M_hm, median, median - true_M_hm)

    # --- Compute coverage ---
    true_M_hm_values = np.array(true_M_hm_values)
    results = compute_coverage(true_M_hm_values, posterior_samples_list)

    log.info("")
    log.info("=" * 60)
    log.info("MOCK CHALLENGE RESULTS")
    log.info("=" * 60)
    log.info("  N mocks: %d", results["n_mocks"])
    log.info("  Mean bias: %+.3f dex", results["mean_bias"])
    log.info("  RMSE: %.3f dex", results["rmse"])
    log.info("")
    log.info("  Coverage:")
    for ci, emp in results["empirical_coverage"].items():
        status = "PASS" if abs(emp - ci) < 0.15 else "WARN"
        log.info("    %d%% CI: empirical=%.0f%% (expected=%d%%) [%s]",
                 int(ci * 100), emp * 100, int(ci * 100), status)

    # --- Save results ---
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_dir / "mock_challenge_results.npz"),
        true_M_hm=true_M_hm_values,
        coverage_68=results["empirical_coverage"][0.68],
        coverage_90=results["empirical_coverage"][0.90],
        coverage_95=results["empirical_coverage"][0.95],
        mean_bias=results["mean_bias"],
        rmse=results["rmse"],
    )
    log.info("\nResults saved to %s", out_dir / "mock_challenge_results.npz")

    # Pass/fail
    cov_90 = results["empirical_coverage"][0.90]
    if cov_90 >= 0.75:
        log.info("\nMock challenge: PASSED (90%% CI coverage = %.0f%%)", cov_90 * 100)
    else:
        log.warning("\nMock challenge: FAILED (90%% CI coverage = %.0f%% < 75%%)", cov_90 * 100)
        log.warning("Posteriors may be miscalibrated. Check SBI training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run count-only mock challenge sanity checks.")
    parser.add_argument("--checkpoint", default="checkpoints/gnn_v2_best.pt")
    parser.add_argument("--normalizer", default="checkpoints/normalizer_v2.npz")
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--n-posterior-samples", type=int, default=5000)
    parser.add_argument("--output-dir", default="outputs/mock_challenge")
    args = parser.parse_args()
    run_mock_challenge(args)
