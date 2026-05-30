"""
Posterior sampling, credible interval computation, and evidence estimation.

Used by scripts/run_inference.py and scripts/combine_posteriors.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import yaml

# Disable pandas 2.2+ PyArrow string backend — it segfaults on Windows when
# constructing Index objects (same root cause as the sbi import crash).
try:
    pd.options.mode.string_storage = "python"
except AttributeError:
    pass  # older pandas versions don't have this option

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter name helpers
# ---------------------------------------------------------------------------

def get_param_names(
    dm_model: str,
    config_path: str = "config/dm_models.yaml",
    theta_mode: str = "native",
) -> list[str]:
    if theta_mode == "suppression":
        return ["log10_M_hm", "n_impacts"]
    if theta_mode != "native":
        raise ValueError(f"Unknown theta_mode={theta_mode!r}; expected 'native' or 'suppression'.")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["models"][dm_model]
    return [p["name"] for p in cfg["inferred_parameters"]]


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

def sample_posterior(
    posterior,
    observation_embedding: torch.Tensor,
    n_samples: int = 10000,
    dm_model: str = "CDM",
    config_path: str = "config/dm_models.yaml",
    theta_mode: str = "native",
) -> pd.DataFrame:
    """Draw samples from the trained NPE posterior.

    Args:
        posterior: sbi posterior object.
        observation_embedding: [128] embedding of the observed stream.
        n_samples: Number of posterior samples to draw.
        dm_model: DM model key for column naming.

    Returns:
        DataFrame with one column per parameter and n_samples rows.
    """
    param_names = get_param_names(dm_model, config_path, theta_mode=theta_mode)
    # Move embedding to the same device as the posterior (trained on CUDA, may be pickled that way)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        samples = posterior.sample(
            (n_samples,),
            x=observation_embedding.to(device),
            show_progress_bars=False,
        )
    samples_np = samples.cpu().numpy()

    # Post-processing: round n_impacts back to integer.
    # The NSF was trained on dequantized (continuous) n_impacts values; posterior
    # samples are continuous. Round to nearest non-negative integer for physical
    # interpretability. Clip to 0 to avoid negative counts from tail samples.
    if "n_impacts" in param_names:
        n_idx = param_names.index("n_impacts")
        samples_np[:, n_idx] = np.clip(np.round(samples_np[:, n_idx]), 0, None)

    # Force object-dtype column index — pandas 2.2 auto-selects PyArrow string
    # backend on Windows which segfaults (same class of issue as the sbi import
    # crash fixed by PYTHONNOUSERSITE=1).
    return pd.DataFrame(samples_np, columns=pd.Index(param_names, dtype=object))


# ---------------------------------------------------------------------------
# Credible intervals
# ---------------------------------------------------------------------------

def compute_credible_intervals(
    samples: pd.DataFrame,
    ci_level: float = 0.9,
) -> pd.DataFrame:
    """Compute lower/upper bounds of credible intervals for each parameter.

    Args:
        samples: DataFrame of posterior samples. Must have at least 2 rows.
        ci_level: Credible interval level (default 0.9 = 90%). Must be in (0, 1).

    Returns:
        DataFrame with index = parameter names, columns = [mean, median, lower, upper, std].

    Raises:
        ValueError: If ci_level is not in (0, 1) or samples is empty.
    """
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if len(samples) < 2:
        raise ValueError(f"Need at least 2 samples for CI computation, got {len(samples)}")
    alpha = (1.0 - ci_level) / 2.0
    results = {}
    for col in samples.columns:
        s = samples[col].values
        results[col] = {
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "lower": float(np.quantile(s, alpha)),
            "upper": float(np.quantile(s, 1.0 - alpha)),
            "std": float(np.std(s)),
        }
    df = pd.DataFrame(results).T
    # Ensure index uses object dtype (not PyArrow string) to avoid Windows segfault
    df.index = pd.Index(list(df.index), dtype=object)
    return df


# ---------------------------------------------------------------------------
# Log evidence (harmonic mean estimator)
# ---------------------------------------------------------------------------

def compute_log_evidence(
    posterior,
    observation_embedding: torch.Tensor,
    n_samples: int = 5000,
) -> float:
    """Estimate log marginal likelihood using the harmonic mean estimator.

    WARNING: The harmonic mean estimator is numerically unstable (can give
    infinite variance). Use this only for quick exploratory comparisons.
    For publication-quality Bayes factors, use nested sampling (dynesty).

    Returns:
        log_evidence (float). Subtract two log_evidences to get log Bayes factor.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs = observation_embedding.to(device)
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=obs, show_progress_bars=False)
        log_probs = posterior.log_prob(samples, x=obs)

    log_probs_np = log_probs.cpu().numpy()
    # Harmonic mean: Z_hat = N / sum(1/p(x|theta_i))
    # log(Z_hat) = log(N) - logsumexp(-log_p)
    # Numerically stable via shift-by-max trick
    neg_lp = -log_probs_np
    max_neg_lp = np.max(neg_lp)
    log_evidence = float(np.log(n_samples) - max_neg_lp -
                         np.log(np.sum(np.exp(neg_lp - max_neg_lp))))
    return log_evidence


# ---------------------------------------------------------------------------
# Posterior combination across streams
# ---------------------------------------------------------------------------

def combine_posteriors_product(
    sample_list: list[pd.DataFrame],
    param_names: list[str],
    n_grid: int = 100,
    param_ranges: dict | None = None,
) -> dict:
    """Combine independent per-stream posteriors via product (sum of log-posteriors).

    Valid when streams probe independent regions of the dark matter halo,
    so their perturbation observations are statistically independent.

    This uses a KDE approximation of each posterior on a shared parameter grid.
    For low-dimensional parameter spaces (<=3 params) this is tractable.

    Args:
        sample_list: One DataFrame per stream, columns = parameter names.
        param_names: Parameter names to combine over.
        n_grid: Grid resolution per parameter dimension.
        param_ranges: Optional dict of {param_name: (lo, hi)} overrides.

    Returns:
        dict with keys: grid_values, log_combined_posterior, param_names.
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    if len(param_names) > 3:
        log.warning("combine_posteriors_product: >3 parameters; using first 3 only for grid combination.")
        param_names = param_names[:3]

    # Determine ranges
    ranges = {}
    for name in param_names:
        all_vals = np.concatenate([df[name].values for df in sample_list])
        if param_ranges and name in param_ranges:
            ranges[name] = param_ranges[name]
        else:
            ranges[name] = (float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99)))

    # Build 1D KDE + product for each parameter (marginal combination)
    combined = {}
    for name in param_names:
        lo, hi = ranges[name]
        grid = np.linspace(lo, hi, n_grid)
        log_prod = np.zeros(n_grid)
        for df in sample_list:
            vals = df[name].values
            kde = gaussian_kde(vals)
            log_prod += np.log(kde(grid) + 1e-300)
        log_prod -= log_prod.max()  # normalise for numerical stability
        combined[name] = {"grid": grid, "log_posterior": log_prod}

    return {"combined": combined, "param_names": param_names}
