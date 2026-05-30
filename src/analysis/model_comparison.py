"""
Dark matter model comparison via Bayes factors and posterior model probabilities.

Uses log Bayes factors computed from per-stream log evidences.
The harmonic mean estimator is used for speed; for publication, switch to
nested sampling (dynesty) via inference.nested_sampling=true in training.yaml.

Combines per-stream posteriors assuming statistical independence.

IMPORTANT SCIENTIFIC CAVEAT:
    The 4-model Bayes factors (CDM vs WDM vs FDM vs SIDM) are SECONDARY to the
    model-independent suppression scale constraint (see inference/suppression_scale.py).

    Strong log10_BF between WDM and FDM is unlikely given the suppression shape
    degeneracy at Gaia DR3 sensitivity. Interpret strong evidence as:
        - "suppressed vs unsuppressed" (WDM/FDM vs CDM/SIDM): VALID
        - "WDM vs FDM": PROBABLY OVER-INTERPRETED (degenerate suppression shapes)

    The paper should lead with M_hm constraints and report model comparison
    as supplementary context only.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

DM_MODELS = ["CDM", "WDM", "FDM", "SIDM"]


# ---------------------------------------------------------------------------
# Bayes factor computation
# ---------------------------------------------------------------------------

def compute_bayes_factors(
    log_evidences: dict[str, float],
    reference_model: str = "CDM",
) -> pd.DataFrame:
    """Compute log10 Bayes factors relative to a reference model.

    Args:
        log_evidences: Dict mapping model name to log evidence (base e).
            Must contain at least one model.
        reference_model: Model to use as denominator. Must be a key in log_evidences.

    Returns:
        DataFrame with columns: model, log_evidence, log10_BF_vs_{reference_model}.

    Raises:
        ValueError: If log_evidences is empty or reference_model not found.
    """
    if not log_evidences:
        raise ValueError("log_evidences must not be empty")
    if reference_model not in log_evidences:
        raise ValueError(f"Reference model '{reference_model}' not in log_evidences: {list(log_evidences.keys())}")
    log_e_ref = log_evidences.get(reference_model, 0.0)
    records = []
    for model, log_e in log_evidences.items():
        log10_bf = (log_e - log_e_ref) / np.log(10.0)
        records.append({
            "model": model,
            "log_evidence": log_e,
            f"log10_BF_vs_{reference_model}": log10_bf,
        })
    return pd.DataFrame(records)


def interpret_bayes_factor(log10_bf: float) -> str:
    """Jeffreys scale interpretation of a log10 Bayes factor."""
    if log10_bf > 2.0:
        return "Decisive evidence for preferred model"
    elif log10_bf > 1.0:
        return "Strong evidence"
    elif log10_bf > 0.5:
        return "Substantial evidence"
    elif log10_bf > 0.0:
        return "Weak evidence"
    elif log10_bf > -0.5:
        return "Weak evidence against"
    elif log10_bf > -1.0:
        return "Substantial evidence against"
    else:
        return "Strong evidence against"


# ---------------------------------------------------------------------------
# Posterior model probabilities
# ---------------------------------------------------------------------------

def compute_model_posterior(
    log_evidences: dict[str, float],
    prior_over_models: str = "uniform",
) -> pd.DataFrame:
    """Convert log evidences to posterior model probabilities.

    P(M_i | data) ∝ P(data | M_i) P(M_i)

    Args:
        log_evidences: Dict of model -> log evidence.
        prior_over_models: "uniform" or dict of model priors.

    Returns:
        DataFrame with model name and posterior probability.
    """
    models = list(log_evidences.keys())
    log_e = np.array([log_evidences[m] for m in models])

    if prior_over_models == "uniform":
        log_prior = np.zeros(len(models))
    else:
        log_prior = np.array([np.log(prior_over_models.get(m, 1.0 / len(models))) for m in models])

    log_unnorm = log_e + log_prior
    log_unnorm -= log_unnorm.max()  # numerical stability
    unnorm = np.exp(log_unnorm)
    probs = unnorm / unnorm.sum()

    return pd.DataFrame({"model": models, "posterior_probability": probs})


# ---------------------------------------------------------------------------
# Multi-stream combination
# ---------------------------------------------------------------------------

def combine_log_evidences_across_streams(
    per_stream_log_evidences: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Sum per-stream log evidences to get combined log evidence.

    Under the independence assumption (streams probe independent subhalo
    realizations), the joint log evidence is the sum of per-stream log evidences.

    Args:
        per_stream_log_evidences: Nested dict {stream_name: {model: log_evidence}}.

    Returns:
        Dict {model: combined_log_evidence}.
    """
    combined = {}
    for stream_name, evidences in per_stream_log_evidences.items():
        for model, log_e in evidences.items():
            combined[model] = combined.get(model, 0.0) + log_e
    return combined


def compute_suppressed_vs_unsuppressed(
    log_evidences: dict[str, float],
) -> dict:
    """Binary comparison: suppressed (WDM+FDM) vs unsuppressed (CDM+SIDM).

    This is the scientifically defensible comparison. Instead of asking
    "which specific model?" (degenerate), we ask "is the subhalo mass
    function suppressed at all?" (distinguishable).

    Combines evidence for WDM/FDM against CDM/SIDM using the posterior odds
    ratio, marginalizing within each composite hypothesis. SIDM is grouped
    with CDM here because the current SIDM configuration keeps the same
    large-scale subhalo mass function.

    Args:
        log_evidences: Dict of model -> log evidence. Must include at least
            one unsuppressed reference ("CDM" or "SIDM") and one suppressed
            model ("WDM" or "FDM").

    Returns:
        Dict with:
            log10_BF_suppressed_vs_unsuppressed: log10 Bayes factor for any suppression
            p_suppressed: posterior probability of suppression (uniform prior)
            interpretation: Jeffreys scale string
    """
    unsuppressed_models = [m for m in ("CDM", "SIDM") if m in log_evidences]
    suppressed_models = [m for m in ("WDM", "FDM") if m in log_evidences]
    if not unsuppressed_models:
        raise ValueError("Need at least one unsuppressed reference model (CDM or SIDM)")
    if not suppressed_models:
        raise ValueError("Need at least one suppressed model (WDM or FDM)")

    # p(data | H) = (1/K) sum_k p(data | M_k) within each composite hypothesis.
    log_e_sup = np.array([log_evidences[m] for m in suppressed_models])
    max_log_e = np.max(log_e_sup)
    log_marginal_sup = max_log_e + np.log(np.sum(np.exp(log_e_sup - max_log_e))) - np.log(len(suppressed_models))

    log_e_unsup = np.array([log_evidences[m] for m in unsuppressed_models])
    max_log_e_unsup = np.max(log_e_unsup)
    log_marginal_unsup = (
        max_log_e_unsup
        + np.log(np.sum(np.exp(log_e_unsup - max_log_e_unsup)))
        - np.log(len(unsuppressed_models))
    )

    log10_bf = (log_marginal_sup - log_marginal_unsup) / np.log(10.0)

    # Posterior probability (equal composite prior: P(sup) = P(unsup) = 0.5)
    log_odds = log_marginal_sup - log_marginal_unsup
    p_suppressed = 1.0 / (1.0 + np.exp(-log_odds))

    return {
        "log10_BF_suppressed_vs_unsuppressed": float(log10_bf),
        "log10_BF_suppressed_vs_CDM": float(log10_bf),  # backward-compatible alias
        "p_suppressed": float(p_suppressed),
        "interpretation": interpret_bayes_factor(float(log10_bf)),
        "n_suppressed_models": len(suppressed_models),
        "n_unsuppressed_models": len(unsuppressed_models),
    }


def model_comparison_report(
    per_stream_log_evidences: dict[str, dict[str, float]],
    config_path: str = "config/dm_models.yaml",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Full model comparison report.

    Combines per-stream evidences, computes Bayes factors and model posteriors,
    and returns a summary DataFrame. Now also includes the binary suppressed vs
    unsuppressed comparison as the primary scientific result.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    prior_over_models = cfg.get("model_comparison", {}).get("prior_over_models", "uniform")
    bf_threshold = cfg.get("model_comparison", {}).get("bayes_factor_strong_threshold", 5.0)

    combined = combine_log_evidences_across_streams(per_stream_log_evidences)
    bf_df = compute_bayes_factors(combined)
    model_post = compute_model_posterior(combined, prior_over_models)

    report = bf_df.merge(model_post, on="model")
    report["interpretation"] = report["log10_BF_vs_CDM"].apply(interpret_bayes_factor)
    report = report.sort_values("log10_BF_vs_CDM", ascending=False)

    # Primary result: binary suppressed vs unsuppressed
    binary = compute_suppressed_vs_unsuppressed(combined)
    log.info(
        "PRIMARY RESULT: Suppressed vs unsuppressed: log10(BF)=%.2f, P(suppressed)=%.3f [%s]",
        binary["log10_BF_suppressed_vs_unsuppressed"],
        binary["p_suppressed"],
        binary["interpretation"],
    )
    log.info("Per-model Bayes factors (secondary):\n%s", report.to_string())

    if output_path:
        report.to_csv(output_path, index=False)

    return report
