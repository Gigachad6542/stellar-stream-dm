"""
Posterior predictive checks and model misspecification diagnostics.

Complements src/inference/calibration.py (SBC and coverage) with:
    1. Posterior predictive checks (PPC): simulate under the posterior, compare
       to observed data via summary statistics.
    2. Model misspecification detection: detect when the posterior is trained on
       simulations that don't match real data (the Achilles heel of SBI).
    3. Expected coverage experiments for quantifying epistemic uncertainty.

Scientific context:
    SBI (neural posterior estimation) is only valid when the simulator
    faithfully represents reality. If the simulation model is misspecified
    (e.g., missing baryonic physics, wrong stream model), posteriors can be
    confidently wrong. Cranmer+2020 (Sec 5.3) and Hermans+2022 flag this as
    the primary failure mode.

    These diagnostics don't fix misspecification, but they detect it before
    publishing nonsense constraints. A paper must report PPC p-values and
    discuss any failures.

References:
    - Cranmer+2020: "The frontier of simulation-based inference"
    - Hermans+2022: "A Trust Crisis In Simulation-Based Inference"
    - Talts+2018: SBC (implemented in calibration.py)
    - Gelman+2013: BDA3, Chapter 6 (posterior predictive checks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary statistics for posterior predictive checks
# ---------------------------------------------------------------------------

def _default_summary_statistics(phi1: np.ndarray) -> dict[str, float]:
    """Compute default summary statistics from a stream's phi1 distribution.

    These are stream-level observables that the posterior should be able to
    reproduce if the model is well-specified.
    """
    from src.analysis.gap_catalog import compute_density_profile, detect_gaps  # noqa: PLC0415
    from src.analysis.power_spectrum_baseline import power_spectrum_summary_vector  # noqa: PLC0415

    stats = {}

    # Basic moments
    stats["n_stars"] = len(phi1)
    stats["phi1_mean"] = float(np.mean(phi1))
    stats["phi1_std"] = float(np.std(phi1))
    stats["phi1_skew"] = float(_safe_skew(phi1))
    stats["phi1_kurtosis"] = float(_safe_kurtosis(phi1))

    # Gap statistics
    gaps = detect_gaps(phi1, gap_sigma=2.0)
    stats["n_gaps_2sigma"] = len(gaps)
    gaps_3 = [g for g in gaps if g.significance >= 3.0]
    stats["n_gaps_3sigma"] = len(gaps_3)
    if gaps:
        stats["max_gap_significance"] = max(g.significance for g in gaps)
        stats["max_gap_depth"] = max(g.depth for g in gaps)
        stats["mean_gap_width"] = float(np.mean([g.phi1_width for g in gaps]))
    else:
        stats["max_gap_significance"] = 0.0
        stats["max_gap_depth"] = 0.0
        stats["mean_gap_width"] = 0.0

    # Power spectrum (first 5 bandpowers as individual statistics)
    bp = power_spectrum_summary_vector(phi1, n_bandpowers=5)
    for i, val in enumerate(bp):
        stats[f"bandpower_{i}"] = float(val)

    return stats


def _safe_skew(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    m3 = np.mean((arr - np.mean(arr)) ** 3)
    s3 = np.std(arr) ** 3
    return m3 / s3 if s3 > 1e-10 else 0.0


def _safe_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    m4 = np.mean((arr - np.mean(arr)) ** 4)
    s4 = np.std(arr) ** 4
    return (m4 / s4 - 3.0) if s4 > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Posterior Predictive Checks (PPC)
# ---------------------------------------------------------------------------

@dataclass
class PPCResult:
    """Result of a posterior predictive check."""
    statistic_name: str
    observed_value: float
    simulated_values: np.ndarray   # values from posterior predictive simulations
    p_value: float                  # two-sided tail probability
    z_score: float                  # (observed - sim_mean) / sim_std
    is_consistent: bool             # p_value > 0.05


def posterior_predictive_check(
    observed_phi1: np.ndarray,
    posterior_samples: np.ndarray,
    simulator_fn: Callable[[np.ndarray], np.ndarray],
    n_ppc_sims: int = 200,
    summary_fn: Callable[[np.ndarray], dict[str, float]] | None = None,
    significance_level: float = 0.05,
) -> list[PPCResult]:
    """Run posterior predictive checks for a single stream.

    For each posterior sample theta_i:
        1. Simulate a synthetic stream: phi1_sim ~ p(data | theta_i)
        2. Compute summary statistics T(phi1_sim)
    Then compare T(observed) to the distribution of T(simulated).

    If the observed value lies in the tails of the predictive distribution,
    the model is likely misspecified (it can't reproduce the data even at
    its best-fit parameters).

    Args:
        observed_phi1: Observed star phi1 positions.
        posterior_samples: [n_samples, n_params] posterior samples for this stream.
        simulator_fn: Maps theta [n_params] -> phi1 array [n_stars_sim].
        n_ppc_sims: Number of predictive simulations to run.
        summary_fn: Custom summary function. Default uses gap + power spectrum stats.
        significance_level: p-value threshold for flagging failures.

    Returns:
        List of PPCResult, one per summary statistic.
    """
    if summary_fn is None:
        summary_fn = _default_summary_statistics

    # Observed statistics
    obs_stats = summary_fn(observed_phi1)

    # Predictive simulations (subsample posterior if we have more than needed)
    n_avail = len(posterior_samples)
    if n_avail > n_ppc_sims:
        idx = np.random.choice(n_avail, n_ppc_sims, replace=False)
    else:
        idx = np.arange(n_avail)

    sim_stats_list = []
    for i in idx:
        theta = posterior_samples[i]
        try:
            sim_phi1 = simulator_fn(theta)
            sim_stats = summary_fn(sim_phi1)
            sim_stats_list.append(sim_stats)
        except Exception as e:
            log.debug("PPC simulation %d failed: %s", i, e)
            continue

    if len(sim_stats_list) < 10:
        log.warning("PPC: only %d/%d simulations succeeded; results unreliable",
                    len(sim_stats_list), n_ppc_sims)
        return []

    # Compare each statistic
    results = []
    for stat_name, obs_val in obs_stats.items():
        sim_vals = np.array([s[stat_name] for s in sim_stats_list if stat_name in s])
        if len(sim_vals) < 10:
            continue

        # Two-sided p-value: fraction of sims more extreme than observed
        n_above = np.sum(sim_vals >= obs_val)
        n_below = np.sum(sim_vals <= obs_val)
        p_value = 2.0 * min(n_above, n_below) / len(sim_vals)
        p_value = min(p_value, 1.0)

        # Z-score
        sim_std = np.std(sim_vals)
        z_score = (obs_val - np.mean(sim_vals)) / sim_std if sim_std > 1e-10 else 0.0

        results.append(PPCResult(
            statistic_name=stat_name,
            observed_value=obs_val,
            simulated_values=sim_vals,
            p_value=p_value,
            z_score=z_score,
            is_consistent=p_value > significance_level,
        ))

    n_fail = sum(1 for r in results if not r.is_consistent)
    if n_fail > 0:
        log.warning("PPC: %d/%d statistics failed (p < %.2f). Model may be misspecified.",
                    n_fail, len(results), significance_level)
    else:
        log.info("PPC: all %d statistics consistent with posterior predictive distribution.", len(results))

    return results


# ---------------------------------------------------------------------------
# Model misspecification detection
# ---------------------------------------------------------------------------

@dataclass
class MisspecificationDiagnostic:
    """Summary of model misspecification evidence."""
    is_misspecified: bool
    evidence_level: str      # "none", "mild", "strong"
    ppc_failures: list[str]  # names of failed PPC statistics
    coverage_drop: float     # empirical_coverage - nominal_coverage (negative = overconfident)
    ood_score: float         # out-of-distribution score for the observation embedding
    details: dict = field(default_factory=dict)


def detect_misspecification(
    observed_embedding: np.ndarray,
    training_embeddings: np.ndarray,
    ppc_results: list[PPCResult] | None = None,
    coverage_empirical: float | None = None,
    coverage_nominal: float = 0.90,
    ood_percentile_threshold: float = 95.0,
    embedding_std: np.ndarray | None = None,
) -> MisspecificationDiagnostic:
    """Detect whether the trained model is misspecified for a given observation.

    Three independent signals of misspecification:
        1. PPC failures: observed data statistics lie in the tails of the
           posterior predictive distribution (model can't reproduce the data).
        2. Coverage drop: empirical coverage << nominal coverage means the
           posterior is overconfident (confident but wrong).
        3. OOD detection: observed embedding lies far from the training
           distribution in embedding space (observation looks unlike anything
           the simulator can produce).

    Any one signal is concerning; two or more together is strong evidence.

    Args:
        observed_embedding: [128] embedding of the observed stream.
        training_embeddings: [N, 128] embeddings from training simulations.
        ppc_results: Results from posterior_predictive_check().
        coverage_empirical: Empirical coverage at coverage_nominal CI level.
        coverage_nominal: Nominal CI level (default 0.90).
        ood_percentile_threshold: Percentile of training distances above which
            the observation is flagged as OOD.
        embedding_std: [128] per-dimension MC Dropout std (optional). If provided,
            the OOD score is augmented with an uncertainty penalty — high std
            signals the GNN is uncertain about this input.

    Returns:
        MisspecificationDiagnostic with evidence level and details.
    """
    signals = []
    details = {}

    # ── Signal 1: PPC failures ──────────────────────────────────────────────
    ppc_failures = []
    if ppc_results is not None:
        ppc_failures = [r.statistic_name for r in ppc_results if not r.is_consistent]
        frac_failed = len(ppc_failures) / max(len(ppc_results), 1)
        details["ppc_fraction_failed"] = frac_failed
        if frac_failed > 0.3:
            signals.append("ppc_strong")
        elif frac_failed > 0.1:
            signals.append("ppc_mild")

    # ── Signal 2: Coverage drop ─────────────────────────────────────────────
    coverage_drop = 0.0
    if coverage_empirical is not None:
        coverage_drop = coverage_empirical - coverage_nominal
        details["coverage_drop"] = coverage_drop
        if coverage_drop < -0.10:  # >10% under-coverage = strong signal
            signals.append("coverage_strong")
        elif coverage_drop < -0.05:
            signals.append("coverage_mild")

    # ── Signal 3: OOD embedding score (enhanced with MC Dropout uncertainty) ─
    ood_score = _compute_ood_score(
        observed_embedding, training_embeddings, embedding_std=embedding_std,
    )
    details["ood_score"] = ood_score
    if embedding_std is not None:
        details["mean_embedding_std"] = float(np.mean(embedding_std))

    # Threshold: if observed is farther from training mean than X% of training points
    train_scores = np.array([
        _compute_ood_score(training_embeddings[i], training_embeddings)
        for i in np.random.choice(len(training_embeddings), min(200, len(training_embeddings)), replace=False)
    ])
    threshold = np.percentile(train_scores, ood_percentile_threshold)
    details["ood_threshold"] = threshold

    if ood_score > threshold:
        signals.append("ood")
        details["ood_percentile"] = float(
            np.mean(train_scores < ood_score) * 100
        )

    # ── Combine signals ─────────────────────────────────────────────────────
    strong_signals = [s for s in signals if "strong" in s]
    mild_signals = [s for s in signals if "mild" in s]

    if len(strong_signals) >= 1 or len(signals) >= 2:
        evidence_level = "strong"
        is_misspecified = True
    elif len(signals) >= 1:
        evidence_level = "mild"
        is_misspecified = False  # warning but not conclusive
    else:
        evidence_level = "none"
        is_misspecified = False

    if is_misspecified:
        log.warning(
            "Model misspecification DETECTED (level=%s). Signals: %s. "
            "Posterior constraints may be unreliable.",
            evidence_level, signals,
        )
    elif evidence_level == "mild":
        log.warning(
            "Mild misspecification signals: %s. Results should be interpreted with caution.",
            signals,
        )

    return MisspecificationDiagnostic(
        is_misspecified=is_misspecified,
        evidence_level=evidence_level,
        ppc_failures=ppc_failures,
        coverage_drop=coverage_drop,
        ood_score=ood_score,
        details=details,
    )


def _compute_ood_score(
    embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    embedding_std: np.ndarray | None = None,
    uncertainty_weight: float = 0.5,
) -> float:
    """Compute out-of-distribution score using k-NN distance + optional MC uncertainty.

    Base score: mean L2 distance to the k nearest neighbors in embedding space.
    More robust than raw Mahalanobis for high-dimensional embeddings.

    If embedding_std is provided (from MC Dropout), a penalty proportional to
    the mean embedding uncertainty is added. High uncertainty means the GNN is
    unsure about this input — a strong OOD signal independent of distance.

    Args:
        embedding: [D] embedding vector for the observation.
        reference_embeddings: [N, D] reference (training) embeddings.
        embedding_std: [D] per-dimension std from MC Dropout (optional).
        uncertainty_weight: Weight for the uncertainty penalty (default 0.5).
            The penalty is: uncertainty_weight * mean(embedding_std) / median_training_knn_dist.
            This normalizes the uncertainty contribution relative to typical distances.

    Returns:
        OOD score (higher = more out-of-distribution).
    """
    k = min(10, len(reference_embeddings) - 1)
    dists = np.linalg.norm(reference_embeddings - embedding[None, :], axis=1)
    # k-NN distance (mean of k nearest, excluding self)
    sorted_dists = np.sort(dists)
    # Skip index 0 if it's zero (self-match)
    start = 1 if sorted_dists[0] < 1e-8 else 0
    knn_dists = sorted_dists[start:start + k]
    base_score = float(np.mean(knn_dists))

    if embedding_std is not None:
        # Normalize uncertainty by a reference scale (median knn distance) to make
        # the uncertainty penalty comparable across embedding spaces
        ref_scale = float(np.median(knn_dists)) if np.median(knn_dists) > 1e-8 else 1.0
        uncertainty_penalty = uncertainty_weight * float(np.mean(embedding_std)) / ref_scale
        return base_score * (1.0 + uncertainty_penalty)

    return base_score


# ---------------------------------------------------------------------------
# Expected coverage experiment (ECE)
# ---------------------------------------------------------------------------

def expected_coverage_experiment(
    posterior,
    theta_test: "torch.Tensor",
    embeddings_test: "torch.Tensor",
    ci_levels: list[float] | None = None,
    n_posterior_samples: int = 1000,
) -> dict:
    """Compute empirical coverage on a held-out test set.

    This is a fast alternative to the full coverage test in calibration.py:
    instead of re-simulating, it uses pre-computed test embeddings.

    Args:
        posterior: Trained sbi posterior.
        theta_test: [N_test, n_params] true parameters.
        embeddings_test: [N_test, 128] pre-computed embeddings.
        ci_levels: CI levels to evaluate (default [0.50, 0.68, 0.90, 0.95]).
        n_posterior_samples: Posterior samples per test point.

    Returns:
        Dict with ci_levels, empirical_coverage, expected_calibration_error.
    """
    import torch  # noqa: PLC0415

    if ci_levels is None:
        ci_levels = [0.50, 0.68, 0.90, 0.95]

    n_test = len(theta_test)
    n_params = theta_test.shape[1]
    in_ci = np.zeros((n_test, len(ci_levels), n_params), dtype=bool)

    try:
        _net = posterior.posterior_estimator if hasattr(posterior, "posterior_estimator") \
               else posterior.net
        _device = next(_net.parameters()).device
    except (StopIteration, AttributeError):
        _device = torch.device("cpu")

    log.info("ECE: %d test points, %d CI levels, device=%s", n_test, len(ci_levels), _device)

    for i in range(n_test):
        x_obs = embeddings_test[i].to(_device)
        with torch.no_grad():
            samples = posterior.sample(
                (n_posterior_samples,), x=x_obs, show_progress_bars=False,
            )
        samples_np = samples.cpu().numpy()
        theta_np = theta_test[i].cpu().numpy()

        for li, ci in enumerate(ci_levels):
            alpha = (1.0 - ci) / 2.0
            lo = np.quantile(samples_np, alpha, axis=0)
            hi = np.quantile(samples_np, 1.0 - alpha, axis=0)
            in_ci[i, li, :] = (theta_np >= lo) & (theta_np <= hi)

    empirical_coverage = in_ci.mean(axis=0)  # [n_levels, n_params]

    # Expected Calibration Error: mean |empirical - nominal| across levels and params
    ece = 0.0
    for li, ci in enumerate(ci_levels):
        ece += np.mean(np.abs(empirical_coverage[li] - ci))
    ece /= len(ci_levels)

    # Per-parameter coverage report
    coverage_report = {}
    for li, ci in enumerate(ci_levels):
        coverage_report[f"CI_{int(ci*100)}"] = empirical_coverage[li].tolist()

    log.info("ECE = %.4f (0 = perfectly calibrated)", ece)

    return {
        "ci_levels": ci_levels,
        "empirical_coverage": empirical_coverage.tolist(),
        "expected_calibration_error": float(ece),
        "coverage_report": coverage_report,
        "n_test": n_test,
    }


# ---------------------------------------------------------------------------
# Embedding-space diagnostic: are observed streams in-distribution?
# ---------------------------------------------------------------------------

def embedding_ood_report(
    observed_embeddings: dict[str, np.ndarray],
    training_embeddings: np.ndarray,
    percentile_threshold: float = 95.0,
) -> dict[str, dict]:
    """Check whether observed stream embeddings are in-distribution.

    For each observed stream, computes the OOD score and compares to the
    training distribution. Streams that are OOD should have their posterior
    constraints flagged as potentially unreliable.

    Args:
        observed_embeddings: Dict mapping stream_name -> [128] embedding.
        training_embeddings: [N_train, 128] embeddings from training sims.
        percentile_threshold: Percentile threshold for OOD classification.

    Returns:
        Dict mapping stream_name -> {ood_score, percentile, is_ood, recommendation}.
    """
    # Pre-compute training distribution scores
    n_sample = min(500, len(training_embeddings))
    sample_idx = np.random.choice(len(training_embeddings), n_sample, replace=False)
    train_scores = np.array([
        _compute_ood_score(training_embeddings[i], training_embeddings)
        for i in sample_idx
    ])
    threshold = np.percentile(train_scores, percentile_threshold)

    report = {}
    for stream_name, emb in observed_embeddings.items():
        score = _compute_ood_score(emb, training_embeddings)
        percentile = float(np.mean(train_scores < score) * 100)
        is_ood = score > threshold

        if is_ood:
            recommendation = (
                f"WARNING: {stream_name} embedding is at the {percentile:.0f}th "
                f"percentile of training distances. Posterior may be unreliable. "
                f"Consider adding more diverse simulations or using a different "
                f"summary statistic for this stream."
            )
        else:
            recommendation = f"{stream_name} is in-distribution (percentile {percentile:.0f}%)."

        report[stream_name] = {
            "ood_score": score,
            "percentile": percentile,
            "threshold": threshold,
            "is_ood": is_ood,
            "recommendation": recommendation,
        }

    n_ood = sum(1 for v in report.values() if v["is_ood"])
    if n_ood > 0:
        log.warning("%d/%d observed streams are OOD — their constraints are suspect.",
                    n_ood, len(report))

    return report
