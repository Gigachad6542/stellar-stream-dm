"""
Statistical significance for the timeline forward model.

The forward model ranks candidate encounters by a combined score (lower = better
fit to the observed stream). On its own a score is not interpretable: "how
different is the best candidate from the data, and is that difference meaningful
versus no impact at all?" needs a reference.

This module provides that reference:

* ``compute_significance`` expresses a candidate score relative to a **null
  distribution** of no-impact (unperturbed) scores: a z-score
  ``(null_mean - score) / null_std`` (positive = better than typical null) and
  an empirical one-sided p-value ``P(null <= score)`` (the fraction of no-impact
  realizations that fit the data at least as well as the candidate).

* The pipeline builds the null distribution by scoring many unperturbed stream
  realizations with different random seeds (``build_null_distribution``), and
  evaluates candidates over multiple seeds (``evaluate_candidate_multiseed``) so
  the ranking reflects the physical encounter, not sampling noise.

Caveat (look-elsewhere): the *best* of many candidates beats the null by chance
more often than a single candidate would. A fully calibrated p-value would
compare the best-candidate score against the distribution of best scores under
the null (re-running the grid per null realization). ``compute_significance``
against the unperturbed-null distribution is the first-order test; the
look-elsewhere-corrected version is supported by passing a null distribution of
*best* scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ImpactTimePosterior:
    """Monte-Carlo posterior over the best-fit impact time (and mass/phi1).

    Built by resampling the observed stream within its per-star measurement
    errors and re-fitting each realization, so the spread reflects how the
    present-day measurement precision (which multi-epoch data tightens) maps
    into the precision of the recovered time-since-impact.
    """
    t_since_samples: list = field(default_factory=list)
    t_since_median: float = 0.0
    t_since_p16: float = 0.0
    t_since_p84: float = 0.0
    t_since_std: float = 0.0
    log10_mass_median: float = 0.0
    impact_phi1_median: float = 0.0
    n_realizations: int = 0
    error_scale: float = 1.0

    def to_dict(self) -> dict:
        return {
            "t_since_median_gyr": self.t_since_median,
            "t_since_p16_gyr": self.t_since_p16,
            "t_since_p84_gyr": self.t_since_p84,
            "t_since_std_gyr": self.t_since_std,
            "t_since_68pct_width_gyr": self.t_since_p84 - self.t_since_p16,
            "log10_mass_median": self.log10_mass_median,
            "impact_phi1_median": self.impact_phi1_median,
            "n_realizations": self.n_realizations,
            "error_scale": self.error_scale,
            "t_since_samples": self.t_since_samples,
        }


def summarize_impact_time(t_samples, m_samples, phi1_samples, error_scale=1.0) -> ImpactTimePosterior:
    """Build an ImpactTimePosterior from per-realization best-fit samples."""
    t = np.asarray(t_samples, dtype=float)
    return ImpactTimePosterior(
        t_since_samples=[float(x) for x in t],
        t_since_median=float(np.median(t)),
        t_since_p16=float(np.percentile(t, 16)),
        t_since_p84=float(np.percentile(t, 84)),
        t_since_std=float(np.std(t)),
        log10_mass_median=float(np.median(m_samples)),
        impact_phi1_median=float(np.median(phi1_samples)),
        n_realizations=len(t),
        error_scale=float(error_scale),
    )


@dataclass
class SignificanceResult:
    candidate_score: float
    null_mean: float
    null_std: float
    n_null: int
    z_score: float          # (null_mean - candidate) / null_std; >0 means better than null
    p_value: float          # empirical P(null <= candidate): smaller = more significant
    improvement: float      # null_mean - candidate_score
    null_scores: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_score": self.candidate_score,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "n_null": self.n_null,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "improvement_over_null_mean": self.improvement,
        }


def compute_significance(candidate_score: float, null_scores) -> SignificanceResult:
    """Significance of a (lower-is-better) candidate score vs a null distribution.

    Args:
        candidate_score: the candidate's combined score (lower = better fit).
        null_scores: iterable of no-impact scores (the null distribution).

    Returns:
        SignificanceResult with z-score and empirical one-sided p-value.
    """
    null = np.asarray(list(null_scores), dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        raise ValueError("null_scores is empty")
    mean = float(null.mean())
    std = float(null.std(ddof=1)) if null.size > 1 else 0.0
    z = (mean - candidate_score) / std if std > 0 else float("inf") if candidate_score < mean else 0.0
    # One-sided: fraction of null realizations that fit at least as well as the
    # candidate (score <= candidate_score). Add-one smoothing avoids p=0.
    p = float((np.sum(null <= candidate_score) + 1) / (null.size + 1))
    return SignificanceResult(
        candidate_score=float(candidate_score),
        null_mean=mean,
        null_std=std,
        n_null=int(null.size),
        z_score=float(z),
        p_value=p,
        improvement=mean - float(candidate_score),
        null_scores=[float(x) for x in null],
    )
