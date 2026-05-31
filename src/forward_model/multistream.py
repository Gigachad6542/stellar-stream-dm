"""
Multi-stream joint significance for the timeline forward model.

A single short stream provides only a marginal constraint on subhalo impacts
(the look-elsewhere-corrected significance is ~0). Subhalo *abundance*, however,
is a population property: combining many streams is how you measure it. This
module combines per-stream significances into a joint statistic.

Two standard meta-analytic combinations are provided:

* **Stouffer's Z**: ``Z = sum(z_i) / sqrt(N)`` — combines per-stream z-scores
  assuming each is ~N(0,1) under the no-impact null. One-sided combined p-value
  ``P(Z > Z_comb)``. (Optionally inverse-variance / sensitivity weighted.)
* **Fisher's method**: ``X = -2 sum(ln p_i) ~ chi^2(2N)`` — combines per-stream
  p-values.

Honesty note: the per-stream inputs MUST be the look-elsewhere-corrected
significances (best-of-grid null), not the naive single-candidate ones, or the
joint number is inflated by the same look-elsewhere effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def combine_z_stouffer(z_scores, weights=None) -> tuple[float, float]:
    """Stouffer combination of per-stream z-scores.

    Returns (Z_combined, one_sided_p). With weights, uses the weighted Stouffer
    form Z = sum(w_i z_i) / sqrt(sum(w_i^2)).
    """
    z = np.asarray(list(z_scores), dtype=float)
    if z.size == 0:
        raise ValueError("no z-scores to combine")
    if weights is None:
        Z = float(z.sum() / math.sqrt(z.size))
    else:
        w = np.asarray(list(weights), dtype=float)
        Z = float(np.sum(w * z) / math.sqrt(np.sum(w * w)))
    p = 0.5 * math.erfc(Z / math.sqrt(2.0))   # one-sided P(N(0,1) > Z)
    return Z, p


def combine_p_fisher(p_values) -> tuple[float, float]:
    """Fisher combination of per-stream one-sided p-values.

    Returns (chi2_statistic, combined_p) with the statistic ~ chi^2(2N).
    """
    from scipy.stats import chi2

    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        raise ValueError("no p-values to combine")
    p = np.clip(p, 1e-12, 1.0)
    x = float(-2.0 * np.sum(np.log(p)))
    return x, float(chi2.sf(x, 2 * p.size))


@dataclass
class MultiStreamResult:
    """Joint multi-stream significance result."""
    per_stream: list = field(default_factory=list)   # list of per-stream dicts
    stouffer_z: float = 0.0
    stouffer_p: float = 1.0
    fisher_chi2: float = 0.0
    fisher_p: float = 1.0
    n_streams: int = 0
    used: str = "look_elsewhere"     # which per-stream significance was combined
    frac_positive: float = 0.0       # fraction of streams with z > 0 (coherence)
    coherent: bool = False
    interpretation: str = ""

    def to_dict(self) -> dict:
        return {
            "n_streams": self.n_streams,
            "combined_using": self.used,
            "stouffer_z": self.stouffer_z,
            "stouffer_p": self.stouffer_p,
            "fisher_chi2": self.fisher_chi2,
            "fisher_p": self.fisher_p,
            "frac_positive": self.frac_positive,
            "coherent": self.coherent,
            "interpretation": self.interpretation,
            "per_stream": self.per_stream,
        }


def combine_streams(per_stream: list, use: str = "look_elsewhere") -> MultiStreamResult:
    """Combine a list of per-stream significance dicts into a joint result.

    Each per-stream dict should contain ``z``/``p`` (naive) and optionally
    ``le_z``/``le_p`` (look-elsewhere-corrected). ``use`` selects which to combine
    ("look_elsewhere" preferred; falls back to naive when LE is absent).
    """
    zk, pk = ("le_z", "le_p") if use == "look_elsewhere" else ("z", "p")
    zs, ps = [], []
    for s in per_stream:
        z = s.get(zk, s.get("z"))
        p = s.get(pk, s.get("p"))
        if z is not None and np.isfinite(z) and p is not None and np.isfinite(p):
            zs.append(float(z)); ps.append(float(p))
    if not zs:
        raise ValueError("no usable per-stream significances")

    Z, Zp = combine_z_stouffer(zs)
    X, Xp = combine_p_fisher(ps)

    # Coherence check: a genuine population signal produces CONSISTENT positive
    # per-stream significance. A mix of large-positive and negative z indicates
    # the apparent excess is driven by per-stream null-construction artifacts and
    # sim-to-real differences, not a coherent subhalo population.
    zarr = np.asarray(zs, dtype=float)
    frac_pos = float(np.mean(zarr > 0))
    coherent = frac_pos >= 0.85 and float(zarr.min()) > -1.0

    # Headline on Fisher's combination of EMPIRICAL p-values, gated by coherence.
    # (Per-stream parametric z is not a calibrated standard normal at modest null
    # sizes, so Stouffer is reported only as a secondary, optimistic indicator.)
    if Xp < 0.05 and not coherent:
        interp = (f"NOT a detection despite Fisher p={Xp:.3f}: the per-stream significances are "
                  f"incoherent ({frac_pos*100:.0f}% positive; range z=[{zarr.min():.1f},{zarr.max():.1f}]), "
                  "so the apparent excess is driven by null-construction artifacts and per-stream "
                  "sim-to-real differences, not a coherent subhalo population. Consistent with an "
                  "upper limit / CDM given current data and simulation fidelity.")
    elif Xp < 0.0027 and coherent:
        interp = (f"Coherent joint detection (Fisher p={Xp:.2e}, {frac_pos*100:.0f}% positive): the "
                  "stream population shows localized perturbation beyond a smooth-stream null.")
    elif Xp < 0.05 and coherent:
        interp = (f"Coherent marginal excess (Fisher p={Xp:.3f}); suggestive, not a detection.")
    else:
        interp = (f"No joint detection (Fisher p={Xp:.2f}): consistent with a smooth (no localized "
                  "impact) null -- an upper limit / CDM-consistency given current data.")

    return MultiStreamResult(
        per_stream=per_stream, stouffer_z=Z, stouffer_p=Zp,
        fisher_chi2=X, fisher_p=Xp, n_streams=len(zs), used=use,
        frac_positive=frac_pos, coherent=coherent, interpretation=interp,
    )
