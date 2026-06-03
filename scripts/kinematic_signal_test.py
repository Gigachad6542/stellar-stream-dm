#!/usr/bin/env python
"""
Is there a usable KINEMATIC impact signature beyond the density gap?

A subhalo flyby imprints a localized antisymmetric "kink" in the mean proper motion
along the stream (velocity analog of the density gap). This script measures the
separability of a matched kink statistic (a) noise-free and (b) under realistic Gaia
proper-motion errors, plus a precision sweep. Result (2026-06): the kink is a powerful
discriminant in clean data (AUC ~1.0, beating the density gap) but collapses to chance
under current Gaia faint-star pm errors -- so a kinematic-feature detector will not
extend completeness today; the signal is precision-limited (see full_report.md §12.2).

Usage:  python scripts/kinematic_signal_test.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_training_data import _fix_galpy_dll_path
_fix_galpy_dll_path()
import numpy as np
from src.simulation.potentials import get_mw_potential
from src.simulation.stream_gen import generate_stream_df, sample_impact_params
from scripts.validate_generator import max_gap_depth, auc
import galstreams


def kink(phi1, pm, bin_deg=2.0):
    """Amplitude of the localized antisymmetric proper-motion kink (peak-to-trough
    of the detrended binned mean within a ~10 deg window)."""
    phi1 = np.asarray(phi1, float); pm = np.asarray(pm, float)
    m = np.isfinite(phi1) & np.isfinite(pm); phi1, pm = phi1[m], pm[m]
    if len(phi1) < 80:
        return 0.0
    lo, hi = np.percentile(phi1, [2, 98]); edges = np.arange(lo, hi + bin_deg, bin_deg)
    nb = len(edges) - 1
    if nb < 8:
        return 0.0
    idx = np.digitize(phi1, edges) - 1; mean = np.full(nb, np.nan)
    for b in range(nb):
        s = idx == b
        if s.sum() >= 5:
            mean[b] = np.mean(pm[s])
    ok = np.isfinite(mean); x = np.arange(nb)[ok]; y = mean[ok]
    if len(x) < 8:
        return 0.0
    r = y - np.polyval(np.polyfit(x, y, 2), x)
    return float(max(r[i:i + 5].max() - r[i:i + 5].min() for i in range(len(r) - 4)))


def main() -> int:
    mws = galstreams.MWStreams(verbose=False); pot = get_mw_potential("config/streams.yaml")
    rng = np.random.default_rng(2); N = 16
    smooth = [generate_stream_df("GD1", pot, n_stars=1200, seed=1500 + i, impact=False, mws=mws) for i in range(N)]
    imps = [generate_stream_df("GD1", pot, n_stars=1200, seed=1800 + i, impact=True,
                               impact_params=sample_impact_params(rng), mws=mws) for i in range(N)]
    K = lambda p, s: max(kink(p.phi1, p.pm1 + rng2.normal(0, s, len(p.pm1))),
                         kink(p.phi1, p.pm2 + rng2.normal(0, s, len(p.pm2))))
    rng2 = np.random.default_rng(7)
    print("density gap AUC (noise-free):",
          round(auc([max_gap_depth(p.phi1) for p in imps], [max_gap_depth(p.phi1) for p in smooth]), 3))
    print("\n  sigma_pm[mas/yr]   kink AUC")
    for sig in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        ki = [K(p, sig) for p in imps]; ko = [K(p, sig) for p in smooth]
        print(f"  {sig:14.2f}   {auc(np.array(ki), np.array(ko)):.3f}")
    print("\nNoise-free kink AUC ~1.0 (beats density); collapses toward chance as pm error grows.")
    print("Gaia DR3 faint-star pm error ~0.1-0.3 mas/yr -> kink largely lost for weak/old impacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
