#!/usr/bin/env python
"""
Axis 3 (DM family) done HONESTLY: a population-level distinguishability analysis.

WDM/FDM/SIDM do NOT change how any single subhalo impact looks -- they change the
subhalo MASS FUNCTION (how many subhalos exist at each mass). So "which DM model"
is not a per-impact GNN classification; it is a population inference from the
DISTRIBUTION of detected impact masses across many streams.

This script combines, for each DM model:
  (mass function dN/dM)  x  (our MEASURED detector completeness vs mass)
-> the distribution of DETECTED impact masses, then quantifies how distinguishable
each model is from CDM (KS distance) and roughly how many detected impacts are
needed to tell them apart.

Completeness(logM) is taken from scripts/detector_completeness.py on the corrected
detector (2026-06-02): rises 0.48 -> 0.72 across 10^7.5..10^8.5.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from src.simulation.mass_functions import (
    sample_cdm_masses, sample_wdm_masses, sample_fdm_masses,
    wdm_half_mode_mass, fdm_jeans_mass,
)

# Measured detector completeness vs log10(mass) (detector_completeness.py, test split).
_CL_LOGM = np.array([7.0, 7.65, 7.95, 8.25, 8.55, 9.0])
_CL_PDET = np.array([0.25, 0.478, 0.502, 0.604, 0.723, 0.85])  # extrapolated at edges


def completeness(logm: np.ndarray) -> np.ndarray:
    return np.clip(np.interp(logm, _CL_LOGM, _CL_PDET), 0.0, 1.0)


def detected_logm_sample(masses: np.ndarray, rng) -> np.ndarray:
    """Apply completeness as an acceptance probability -> detected-mass sample."""
    logm = np.log10(masses)
    keep = rng.uniform(size=len(logm)) < completeness(logm)
    return logm[keep]


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import ks_2samp
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def n_to_distinguish(ks: float, conf: float = 0.95) -> float:
    """Rough two-sample N (per sample) so KS critical value < observed KS distance."""
    c = {0.90: 1.22, 0.95: 1.36, 0.99: 1.63}.get(conf, 1.36)
    if not np.isfinite(ks) or ks <= 1e-3:
        return float("inf")
    # D_crit = c*sqrt(2/N)  (equal n) -> N = 2 (c/KS)^2
    return float(2.0 * (c / ks) ** 2)


def main() -> int:
    rng = np.random.default_rng(0)
    N = 400_000
    lo, hi, alpha = 6.5, 9.0, -1.9

    cdm = sample_cdm_masses(N, lo, hi, alpha, seed=1)
    models = {
        "CDM": cdm,
        "SIDM": sample_cdm_masses(N, lo, hi, alpha, seed=2),  # same mass function as CDM
        "WDM 6keV": sample_wdm_masses(N, 6.0, lo, hi, alpha, seed=3),
        "WDM 3keV": sample_wdm_masses(N, 3.0, lo, hi, alpha, seed=4),
        "FDM 1e-22": sample_fdm_masses(N, 1e-22, lo, hi, alpha, seed=5),
        "FDM 1e-21": sample_fdm_masses(N, 1e-21, lo, hi, alpha, seed=6),
    }
    print(f"  WDM half-mode mass: 6keV -> 10^{np.log10(wdm_half_mode_mass(6.0)):.2f}, "
          f"3keV -> 10^{np.log10(wdm_half_mode_mass(3.0)):.2f} Msun")
    print(f"  FDM Jeans mass: 1e-22 -> 10^{np.log10(fdm_jeans_mass(1e-22)):.2f}, "
          f"1e-21 -> 10^{np.log10(fdm_jeans_mass(1e-21)):.2f} Msun")

    det = {k: detected_logm_sample(v, rng) for k, v in models.items()}
    cdm_det = det["CDM"]
    print("\n" + "=" * 70)
    print("DM-FAMILY DISTINGUISHABILITY from the DETECTED-impact mass distribution")
    print("=" * 70)
    print(f"  {'model':10s} {'median logM':>11s} {'frac>10^8':>10s} {'KS vs CDM':>10s} "
          f"{'N_det(95%)':>11s}")
    for k, d in det.items():
        ks = ks_distance(d, cdm_det) if k != "CDM" else 0.0
        nreq = n_to_distinguish(ks) if k != "CDM" else float("nan")
        print(f"  {k:10s} {np.median(d):11.2f} {np.mean(d > 8.0):10.2f} "
              f"{ks:10.3f} {('%.0f' % nreq) if np.isfinite(nreq) else 'inf':>11s}")
    print("\n  Interpretation:")
    print("  - SIDM mass function == CDM: NOT distinguishable by impact mass distribution")
    print("    (SIDM differs in subhalo CONCENTRATION -> gap shape at fixed mass, a")
    print("     separate morphology signal, not captured here).")
    print("  - WDM/FDM suppress low-mass subhalos -> detected impacts skew to higher mass;")
    print("    N_det(95%) = order of detected impacts needed to distinguish from CDM.")
    print("  - This is why DM family is a POPULATION inference, not a per-impact GNN label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
