#!/usr/bin/env python
"""
Rigorous DM-model discrimination forecast: abundance (Poisson rate) + mass shape.

The earlier estimate (dm_family_distinguishability.py) used only the NORMALIZED
detected-mass distribution (a KS heuristic), discarding the strongest signal: WDM/FDM
produce FEWER detectable impacts, not just a shifted mass shape. Here we combine both
in an Asimov likelihood-ratio forecast.

Per stream, the number of DETECTABLE impacts is Poisson with mean
    lambda_det(model) = INT  R_CDM(M) * S_model(M) * C(M)  dlogM,
where R_CDM is the CDM differential encounter rate over the detectable band
(normalized to compute_encounter_rate), S_model the WDM/FDM suppression filter, and
C(M) the MEASURED detector completeness. Each detection's mass is drawn from
    p_det(M; model) ∝ R_CDM(M) * S_model(M) * C(M).

The expected per-stream log-likelihood ratio of a true model T vs CDM (Asimov) is
    LLR1 = [lambda_CDM - lambda_T] + lambda_T*ln(lambda_T/lambda_CDM)      (rate)
         + lambda_T * KL(p_T || p_CDM)                                     (mass shape)
and the discrimination significance after N streams is Z = sqrt(2 * N * LLR1), so
    N_streams(Zsigma) = Z^2 / (2 * LLR1),   expected detections = N_streams * lambda_T.

SIDM has CDM rate AND CDM mass function -> LLR1 = 0 -> not discriminable here
(requires the gap-SHAPE signal; see dm_sidm_morphology.py).
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.simulation.mass_functions import (
    compute_encounter_rate, wdm_suppression, fdm_suppression,
    wdm_half_mode_mass, fdm_jeans_mass,
)

# Measured detector completeness vs log10(M) (detector_completeness.py).
_CL_LOGM = np.array([7.0, 7.65, 7.95, 8.25, 8.55, 9.0])
_CL_P = np.array([0.25, 0.478, 0.502, 0.604, 0.723, 0.85])

# Representative GD-1-like stream (for the per-stream rate normalization).
STREAM_AGE_GYR = 3.0
STREAM_LEN_KPC = 14.0
STREAM_WIDTH_PC = 50.0
BAND = (7.5, 8.7)   # detectable mass band (log10 Msun)
ALPHA = -1.9


def completeness(logm):
    return np.clip(np.interp(logm, _CL_LOGM, _CL_P), 0.0, 1.0)


def model_rate_and_pdf(model: str, params: dict, logm: np.ndarray):
    """Return (lambda_det per stream, detected-mass pdf over logm) for a model."""
    M = 10.0 ** logm
    # CDM differential rate over the band, normalized so its integral = compute_encounter_rate
    shape = M ** (ALPHA + 1.0)  # dN/dlogM
    lam_band_cdm = compute_encounter_rate(STREAM_AGE_GYR, STREAM_LEN_KPC, STREAM_WIDTH_PC,
                                          BAND[0], BAND[1], alpha=ALPHA)
    R_cdm = shape / np.trapezoid(shape, logm) * lam_band_cdm
    if model in ("CDM", "SIDM"):
        S = np.ones_like(M)
    elif model == "WDM":
        S = wdm_suppression(M, wdm_half_mode_mass(params["m_wdm_kev"]))
    elif model == "FDM":
        S = fdm_suppression(M, fdm_jeans_mass(params["m_axion_ev"]))
    C = completeness(logm)
    integrand = R_cdm * S * C
    lam_det = float(np.trapezoid(integrand, logm))
    pdf = integrand / max(lam_det, 1e-30)
    return lam_det, pdf


def kl(p, q, logm):
    m = (p > 0) & (q > 0)
    return float(np.trapezoid(np.where(m, p * np.log(np.where(m, p / np.where(q > 0, q, 1), 1)), 0.0), logm))


def main() -> int:
    logm = np.linspace(BAND[0], BAND[1], 600)
    lam_cdm, p_cdm = model_rate_and_pdf("CDM", {}, logm)
    models = {
        "SIDM":      ("SIDM", {}),
        "WDM 6 keV": ("WDM", {"m_wdm_kev": 6.0}),
        "WDM 3 keV": ("WDM", {"m_wdm_kev": 3.0}),
        "WDM 4 keV": ("WDM", {"m_wdm_kev": 4.0}),
        "FDM 1e-22": ("FDM", {"m_axion_ev": 1e-22}),
        "FDM 1e-21": ("FDM", {"m_axion_ev": 1e-21}),
    }
    print(f"Detectable-impact rate per GD-1-like stream:  CDM lambda_det = {lam_cdm:.3f}")
    print("=" * 78)
    print(f"  {'model':10} {'lam_det':>8} {'rate/CDM':>9} {'KL(mass)':>9} "
          f"{'N_str(3s)':>10} {'N_det(3s)':>10}")
    rows = {"CDM": {"lam_det": lam_cdm}}
    for name, (mdl, par) in models.items():
        lam_t, p_t = model_rate_and_pdf(mdl, par, logm)
        klmass = kl(p_t, p_cdm, logm)
        llr1 = (lam_cdm - lam_t) + (lam_t * np.log(lam_t / lam_cdm) if lam_t > 0 else 0.0) + lam_t * klmass
        if llr1 <= 1e-9:
            nstr = float("inf"); ndet = float("inf")
        else:
            nstr = 9.0 / (2.0 * llr1)        # 3 sigma
            ndet = nstr * lam_t
        rows[name] = {"lam_det": lam_t, "rate_ratio": lam_t / lam_cdm, "kl_mass": klmass,
                      "N_streams_3sig": nstr, "N_det_3sig": ndet}
        ns = f"{nstr:.0f}" if np.isfinite(nstr) else "inf"
        nd = f"{ndet:.0f}" if np.isfinite(ndet) else "inf"
        print(f"  {name:10} {lam_t:8.3f} {lam_t/lam_cdm:9.2f} {klmass:9.3f} {ns:>10} {nd:>10}")
    print("=" * 78)
    print("N_str = equivalent GD-1-like streams for 3 sigma discrimination from CDM;")
    print("N_det = expected detected impacts over those streams.")
    print("SIDM: rate=CDM, mass=CDM -> not discriminable here (needs gap shape).")
    os.makedirs("outputs/dm", exist_ok=True)
    json.dump(rows, open("outputs/dm/discrimination_forecast.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
