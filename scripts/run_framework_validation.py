"""
Run the framework's statistical inference engine on mock and synthetic data.

This exercises the actual code — rate model, Poisson likelihood inversion,
hierarchical Bayesian inference via emcee MCMC, and coverage metrics — on
controlled inputs with known ground truth.

This does NOT use the GNN (which requires a trained checkpoint). It tests the
mathematical and statistical machinery downstream of the embedding step.

Outputs:
  - outputs/validation/mock_challenge_results.json
  - outputs/validation/hierarchical_result.json
  - outputs/validation/figures/*.png
"""
from __future__ import annotations
import sys, os, json, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from src.inference.hierarchical import expected_subhalo_rate, run_hierarchical_inference
from scripts.run_mock_challenge import compute_coverage

OUT = ROOT / "outputs" / "validation"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ── Stream properties (from config/streams.yaml) ──────────────────────────
STREAMS = {
    "GD1":    {"length_deg": 60,  "age_gyr": 5,  "distance_kpc": 12},
    "Pal5":   {"length_deg": 20,  "age_gyr": 11, "distance_kpc": 20},
    "Orphan": {"length_deg": 120, "age_gyr": 5,  "distance_kpc": 20},
    "ATLAS":  {"length_deg": 15,  "age_gyr": 4,  "distance_kpc": 20},
    "Jhelum": {"length_deg": 30,  "age_gyr": 4,  "distance_kpc": 13},
    "Fjorm":  {"length_deg": 25,  "age_gyr": 3,  "distance_kpc": 15},
    "Sylgr":  {"length_deg": 12,  "age_gyr": 3,  "distance_kpc": 15},
}

# ── Mock truths (from run_mock_challenge.py) ──────────────────────────────
MOCK_TRUTHS = {
    "CDM_1": {"dm": "CDM", "log10_Mhm": 4.5, "n_impacts": 5},
    "CDM_2": {"dm": "CDM", "log10_Mhm": 4.5, "n_impacts": 4},
    "CDM_3": {"dm": "CDM", "log10_Mhm": 4.5, "n_impacts": 7},
    "CDM_4": {"dm": "CDM", "log10_Mhm": 4.5, "n_impacts": 3},
    "CDM_5": {"dm": "CDM", "log10_Mhm": 4.5, "n_impacts": 6},
    "WDM_1": {"dm": "WDM", "log10_Mhm": 6.5, "n_impacts": 4},
    "WDM_2": {"dm": "WDM", "log10_Mhm": 7.0, "n_impacts": 3},
    "WDM_3": {"dm": "WDM", "log10_Mhm": 7.5, "n_impacts": 3},
    "WDM_4": {"dm": "WDM", "log10_Mhm": 8.0, "n_impacts": 2},
    "WDM_5": {"dm": "WDM", "log10_Mhm": 8.5, "n_impacts": 1},
    "FDM_1": {"dm": "FDM", "log10_Mhm": 6.5, "n_impacts": 4},
    "FDM_2": {"dm": "FDM", "log10_Mhm": 7.0, "n_impacts": 3},
    "FDM_3": {"dm": "FDM", "log10_Mhm": 7.5, "n_impacts": 2},
    "FDM_4": {"dm": "FDM", "log10_Mhm": 8.0, "n_impacts": 2},
    "FDM_5": {"dm": "FDM", "log10_Mhm": 8.5, "n_impacts": 1},
    "SIDM_1": {"dm": "SIDM", "log10_Mhm": 6.5, "n_impacts": 5},
    "SIDM_2": {"dm": "SIDM", "log10_Mhm": 7.0, "n_impacts": 4},
    "SIDM_3": {"dm": "SIDM", "log10_Mhm": 7.5, "n_impacts": 3},
    "SIDM_4": {"dm": "SIDM", "log10_Mhm": 8.0, "n_impacts": 2},
    "SIDM_5": {"dm": "SIDM", "log10_Mhm": 8.5, "n_impacts": 1},
}


def poisson_posterior_grid(n_obs, length_deg=60, age_gyr=5, dist_kpc=15, n_samples=5000):
    """Invert rate model via Poisson likelihood on a grid to get M_hm posterior."""
    from scipy.special import gammaln
    log_m_grid = np.linspace(4.0, 10.5, 800)
    log_lik = np.zeros(len(log_m_grid))
    for j, lm in enumerate(log_m_grid):
        rate = expected_subhalo_rate(lm, length_deg, age_gyr, dist_kpc)
        log_lik[j] = n_obs * np.log(max(rate, 1e-10)) - rate - gammaln(n_obs + 1)
    log_lik -= log_lik.max()
    weights = np.exp(log_lik)
    weights /= weights.sum()
    samples = np.random.choice(log_m_grid, size=n_samples, replace=True, p=weights)
    samples += np.random.uniform(-0.008, 0.008, len(samples))  # de-discretise
    return samples


def run_mock_challenge():
    """Run all 20 mocks through Poisson likelihood inversion and compute coverage."""
    log.info("=" * 60)
    log.info("MOCK DATA CHALLENGE (20 streams)")
    log.info("=" * 60)

    true_vals = []
    posteriors = []
    per_mock = []

    for name, truth in MOCK_TRUTHS.items():
        samples = poisson_posterior_grid(truth["n_impacts"])
        true_vals.append(truth["log10_Mhm"])
        posteriors.append(samples)
        med = np.median(samples)
        bias = med - truth["log10_Mhm"]
        per_mock.append({
            "name": name, "dm": truth["dm"],
            "truth": truth["log10_Mhm"], "n_impacts": truth["n_impacts"],
            "median": round(float(med), 3), "bias": round(float(bias), 3),
        })
        log.info("  %s: truth=%.1f, n_imp=%d, median=%.2f, bias=%+.2f",
                 name, truth["log10_Mhm"], truth["n_impacts"], med, bias)

    true_vals = np.array(true_vals)
    results = compute_coverage(true_vals, posteriors)

    log.info("")
    log.info("COVERAGE RESULTS:")
    for ci, emp in results["empirical_coverage"].items():
        status = "PASS" if emp >= 0.75 else "FAIL"
        log.info("  %d%% CI: empirical=%.0f%% [%s]", int(ci*100), emp*100, status)
    log.info("  Mean bias: %+.3f dex", results["mean_bias"])
    log.info("  RMSE: %.3f dex", results["rmse"])

    return {
        "n_mocks": results["n_mocks"],
        "coverage": {f"{int(k*100)}%": round(v, 3) for k, v in results["empirical_coverage"].items()},
        "mean_bias_dex": round(results["mean_bias"], 4),
        "rmse_dex": round(results["rmse"], 4),
        "per_mock": per_mock,
    }


def run_hierarchical():
    """Run hierarchical Bayesian inference on a CDM-like scenario (7 streams)."""
    log.info("")
    log.info("=" * 60)
    log.info("HIERARCHICAL BAYESIAN INFERENCE (7 streams, CDM scenario)")
    log.info("=" * 60)

    # Simulate observed n_impacts under CDM (M_hm = 4.5, no suppression)
    np.random.seed(42)
    n_impacts = {}
    for name, props in STREAMS.items():
        rate = expected_subhalo_rate(4.5, props["length_deg"], props["age_gyr"], props["distance_kpc"])
        n_obs = np.random.poisson(rate)
        n_impacts[name] = int(max(n_obs, 0))
        log.info("  %s: CDM rate=%.1f, observed=%d", name, rate, n_impacts[name])

    result = run_hierarchical_inference(
        per_stream_n_impacts=n_impacts,
        per_stream_properties=STREAMS,
        backend="emcee",
        n_walkers=32, n_steps=8000, n_burnin=2000,
    )

    log.info("")
    log.info("HIERARCHICAL RESULT:")
    log.info("  Backend: %s", result.backend)
    log.info("  log10(M_hm) median: %.2f", result.log10_M_hm_median)
    log.info("  log10(M_hm) 95%% upper limit: %.2f", result.log10_M_hm_upper_95)
    log.info("  log10(M_hm) 68%% HDI: [%.2f, %.2f]", *result.log10_M_hm_hdi_68)
    log.info("  r_hat: %.4f", result.r_hat_max)
    log.info("  n_eff: %d", result.n_effective_samples)
    log.info("  Per-stream rates at median:")
    for s, r in result.per_stream_rates.items():
        log.info("    %s: %.2f", s, r)

    return {
        "backend": result.backend,
        "median": round(result.log10_M_hm_median, 3),
        "upper_95": round(result.log10_M_hm_upper_95, 3),
        "lower_5": round(result.log10_M_hm_lower_5, 3),
        "hdi_68": [round(result.log10_M_hm_hdi_68[0], 3), round(result.log10_M_hm_hdi_68[1], 3)],
        "r_hat": round(result.r_hat_max, 4),
        "n_eff": result.n_effective_samples,
        "n_streams": result.n_streams,
        "observed_n_impacts": {k: int(v) for k, v in n_impacts.items()},
        "per_stream_rates_at_median": {k: round(v, 2) for k, v in result.per_stream_rates.items()},
        "n_posterior_samples": len(result.log10_M_hm_samples),
    }, result.log10_M_hm_samples


def run_hierarchical_wdm():
    """Run hierarchical inference under a WDM scenario (M_hm = 10^7.5)."""
    log.info("")
    log.info("=" * 60)
    log.info("HIERARCHICAL INFERENCE (7 streams, WDM scenario, M_hm=10^7.5)")
    log.info("=" * 60)

    np.random.seed(123)
    n_impacts = {}
    for name, props in STREAMS.items():
        rate = expected_subhalo_rate(7.5, props["length_deg"], props["age_gyr"], props["distance_kpc"])
        n_obs = np.random.poisson(max(rate, 0.1))
        n_impacts[name] = int(max(n_obs, 0))
        log.info("  %s: WDM rate=%.2f, observed=%d", name, rate, n_impacts[name])

    result = run_hierarchical_inference(
        per_stream_n_impacts=n_impacts,
        per_stream_properties=STREAMS,
        backend="emcee",
        n_walkers=32, n_steps=8000, n_burnin=2000,
    )

    log.info("")
    log.info("HIERARCHICAL RESULT (WDM):")
    log.info("  log10(M_hm) median: %.2f", result.log10_M_hm_median)
    log.info("  log10(M_hm) 95%% upper limit: %.2f", result.log10_M_hm_upper_95)
    log.info("  log10(M_hm) 68%% HDI: [%.2f, %.2f]", *result.log10_M_hm_hdi_68)
    log.info("  r_hat: %.4f", result.r_hat_max)

    return {
        "true_log10_Mhm": 7.5,
        "median": round(result.log10_M_hm_median, 3),
        "upper_95": round(result.log10_M_hm_upper_95, 3),
        "hdi_68": [round(result.log10_M_hm_hdi_68[0], 3), round(result.log10_M_hm_hdi_68[1], 3)],
        "r_hat": round(result.r_hat_max, 4),
        "n_eff": result.n_effective_samples,
        "observed_n_impacts": {k: int(v) for k, v in n_impacts.items()},
    }, result.log10_M_hm_samples


if __name__ == "__main__":
    # 1. Mock challenge
    mock_results = run_mock_challenge()
    with open(OUT / "mock_challenge_results.json", "w") as f:
        json.dump(mock_results, f, indent=2)

    # 2. Hierarchical CDM scenario
    hier_cdm, cdm_samples = run_hierarchical()
    with open(OUT / "hierarchical_cdm.json", "w") as f:
        json.dump(hier_cdm, f, indent=2)
    np.save(OUT / "hierarchical_cdm_samples.npy", cdm_samples)

    # 3. Hierarchical WDM scenario
    hier_wdm, wdm_samples = run_hierarchical_wdm()
    with open(OUT / "hierarchical_wdm.json", "w") as f:
        json.dump(hier_wdm, f, indent=2)
    np.save(OUT / "hierarchical_wdm_samples.npy", wdm_samples)

    log.info("")
    log.info("=" * 60)
    log.info("ALL VALIDATION COMPLETE")
    log.info("Results saved to: %s", OUT)
    log.info("=" * 60)
