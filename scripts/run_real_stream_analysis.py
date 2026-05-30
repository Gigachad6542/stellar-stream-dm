"""
Apply the hierarchical Bayesian inference framework to literature-reported
gap counts from real Gaia DR3 observations of stellar streams.

This is the first application of our framework to real observational data.
Gap counts are drawn from published analyses of Gaia DR3 streams. The resulting
M_hm posterior represents our constraint on dark matter substructure.

Literature sources for gap counts:
  - GD-1: Bonaca+2019 (2 gaps + spur); Price-Whelan & Bonaca 2018 (density variations)
          de Boer+2020 (5-7 significant features); we adopt n=5
  - Pal 5: Erkal+2017 (tidal tails); Bonaca+2020 (perturbation features); n=3
  - Orphan-Chenab: Koposov+2019, Erkal+2019 (density variations along 120 deg); n=4
  - ATLAS: Li+2021 (thin stream, 1-2 features); n=1
  - Jhelum: Bonaca+2019b (double-component, possible DM interaction); n=2
  - Fjorm: Ibata+2021 (recently discovered, clean kinematics); n=1
  - Sylgr: Ibata+2021 (shortest stream, cold); n=1

Important caveats:
  - Not all observed gaps are necessarily from DM subhalo impacts; baryonic
    perturbations (GMCs, spiral arms) can also produce density features
  - Gap counting methodology varies between papers
  - Some streams have uncertain membership assignments

We also compare with published independent constraints:
  - Gravitational lensing: Gilman+2020, Hsueh+2020 → M_hm < 10^7.5-8.0
  - Satellite galaxy counts: Nadler+2021 → WDM m > 6.5 keV (M_hm < 10^7.2)
  - Lyman-alpha forest: Irsic+2017 → WDM m > 5.3 keV (M_hm < 10^7.5)

Outputs:
  - outputs/real_analysis/hierarchical_real_streams.json
  - outputs/real_analysis/hierarchical_real_streams_samples.npy
  - outputs/real_analysis/comparison_with_literature.json
  - outputs/real_analysis/figures/*.png
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

OUT = ROOT / "outputs" / "real_analysis"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ── Real stream properties (from config/streams.yaml + literature) ──────────
STREAMS = {
    "GD1":    {"length_deg": 60,  "age_gyr": 5,  "distance_kpc": 12},
    "Pal5":   {"length_deg": 20,  "age_gyr": 11, "distance_kpc": 20},
    "Orphan": {"length_deg": 120, "age_gyr": 5,  "distance_kpc": 20},
    "ATLAS":  {"length_deg": 15,  "age_gyr": 4,  "distance_kpc": 20},
    "Jhelum": {"length_deg": 30,  "age_gyr": 4,  "distance_kpc": 13},
    "Fjorm":  {"length_deg": 25,  "age_gyr": 3,  "distance_kpc": 15},
    "Sylgr":  {"length_deg": 12,  "age_gyr": 3,  "distance_kpc": 15},
}

# ── Literature gap counts ───────────────────────────────────────────────────
# These are our best estimates from published Gaia DR3 analyses.
# Uncertainty in these counts is a major systematic.
OBSERVED_GAPS = {
    "GD1":    5,   # Bonaca+2019, de Boer+2020: 5-7 features; conservative estimate
    "Pal5":   3,   # Erkal+2017, Bonaca+2020: tidal tail perturbations
    "Orphan":  4,   # Koposov+2019, Erkal+2019: density variations
    "ATLAS":  1,   # Li+2021: limited features in short stream
    "Jhelum": 2,   # Bonaca+2019b: double-component structure
    "Fjorm":  1,   # Ibata+2021: recently discovered
    "Sylgr":  1,   # Ibata+2021: very short, limited statistics
}

# Alternative gap counts for systematic uncertainty analysis
OBSERVED_GAPS_HIGH = {k: v + 1 for k, v in OBSERVED_GAPS.items()}  # +1 to all
OBSERVED_GAPS_LOW = {k: max(v - 1, 0) for k, v in OBSERVED_GAPS.items()}  # -1

# ── Published independent constraints ──────────────────────────────────────
LITERATURE_CONSTRAINTS = {
    "Gilman+2020 (strong lensing)": {
        "method": "Gravitational lensing flux ratio anomalies",
        "constraint": "log10(M_hm) < 7.5-8.0 at 95% CL",
        "log10_Mhm_upper_95": 7.8,
        "wdm_mass_lower_keV": 5.2,
    },
    "Hsueh+2020 (strong lensing)": {
        "method": "Extended gravitational arc perturbations",
        "constraint": "log10(M_hm) < 8.0 at 95% CL",
        "log10_Mhm_upper_95": 8.0,
        "wdm_mass_lower_keV": 4.8,
    },
    "Nadler+2021 (satellite counts)": {
        "method": "Milky Way satellite galaxy census",
        "constraint": "m_WDM > 6.5 keV → log10(M_hm) < 7.2",
        "log10_Mhm_upper_95": 7.2,
        "wdm_mass_lower_keV": 6.5,
    },
    "Irsic+2017 (Lyman-alpha)": {
        "method": "Lyman-alpha forest power spectrum",
        "constraint": "m_WDM > 5.3 keV → log10(M_hm) < 7.5",
        "log10_Mhm_upper_95": 7.5,
        "wdm_mass_lower_keV": 5.3,
    },
    "Banik+2021 (streams, power spectrum)": {
        "method": "1D density power spectrum of GD-1",
        "constraint": "Weak preference for CDM over WDM",
        "log10_Mhm_upper_95": 7.0,
        "wdm_mass_lower_keV": 7.0,
    },
}


def run_primary_analysis():
    """Run hierarchical inference with literature gap counts."""
    log.info("=" * 70)
    log.info("REAL STREAM ANALYSIS: Hierarchical inference on Gaia DR3 gap counts")
    log.info("=" * 70)

    log.info("\nObserved gap counts from literature:")
    total = 0
    for name, n in OBSERVED_GAPS.items():
        props = STREAMS[name]
        rate_cdm = expected_subhalo_rate(4.5, props["length_deg"], props["age_gyr"], props["distance_kpc"])
        log.info("  %s: n_obs=%d (CDM expected: %.1f)", name, n, rate_cdm)
        total += n
    log.info("  Total: %d gaps across 7 streams", total)

    # Run inference
    result = run_hierarchical_inference(
        per_stream_n_impacts=OBSERVED_GAPS,
        per_stream_properties=STREAMS,
        backend="emcee",
        n_walkers=64, n_steps=15000, n_burnin=3000,
    )

    log.info("\n" + "=" * 70)
    log.info("PRIMARY RESULT:")
    log.info("  log10(M_hm) median: %.3f", result.log10_M_hm_median)
    log.info("  log10(M_hm) 95%% upper limit: %.3f", result.log10_M_hm_upper_95)
    log.info("  log10(M_hm) 5%% lower limit: %.3f", result.log10_M_hm_lower_5)
    log.info("  log10(M_hm) 68%% HDI: [%.3f, %.3f]", *result.log10_M_hm_hdi_68)
    log.info("  r_hat: %.4f", result.r_hat_max)
    log.info("  n_eff: %d", result.n_effective_samples)
    log.info("  Per-stream rates at median:")
    for s, r in result.per_stream_rates.items():
        log.info("    %s: %.2f (observed: %d)", s, r, OBSERVED_GAPS[s])

    return result


def run_systematic_analysis():
    """Run inference with high and low gap count estimates for systematics."""
    log.info("\n" + "=" * 70)
    log.info("SYSTEMATIC UNCERTAINTY: +/- 1 gap per stream")
    log.info("=" * 70)

    results = {}
    for label, gaps in [("high", OBSERVED_GAPS_HIGH), ("low", OBSERVED_GAPS_LOW)]:
        result = run_hierarchical_inference(
            per_stream_n_impacts=gaps,
            per_stream_properties=STREAMS,
            backend="emcee",
            n_walkers=32, n_steps=10000, n_burnin=2000,
        )
        results[label] = result
        log.info("  %s counts: median=%.3f, 95%% upper=%.3f, 68%% HDI=[%.3f, %.3f]",
                 label, result.log10_M_hm_median, result.log10_M_hm_upper_95,
                 *result.log10_M_hm_hdi_68)

    return results


def run_individual_stream_analysis():
    """Run inference on each stream individually for comparison."""
    log.info("\n" + "=" * 70)
    log.info("INDIVIDUAL STREAM INFERENCE")
    log.info("=" * 70)

    from scipy.special import gammaln
    results = {}
    log_m_grid = np.linspace(4.0, 10.0, 500)

    for name, n_obs in OBSERVED_GAPS.items():
        props = STREAMS[name]
        log_lik = np.zeros(len(log_m_grid))
        for j, lm in enumerate(log_m_grid):
            rate = expected_subhalo_rate(lm, props["length_deg"], props["age_gyr"], props["distance_kpc"])
            log_lik[j] = n_obs * np.log(max(rate, 1e-10)) - rate - gammaln(n_obs + 1)
        log_lik -= log_lik.max()
        weights = np.exp(log_lik)
        weights /= weights.sum()
        samples = np.random.choice(log_m_grid, size=10000, replace=True, p=weights)
        samples += np.random.uniform(-0.01, 0.01, len(samples))

        med = np.median(samples)
        upper = np.percentile(samples, 95)
        results[name] = {"median": float(med), "upper_95": float(upper),
                         "n_obs": n_obs, "samples": samples}
        log.info("  %s (n=%d): median=%.2f, 95%% upper=%.2f", name, n_obs, med, upper)

    return results


def compare_with_literature(result):
    """Compare our constraint with published results."""
    log.info("\n" + "=" * 70)
    log.info("COMPARISON WITH INDEPENDENT CONSTRAINTS")
    log.info("=" * 70)

    our_upper_95 = result.log10_M_hm_upper_95
    our_median = result.log10_M_hm_median

    log.info("  Our result: median=%.2f, 95%% upper=%.2f", our_median, our_upper_95)
    log.info("")

    comparisons = []
    for source, data in LITERATURE_CONSTRAINTS.items():
        their_upper = data["log10_Mhm_upper_95"]
        consistent = our_upper_95 <= their_upper + 0.5  # within 0.5 dex
        status = "CONSISTENT" if consistent else "TENSION"
        log.info("  %s: upper=%.1f [%s]", source, their_upper, status)
        comparisons.append({
            "source": source,
            "method": data["method"],
            "their_upper_95": their_upper,
            "our_upper_95": our_upper_95,
            "consistent": consistent,
        })

    return comparisons


def make_figures(result, systematics, individual):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "figure.dpi": 200,
    })

    # Figure 1: Main posterior
    fig, ax = plt.subplots(figsize=(6, 4))
    samples = result.log10_M_hm_samples
    ax.hist(samples, bins=100, density=True, color="#2980b9", alpha=0.7, edgecolor="none")
    ax.axvline(result.log10_M_hm_median, color="#e74c3c", ls="-", lw=2,
               label=f"Median: {result.log10_M_hm_median:.2f}")
    ax.axvline(result.log10_M_hm_upper_95, color="#e74c3c", ls="--", lw=1.5,
               label=f"95% upper: {result.log10_M_hm_upper_95:.2f}")
    ax.axvspan(*result.log10_M_hm_hdi_68, alpha=0.15, color="#e74c3c",
               label=f"68% HDI: [{result.log10_M_hm_hdi_68[0]:.2f}, {result.log10_M_hm_hdi_68[1]:.2f}]")

    # Mark literature constraints
    colors_lit = ["#27ae60", "#8e44ad", "#e67e22", "#16a085", "#7f8c8d"]
    for i, (source, data) in enumerate(LITERATURE_CONSTRAINTS.items()):
        short = source.split("(")[0].strip()
        ax.axvline(data["log10_Mhm_upper_95"], color=colors_lit[i], ls=":", lw=1.2,
                   alpha=0.7, label=f"{short}: <{data['log10_Mhm_upper_95']:.1f}")

    ax.set_xlabel(r"$\log_{10}(M_{\rm hm}\;/\;M_\odot)$")
    ax.set_ylabel("Posterior density")
    ax.set_title("Constraint on Half-Mode Mass from 7 Gaia DR3 Stellar Streams")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(4.0, 10.0)
    fig.tight_layout()
    fig.savefig(FIG / "primary_posterior.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Systematic uncertainty comparison
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, color, res in [
        ("Primary (best estimate)", "#2980b9", result),
        ("High counts (+1 each)", "#27ae60", systematics["high"]),
        ("Low counts (-1 each)", "#e67e22", systematics["low"]),
    ]:
        ax.hist(res.log10_M_hm_samples, bins=80, density=True, alpha=0.4,
                color=color, edgecolor="none", label=label)
        ax.axvline(res.log10_M_hm_median, color=color, ls="--", lw=1.5, alpha=0.8)
    ax.set_xlabel(r"$\log_{10}(M_{\rm hm}\;/\;M_\odot)$")
    ax.set_ylabel("Posterior density")
    ax.set_title("Systematic Uncertainty from Gap Count Estimates")
    ax.legend(fontsize=8)
    ax.set_xlim(4.0, 10.0)
    fig.tight_layout()
    fig.savefig(FIG / "systematic_uncertainty.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Per-stream individual posteriors
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2980b9", "#e67e22", "#27ae60", "#8e44ad", "#e74c3c", "#16a085", "#7f8c8d"]
    for i, (name, data) in enumerate(individual.items()):
        ax.hist(data["samples"], bins=60, density=True, alpha=0.35,
                color=colors[i], edgecolor="none",
                label=f"{name} (n={data['n_obs']})")
    # Combined
    ax.hist(result.log10_M_hm_samples, bins=60, density=True, alpha=0.6,
            color="black", edgecolor="none", histtype="step", lw=2,
            label="Combined (hierarchical)")
    ax.set_xlabel(r"$\log_{10}(M_{\rm hm}\;/\;M_\odot)$")
    ax.set_ylabel("Posterior density")
    ax.set_title("Individual vs. Hierarchical Stream Constraints")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(4.0, 10.0)
    fig.tight_layout()
    fig.savefig(FIG / "individual_vs_hierarchical.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: Comparison with literature
    fig, ax = plt.subplots(figsize=(7, 3.5))
    methods = list(LITERATURE_CONSTRAINTS.keys())
    upper_limits = [LITERATURE_CONSTRAINTS[m]["log10_Mhm_upper_95"] for m in methods]
    methods.append("This work")
    upper_limits.append(result.log10_M_hm_upper_95)

    y = range(len(methods))
    colors_bar = colors_lit + ["#2980b9"]
    bars = ax.barh(y, upper_limits, color=colors_bar, alpha=0.7, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([m.split("(")[0].strip() if "(" in m else m for m in methods], fontsize=8)
    ax.set_xlabel(r"95% upper limit on $\log_{10}(M_{\rm hm}\;/\;M_\odot)$")
    ax.set_title("Comparison with Published Independent Constraints")
    ax.axvline(result.log10_M_hm_upper_95, color="#2980b9", ls="--", lw=1.5, alpha=0.5)
    # Highlight our result
    bars[-1].set_edgecolor("#1a5276")
    bars[-1].set_linewidth(2)
    ax.set_xlim(4.0, 10.0)
    fig.tight_layout()
    fig.savefig(FIG / "literature_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    log.info("\nFigures saved to: %s", FIG)


if __name__ == "__main__":
    # 1. Primary analysis
    result = run_primary_analysis()

    # 2. Systematic uncertainty
    systematics = run_systematic_analysis()

    # 3. Individual stream analysis
    individual = run_individual_stream_analysis()

    # 4. Literature comparison
    comparisons = compare_with_literature(result)

    # 5. Generate figures
    make_figures(result, systematics, individual)

    # 6. Save results
    primary_json = {
        "backend": result.backend,
        "n_streams": result.n_streams,
        "median": round(result.log10_M_hm_median, 4),
        "upper_95": round(result.log10_M_hm_upper_95, 4),
        "lower_5": round(result.log10_M_hm_lower_5, 4),
        "hdi_68": [round(result.log10_M_hm_hdi_68[0], 4), round(result.log10_M_hm_hdi_68[1], 4)],
        "r_hat": round(result.r_hat_max, 4),
        "n_eff": result.n_effective_samples,
        "observed_gaps": OBSERVED_GAPS,
        "per_stream_rates_at_median": {k: round(v, 3) for k, v in result.per_stream_rates.items()},
        "n_posterior_samples": len(result.log10_M_hm_samples),
        "systematic_high_median": round(systematics["high"].log10_M_hm_median, 4),
        "systematic_low_median": round(systematics["low"].log10_M_hm_median, 4),
    }
    with open(OUT / "hierarchical_real_streams.json", "w") as f:
        json.dump(primary_json, f, indent=2)
    np.save(OUT / "hierarchical_real_streams_samples.npy", result.log10_M_hm_samples)

    with open(OUT / "comparison_with_literature.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("ANALYSIS COMPLETE")
    log.info("=" * 70)
    log.info("Primary constraint: log10(M_hm) = %.2f (+%.2f/-%.2f) [68%% HDI]",
             result.log10_M_hm_median,
             result.log10_M_hm_hdi_68[1] - result.log10_M_hm_median,
             result.log10_M_hm_median - result.log10_M_hm_hdi_68[0])
    log.info("95%% upper limit: log10(M_hm) < %.2f", result.log10_M_hm_upper_95)
    log.info("Results saved to: %s", OUT)
