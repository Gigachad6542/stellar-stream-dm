"""
Generate gap catalog and final analysis figures.

Reads per-stream posterior samples and log evidences from outputs/,
detects gaps in real stream density profiles, classifies them, and
produces all Phase 8 deliverables:
  1. Gap catalog (outputs/gap_catalog.csv)
  2. Model comparison report + figure
  3. Per-stream density profiles with gap overlays
  4. Mass function constraints figure
  5. Multi-stream consistency plot

Usage:
    python -u scripts/run_analysis.py
    python -u scripts/run_analysis.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.gap_catalog import detect_gaps, classify_gaps, gaps_to_dataframe
from src.analysis.model_comparison import model_comparison_report
from src.analysis.power_spectrum_baseline import (
    compute_density_1d,
    compute_power_spectrum,
    detect_power_excess,
    power_spectrum_summary_vector,
)
from src.analysis.visualization import (
    plot_corner,
    plot_combined_posteriors,
    plot_density_profile,
    plot_gap_classifier_diagnostics,
    plot_mass_function_constraints,
    plot_model_comparison,
    plot_ppc_results,
    plot_power_spectrum,
    plot_suppression_constraint,
)
from src.inference.posteriors import combine_posteriors_product
from src.inference.suppression_scale import (
    infer_suppression_from_counts,
    compute_upper_limit,
    M_hm_to_wdm_mass,
    M_hm_to_fdm_mass,
    M_hm_to_sidm_cross_section,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DM_MODELS = ["CDM", "WDM", "FDM", "SIDM"]


def load_stream_phi1(stream_name: str, processed_path: str) -> np.ndarray | None:
    """Load phi1 array for a real stream from processed HDF5."""
    try:
        with h5py.File(processed_path, "r") as f:
            return np.array(f[f"streams/{stream_name}/members/phi1"])
    except Exception as e:
        log.warning("Could not load phi1 for %s: %s", stream_name, e)
        return None


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "config" / "streams.yaml") as f:
        streams_cfg = yaml.safe_load(f)
    stream_names = list(streams_cfg["streams"].keys())

    # ==========================================
    # 1. Model comparison report
    # ==========================================
    ev_path = output_dir / "log_evidences.csv"
    if not ev_path.exists():
        log.error("No log_evidences.csv found — run run_inference.py first")
        return

    ev_df = pd.read_csv(ev_path, index_col=0)
    log.info("Log evidence matrix:\n%s", ev_df.to_string())

    per_stream_log_evidences = {}
    for stream in ev_df.index:
        per_stream_log_evidences[stream] = {
            model: float(ev_df.loc[stream, model])
            for model in DM_MODELS if model in ev_df.columns
        }

    report = model_comparison_report(
        per_stream_log_evidences,
        config_path=str(ROOT / "config" / "dm_models.yaml"),
        output_path=str(output_dir / "model_comparison.csv"),
    )
    log.info("Model comparison report:\n%s", report.to_string())

    try:
        fig = plot_model_comparison(report, output_path=figures_dir / "model_comparison.pdf")
        plt.close(fig)
        log.info("Model comparison figure saved")
    except Exception as e:
        log.warning("Model comparison plot failed: %s", e)

    # ==========================================
    # 2. Gap catalog for all streams
    # ==========================================
    processed_path = str(ROOT / args.processed_path)
    all_gaps_dfs = []

    for stream_name in stream_names:
        scfg = streams_cfg["streams"][stream_name]
        phi1_range = tuple(scfg["phi1_range_deg"])

        phi1 = load_stream_phi1(stream_name, processed_path)
        if phi1 is None or len(phi1) < 20:
            log.info("Skipping gap detection for %s (too few stars or no data)", stream_name)
            continue

        gaps = detect_gaps(
            phi1,
            n_bins=min(200, max(50, len(phi1) // 5)),
            phi1_range=phi1_range,
            smooth_sigma=3.0,
            gap_sigma=2.0,
        )

        # Load CDM posterior for gap classification (CDM is the baseline)
        cdm_samples_path = output_dir / stream_name / "samples_CDM.csv"
        dm_samples = None
        if cdm_samples_path.exists():
            dm_samples = pd.read_csv(cdm_samples_path)

        gaps = classify_gaps(
            gaps,
            dm_posterior_samples=dm_samples,
            baryonic_posterior_samples=None,  # baryonic model not separately trained
        )

        gaps_df = gaps_to_dataframe(gaps, stream_name)
        all_gaps_dfs.append(gaps_df)
        log.info("%s: %d gaps detected", stream_name, len(gaps))

        # Density profile figure
        try:
            fig = plot_density_profile(
                phi1, gaps=gaps, stream_name=scfg.get("display_name", stream_name),
                phi1_range=phi1_range,
                output_path=figures_dir / f"density_{stream_name}.pdf",
            )
            plt.close(fig)
        except Exception as e:
            log.warning("Density profile plot failed for %s: %s", stream_name, e)

    if all_gaps_dfs:
        gap_catalog = pd.concat(all_gaps_dfs, ignore_index=True)
        gap_catalog.to_csv(output_dir / "gap_catalog.csv", index=False)
        log.info("Gap catalog saved: %d gaps across %d streams",
                 len(gap_catalog), gap_catalog["stream"].nunique())
    else:
        log.warning("No gaps detected in any stream")

    # ==========================================
    # 3. Combined posteriors per DM model
    # ==========================================
    for dm_model in DM_MODELS:
        per_stream_samples = {}
        for stream in stream_names:
            csv_path = output_dir / stream / f"samples_{dm_model}.csv"
            if csv_path.exists():
                per_stream_samples[stream] = pd.read_csv(csv_path)

        if len(per_stream_samples) < 2:
            continue

        param_names = list(next(iter(per_stream_samples.values())).columns)

        try:
            combined_result = combine_posteriors_product(
                list(per_stream_samples.values()),
                param_names,
            )

            # Find which parameter is the "mass" parameter for this model
            mass_param = None
            for p in param_names:
                if p != "n_impacts":
                    mass_param = p
                    break

            if mass_param:
                fig = plot_combined_posteriors(
                    per_stream_samples, combined_result,
                    param_name=mass_param,
                    output_path=figures_dir / f"combined_posterior_{dm_model}.pdf",
                )
                plt.close(fig)
                log.info("Combined posterior plot for %s saved", dm_model)
        except Exception as e:
            log.warning("Combined posterior for %s failed: %s", dm_model, e)

    # ==========================================
    # 4. Mass function constraints
    # ==========================================
    # Use GD-1 as the representative stream for mass constraints
    mass_samples = {}
    for dm_model in DM_MODELS:
        csv_path = output_dir / "GD1" / f"samples_{dm_model}.csv"
        if csv_path.exists():
            mass_samples[dm_model] = pd.read_csv(csv_path)

    if mass_samples:
        # Find mass-like parameter for each model
        for dm_model, samples in mass_samples.items():
            mass_param = [c for c in samples.columns if c != "n_impacts"]
            if mass_param:
                try:
                    fig = plot_mass_function_constraints(
                        {dm_model: samples},
                        param_name=mass_param[0],
                        output_path=figures_dir / f"mass_constraints_{dm_model}.pdf",
                    )
                    plt.close(fig)
                except Exception as e:
                    log.warning("Mass function plot failed for %s: %s", dm_model, e)

    # ==========================================
    # 5. Multi-stream consistency plot
    # ==========================================
    # Overlay all streams' n_impacts posteriors under CDM
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        for stream in stream_names:
            csv_path = output_dir / stream / "samples_CDM.csv"
            if csv_path.exists():
                samples = pd.read_csv(csv_path)
                if "n_impacts" in samples.columns:
                    vals = samples["n_impacts"].values
                    bins = np.arange(-0.5, 11.5, 1)
                    ax.hist(vals, bins=bins, alpha=0.4, density=True, label=stream)
        ax.set_xlabel("n_impacts (CDM)")
        ax.set_ylabel("Posterior density")
        ax.set_title("Multi-stream n_impacts consistency (CDM)")
        ax.legend(fontsize=8)
        fig.savefig(figures_dir / "multi_stream_consistency.pdf", bbox_inches="tight")
        plt.close(fig)
        log.info("Multi-stream consistency plot saved")
    except Exception as e:
        log.warning("Multi-stream consistency plot failed: %s", e)

    # ==========================================
    # 6. Power spectrum analysis
    # ==========================================
    log.info("Computing power spectra for all streams ...")
    for stream_name in stream_names:
        scfg = streams_cfg["streams"][stream_name]
        phi1 = load_stream_phi1(stream_name, processed_path)
        if phi1 is None or len(phi1) < 50:
            continue
        try:
            phi1_range = tuple(scfg["phi1_range_deg"])
            density, bin_centers = compute_density_1d(phi1, n_bins=200, phi1_range=phi1_range)
            freqs, power = compute_power_spectrum(density)
            excess = detect_power_excess(power, threshold_sigma=3.0)

            fig = plot_power_spectrum(
                freqs, power, excess_mask=excess.get("excess_mask"),
                stream_name=scfg.get("display_name", stream_name),
                output_path=figures_dir / f"power_spectrum_{stream_name}.pdf",
            )
            plt.close(fig)
            log.info("Power spectrum for %s: %d excess frequencies",
                     stream_name, int(np.sum(excess.get("excess_mask", []))))
        except Exception as e:
            log.warning("Power spectrum analysis failed for %s: %s", stream_name, e)

    # ==========================================
    # 7. Suppression scale constraint (PRIMARY RESULT)
    # ==========================================
    log.info("Computing half-mode mass constraint ...")
    try:
        # Collect n_impacts posteriors from all primary streams
        n_impacts_posteriors = {}
        for stream_name in stream_names:
            tier = streams_cfg["streams"][stream_name].get("analysis_tier", "secondary")
            if tier != "primary":
                continue
            cdm_path = output_dir / stream_name / "samples_CDM.csv"
            if cdm_path.exists():
                samples = pd.read_csv(cdm_path)
                if "n_impacts" in samples.columns:
                    n_impacts_posteriors[stream_name] = samples["n_impacts"].values

        if n_impacts_posteriors:
            # Use first primary stream for the main constraint
            # (multi-stream combination happens inside infer_suppression_from_counts)
            all_n_impacts = np.concatenate(list(n_impacts_posteriors.values()))

            log10_M_hm_posterior = infer_suppression_from_counts(
                all_n_impacts,
                n_expected_cdm=5.0,  # expected CDM subhalo impacts for primary streams
            )

            ul_95 = compute_upper_limit(log10_M_hm_posterior, confidence=0.95)

            # Particle mass limits
            M_hm_95 = 10.0 ** ul_95
            particle_limits = {
                "m_wdm_keV_lower": M_hm_to_wdm_mass(M_hm_95),
                "m_axion_eV_lower": M_hm_to_fdm_mass(M_hm_95),
                "sigma_m_upper": M_hm_to_sidm_cross_section(M_hm_95),
            }

            fig = plot_suppression_constraint(
                log10_M_hm_posterior,
                upper_limit_95=ul_95,
                particle_mass_limits=particle_limits,
                output_path=figures_dir / "suppression_constraint.pdf",
            )
            plt.close(fig)

            log.info("Suppression constraint: 95%% UL log10(M_hm) = %.2f", ul_95)
            log.info("  WDM mass lower limit: %.2f keV", particle_limits["m_wdm_keV_lower"])
            log.info("  FDM mass lower limit: %.2e eV", particle_limits["m_axion_eV_lower"])
            log.info("  SIDM cross-section upper limit: %.2f cm^2/g", particle_limits["sigma_m_upper"])

            # Save constraint
            constraint_df = pd.DataFrame([{
                "log10_M_hm_median": float(np.median(log10_M_hm_posterior)),
                "log10_M_hm_95_UL": ul_95,
                "m_wdm_keV_lower": particle_limits["m_wdm_keV_lower"],
                "m_axion_eV_lower": particle_limits["m_axion_eV_lower"],
                "sigma_m_upper": particle_limits["sigma_m_upper"],
                "n_primary_streams": len(n_impacts_posteriors),
            }])
            constraint_df.to_csv(output_dir / "suppression_constraint.csv", index=False)
        else:
            log.warning("No primary stream posteriors found — skipping suppression constraint")
    except Exception as e:
        log.warning("Suppression scale analysis failed: %s", e)

    # ==========================================
    # 8. Summary statistics
    # ==========================================
    log.info("=" * 60)
    log.info("ANALYSIS SUMMARY")
    log.info("=" * 60)
    log.info("Streams analysed: %s", stream_names)
    log.info("DM models: %s", DM_MODELS)
    if all_gaps_dfs:
        log.info("Total gaps detected: %d", len(gap_catalog))
        dm_gaps = gap_catalog[gap_catalog["p_dm_subhalo"] > 0.5]
        log.info("  DM-candidate gaps (P>0.5): %d", len(dm_gaps))
    log.info("Preferred model (combined): %s",
             report.iloc[0]["model"] if len(report) > 0 else "N/A")
    log.info("Outputs saved to: %s", output_dir)
    log.info("Figures saved to: %s", figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analysis outputs.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--processed-path", default="data/processed/streams.h5")
    main(parser.parse_args())
