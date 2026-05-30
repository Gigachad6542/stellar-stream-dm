"""
Combine per-stream posteriors into a joint DM model constraint.

Reads the posterior sample CSVs written by run_inference.py and:
1. Combines per-stream posteriors via product (sum of log-posteriors)
2. Computes Bayes factors between DM models
3. Produces the multi-stream consistency plot
4. Saves the final model comparison report

Usage:
    python scripts/combine_posteriors.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.model_comparison import (
    combine_log_evidences_across_streams,
    model_comparison_report,
)
from src.analysis.visualization import plot_combined_posteriors, plot_model_comparison
from src.inference.posteriors import combine_posteriors_product

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DM_MODELS = ["CDM", "WDM", "FDM", "SIDM"]


def main(args) -> None:
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open("config/streams.yaml") as f:
        streams_cfg = yaml.safe_load(f)
    stream_names = list(streams_cfg["streams"].keys())

    # Load log evidences
    ev_path = output_dir / "log_evidences.csv"
    if not ev_path.exists():
        log.error("Log evidence matrix not found at %s. Run run_inference.py first.", ev_path)
        return

    ev_df = pd.read_csv(ev_path, index_col=0)
    log.info("Loaded log evidence matrix:\n%s", ev_df.to_string())

    per_stream_log_evidences = {}
    for stream in ev_df.index:
        per_stream_log_evidences[stream] = {
            model: float(ev_df.loc[stream, model])
            for model in DM_MODELS
            if model in ev_df.columns
        }

    # Combined model comparison
    report = model_comparison_report(
        per_stream_log_evidences,
        output_path=str(output_dir / "model_comparison.csv"),
    )
    log.info("Model comparison report:\n%s", report.to_string())

    fig = plot_model_comparison(report, output_path=figures_dir / "model_comparison.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)

    # Per-parameter combined posteriors
    for dm_model in DM_MODELS:
        per_stream_samples = {}
        for stream in stream_names:
            csv_path = output_dir / stream / f"samples_{dm_model}.csv"
            if csv_path.exists():
                per_stream_samples[stream] = pd.read_csv(csv_path)

        if not per_stream_samples:
            continue

        param_names = list(next(iter(per_stream_samples.values())).columns)
        combined_result = combine_posteriors_product(
            list(per_stream_samples.values()),
            param_names[:3],  # limit to first 3 for tractable grid combination
        )

        if "log10_M_sub_mean" in param_names:
            fig = plot_combined_posteriors(
                per_stream_samples, combined_result,
                param_name="log10_M_sub_mean",
                output_path=figures_dir / f"combined_posterior_{dm_model}.pdf",
            )
            plt.close(fig)

    log.info("Posterior combination complete. Figures saved to %s.", figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine per-stream posteriors.")
    parser.add_argument("--output-dir", default="outputs")
    main(parser.parse_args())
