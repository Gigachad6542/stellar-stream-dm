"""
Build TARP recalibrators from SBC rank data and validate.

Loads the SBC ranks computed by run_sbc.py, builds per-model recalibrators,
saves them alongside the posteriors, and produces diagnostic plots.

Usage:
    python -u scripts/run_tarp_recalibration.py
    python -u scripts/run_tarp_recalibration.py --sbc-dir outputs/sbc --posterior-dir checkpoints/v2_sbi_profile
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.tarp_calibration import TARPRecalibrator
from src.inference.posteriors import get_param_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DM_MODELS = ["CDM", "WDM", "FDM", "SIDM"]


def main(args: argparse.Namespace) -> None:
    sbc_dir = Path(args.sbc_dir)
    if not sbc_dir.is_absolute():
        sbc_dir = ROOT / sbc_dir
    posterior_dir = Path(args.posterior_dir)
    if not posterior_dir.is_absolute():
        posterior_dir = ROOT / posterior_dir

    all_summaries = {}

    for dm_model in args.dm_models:
        model_sbc_dir = sbc_dir / dm_model
        ranks_path = model_sbc_dir / "sbc_ranks.npy"

        if not ranks_path.exists():
            log.warning("[%s] No SBC ranks found at %s — skipping", dm_model, ranks_path)
            continue

        ranks = np.load(ranks_path)
        log.info("[%s] Loaded SBC ranks: shape %s", dm_model, ranks.shape)

        # Get parameter names
        param_names = get_param_names(
            dm_model,
            str(ROOT / "config" / "dm_models.yaml"),
            theta_mode=args.theta_mode,
        )

        # Build recalibrator
        recal = TARPRecalibrator.from_sbc_ranks(
            ranks,
            n_posterior_samples=args.n_posterior_samples,
            param_names=param_names,
        )

        # Save recalibrator
        recal_path = posterior_dir / f"recalibrator_{dm_model}.pkl"
        recal.save(recal_path)

        # Save coverage plot
        plot_path = model_sbc_dir / "tarp_coverage.pdf"
        recal.plot_coverage(output_path=str(plot_path))

        # Get calibration summary
        summary = recal.calibration_summary()
        all_summaries[dm_model] = summary

        # Report key corrections
        log.info("[%s] Calibration corrections:", dm_model)
        for param_name, param_data in summary.items():
            for level_key, level_data in param_data.items():
                actual = level_data["actual_coverage"]
                correction = level_data["correction_needed"]
                bias = level_data["bias"]
                nominal_pct = int(level_key.split("_")[1])
                log.info(
                    "  %s %d%%: actual=%.1f%%, correction=%.3f, bias=%+.1f%%",
                    param_name, nominal_pct, actual * 100, correction, bias * 100,
                )

    # Save combined summary
    summary_path = sbc_dir / "tarp_calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    log.info("Saved calibration summary to %s", summary_path)

    # Overall assessment
    log.info("=" * 60)
    log.info("TARP RECALIBRATION SUMMARY")
    log.info("=" * 60)
    for dm_model, summary in all_summaries.items():
        for param_name, param_data in summary.items():
            s90 = param_data["nominal_90"]
            log.info(
                "  %s/%s: 90%% CI actually covers %.1f%% -> use %.0f%% nominal for true 90%%",
                dm_model, param_name,
                s90["actual_coverage"] * 100,
                s90["correction_needed"] * 100,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TARP recalibrators from SBC ranks.")
    parser.add_argument(
        "--dm-models", nargs="+", default=DM_MODELS,
        choices=DM_MODELS, metavar="MODEL",
    )
    parser.add_argument("--sbc-dir", default="outputs/sbc")
    parser.add_argument("--posterior-dir", default="checkpoints/v2_sbi_profile")
    parser.add_argument("--theta-mode", default="suppression")
    parser.add_argument("--n-posterior-samples", type=int, default=500,
                        help="Must match the value used in run_sbc.py")
    args = parser.parse_args()
    main(args)
