#!/usr/bin/env python
"""
Production forward-model grid on GD-1.

Runs a comprehensive parameter sweep using the full orbit-integrated
timeline evolution (default) or fast impulse mode (--fast).

Grid: 11 masses x 20 times x 2 phi1 positions = 440 candidates
Estimated time (full mode, no C extension): ~2-4 hours with 4 workers
Estimated time (fast mode): <1 minute

Usage:
    python scripts/run_production_grid_gd1.py                        # full orbit, 4 workers
    python scripts/run_production_grid_gd1.py --fast --n-workers 8   # fast mode, 8 workers
    python scripts/run_production_grid_gd1.py --n-workers 1          # sequential (debugging)

Output:
    outputs/forward_model/GD1/forward_model_results.json
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Production forward-model grid on GD-1")
    p.add_argument("--fast", action="store_true",
                   help="Use fast impulse mode instead of full orbit integration")
    p.add_argument("--n-workers", type=int, default=4,
                   help="Number of parallel workers (default: 4)")
    p.add_argument("--n-stars", type=int, default=3000,
                   help="Stars per simulation (default: 3000)")
    p.add_argument("--refine", action="store_true", default=True,
                   help="Refine top candidates with finer grid (default: True)")
    p.add_argument("--no-refine", dest="refine", action="store_false",
                   help="Skip refinement pass")
    p.add_argument("--refine-top", type=int, default=5,
                   help="Number of top candidates to refine (default: 5)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("production_grid")

    from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
    from src.forward_model.scoring import ScoreWeights

    # Production grid parameters for GD-1
    # Mass range: 10^6 to 10^8.5 Msun (11 values at 0.25 dex)
    # Time range: 0.5 to 10 Gyr (20 values at 0.5 Gyr)
    # phi1: auto-detected from data (known gaps or percentiles)
    #
    # This covers the physically relevant regime:
    #   - M < 10^6: too small to create observable gaps in Gaia
    #   - M > 10^8.5: rare + would create features larger than the stream
    #   - t < 0.5 Gyr: gap hasn't evolved enough; too narrow
    #   - t > 10 Gyr: gap has phase-mixed away; undetectable

    config = ForwardModelConfig(
        stream_name="GD1",
        config_path="config/streams.yaml",
        processed_h5_path="data/processed/streams.h5",

        # Mass grid: log10(M_sub/Msun)
        log10_mass_range=(6.0, 8.5),
        log10_mass_step=0.25,

        # Time grid: Gyr since impact
        t_since_range=(0.5, 10.0),
        t_since_step=0.5,

        # Impact locations: auto-detected from data
        impact_phi1_values=None,  # will be set below

        # Encounter geometry
        flyby_vel_kms=200.0,      # typical dark matter subhalo flyby speed
        impact_param_kpc=0.1,     # close flyby for maximal signal

        # Simulation
        n_stars_sim=args.n_stars,
        base_seed=42,

        # Observation filtering
        phi2_cut_deg=1.0,
        membership_prob_min=0.5,

        # Scoring
        density_bin_width_deg=1.0,
        kinematic_bin_width_deg=2.0,
        gap_detection_min_depth=0.3,
        gap_detection_min_significance=2.0,
        score_weights=ScoreWeights(
            density=1.0,
            gap=1.5,
            kinematic=0.8,
            profile=0.5,
        ),

        # Mode
        use_fast_mode=args.fast,
        n_workers=args.n_workers,

        # Output
        output_dir="outputs/forward_model",
        top_k=20,
    )

    # Resolve phi1 impact positions from data
    import h5py
    import numpy as np
    import yaml

    with open(config.config_path) as f:
        streams_cfg = yaml.safe_load(f)
    s_cfg = streams_cfg["streams"]["GD1"]

    with h5py.File(config.processed_h5_path, "r") as f:
        obs_phi1 = f["streams/GD1/members/phi1"][:]
    data_p5, data_p95 = float(np.percentile(obs_phi1, 5)), float(np.percentile(obs_phi1, 95))

    # Use known gaps if within data extent
    known = s_cfg.get("known_gaps", [])
    phi1_values = [
        g["phi1_center"] for g in known
        if g.get("candidate_dm", False) and data_p5 <= g["phi1_center"] <= data_p95
    ]

    if not phi1_values:
        # Place impacts at 25th and 50th percentile of data
        q25 = float(np.percentile(obs_phi1, 25))
        q50 = float(np.percentile(obs_phi1, 50))
        phi1_values = [q25, q50]
        log.info("No DM-candidate gaps in data extent; using phi1=[%.1f, %.1f]",
                 q25, q50)

    config.impact_phi1_values = phi1_values

    # Compute grid size
    n_masses = int((config.log10_mass_range[1] - config.log10_mass_range[0]) / config.log10_mass_step) + 1
    n_times = int((config.t_since_range[1] - config.t_since_range[0]) / config.t_since_step) + 1
    n_phi1 = len(phi1_values)
    total = n_masses * n_times * n_phi1

    mode_str = "FAST (impulse)" if config.use_fast_mode else "FULL (orbit-integrated)"
    log.info("=" * 70)
    log.info("PRODUCTION GRID: GD-1")
    log.info("  Mode: %s", mode_str)
    log.info("  Grid: %d masses x %d times x %d phi1 = %d candidates",
             n_masses, n_times, n_phi1, total)
    log.info("  Workers: %d", config.n_workers)
    log.info("  N_stars/sim: %d", config.n_stars_sim)
    log.info("=" * 70)

    t_start = time.perf_counter()

    model = TimelineForwardModel(config)
    model.prepare()
    results = model.run_grid()

    # Refinement pass
    if args.refine:
        log.info("")
        log.info("=" * 70)
        log.info("REFINEMENT PASS: zooming in around top-%d candidates", args.refine_top)
        log.info("=" * 70)
        results = model.refine_top_candidates(results, n_top=args.refine_top)

    out_path = model.save_results(results)

    total_time = time.perf_counter() - t_start

    log.info("=" * 70)
    log.info("PRODUCTION GRID COMPLETE")
    log.info("  Total time: %.1f s (%.1f min)", total_time, total_time / 60)
    log.info("  Total candidates evaluated: %d", len(results))
    log.info("  Results: %s", out_path)
    log.info("")
    log.info("  Null hypothesis score: %.4f", model.null_score.combined)
    log.info("")
    log.info("  Top 5 candidates:")
    for i, r in enumerate(results[:5]):
        improvement = model.null_score.combined - r.score.combined
        log.info("    %d. log10(M)=%.3f, t=%.2f Gyr, phi1=%.2f -- "
                 "score=%.4f (%.2f%% better than null)",
                 i + 1, r.log10_mass, r.t_since_gyr, r.impact_phi1,
                 r.score.combined,
                 100.0 * improvement / max(model.null_score.combined, 1e-8))
    log.info("")

    n_better = sum(1 for r in results if r.score.combined < model.null_score.combined)
    log.info("  Candidates better than null: %d / %d (%.1f%%)",
             n_better, len(results), 100.0 * n_better / len(results))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
