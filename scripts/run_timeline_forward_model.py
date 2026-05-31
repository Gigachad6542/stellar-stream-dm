#!/usr/bin/env python
"""
Timeline Forward Model runner.

Evaluates forced subhalo encounters against real stream data to find
the encounter parameters that best reproduce the observed morphology.

Usage:
    python scripts/run_timeline_forward_model.py --stream GD1
    python scripts/run_timeline_forward_model.py --stream GD1 --phi1 -40 -20
    python scripts/run_timeline_forward_model.py --stream GD1 --log10-mass-range 6.5 8.0 --t-range 1.0 5.0
    python scripts/run_timeline_forward_model.py --stream GD1 --quick  # fast smoke test

Output:
    outputs/forward_model/{stream}/forward_model_results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Timeline forward model: find encounter params that reproduce observed gaps."
    )
    p.add_argument("--stream", type=str, default="GD1",
                   help="Stream name (must match config/streams.yaml key)")
    p.add_argument("--config", type=str, default="config/streams.yaml",
                   help="Path to streams config YAML")
    p.add_argument("--h5", type=str, default="data/processed/streams.h5",
                   help="Path to processed real-stream HDF5")
    p.add_argument("--output-dir", type=str, default="outputs/forward_model",
                   help="Output directory for results")

    # Grid parameters
    p.add_argument("--log10-mass-range", type=float, nargs=2, default=[6.0, 8.5],
                   metavar=("MIN", "MAX"),
                   help="log10(M_sub/Msun) range")
    p.add_argument("--log10-mass-step", type=float, default=0.25,
                   help="Step size in log10(M)")
    p.add_argument("--t-range", type=float, nargs=2, default=[0.5, 10.0],
                   metavar=("MIN", "MAX"),
                   help="t_since_impact [Gyr] range")
    p.add_argument("--t-step", type=float, default=0.5,
                   help="Step size in t_since [Gyr]")
    p.add_argument("--phi1", type=float, nargs="+", default=None,
                   help="Impact phi1 positions [deg]. Default: known gap locations from config.")

    # Encounter geometry
    p.add_argument("--flyby-vel", type=float, default=200.0,
                   help="Flyby velocity [km/s]")
    p.add_argument("--impact-param", type=float, default=0.1,
                   help="Impact parameter [kpc]")

    # Simulation
    p.add_argument("--n-stars", type=int, default=5000,
                   help="Number of stars in simulated stream")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for stream generation")

    # Observation filtering
    p.add_argument("--phi2-cut", type=float, default=1.0,
                   help="Half-width phi2 cut for stream member selection [deg]")
    p.add_argument("--membership-min", type=float, default=0.5,
                   help="Minimum membership probability to keep a star")

    # Scoring
    p.add_argument("--use-gnn-scorer", action="store_true",
                   help="Include the GNN embedding distance in the combined score. OFF by default: "
                        "on real data the encoder has a large sim-to-real gap (embedding distance "
                        "~40 vs ~1 sim-to-sim), so it dominates and corrupts the score. Enable only "
                        "after domain-randomization retraining closes the gap.")
    p.add_argument("--density-bin-width", type=float, default=1.0,
                   help="Density profile bin width [deg]")
    p.add_argument("--kin-bin-width", type=float, default=2.0,
                   help="Kinematic comparison bin width [deg]")
    p.add_argument("--gap-min-depth", type=float, default=0.3,
                   help="Minimum gap depth for detection")
    p.add_argument("--gap-min-sig", type=float, default=2.0,
                   help="Minimum gap significance for detection")

    # Evolution mode
    p.add_argument("--fast", action="store_true",
                   help="Fast mode: use impulse approximation instead of full orbit integration. "
                        "~500x faster but less physically accurate.")
    p.add_argument("--n-workers", type=int, default=1,
                   help="Number of parallel worker processes. >1 uses multiprocessing for "
                        "embarrassingly parallel grid evaluation. Default: 1 (sequential).")

    # Statistical significance
    p.add_argument("--significance", type=int, default=0, metavar="N",
                   help="Build a null distribution of N unperturbed realizations and report the "
                        "best candidate's significance (z-score + empirical p-value).")
    p.add_argument("--n-seeds", type=int, default=1,
                   help="Evaluate the top candidates over this many base-stream seeds and rank by "
                        "the seed-averaged score (removes sampling noise). Default 1 (no averaging).")
    p.add_argument("--significance-seed0", type=int, default=1000,
                   help="First seed for the null-distribution realizations.")

    # Refinement
    p.add_argument("--refine", action="store_true",
                   help="After coarse grid, refine top candidates with finer sub-grid")
    p.add_argument("--refine-top", type=int, default=5,
                   help="Number of top candidates to refine around (default: 5)")

    # Multi-encounter
    p.add_argument("--multi-encounter", type=int, default=0, metavar="N",
                   help="Run multi-encounter evaluation with N sequential impacts. "
                        "Uses random sampling of encounter configurations.")
    p.add_argument("--multi-samples", type=int, default=100,
                   help="Number of random multi-encounter samples to evaluate (default: 100)")

    # Model comparison
    p.add_argument("--compare-models", action="store_true",
                   help="After grid, evaluate top candidates under all 4 DM models "
                        "(CDM, WDM, FDM, SIDM) and report which best matches observations.")
    p.add_argument("--compare-top", type=int, default=5,
                   help="Number of top candidates to compare across DM models (default: 5)")

    # Detection -> timeline handoff (auto-seed the grid from the data)
    p.add_argument("--auto-detect", action="store_true",
                   help="Run the detection front-end on the real stream first, then seed the "
                        "forward-model grid: phi1 from detected gaps (or in-frame quartiles), "
                        "and the time window from the GNN time-since-impact estimate.")
    p.add_argument("--detector-checkpoint", type=str, default=None,
                   help="Path to a trained timeline GNN checkpoint for P(impact) + time estimate. "
                        "If omitted, detection is model-free (gap-based localisation only).")
    p.add_argument("--t-window", type=float, default=2.0,
                   help="Half-width [Gyr] of the time grid around the GNN time estimate "
                        "when --auto-detect is set (default: 2.0).")
    p.add_argument("--max-phi1-seeds", type=int, default=3,
                   help="Max number of phi1 positions to seed from detection (default: 3).")
    p.add_argument("--detect-sig-threshold", type=float, default=3.0,
                   help="Gap significance at/above which a model-free detection counts (default: 3.0).")
    p.add_argument("--require-detection", action="store_true",
                   help="With --auto-detect, exit without running the grid if no probable impact "
                        "is found (neither a significant gap nor a GNN detection).")

    # Convenience
    p.add_argument("--quick", action="store_true",
                   help="Quick smoke test: small grid (3 masses x 3 times), implies --fast")
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of top candidates to save in detail")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose (DEBUG) logging")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import after path setup
    import yaml
    from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
    from src.forward_model.scoring import ScoreWeights

    # Resolve impact_phi1 values
    if args.phi1 is not None:
        phi1_values = args.phi1
    else:
        # Default: use known gap locations from stream config, but only if they
        # fall within the actual data extent. The config known_gaps may use a
        # different phi1 convention than the stored data.
        import h5py
        import numpy as np

        with open(args.config) as f:
            streams_cfg = yaml.safe_load(f)
        s_cfg = streams_cfg["streams"][args.stream]

        # Determine actual data phi1 range
        with h5py.File(args.h5, "r") as f:
            obs_phi1 = f[f"streams/{args.stream}/members/phi1"][:]
        data_p5, data_p95 = float(np.percentile(obs_phi1, 5)), float(np.percentile(obs_phi1, 95))

        known = s_cfg.get("known_gaps", [])
        phi1_values = [
            g["phi1_center"] for g in known
            if g.get("candidate_dm", False)
            and data_p5 <= g["phi1_center"] <= data_p95
        ]

        if not phi1_values:
            # Gaps are outside data extent or none marked as DM candidates.
            # Place test impacts at 1/4 and 1/2 of the data range.
            q25 = float(np.percentile(obs_phi1, 25))
            q50 = float(np.percentile(obs_phi1, 50))
            phi1_values = [q25, q50]
            logging.getLogger(__name__).info(
                "No DM-candidate gaps within data extent for %s; "
                "using phi1=[%.1f, %.1f] (25th, 50th percentile)",
                args.stream, q25, q50,
            )

    # Quick mode overrides (implies fast mode)
    if args.quick:
        args.log10_mass_range = [6.5, 7.5]
        args.log10_mass_step = 0.5
        args.t_range = [1.0, 3.0]
        args.t_step = 1.0
        args.n_stars = 2000
        args.fast = True
        logging.getLogger(__name__).info("Quick mode: small grid + fast (impulse) mode")

    # Build config
    config = ForwardModelConfig(
        stream_name=args.stream,
        config_path=args.config,
        processed_h5_path=args.h5,
        log10_mass_range=tuple(args.log10_mass_range),
        log10_mass_step=args.log10_mass_step,
        t_since_range=tuple(args.t_range),
        t_since_step=args.t_step,
        impact_phi1_values=phi1_values,
        flyby_vel_kms=args.flyby_vel,
        impact_param_kpc=args.impact_param,
        n_stars_sim=args.n_stars,
        base_seed=args.seed,
        phi2_cut_deg=args.phi2_cut,
        membership_prob_min=args.membership_min,
        density_bin_width_deg=args.density_bin_width,
        kinematic_bin_width_deg=args.kin_bin_width,
        gap_detection_min_depth=args.gap_min_depth,
        gap_detection_min_significance=args.gap_min_sig,
        score_weights=ScoreWeights(),
        use_gnn_scorer=args.use_gnn_scorer,
        use_fast_mode=args.fast,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

    # Run
    log = logging.getLogger("forward_model")
    mode_str = "FAST (impulse approximation)" if config.use_fast_mode else "FULL (orbit-integrated evolution)"
    log.info("=" * 70)
    log.info("Timeline Forward Model")
    log.info("  Stream: %s", config.stream_name)
    log.info("  Mode: %s", mode_str)
    log.info("  Mass range: log10(M) = [%.2f, %.2f] step %.2f",
             config.log10_mass_range[0], config.log10_mass_range[1], config.log10_mass_step)
    log.info("  Time range: [%.1f, %.1f] Gyr step %.1f",
             config.t_since_range[0], config.t_since_range[1], config.t_since_step)
    log.info("  Impact phi1: %s", config.impact_phi1_values)
    log.info("=" * 70)

    t_start = time.perf_counter()

    model = TimelineForwardModel(config)
    model.prepare()

    # --- Detection -> timeline handoff (optional) ---
    # Run AFTER prepare() (which loads the observed data + phi1 range) but BEFORE
    # run_grid() (which builds the grid from config). The base stream and null
    # hypothesis do not depend on the grid parameters, so seeding the config in
    # place here requires no recomputation.
    if args.auto_detect:
        import json

        from src.forward_model.detection import (
            StreamImpactDetector,
            detect_impacts_modelfree,
            seed_config_from_detection,
        )

        log.info("")
        log.info("=" * 70)
        log.info("DETECTION FRONT-END (auto-seeding the grid)")
        log.info("=" * 70)

        if args.detector_checkpoint:
            detector = StreamImpactDetector(args.detector_checkpoint, device="cpu")
            detection = detector.detect(
                model.obs_particles, model.phi1_range, stream_name=config.stream_name,
                density_bin_width_deg=config.density_bin_width_deg,
                gap_min_depth=config.gap_detection_min_depth,
                gap_min_significance=config.gap_detection_min_significance,
                significant_threshold=args.detect_sig_threshold,
            )
        else:
            log.info("No detector checkpoint given; using model-free gap detection only.")
            detection = detect_impacts_modelfree(
                model.obs_particles, model.phi1_range, stream_name=config.stream_name,
                density_bin_width_deg=config.density_bin_width_deg,
                gap_min_depth=config.gap_detection_min_depth,
                gap_min_significance=config.gap_detection_min_significance,
                significant_threshold=args.detect_sig_threshold,
            )

        log.info("Detection: impact_detected=%s | %s",
                 detection.impact_detected, detection.detection_reason)
        if detection.gnn_available:
            log.info("  GNN: p_impact=%.3f, t_since~%s Gyr, effective_n=%.2f",
                     detection.p_impact,
                     f"{detection.t_since_gyr_estimate:.2f}" if detection.t_since_gyr_estimate else "n/a",
                     detection.effective_n_impacts or 0.0)
            if detection.ood_max_sigma is not None:
                flag = " [OUT-OF-DISTRIBUTION: treat p_impact as unreliable]" if \
                    detection.ood_max_sigma >= 5.0 else ""
                log.info("  GNN input OOD check: max %.1f sigma, %.1f%% clipped%s",
                         detection.ood_max_sigma, 100.0 * (detection.ood_frac_clipped or 0.0), flag)

        # Persist the detection result alongside the forward-model outputs.
        det_dir = Path(config.output_dir) / config.stream_name
        det_dir.mkdir(parents=True, exist_ok=True)
        det_path = det_dir / "detection_result.json"
        with open(det_path, "w") as f:
            json.dump(detection.to_dict(), f, indent=2)
        log.info("  Detection result saved to %s", det_path)

        if args.require_detection and not detection.impact_detected:
            log.info("No probable impact detected and --require-detection set; exiting "
                     "before grid evaluation.")
            return

        # Stream disruption age caps the physically-allowed time since impact.
        stream_age = model.stream_config.get(
            "disruption_age_gyr", model.stream_config.get("isochrone_age_gyr"))

        # Seed the grid: replace the config with a narrowed one.
        config = seed_config_from_detection(
            detection, config,
            t_window_gyr=args.t_window,
            max_phi1_seeds=args.max_phi1_seeds,
            min_t_since_gyr=config.t_since_range[0],
            max_t_since_gyr=config.t_since_range[1],
            stream_age_gyr=stream_age,
        )
        model.cfg = config
        log.info("Seeded grid: phi1=%s, t_since_range=%s",
                 [f"{p:.1f}" for p in config.impact_phi1_values], config.t_since_range)
        log.info("=" * 70)

    results = model.run_grid()

    # Optional refinement pass
    if args.refine:
        log.info("")
        log.info("=" * 70)
        log.info("REFINEMENT PASS")
        log.info("=" * 70)
        results = model.refine_top_candidates(results, n_top=args.refine_top)

    # --- Multi-seed re-ranking of the top candidates (remove sampling noise) ---
    if args.n_seeds > 1:
        log.info("")
        log.info("=" * 70)
        log.info("MULTI-SEED RE-RANKING (top %d candidates x %d seeds)", args.refine_top, args.n_seeds)
        log.info("=" * 70)
        seeds = [config.base_seed + 1000 * j for j in range(args.n_seeds)]
        n_top = min(args.refine_top, len(results))
        for r in results[:n_top]:
            mean, std, _ = model.evaluate_candidate_multiseed(
                {"log10_mass": r.log10_mass, "t_since_gyr": r.t_since_gyr,
                 "impact_phi1": r.impact_phi1}, seeds)
            r.score.details["multiseed_mean"] = mean
            r.score.details["multiseed_std"] = std
            r.score.combined = mean   # rank by seed-averaged score
        results[:n_top] = sorted(results[:n_top], key=lambda r: r.score.combined)
        log.info("  Re-ranked best: log10_M=%.2f t=%.1f phi1=%.1f  score=%.4f +/- %.4f",
                 results[0].log10_mass, results[0].t_since_gyr, results[0].impact_phi1,
                 results[0].score.combined, results[0].score.details.get("multiseed_std", 0.0))

    # --- Statistical significance vs a no-impact null distribution ---
    significance = None
    if args.significance > 0:
        import json
        from src.forward_model.significance import compute_significance
        log.info("")
        log.info("=" * 70)
        log.info("SIGNIFICANCE (null distribution of %d unperturbed realizations)", args.significance)
        log.info("=" * 70)
        null_scores = model.build_null_distribution(
            n_realizations=args.significance, seed0=args.significance_seed0)
        significance = compute_significance(results[0].score.combined, null_scores)
        log.info("  Best candidate score = %.4f", significance.candidate_score)
        log.info("  Null: mean=%.4f std=%.4f (n=%d)",
                 significance.null_mean, significance.null_std, significance.n_null)
        log.info("  -> z-score = %.2f sigma, empirical p-value = %.3f",
                 significance.z_score, significance.p_value)
        if significance.p_value > 0.05:
            log.info("  NOT significant: the best fit is within the no-impact scatter "
                     "(p > 0.05). No evidence for a distinct impact.")
        else:
            log.info("  Significant at p=%.3f: the best fit is better than the no-impact "
                     "baseline beyond its scatter.", significance.p_value)
        sig_dir = Path(config.output_dir) / config.stream_name
        sig_dir.mkdir(parents=True, exist_ok=True)
        with open(sig_dir / "significance.json", "w") as f:
            json.dump(significance.to_dict(), f, indent=2)
        log.info("  Significance saved to %s", sig_dir / "significance.json")

    out_path = model.save_results(results)

    total_time = time.perf_counter() - t_start
    log.info("=" * 70)
    log.info("DONE in %.1f s", total_time)
    log.info("Results: %s", out_path)
    log.info("Best candidate:")
    best = results[0]
    log.info("  log10(M) = %.2f, t = %.1f Gyr, phi1 = %.1f deg",
             best.log10_mass, best.t_since_gyr, best.impact_phi1)
    log.info("  combined_score = %.4f", best.score.combined)
    log.info("  density_residual = %.4f", best.score.density_residual)
    log.info("  gap_agreement = %.4f", best.score.gap_agreement)
    log.info("  kinematic_perturbation = %.4f", best.score.kinematic_perturbation)
    log.info("=" * 70)

    # --- DM model comparison (optional) ---
    if args.compare_models:
        log.info("")
        log.info("=" * 70)
        log.info("DM MODEL COMPARISON (top-%d candidates × 4 models)", args.compare_top)
        log.info("=" * 70)

        top_candidates = [
            {"log10_mass": r.log10_mass, "t_since_gyr": r.t_since_gyr,
             "impact_phi1": r.impact_phi1}
            for r in results[:args.compare_top]
        ]
        comparison_results = model.run_model_comparison(candidates=top_candidates)
        comp_path = model.save_model_comparison_results(comparison_results)
        log.info("Model comparison results: %s", comp_path)

    # --- Multi-encounter evaluation (optional) ---
    if args.multi_encounter > 0:
        log.info("")
        log.info("=" * 70)
        log.info("MULTI-ENCOUNTER EVALUATION (%d encounters x %d samples)",
                 args.multi_encounter, args.multi_samples)
        log.info("=" * 70)

        multi_results = model.run_multi_encounter_grid(
            n_encounters=args.multi_encounter,
            n_random_samples=args.multi_samples,
            seed=args.seed,
        )
        multi_path = model.save_multi_encounter_results(multi_results)
        log.info("Multi-encounter results: %s", multi_path)

        if multi_results:
            mbest = multi_results[0]
            log.info("Best multi-encounter (combined=%.4f):", mbest.score.combined)
            for enc in mbest.encounters:
                log.info("  M=10^%.2f, t=%.1f Gyr, phi1=%.1f deg",
                         enc["log10_mass"], enc["t_since_gyr"], enc["impact_phi1"])

            # Compare single vs multi
            single_best = results[0].score.combined
            multi_best = mbest.score.combined
            if multi_best < single_best:
                log.info("  Multi-encounter IMPROVES over single: %.4f vs %.4f (%.1f%%)",
                         multi_best, single_best,
                         100.0 * (single_best - multi_best) / max(single_best, 1e-8))
            else:
                log.info("  Multi-encounter does not improve over single-encounter best")


if __name__ == "__main__":
    main()
