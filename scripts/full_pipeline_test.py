#!/usr/bin/env python
"""
Full pipeline test run — every stage individually timed and reported.

Runs on Pal5 (smallest stream, 391 raw members) in fast mode to keep
wall-clock time reasonable.
"""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline_test")

from src.forward_model.pipeline import (
    ForwardModelConfig,
    TimelineForwardModel,
)
from src.forward_model.scoring import ScoreWeights

STREAM = "Pal5"
timings = {}


def banner(title):
    log.info("")
    log.info("=" * 70)
    log.info("  %s", title)
    log.info("=" * 70)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
banner("STEP 0: CONFIGURATION")
t0 = time.perf_counter()

cfg = ForwardModelConfig(
    stream_name=STREAM,
    config_path="config/streams.yaml",
    processed_h5_path="data/processed/streams.h5",
    use_fast_mode=True,
    n_stars_sim=2000,
    use_gnn_scorer=True,
    gnn_checkpoint="checkpoints/gnn_best.pt",
    log10_mass_range=(6.0, 8.5),
    log10_mass_step=0.5,
    t_since_range=(0.5, 5.0),
    t_since_step=1.0,
    n_workers=1,
    output_dir="outputs/forward_model",
    top_k=10,
)

timings["config"] = time.perf_counter() - t0
log.info("  Stream:     %s", cfg.stream_name)
log.info("  Mode:       FAST (impulse approximation)")
log.info("  Stars/sim:  %d", cfg.n_stars_sim)
log.info("  Mass range: log10(M) = [%.1f, %.1f] step %.1f",
         cfg.log10_mass_range[0], cfg.log10_mass_range[1], cfg.log10_mass_step)
log.info("  Time range: [%.1f, %.1f] Gyr step %.1f",
         cfg.t_since_range[0], cfg.t_since_range[1], cfg.t_since_step)
log.info("  GNN scorer: %s", "enabled" if cfg.use_gnn_scorer else "disabled")
log.info("  Config time: %.3f s", timings["config"])


# -----------------------------------------------------------------------
# Step 1: Prepare (load data + baseline stream + null hypothesis)
# -----------------------------------------------------------------------
banner("STEP 1: PREPARE (load data, baseline stream, null hypothesis)")
t0 = time.perf_counter()

model = TimelineForwardModel(cfg)
model.prepare()

timings["prepare"] = time.perf_counter() - t0
log.info("")
log.info("  RESULTS:")
log.info("    Observed members:    %d stars", len(model.obs_particles["phi1"]))
log.info("    phi1 range:          [%.1f, %.1f] deg", model.phi1_range[0], model.phi1_range[1])
log.info("    Observed profile:    %d bins", model.obs_profile.n_bins)
log.info("    Observed gaps:       %d detected", len(model.obs_gaps))
for g in model.obs_gaps:
    log.info("      phi1=%.1f, depth=%.2f, width=%.1f, sig=%.1f",
             g.phi1_center, g.depth, g.phi1_width, g.significance)
log.info("    Baseline stream:     %d stars", len(model.base_stream.phi1))
log.info("    GNN scorer:          %s",
         "loaded" if model.gnn_scorer and model.gnn_scorer.is_available else "not available")
log.info("    Null hypothesis:     combined=%.4f", model.null_score.combined)
log.info("      density_residual:  %.4f", model.null_score.density_residual)
log.info("      gap_agreement:     %.4f", model.null_score.gap_agreement)
log.info("      kinematic:         %.4f", model.null_score.kinematic_perturbation)
log.info("      profile_distance:  %.4f", model.null_score.profile_distance)
log.info("    Prepare time:        %.2f s", timings["prepare"])


# -----------------------------------------------------------------------
# Step 2: Single candidate evaluation
# -----------------------------------------------------------------------
banner("STEP 2: SINGLE CANDIDATE EVALUATION")
t0 = time.perf_counter()

test_params = {"log10_mass": 7.5, "t_since_gyr": 2.0, "impact_phi1": -5.0}
single_result = model.evaluate_candidate(test_params)

timings["single_candidate"] = time.perf_counter() - t0
log.info("")
log.info("  RESULTS:")
log.info("    Candidate: log10(M)=%.1f, t=%.1f Gyr, phi1=%.1f",
         test_params["log10_mass"], test_params["t_since_gyr"], test_params["impact_phi1"])
log.info("    combined_score:      %.4f", single_result.score.combined)
log.info("    density_residual:    %.4f", single_result.score.density_residual)
log.info("    gap_agreement:       %.4f", single_result.score.gap_agreement)
log.info("    kinematic:           %.4f", single_result.score.kinematic_perturbation)
log.info("    profile_distance:    %.4f", single_result.score.profile_distance)
log.info("    n_stars_sim:         %d", single_result.n_stars_sim)
vs_null = model.null_score.combined - single_result.score.combined
pct = 100.0 * vs_null / max(model.null_score.combined, 1e-8)
log.info("    vs null:             %+.4f (%+.2f%%)", vs_null, pct)
log.info("    Eval time:           %.4f s", timings["single_candidate"])


# -----------------------------------------------------------------------
# Step 3: Grid evaluation
# -----------------------------------------------------------------------
banner("STEP 3: GRID EVALUATION")
t0 = time.perf_counter()

grid = model.build_parameter_grid()
log.info("  Grid size: %d candidates", len(grid))

results = model.run_grid()

timings["grid"] = time.perf_counter() - t0
n_better = sum(1 for r in results if r.score.combined < model.null_score.combined)
log.info("")
log.info("  RESULTS:")
log.info("    Candidates evaluated: %d", len(results))
log.info("    Better than null:     %d / %d (%.1f%%)",
         n_better, len(results), 100.0 * n_better / len(results))
log.info("    Best candidate:")
best = results[0]
log.info("      log10(M)=%.2f, t=%.1f Gyr, phi1=%.1f",
         best.log10_mass, best.t_since_gyr, best.impact_phi1)
log.info("      combined=%.4f (%.2f%% better than null)",
         best.score.combined,
         100.0 * (model.null_score.combined - best.score.combined) / max(model.null_score.combined, 1e-8))
log.info("    Worst candidate:")
worst = results[-1]
log.info("      log10(M)=%.2f, t=%.1f Gyr, phi1=%.1f, combined=%.4f",
         worst.log10_mass, worst.t_since_gyr, worst.impact_phi1, worst.score.combined)
log.info("    Top-5:")
for i, r in enumerate(results[:5]):
    log.info("      %d. M=10^%.2f, t=%.1f, phi1=%.1f -> %.4f",
             i + 1, r.log10_mass, r.t_since_gyr, r.impact_phi1, r.score.combined)
log.info("    Grid time:           %.2f s (%.1f cand/s)",
         timings["grid"], len(results) / timings["grid"])


# -----------------------------------------------------------------------
# Step 4: Grid refinement
# -----------------------------------------------------------------------
banner("STEP 4: ADAPTIVE GRID REFINEMENT (top-3)")
t0 = time.perf_counter()

refined = model.refine_top_candidates(results, n_top=3)

timings["refine"] = time.perf_counter() - t0
n_new = len(refined) - len(results)
log.info("")
log.info("  RESULTS:")
log.info("    Coarse candidates:   %d", len(results))
log.info("    New fine candidates:  %d", n_new)
log.info("    Total candidates:    %d", len(refined))
old_best = results[0].score.combined
new_best = refined[0].score.combined
if new_best < old_best:
    log.info("    Refinement improved:  %.4f -> %.4f (%.3f%% gain)",
             old_best, new_best, 100.0 * (old_best - new_best) / max(old_best, 1e-8))
else:
    log.info("    Refinement:           no improvement (coarse peak is sharp)")
log.info("    Refined best:")
rb = refined[0]
log.info("      log10(M)=%.3f, t=%.2f Gyr, phi1=%.2f, combined=%.4f",
         rb.log10_mass, rb.t_since_gyr, rb.impact_phi1, rb.score.combined)
log.info("    Refine time:         %.2f s", timings["refine"])


# -----------------------------------------------------------------------
# Step 5: Multi-encounter evaluation
# -----------------------------------------------------------------------
banner("STEP 5: MULTI-ENCOUNTER (2 sequential impacts)")
t0 = time.perf_counter()

enc_list = [
    {"log10_mass": 7.0, "t_since_gyr": 4.0, "impact_phi1": -8.0},
    {"log10_mass": 8.0, "t_since_gyr": 1.5, "impact_phi1": 3.0},
]
multi_single = model.evaluate_multi_encounter(enc_list)

timings["multi_single"] = time.perf_counter() - t0
log.info("")
log.info("  RESULTS:")
log.info("    Encounters:")
for e in multi_single.encounters:
    log.info("      M=10^%.1f, t=%.1f Gyr, phi1=%.1f",
             e["log10_mass"], e["t_since_gyr"], e["impact_phi1"])
log.info("    combined_score:      %.4f", multi_single.score.combined)
log.info("    density_residual:    %.4f", multi_single.score.density_residual)
log.info("    gap_agreement:       %.4f", multi_single.score.gap_agreement)
log.info("    kinematic:           %.4f", multi_single.score.kinematic_perturbation)
vs_null = model.null_score.combined - multi_single.score.combined
pct = 100.0 * vs_null / max(model.null_score.combined, 1e-8)
log.info("    vs null:             %+.4f (%+.2f%%)", vs_null, pct)
log.info("    Eval time:           %.4f s", timings["multi_single"])


# -----------------------------------------------------------------------
# Step 6: Multi-encounter grid sampling
# -----------------------------------------------------------------------
banner("STEP 6: MULTI-ENCOUNTER GRID (20 random 2-encounter samples)")
t0 = time.perf_counter()

multi_results = model.run_multi_encounter_grid(
    n_encounters=2,
    n_random_samples=20,
    seed=42,
)

timings["multi_grid"] = time.perf_counter() - t0
n_better_multi = sum(1 for r in multi_results if r.score.combined < model.null_score.combined)
log.info("")
log.info("  RESULTS:")
log.info("    Samples evaluated:   %d", len(multi_results))
log.info("    Better than null:    %d / %d", n_better_multi, len(multi_results))
if multi_results:
    mb = multi_results[0]
    log.info("    Best multi-encounter:")
    for e in mb.encounters:
        log.info("      M=10^%.2f, t=%.1f Gyr, phi1=%.1f",
                 e["log10_mass"], e["t_since_gyr"], e["impact_phi1"])
    log.info("      combined=%.4f", mb.score.combined)
    best_single = refined[0].score.combined
    if mb.score.combined < best_single:
        log.info("    Multi BEATS single:  %.4f < %.4f", mb.score.combined, best_single)
    else:
        log.info("    Single still best:   %.4f < %.4f", best_single, mb.score.combined)
log.info("    Multi-grid time:     %.2f s", timings["multi_grid"])


# -----------------------------------------------------------------------
# Step 7: Save results
# -----------------------------------------------------------------------
banner("STEP 7: SAVE RESULTS")
t0 = time.perf_counter()

out_path = model.save_results(refined)
multi_path = model.save_multi_encounter_results(multi_results)

timings["save"] = time.perf_counter() - t0
log.info("")
log.info("  RESULTS:")
log.info("    Single-encounter:    %s", out_path)
log.info("    Multi-encounter:     %s", multi_path)

# Verify JSON is valid
with open(out_path) as f:
    data = json.load(f)
log.info("    JSON keys:           %s", list(data.keys()))
log.info("    top_candidates:      %d entries", len(data["top_candidates"]))
log.info("    null_hypothesis:     %s", data.get("null_hypothesis"))
log.info("    Save time:           %.3f s", timings["save"])


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
banner("SUMMARY")
total = sum(timings.values())
log.info("  Stream: %s (%d observed members after filtering)", STREAM, len(model.obs_particles["phi1"]))
log.info("")
log.info("  Step                       Time (s)     Status")
log.info("  %-27s %8.3f     %s", "0. Configuration", timings["config"], "OK")
log.info("  %-27s %8.3f     %s", "1. Prepare", timings["prepare"],
         f"{len(model.obs_gaps)} gaps, null={model.null_score.combined:.4f}")
log.info("  %-27s %8.3f     %s", "2. Single candidate", timings["single_candidate"],
         f"score={single_result.score.combined:.4f}")
log.info("  %-27s %8.3f     %s", "3. Grid evaluation", timings["grid"],
         f"{len(results)} cand, {n_better}/{len(results)} beat null")
log.info("  %-27s %8.3f     %s", "4. Refinement", timings["refine"],
         f"+{n_new} fine, best={refined[0].score.combined:.4f}")
log.info("  %-27s %8.3f     %s", "5. Multi-encounter (single)", timings["multi_single"],
         f"score={multi_single.score.combined:.4f}")
log.info("  %-27s %8.3f     %s", "6. Multi-encounter grid", timings["multi_grid"],
         f"{len(multi_results)} samples, {n_better_multi} beat null")
log.info("  %-27s %8.3f     %s", "7. Save results", timings["save"], "2 JSON files")
log.info("  %-27s %8.3f", "TOTAL", total)
log.info("")
log.info("  Null hypothesis score:  %.4f", model.null_score.combined)
log.info("  Best single-encounter:  %.4f (%.2f%% better)",
         refined[0].score.combined,
         100.0 * (model.null_score.combined - refined[0].score.combined) / max(model.null_score.combined, 1e-8))
if multi_results:
    log.info("  Best multi-encounter:   %.4f (%.2f%% better)",
             multi_results[0].score.combined,
             100.0 * (model.null_score.combined - multi_results[0].score.combined) / max(model.null_score.combined, 1e-8))
log.info("")
log.info("  ALL STEPS COMPLETED SUCCESSFULLY")
