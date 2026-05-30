#!/usr/bin/env python
"""Functional test: run all 4 DM models on GD-1 and compare."""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("model_comparison")

from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
from src.simulation.subhalo import DM_MODELS, subhalo_profile_for_model

# Show what each model predicts for a 10^7 Msun subhalo
log.info("=" * 70)
log.info("  SUBHALO PROFILE COMPARISON (M = 10^7 Msun)")
log.info("=" * 70)
log.info("  %-6s  %-18s  a_kpc     r_core    kick_supp  c", "Model", "Profile Type")
for m in DM_MODELS:
    p = subhalo_profile_for_model(m, 1e7)
    log.info("  %-6s  %-18s  %.5f   %.5f   %.3f      %.1f",
             m, p["profile_type"], p["scale_radius_kpc"],
             p["core_radius_kpc"], p["kick_suppression"], p["concentration"])

log.info("")
log.info("  Same for M = 10^8 Msun:")
log.info("  %-6s  %-18s  a_kpc     r_core    kick_supp  c", "Model", "Profile Type")
for m in DM_MODELS:
    p = subhalo_profile_for_model(m, 1e8)
    log.info("  %-6s  %-18s  %.5f   %.5f   %.3f      %.1f",
             m, p["profile_type"], p["scale_radius_kpc"],
             p["core_radius_kpc"], p["kick_suppression"], p["concentration"])

# Run on GD-1 in fast mode
log.info("")
log.info("=" * 70)
log.info("  PIPELINE: GD-1 model comparison (fast mode)")
log.info("=" * 70)

cfg = ForwardModelConfig(
    stream_name="GD1",
    config_path="config/streams.yaml",
    processed_h5_path="data/processed/streams.h5",
    use_fast_mode=True,
    n_stars_sim=2000,
    use_gnn_scorer=False,
    log10_mass_range=(6.5, 8.5),
    log10_mass_step=0.5,
    t_since_range=(1.0, 5.0),
    t_since_step=1.0,
)

model = TimelineForwardModel(cfg)
model.prepare()

# Run coarse grid first
results = model.run_grid()
log.info("  Coarse grid: %d candidates, best=%.4f", len(results), results[0].score.combined)

# Take top 5 and compare across all 4 DM models
top_candidates = [
    {"log10_mass": r.log10_mass, "t_since_gyr": r.t_since_gyr, "impact_phi1": r.impact_phi1}
    for r in results[:5]
]

log.info("")
log.info("=" * 70)
log.info("  Comparing top-5 candidates across CDM / WDM / FDM / SIDM")
log.info("=" * 70)

comparison = model.run_model_comparison(candidates=top_candidates)

# Detailed results
log.info("")
log.info("=" * 70)
log.info("  DETAILED RESULTS")
log.info("=" * 70)
for i, r in enumerate(comparison):
    log.info("")
    log.info("  Candidate %d: M=10^%.2f, t=%.1f Gyr, phi1=%.1f",
             i + 1, r.log10_mass, r.t_since_gyr, r.impact_phi1)
    log.info("    %-6s  combined   density    gap        kinematic  profile_type", "Model")
    for m in DM_MODELS:
        sr = r.model_scores[m]
        pt = r.model_profiles[m]["profile_type"]
        marker = " <-- BEST" if m == r.best_model else ""
        log.info("    %-6s  %.4f    %.4f     %.4f     %.4f     %-18s%s",
                 m, sr.combined, sr.density_residual, sr.gap_agreement,
                 sr.kinematic_perturbation, pt, marker)
    log.info("    Winner: %s (score=%.4f, vs null=%.4f)",
             r.best_model, r.best_score, r.null_combined)

# Save
out_path = model.save_model_comparison_results(comparison)
log.info("")
log.info("Results saved to: %s", out_path)

# Summary
log.info("")
log.info("=" * 70)
log.info("  SUMMARY")
log.info("=" * 70)
wins = {}
for r in comparison:
    wins[r.best_model] = wins.get(r.best_model, 0) + 1
for m in DM_MODELS:
    log.info("  %s wins: %d / %d", m, wins.get(m, 0), len(comparison))
log.info("  Overall best model: %s", comparison[0].best_model)
log.info("  ALL STEPS COMPLETED SUCCESSFULLY")
