#!/usr/bin/env python
"""Quick functional test for multi-encounter evaluation."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

from src.forward_model.pipeline import TimelineForwardModel, ForwardModelConfig

cfg = ForwardModelConfig(
    stream_name="GD1",
    config_path="config/streams.yaml",
    processed_h5_path="data/processed/streams.h5",
    use_fast_mode=True,
    n_stars_sim=2000,
    use_gnn_scorer=False,
    log10_mass_range=(7.0, 8.0),
    log10_mass_step=1.0,
    t_since_range=(1.0, 3.0),
    t_since_step=2.0,
)

model = TimelineForwardModel(cfg)
model.prepare()

# Test 1: evaluate a 2-encounter configuration
print("\n" + "=" * 60)
print("TEST 1: Single multi-encounter evaluation")
print("=" * 60)
enc_list = [
    {"log10_mass": 7.5, "t_since_gyr": 5.0, "impact_phi1": 20.0},
    {"log10_mass": 8.0, "t_since_gyr": 1.5, "impact_phi1": 50.0},
]
result = model.evaluate_multi_encounter(enc_list)
print(f"  n_encounters: {result.n_encounters}")
print(f"  combined_score: {result.score.combined:.4f}")
print(f"  density: {result.score.density_residual:.4f}")
print(f"  gap: {result.score.gap_agreement:.4f}")
print(f"  kinematic: {result.score.kinematic_perturbation:.4f}")
print(f"  n_stars: {result.n_stars_sim}")
print(f"  runtime: {result.runtime_s:.3f}s")
print("  Encounters:")
for e in result.encounters:
    print(f"    M=10^{e['log10_mass']:.1f}, t={e['t_since_gyr']:.1f} Gyr, phi1={e['impact_phi1']:.1f}")

# Test 2: compare single vs multi
print("\n" + "=" * 60)
print("TEST 2: Single encounter vs same encounters as multi")
print("=" * 60)
single1 = model.evaluate_candidate({"log10_mass": 7.5, "t_since_gyr": 5.0, "impact_phi1": 20.0})
single2 = model.evaluate_candidate({"log10_mass": 8.0, "t_since_gyr": 1.5, "impact_phi1": 50.0})
print(f"  Single enc1 (M=10^7.5, t=5.0): score={single1.score.combined:.4f}")
print(f"  Single enc2 (M=10^8.0, t=1.5): score={single2.score.combined:.4f}")
print(f"  Multi (both together):          score={result.score.combined:.4f}")
print(f"  Best single: {min(single1.score.combined, single2.score.combined):.4f}")
if result.score.combined < min(single1.score.combined, single2.score.combined):
    print("  -> Multi-encounter IMPROVES over best single!")
else:
    print("  -> Best single is already better (expected for some configs)")

# Test 3: multi-encounter grid
print("\n" + "=" * 60)
print("TEST 3: Multi-encounter grid (10 random 2-encounter samples)")
print("=" * 60)
multi_results = model.run_multi_encounter_grid(n_encounters=2, n_random_samples=10, seed=42)
print(f"  Got {len(multi_results)} results")
print(f"  Best: combined={multi_results[0].score.combined:.4f}")
for e in multi_results[0].encounters:
    print(f"    M=10^{e['log10_mass']:.2f}, t={e['t_since_gyr']:.1f} Gyr, phi1={e['impact_phi1']:.1f}")

# Test 4: 3-encounter configuration
print("\n" + "=" * 60)
print("TEST 4: 3-encounter configuration")
print("=" * 60)
enc_3 = [
    {"log10_mass": 7.0, "t_since_gyr": 8.0, "impact_phi1": 15.0},
    {"log10_mass": 7.5, "t_since_gyr": 4.0, "impact_phi1": 40.0},
    {"log10_mass": 8.0, "t_since_gyr": 1.0, "impact_phi1": 65.0},
]
result_3 = model.evaluate_multi_encounter(enc_3)
print(f"  n_encounters: {result_3.n_encounters}")
print(f"  combined_score: {result_3.score.combined:.4f}")
print(f"  runtime: {result_3.runtime_s:.3f}s")

print("\n" + "=" * 60)
print("ALL MULTI-ENCOUNTER TESTS PASSED!")
print("=" * 60)
