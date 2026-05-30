# 2026-05-30 First-step accuracy fixes: localization, time cap, RV zero-point

Three accuracy fixes to the early (detection/scoring) steps that feed the rewind.

## 1. GD-1 gap localization (`detect_gaps` was finding nothing)

Diagnosis: GD-1's membership column is **degenerate (all 1.0)** — a pre-selected,
contamination-diluted catalog (14k stars vs the ~4k expected PWB18 members), so
its deepest density dips are only **~10-15%**, far below the 30%-depth default.
Also, the config `known_gaps` (phi1≈−40) are in the **PWB18 frame** while the
data/sims use the **Ibata-2021 frame** (phi1∈[0.6,72.8]) — so they are unusable
as-is. The real most-depleted region is phi1≈36.

Fix: new `scoring.find_density_minima()` — prominence-ranked minima (scipy
`find_peaks`) with no hard depth cut — used for **localization**, while the
strict `detect_gaps` still drives the binary detection decision. `detection`
now seeds the phi1 grid from the real deepest minima (GD-1 → phi1≈36) instead of
blind quartiles or the out-of-frame −40, and reports each minimum's depth +
significance honestly.

## 2. t_since estimate exceeding the stream age

The timeline GNN head predicts t_since≈6 Gyr for GD-1, but GD-1's disruption age
is ~3 Gyr — so a ±2 Gyr window would sit entirely above the age and every
candidate would clamp to the same value. The head is trained across streams of
different ages and is unreliable on OOD real data.

Fix: `seed_config_from_detection(..., stream_age_gyr=...)` caps the t_since
estimate (and window) at the stream's disruption age; the runner passes
`disruption_age_gyr` from the stream config. GD-1: 5.92 Gyr → capped 3.0 →
window [1.0, 3.0] Gyr (physical), with a log note.

## 3. Radial-velocity zero-point

The real-ATLAS RV residual (~15 km/s) was dominated by a constant line-of-sight
offset — a frame/convention nuisance (heliocentric sim `vrad` vs S5 `vlos`),
not a subhalo signal.

Fix: `radial_velocity_score` now subtracts the overall median sim−obs offset
before the RMS, so only the **differential** RV track (the real perturbation
signature) is scored. Tests updated: constant offset is now insensitive; a slope
(differential) change still raises the score.

## Verification

- Suite: **251 passed, 3 deselected**.
- GD-1 detection: seeds phi1=36 (real minimum), t-window [1,3] Gyr (capped),
  p_impact=0.98 with a 391σ OOD flag — all physical and honest.

## Follow-ups

- The proper GD-1 fix is a real membership catalog (PWB18 with probabilities) or
  a PWB18↔I21 frame transform for `known_gaps`; the current catalog's degenerate
  membership limits gap contrast.
- Significance reported for shallow GD-1 dips is inflated by the contamination-
  swollen star count (treats ~14k as independent members); depth is the honest
  measure.
