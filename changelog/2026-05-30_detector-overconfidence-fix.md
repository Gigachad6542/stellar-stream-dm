# 2026-05-30 Fixing the timeline detector's over-confidence on real data

## The problem

On real GD-1 the timeline GNN detector returned `p_impact = 1.0000` for
everything — useless as a probability. We needed to know *why* before trusting
the first step of the timeline pipeline.

## Diagnosis (scripts/calibrate_detector.py + ad-hoc diagnostics)

1. **The model is not broken.** On the held-out *simulated* validation split it
   discriminates well: AUC 0.93–0.94, mean p(neg)=0.18 vs p(pos)=0.84, only ~1%
   of sim-negatives falsely confident. Temperature scaling gives **T = 1.065**
   (NLL 0.2872 → 0.2869) — i.e. calibration was never the issue.
2. **Real GD-1 produced an absurd logit of ~190–255** (vs sim max ~41) →
   `sigmoid` saturates to 1.0, and MC-dropout can't even perturb it.
3. **Root cause = input construction, not the network.** `obs_particles_to_data`
   injected *fake constant* error columns (`e_vrad=2.0`, `e_pm=0.1`, …). Those
   error features are near-constant in training, so the normalizer clamps their
   std to 0.01; any offset on real data is then amplified to ~100σ
   (post-norm `e_vrad` was **+100σ**), exploding the logit.

Ablation on real GD-1 (logit): fake errors **255** → +real errors 260 →
+impute unmeasured vrad/e_vrad to training mean **8.1** → +clip post-norm ±5σ
**8.0**. The imputation is the decisive fix.

## The fix (make the first step accurate)

`src/forward_model/gnn_scorer.py` — `obs_particles_to_data` now uses the **real
per-star errors** when present and passes **NaN through for unmeasured fields**
(instead of fabricating constants).

`src/forward_model/detection.py` — `StreamImpactDetector` now:
- **imputes unmeasured (NaN) features to the training mean** (post-norm ≈ 0), so
  a stream with no RV contributes neutral RV features rather than an OOD spike;
- **clips standardized features to ±`clip_sigma` (default 5)** so no single
  out-of-distribution feature can saturate the logit;
- records **OOD diagnostics** (`ood_max_sigma`, `ood_frac_clipped`) on every
  prediction and warns when inputs are out-of-distribution;
- applies a **temperature** loaded from `calibration.json` (auto-written by
  `scripts/calibrate_detector.py`).

`src/forward_model/pipeline.py` — `_load_observed_data` now loads the real
`e_dist/e_pm1/e_pm2/e_vrad` columns so the detector receives true errors.

`scripts/run_timeline_forward_model.py` — logs the OOD check and an explicit
"treat p_impact as unreliable" flag when inputs are out-of-distribution.

`scripts/calibrate_detector.py` (new) — temperature-scales the detector on its
sim validation split and writes `calibration.json` next to the checkpoint.

## Result on real GD-1

`p_impact`: **1.0000 → 0.9829** (finite, defensible), `t_since ≈ 5.9 Gyr`, and
critically an **OOD flag of ~391σ** is now surfaced — honest notice that GD-1 is
far from the training distribution and the probability should not be over-trusted
(rely on the model-free gap detector + injection-recovery for hard conclusions).

## Tests / verification

- `tests/test_detection.py`: +2 input-accuracy tests (real errors used; NaN
  passthrough for unmeasured fields). Full non-slow suite: **250 passed, 3 deselected**.
- `calibration.json` written for `gnn_v2_timeline_s3_20260529_123353` (T=1.065).

## Takeaways / follow-ups

- The "over-confidence" was an OOD input-handling bug, not a model or calibration
  flaw. The detector discriminates well in-distribution.
- Remaining OOD (~391σ from a few real-data outliers, plus phi1/membership
  shifts) is intrinsic sim-to-real mismatch; the real cure is domain-randomized
  retraining. Until then the OOD flag + model-free detection are the safeguards.
- Consider re-checking the timeline `t_since` head: its GD-1 estimate (~6 Gyr)
  still exceeds GD-1's ~3 Gyr disruption age, suggesting the timeline label or
  training range needs review.
