# 2026-06-01 v3 retrain, calibration, and multi-stream results

The corrected `simulations_v3_track6d` dataset (track-6D IC + calibrated spray,
100k sims) run end-to-end through the detector + significance pipeline.

## Pipeline executed
1. **Precompute error-DR profile cache** → `data/processed/profile_features_v3_errordr.h5.npz`
   (100000×157, 43 min). This is what makes error-DR training fast (it avoids
   recomputing profiles on-the-fly every batch: 0.29→2.34 batches/s, 8×).
2. **Train** `train_v2.py --error-dr --use-profile-branch --profile-feature-set summary
   --profile-features-path <cache> --binary-target impact_timeline_detectable
   --label-schema timeline --regression-target timeline_effective --epochs 80`
   → `checkpoints/gnn_v2_timeline_errordr_v3_20260601`. Best val_binary_acc 0.595.
3. **Calibrate** `calibrate_detector.py --error-dr` → T=0.672, **AUC=0.618**.
4. **Multi-stream** `run_multistream_analysis.py` → `outputs/multistream/joint_significance_v3.json`.

## Headline results (honest)
- **Detector AUC = 0.618** on physically-faithful v3 sims, vs **0.937** on the old
  *balanced* curriculum dataset. The strong prior number was substantially a
  dataset-construction artifact; the corrected simulator yields a genuinely harder
  detection problem.
- **Multi-stream joint (look-elsewhere corrected):** Stouffer Z=1.00 (p=0.159),
  Fisher χ²=16.2 (p=0.301) → **no joint detection**, consistent with a smooth null
  / CDM upper limit.
- Per-stream LE z now **coherent and modest**: GD1 −0.47, Pal5 +1.03, Orphan −0.40,
  ATLAS +1.58, Jhelum +2.67, Fjörm −2.48, Sylgr +0.72 (range [−2.5,+2.7]) — vs the
  pre-fix incoherent [−4.8,+29] (71% positive). The corrected sims both lower and
  *regularize* the significance.
- Look-elsewhere correction is essential: naive z=12.0 (Pal5), 105 (Sylgr) collapse
  to 1.0, 0.7.

## Manuscript
`paper/manuscript.md` abstract, §4.3, §7 (Table 2), §8, §10 updated with these
real numbers. Remaining: §5 injection-recovery numbers, figure regeneration,
final read-through.
