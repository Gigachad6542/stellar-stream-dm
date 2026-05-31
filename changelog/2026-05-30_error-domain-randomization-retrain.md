# 2026-05-30 Error domain randomization (native sim-to-real robustness)

## Motivation

The detector's `p_impact=1.0` saturation on real data was traced to *near-constant
error features* in the simulations: the normalizer clamps a near-constant
feature's std to 0.01, so any real-data offset explodes (real GD-1 `e_vrad` was
**+100σ**). The inference-side fix (impute + clip + OOD flag) makes the detector
usable now; this change fixes it *natively* by retraining with realistic, varying
errors so real data is in-distribution.

## What changed

- `src/data/dataset.py` — `StreamSimDataset(error_dr=...)`: new `_apply_error_dr`
  draws per-star `e_dist`/`e_pm1`/`e_pm2`/`e_vrad` log-uniformly over realistic
  Gaia DR3 ranges, adds error-consistent observational noise, and masks the RV
  for a configurable fraction of streams (mimicking no-spectroscopy streams like
  GD-1). Applied deterministically per-simulation in `__getitem__`, so the
  normalizer (computed from `dataset_raw`) captures the spread and training is
  reproducible.
- `config/training.yaml` — `training_v2.error_domain_randomization` block
  (disabled by default; ranges + `rv_mask_prob`).
- `scripts/train_v2.py` — `--error-dr` flag enables it and forces on-the-fly
  profile features (the precomputed cache is stale once node features change).

## Verification (the fix works)

Recomputing the normalizer on `data/simulations_v2_detector_balanced`:

| error feature | std without DR | std with DR |
|---|---|---|
| e_vrad | **0.0100 (clamped)** | **1.95** |
| e_dist | 1.69 | 0.48 |
| e_pm1/e_pm2 | 4.07 | 0.15 |

And the real-GD-1 mapping that caused the saturation:

| feature | real value | old σ | new σ |
|---|---|---|---|
| e_vrad | 2.0 | **+100** | **−0.7** |
| e_dist | 1.31 | −1.1 | 0.9 |
| e_pm1 | 0.42 | −1.3 | 1.3 |

The +100σ explosion is eliminated — real GD-1 lands in-distribution.

A 1-epoch smoke run (`--error-dr`, on-the-fly profile features) completes end to
end and saves a normalizer/checkpoint. Full non-slow suite: **251 passed**.

## Full retrain

Launched as a background job:

```
python scripts/train_v2.py --sim-dir data/simulations_v2_detector_balanced \
  --checkpoint-dir checkpoints/gnn_v2_timeline_errordr_20260530 \
  --binary-target impact_timeline_detectable --label-schema timeline \
  --regression-target timeline_effective --use-profile-branch \
  --profile-feature-set summary --error-dr --epochs 80
```

After it finishes: run `scripts/calibrate_detector.py` on the new checkpoint, then
re-check the OOD `max_sigma` on real GD-1 (expected to drop from ~391σ to order
unity) and the resulting `p_impact`. The detector then no longer needs the ±5σ
input clip to behave on real data, and `p_impact` becomes trustworthy.
