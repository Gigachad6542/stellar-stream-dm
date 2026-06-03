# 2026-06-02 Detector retrain on corrected sims — the GNN finds the impact

Retrained the binary detector on the streamgapdf dataset (data/simulations_detector_df,
15,830 sims, gaia noise) and it WORKS. This closes the "make the GNN find the impact"
goal — the blocker was always the simulator (point-like subhalos via the rs bug +
clumpy homemade spray), not the architecture.

## Training
scripts/train_v2.py --sim-dir data/simulations_detector_df --binary-target impact_strong
  --strength-threshold 0.5 --use-profile-branch --epochs 80
  -> checkpoints/detector_df_20260602/  (best val_binary_acc 0.951 @ epoch 34)

## Results
- **Test AUC = 0.982** (calibrate_detector.py)  vs v3's 0.62 on the broken sims.
- Calibration T=0.541, NLL 0.226 -> 0.174.
- 1-epoch/800-example smoke already hit 0.853 acc -> the signal is strongly learnable.

## Real STREAMFINDER GD-1 (data/processed/GD1_streamfinder.h5, 811 members)
- p_impact = 0.774 (impact flagged), model-free gap detected, impact_detected=True.
- **OOD max_sigma = 3.1 (0% features clipped) -> real GD-1 is IN-DISTRIBUTION.**
  v3 was 391 sigma raw / 13 sigma even after a dedicated error-DR retrain. The
  gaia-noise streamgapdf generator matches the real catalog's feature distribution
  natively -> error-DR is no longer required for GD-1.

## Notes
- regression_target='raw' -> the t_since estimate is meaningless (binary head only);
  retrain with a timeline schema if the time estimate is needed later.
- split-file naming mismatch: train_v2 writes ..._s0p5.npz, calibrate expects
  ..._t0p5.npz (copied; worth unifying).

## Next
- Re-run the v3 conclusions on corrected impacts: timeline forward model
  (detect->rewind->re-impact->score) + multistream joint significance.
- Optional: SBI/mass inference stage on the corrected impacts (full param range).
