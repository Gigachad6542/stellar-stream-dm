# 2026-06-01 v3 SBI posteriors + SBC + TARP (closing the "retrained on v3?" gap)

After the v3 detector retrain, the SBI mass-function layer and its coverage
diagnostics were still on old (pre-fix) data. This brings the WHOLE pipeline onto
v3.

## Integration bugs fixed (found via dry-run)
`train_sbi.py` / `run_sbc.py` rebuilt the GNN with the compact profile default
(profile_dim=62) because the detector's train-time `--profile-feature-set summary`
(dim 157) was not persisted into the checkpoint config -> state_dict size mismatch,
then a cache-shape mismatch in embedding. Fix: load the checkpoint unconditionally
and read `profile_mlp.0.weight` to recover the true profile dim, then back out the
matching `feature_set` so model build + embedding-cache validation agree.

## Pipeline run on v3
- `train_sbi.py ... --sim-dir data/simulations_v3_track6d --sbi-output-dir checkpoints/v3_sbi
  --use-profile --profile-features-path data/processed/profile_features_v3_errordr.h5.npz
  --theta-mode suppression` -> 4 posteriors (CDM/WDM/FDM/SIDM), ~4 min each, NPE converged.
- `run_sbc.py ... --posterior-dir checkpoints/v3_sbi --output-dir outputs/sbc_v3 --sbc-split test`
  (500 trials x 500 samples).
- `run_tarp_recalibration.py --sbc-dir outputs/sbc_v3 --posterior-dir checkpoints/v3_sbi`.

## Results (honest)
- **SBC: all 4 models FAIL the uniform-rank K-S test** (p approx 0 for >=1 parameter).
  The v3 SBI posteriors are miscalibrated.
- **TARP coverage @ nominal 90% CI:**
  - `log10_M_hm`: CDM 0%, SIDM 0%, WDM 86%, FDM 92% -> the halo mass-function cutoff
    is essentially **uninformative** for CDM/SIDM (cannot be recalibrated).
  - `n_impacts`: 47-87% -> weakly constrained, recalibratable (TARP nominal->true map saved).
- Recalibrators written to `checkpoints/v3_sbi/recalibrator_<MODEL>.pkl`;
  summary `outputs/sbc_v3/tarp_calibration_summary.json`.

## Interpretation
Consistent with the weak v3 detector (AUC 0.618): on physically-faithful sims the
present-day snapshot carries little information about the subhalo mass function, so
the SBI `log10_M_hm` posteriors have poor coverage. This strengthens the cautionary
/ null headline rather than weakening it.

## Pipeline status: fully v3-consistent
generator -> 100k v3 sims -> detector (AUC 0.618) -> SBI posteriors -> SBC/TARP ->
multistream (no detection) -> injection-recovery (degenerate but detected). No old
(pre-fix) artifacts remain in the result chain.

## For the manuscript (deferred per request)
Section 4.2 should report the SBC failure + TARP coverage honestly (log10_M_hm
uninformative on v3) when we do the text rounds.
