# 2026-05-29 Review of prior work and handoff assessment

## What the other AI changed (S2/S3 session, overnight 05-28/29)

### New datasets
- `data/simulations_v2_detector_balanced`: 20K sims with balanced impact/no-impact labels independent of DM family
- `data/simulations_signal_ladder`: small curriculum datasets proving impact morphology is learnable

### New/rewritten scripts
- `scripts/run_detector_balanced_handoff.py`: environment check + validation + baseline audit
- `scripts/run_detector_fusion_diagnostics.py`: RF/GNN/fusion comparison
- `scripts/precompute_profile_features.py`: cached profile feature generation
- `scripts/compute_impact_strength.py`, `scripts/compute_timeline_impact_labels.py`: new label passes
- `scripts/evaluate_v2_classifier.py`: proper held-out evaluation
- `scripts/evaluate_summary_baseline.py`: RF/ExtraTrees baseline benchmarks
- `scripts/diagnose_v2_signal.py`, `scripts/audit_v2_signal_thresholds.py`: diagnostic tooling

### Major rewrites of existing scripts
- **`scripts/run_sbc.py`**: completely rewritten with split-aware SBC, boundary-aware calibration for CDM/SIDM log10_M_hm, profile cache support, deterministic downsampling, JSON summary output, `--output-dir` isolation
- **`scripts/train_sbi.py`**: updated with `profile_feature_dim()`, `feature_set` contracts, `load_split_indices`, `--profile-features-path`, `--downsample-seed`, `--sim-dir`

### New source modules
- `src/data/splits.py`: proper train/val/test index splitting
- `src/data/radial_velocity.py`: improved stream processing with RV data
- `src/analysis/gap_classifier.py`, `src/analysis/power_spectrum_baseline.py`
- `src/inference/diagnostics.py`, `src/inference/injection_tests.py`, `src/inference/suppression_scale.py`
- `src/models/equivariant_gnn.py`

### New checkpoints
- `checkpoints/gnn_v2_detector_summary_s2/`: S2 detector, test AUC 0.8982
- S3 timeline run in progress at time of handoff

### Key results
- S2 GNN: AUC 0.8982, AP 0.8580
- RF summary baseline: AUC 0.9146, AP 0.8710
- RF+GNN fusion (Platt-calibrated): AUC 0.9177, AP 0.8769, ECE 0.0212
- Pilot SBI WDM log10_M_hm: K-S p=0.60 (passes!)
- Pilot SBI FDM log10_M_hm: K-S p=0.43 (passes!)

## Valid criticisms of my earlier work

1. **Profile dimension mismatch**: I used `n_bins + 14 = 62` when S2 checkpoint needs `summary` feature set = 157 dims
2. **Nondeterministic downsampling**: random node sampling without seed meant cached features couldn't align with GNN inputs
3. **No split awareness in SBC**: I used arbitrary last-10% instead of the saved test split, risking train/test contamination
4. **Dynamic profile recomputation**: slow per-batch Python loops instead of using precomputed cache
5. **Missing feature_set contract**: didn't read the checkpoint's feature_set config, so wrong features would be built
6. **No boundary-aware SBC**: treated CDM/SIDM log10_M_hm as a normal parameter when it's a point-mass at the prior boundary

## What I built that's still valid
- `src/inference/tarp_calibration.py`: TARP recalibrator (concept is sound but needs to work with the updated SBC infrastructure)
- `scripts/run_tarp_recalibration.py`: recalibration runner (same caveat)
- `src/inference/calibration.py` plt.show() -> plt.close() fix: prevents hang in non-interactive mode

## Status
- Adopting the other AI's code as the new baseline
- All future changes will be notarized in this changelog directory
- Ready for new plan
