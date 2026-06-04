# `scripts/` — runnable entry points

Index of every script, grouped by role. All scripts add the repo root to
`sys.path` themselves and resolve paths relative to it, so run them from the
repo root, e.g.:

```bash
python scripts/run_timeline_forward_model.py --stream GD1
```

> **Note on organization:** scripts hard-code the repo root via
> `Path(__file__).resolve().parent.parent`, so they must stay one level under
> `scripts/` — do not move them into subfolders without fixing that path. This
> index provides the organization instead. Scripts under *Diagnostics & dev*
> and *Experimental* are one-offs kept for reference and are archival
> candidates.

---

## 1. Core timeline pipeline (current DM-detection workflow)

Run roughly in this order. This is the live pipeline.

| Script | Purpose |
|---|---|
| `generate_training_data.py` | Parallel simulation generation (track6d IC + spray); writes HDF5 chunks. BLAS threads pinned, `n_jobs` = physical cores. |
| `precompute_profile_features.py` | Precompute graph-level density-profile features for the GNN profile branch (`--error-dr`). |
| `train_v2.py` | Train the v2 binary/regression GNN detector heads (`--error-dr`). |
| `calibrate_detector.py` | Temperature-calibrate the trained detector on its validation split. |
| `run_timeline_forward_model.py` | The timeline runner: detect impact → estimate time → rewind → re-impact grid → re-evolve → score vs real. |
| `run_injection_recovery.py` | Injection-recovery test for the timeline forward model. |
| `run_multistream_analysis.py` | Multi-stream joint subhalo-impact significance (Stouffer/Fisher + coherence gate). |

## 2. Data acquisition & processing

| Script | Purpose |
|---|---|
| `process_streams.py` | Download Gaia DR3 and process the streams into the HDF5 database. |
| `process_streams_parallel.py` | Parallel Gaia download + processing for all streams. |
| `process_gd1.py` | Process GD-1 from the published PWB18 masked-region catalog, with Gaia TAP fallback. |
| `ingest_pwb18_gd1.py` | Verify and ingest the authoritative PWB18 GD-1 region catalog into dedicated track and PM+CMD HDF5 selections. |
| `ingest_desi_gd1.py` | Verify and ingest the latest DESI DR2 GD-1 v3 thin-stream and thin+cocoon probability catalogs with real RVs. |
| `run_gd1_cross_track_morphology.py` | Diagnostic conditional cross-stream morphology screen using DESI thin+cocoon probabilities without treating sparse spectroscopy as an along-stream density survey. |
| `summarize_gd1_profile_diagnostics.py` | Join density+RV and cross-stream morphology screens with a Pareto/agreement diagnostic instead of an arbitrary weighted likelihood. |
| `summarize_gd1_full_orbit_shortlist.py` | Consolidate the three full-orbit Pareto-shortlist checks and test whether fast-mode preferences survive physical evolution. |
| `run_gd1_two_perturbation_shortlist.py` | Constrained 3x3 canonical+spur full-orbit two-perturbation test with matched no-kick controls and an explicit complexity guard. |
| `run_gd1_background_calibration.py` | Calibrate smooth GD-1 spray age, progenitor mass, and velocity scale on control regions while holding both candidate-impact windows out. |
| `run_gd1_background_sensitivity_shortlist.py` | Re-test fixed GD-1 single-encounter representatives across legacy, length-calibrated, and control-region-selected smooth backgrounds with matched controls. |
| `run_gd1_streamgapdf_profile_screen.py` | Screen continuous perturber scale radius with a matched two-arm streamdf null and localized trailing-arm streamgapdf impact. |
| `run_gd1_streamgapdf_injection_recovery.py` | Frozen matched two-arm streamgapdf profile validation: 48-cell fixed-geometry identifiability or nuisance-geometry recovery, with real-like PWB18/DESI selection and shard/merge support. |
| `run_gd1_streamgapdf_null_fpr.py` | No-impact false-positive challenge using the same nuisance search and frozen decision threshold as streamgapdf injection/recovery. |
| `fuse_gd1_pwb18_desi.py` | Fuse PWB18 main-track density selection with DESI v3 spectroscopy by one-to-one sky-position matching. |
| `process_gd1_cached.py` | Process GD-1 from cached raw FITS (skip the Gaia TAP re-query). |
| `fetch_multi_epoch.py` | Fetch + fuse real multi-epoch / multi-survey RVs & PMs for a stream. |
| `clean_membership.py` | Extract track-consistent clean members from a contaminated catalog. |
| `make_synthetic_test_data.py` | Create synthetic test HDF5 simulations for GNN validation. |

## 3. Generator calibration & label derivation

| Script | Purpose |
|---|---|
| `calibrate_spray_age.py` | Calibrate per-stream `spray_age_gyr` so generated extent matches the observed track. |
| `compute_impact_strength.py` | Compute per-simulation impact-strength scores for existing datasets. |
| `compute_timeline_impact_labels.py` | Derive timeline-aware impact labels from existing simulations. |

## 4. Model training (GNN / SBI)

| Script | Purpose |
|---|---|
| `train.py` | Train the GNN encoder (+ optional baseline CNN) on the simulation dataset. |
| `train_baseline.py` | Train the 1D CNN baseline (`DensityProfileCNN`). |
| `train_sbi.py` | Train the SBI (SNPE-C) pipeline using pre-trained GNN embeddings. |
| `train_v2.py` | *(also listed under Core)* — v2 detector heads. |

## 5. Inference, analysis & evaluation

| Script | Purpose |
|---|---|
| `run_inference.py` | Apply the trained GNN + NPE posterior to real stream data. |
| `run_analysis.py` | Generate the gap catalog and final analysis figures. |
| `run_real_stream_analysis.py` | Hierarchical Bayesian inference on literature-reported streams. |
| `dm_discrimination_forecast.py` | Baseline Asimov forecast for DM abundance+mass discrimination. |
| `dm_forecast_sensitivity_grid.py` | Reviewer-facing sensitivity grid over rate, completeness, threshold proxy, mass band, and stream count. |
| `dm_null_power_table.py` | Summarize what the current seven-stream null can and cannot constrain. |
| `dm_discriminants_table.py` | Build the report's unified DM-model discriminants table from forecast artifacts. |
| `dm_density_profile_pivot.py` | Reframe the near-term DM analysis around perturber density profiles; writes target readiness and profile-ladder artifacts. |
| `run_density_profile_grid.py` | Screen individual GD-1 perturber mass and compactness separately; refuses physical interpretation without selection and feature guards. |
| `run_production_grid_gd1.py` | Production forward-model grid on GD-1. |
| `run_sensitivity.py` | Minimum detectable subhalo mass per stream. |
| `run_framework_validation.py` | Run the inference engine on mock + synthetic data. |
| `run_mock_challenge.py` | Mock Data Challenge: count-rate validation + artifact sanity checks. |
| `run_detector_balanced_handoff.py` | Post-generation handoff for the detector-balanced dataset. |
| `run_detector_fusion_diagnostics.py` | Compare/fuse the GNN and summary-feature baselines. |
| `combine_posteriors.py` | Combine per-stream posteriors into a joint DM constraint. |
| `evaluate_summary_baseline.py` | Evaluate classical summary-feature baselines on the V2 split. |
| `evaluate_v2_classifier.py` | Evaluate a trained V2 GNN checkpoint on val/test. |
| `embed_pca.py` | PCA/UMAP visualization of GNN embeddings. |

## 6. SBI calibration & coverage

| Script | Purpose |
|---|---|
| `run_sbc.py` | Simulation-based calibration (SBC) using pre-computed embeddings. |
| `run_tarp_recalibration.py` | Build TARP recalibrators from SBC ranks and validate. |
| `test_posteriors.py` | Posterior sanity test + fast SBC on test-set embeddings. |

## 7. Dataset & phase validation

| Script | Purpose |
|---|---|
| `validate_phase3.py` | Phase 3 physics validation. |
| `validate_phase4.py` | Phase 4: one GNN forward pass on simulated data. |
| `validate_v2_dataset.py` | Validate a V2 simulation chunk dataset before training. |
| `audit_v2_signal_thresholds.py` | Audit whether V2 impact labels are learnable before a GNN run. |
| `diagnose_v2_signal.py` | Diagnose whether a V2 binary target is visible in simulated features. |
| `validate_pwb18_gd1.py` | Validate PWB18 provenance, masks, frame mapping, contamination correction, and target-feature stability before profile inference. |
| `validate_desi_gd1.py` | Validate DESI v3 probabilities, RVs, frame mapping, and target-region density/kinematics before profile inference. |

## 8. Experimental — JAX integrator (not in the default path)

| Script | Purpose |
|---|---|
| `benchmark_jax.py` | Benchmark the JAX integrator vs galpy's C integrator. |
| `optimize_jax_ics.py` | Pre-compute JAX-optimised progenitor ICs for all streams. |
| `test_jax_integrator.py` | Validate the JAX orbit integrator against galpy. |
