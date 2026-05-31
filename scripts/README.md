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
| `process_gd1.py` | Download and process GD-1 from Gaia DR3. |
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

## 8. Diagnostics & dev one-offs (archival candidates)

| Script | Purpose |
|---|---|
| `debug_stream_gen.py` | Debug why `generate_stream` produced 0 particles. |
| `diag_particle_positions.py` | Show where spray particles land before selection cuts. |
| `inspect_benchmark.py` | Quick inspection of benchmark HDF5 output. |
| `demo_forward_pass.py` | Demo: pass real data through the trained GNN + posteriors. |
| `full_pipeline_test.py` | Full pipeline run with each stage timed. |
| `test_ic_optimizer.py` | Quick test of the v3 progenitor IC optimiser. |
| `test_model_comparison.py` | Run all 4 DM models on GD-1 and compare. |
| `test_multi_encounter.py` | Quick multi-encounter evaluation test. |
| `train_import_test.py` | Import smoke test (has a stray BOM — safe to delete). |

> The `test_*.py` files here are standalone scripts, **not** pytest tests — the
> real unit tests live in `tests/`.

## 9. Experimental — JAX integrator (not in the default path)

| Script | Purpose |
|---|---|
| `benchmark_jax.py` | Benchmark the JAX integrator vs galpy's C integrator. |
| `optimize_jax_ics.py` | Pre-compute JAX-optimised progenitor ICs for all streams. |
| `test_jax_integrator.py` | Validate the JAX orbit integrator against galpy. |
