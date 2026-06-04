# Claim-traceability audit

Every headline quantitative claim in `full_report.md` is mapped to a machine
artifact, script, or literature reference. Verified/updated 2026-06-03.

Legend: OK = verified against an artifact; LIT = literature/reference claim; CAVEAT =
reported with an explicit systematic limitation.

## Scope split

| Result class | Backend | Report status |
|---|---|---|
| Detector training, simulator validation, completeness, characterization | validated `streamdf` / `streamgapdf` | OK |
| DM abundance/mass forecasts | validated detector completeness + analytic mass functions | OK, forecast |
| Real-stream detector checks | calibrated detector on cleaned catalogs | OK, catalog-size caveat |
| Timeline replay, injection-recovery, real GD-1 fit, multistream significance | corrected particle-spray + impulse/re-evolution forward model | CAVEAT |
| Matched GD-1 background/profile diagnostics | matched controls plus two-arm `streamdf` / `streamgapdf` | INTERNAL VALIDATION; no real profile claim |
| Kinematic frontier | model-free synthetic diagnostic | CAVEAT |

## Detector performance

| Claim | Value | Source | Status |
|---|---:|---|---|
| Corrected detector test AUC | 0.982 | `checkpoints/detector_df_20260602/calibration.json` | OK |
| Calibration temperature | T=0.54 | same (`temperature` 0.541) | OK |
| Best validation accuracy | 0.951 | `gnn_v2_training_history.json` | OK |
| Zero false-positive operating point | 0.000 at threshold 0.5 | `scripts/detector_completeness.py` outputs | OK |
| Score separation / reliability / embedding figures | Figures 6-8 | `paper/make_report_figures.py` | OK |

## Generator / simulator

| Claim | Value | Source | Status |
|---|---:|---|---|
| Corrected generator separability | G6 AUC about 0.94 | `scripts/validate_df_generator.py` | OK |
| Smoothness excess over Poisson | about 0.01 | `scripts/validate_df_generator.py` | OK |
| Detector dataset size / balance | 15,830 total; 7,830 impact / 8,000 smooth | `data/simulations_detector_df` chunk inspection | OK |
| Supported DF training streams | GD1, ATLAS, Jhelum, Orphan | generator smoke tests / allow-list | OK |
| `streamdf` / `streamgapdf` provenance | Bovy 2014; Sanders, Bovy & Erkal 2016 | `references.bib` | LIT |

## Real-data detector application

| Claim | Value | Source | Status |
|---|---:|---|---|
| GD-1 detector result | p_impact=0.77, input OOD 3.1 sigma, known gap | StreamImpactDetector on `GD1_streamfinder.h5` | OK |
| ATLAS detector result | p_impact=0.06, input OOD 1.9 sigma, null | StreamImpactDetector on cleaned ATLAS catalog | OK |
| Member-count floor | p_impact rises spuriously as N falls below about 500 | GD-1 subsample diagnostic / Figure 17 | OK |
| Catalog limitation | only GD-1 and ATLAS currently meet clean-count reliability | clean-catalog member counts | OK |

## Completeness and characterization

| Claim | Value | Source | Status |
|---|---:|---|---|
| Overall completeness | about 0.57 | `scripts/detector_completeness.py` | OK |
| Completeness vs mass | 0.48 -> 0.72 over 10^7.5 -> 10^8.5 Msun | same | OK |
| Completeness vs time | 0.75 -> 0.39 over 0.3 -> 1.4 Gyr | same | OK |
| Gap-strength step | about 0.05 for shallow gaps to about 1.0 for strong gaps | same / Figure 11 | OK |
| Dedicated-head mass recovery | R^2 < 0 | `characterize_eval.py` on `detector_char_20260602` | OK |
| Dedicated-head time recovery | R^2 about 0.2 | same | OK |

## Dark-matter model discrimination

| Claim | Value | Source | Status |
|---|---:|---|---|
| CDM detectable-impact rate | lambda_det about 0.026 per GD-1-like stream | `outputs/dm/discrimination_forecast.json` | OK, forecast |
| Rate suppression vs CDM | WDM3=0.24, WDM4=0.49, WDM6=0.88, FDM1e-22=0.26 | same | OK, forecast |
| Detections for 3 sigma vs CDM | about 2 for WDM3 / FDM1e-22 | same | OK, forecast |
| Streams to collect those detections | about 300-365 streams in the floor-normalized baseline; about 1,160-1,370 without the low-rate floor | same | CAVEAT, normalization-sensitive forecast |
| Present seven-stream null power | CDM E[N_det]=0.18 and P0=0.83 baseline; favorable WDM/FDM also P0 about 0.95-0.96; Z<0.5 vs CDM | `outputs/dm/dm_null_power_table.json` | OK, forecast/interpretation |
| Forecast sensitivity grid | 180 cases; CDM P0(7 streams)=0.60-0.98 across grid; max seven-stream Z=0.72 for WDM3 and 0.79 for FDM1e-22; median N_streams=533 and 463 | `outputs/dm/dm_forecast_sensitivity_grid.json` | OK, sensitivity forecast |
| SIDM mass-spectrum separability | not separable by abundance/mass | same | OK |
| SIDM gap-shape separability | core-sensitive: mean depth-AUC about 0.55, 0.63, 0.64, 0.68, 0.79, 0.82, 0.87, 0.84 for 1.25x through 5x scale-radius cores | `outputs/dm/sidm_morphology.json`; shards `outputs/dm/sidm_morphology_shard_20260603_155409_*.json` | CAVEAT, still low-trial morphology grid |
| Density-profile pivot | Compact/cored/solitonic response ladder defines a controlled validation target; real-GD1 inference is not yet supported | `outputs/dm/density_profile_pivot.json`; `scripts/dm_density_profile_pivot.py` | CAVEAT, strategy/diagnostic |

Forecast caveat: `scripts/dm_discrimination_forecast.py` uses a representative
GD-1-like sensitivity band and completeness interpolation. The detections-needed result is
fairly stable for favorable WDM/FDM models, but the streams-needed forecast depends strongly
on the encounter-rate normalization/floor and should be treated as a sensitivity result, not
a final journal constraint. `scripts/dm_forecast_sensitivity_grid.py` makes this explicit
with a 180-case grid over rate normalization/floor, completeness scale, a detector-threshold
completeness proxy, mass band, and stream count. The threshold axis is only a proxy and does
not model false-positive contamination.

Density-profile caveat: the pivot artifact is a strategy and diagnostic scaffold, not
a final perturber-profile constraint. It uses local catalog counts, the existing SIDM
morphology grid, and analytic impulse-shape metrics to define the next forward-model
experiment.

## Matched GD-1 background and profile diagnostics

These results are internal validation evidence and factual caveats, not headline paper
results or a real-data density-profile constraint.

| Claim | Value | Source | Status |
|---|---:|---|---|
| Smooth-background sensitivity | held-out control-region primary score improves by up to 64%; morphology by up to 32% | `outputs/profile_grid/GD1/gd1_background_calibration_expanded_multiseed.json`; `scripts/run_gd1_background_calibration.py` | INTERNAL VALIDATION |
| Fixed candidate robustness | 0/24 encounter/background combinations improve both local diagnostic families across the three-seed mean | `outputs/profile_grid/GD1/gd1_background_sensitivity_single_shortlist.json`; `scripts/run_gd1_background_sensitivity_shortlist.py` | INTERNAL VALIDATION |
| Two-arm real-stream substrate | matched leading+trailing `streamdf` null spans the observed GD-1 range; localized `streamgapdf` impacts and continuous `scale_radius_kpc` are implemented | `src/simulation/stream_gen.py`; `scripts/run_gd1_streamgapdf_profile_screen.py` | INTERNAL VALIDATION |
| Initial continuous-profile screen | no tested real-GD1 profile improves both local diagnostic families | `outputs/profile_grid/GD1/gd1_streamgapdf_profile_*.json` | INTERNAL VALIDATION |
| Frozen next-stage validation | fixed 48-cell profile challenge, nuisance recovery, and no-impact FPR scripts implemented; no result claimed yet | `scripts/run_gd1_streamgapdf_injection_recovery.py`; `scripts/run_gd1_streamgapdf_null_fpr.py` | READY, PENDING RUN |

## Timeline forward model and multistream significance

| Claim | Value | Source | Status |
|---|---:|---|---|
| Injection-recovery | planted impact recovered at rank 0/36 | `outputs/injection_recovery/GD1/injection_recovery_results.json` | CAVEAT |
| Real GD-1 gap localization | phi1 about 49.9 deg, depth about 0.37 | `outputs/forward_model/GD1/forward_model_results.json` | CAVEAT |
| Real GD-1 attribution | gap real; single-subhalo attribution not significant after LE | forward-model significance outputs | CAVEAT |
| Seven-stream joint result | Stouffer Z=1.40, Fisher p=0.28, incoherent | `outputs/multistream/joint_significance_corrected.json` | CAVEAT |
| Naive to LE collapse | Sylgr 18.9 -> -1.3; Pal5 11.8 -> 1.0 | same | CAVEAT |

Forward-model caveat: these numbers use the corrected particle-spray + impulse
forward-model backend, not the `streamgapdf` detector-training backend. They are
valid as the current methods/upper-limit result, but should not be described as
fully ported to the validated DF simulator.

## Kinematic frontier

| Claim | Value | Source | Status |
|---|---:|---|---|
| Proper-motion kink AUC, noise-free | about 0.99-1.0 | `scripts/kinematic_signal_test.py` | CAVEAT |
| Proper-motion kink AUC, Gaia-like noise | about 0.49 | same / Figure 19 | CAVEAT |
| Density gap AUC, noise-free/noisy | about 0.96 / 0.70 | same | CAVEAT |
| Decision | do not retrain on kinematics yet | report Section 12.2 | OK |

Kinematic caveat: this diagnostic uses simplified synthetic proper-motion noise; a
publication-strength kinematic claim should use per-star Gaia error distributions
and foreground/membership perturbations.

## Literature claims

| Claim | Reference | Status |
|---|---|---|
| `streamdf` action-angle stream model | Bovy (2014) | LIT |
| `streamgapdf` single-gap model | Sanders, Bovy & Erkal (2016) | LIT |
| Erkal-Belokurov impulse | Erkal & Belokurov (2015) | LIT |
| concentration-mass relation | Ludlow et al. (2016) | LIT |
| galpy / MWPotential2014 | Bovy (2015) | LIT |
| galstreams tracks | Mateu (2023) | LIT |
| GINEConv | Hu et al. (2020) | LIT |
| STREAMFINDER catalog | Ibata et al. (2021) | LIT |
| GD-1 gap-and-spur | Bonaca et al. (2019); Price-Whelan & Bonaca (2018) | LIT |
| subhalo mass-function constraints | Banik et al. (2021); Carlberg (2012) | LIT |
| Gaia DR3 | Gaia Collaboration (2023) | LIT |

## Outstanding before submission

- Expand SIDM morphology beyond the current exploratory grid.
- Stress-test the abundance forecast against completeness and rate-normalization
  assumptions.
- Make the forward-model backend caveat visually explicit in any journal draft.
- Rebuild `mnras_paper.tex` from the current condensed manuscript before submission.
- Run `pytest -m "not slow"` and archive exact figure-generation commands.
