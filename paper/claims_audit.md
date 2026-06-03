# Claim-traceability audit (manuscript.md)

Every quantitative claim in the manuscript mapped to its source. "Source" is a
machine artifact (a JSON/checkpoint/script output) or a literature reference.
Verified 2026-06-02 against the corrected-simulator pipeline.

Legend: ✓ verified against artifact · 📖 literature · ⏳ pending (retrain in flight).

## Detector performance
| Claim | Value | Source | Status |
|---|---|---|---|
| Corrected detector test AUC | 0.982 | `checkpoints/detector_df_20260602/calibration.json` (auc 0.9822) | ✓ |
| Calibration temperature | T=0.54 | same (`temperature` 0.541) | ✓ |
| Best validation accuracy | 0.951 | `…/gnn_v2_training_history.json` (max val_binary_acc 0.951) | ✓ |
| v3 detector AUC (prior, broken sims) | 0.62 | `changelog/2026-06-01_v3-retrain-and-results.md` (0.618) | ✓ |
| Zero false-positive rate | 0.000 | `scripts/detector_completeness.py` output | ✓ |

## Generator / simulator
| Claim | Value | Source | Status |
|---|---|---|---|
| Generator separability (corrected) | G6 AUC 0.94 | `scripts/validate_df_generator.py` (0.938) | ✓ |
| Generator separability (bespoke) | 0.57 | `changelog/2026-06-02_public-generator-streamgapdf.md` | ✓ |
| Smoothness excess over Poisson | 0.01 | `validate_df_generator.py` (G1 0.007–0.017) | ✓ |
| scale-radius bug magnitude | 10⁸M⊙→~0.3pc vs ~0.25kpc | `src/simulation/subhalo.py` `scale_radius_from_mass` | ✓ |
| Foreground collapses separability | G6 0.96→0.56 | `changelog/2026-06-02_detector-data-pipeline.md` | ✓ |
| nTrackChunks=5, isochrone b≈0.61 | — | this-session generator runs | ✓ |
| Dataset size / balance | 15,830 (7,830 imp / 8,000 smooth) | chunk inspection `data/simulations_detector_df` | ✓ |
| Supported streams | GD1, ATLAS, Jhelum, Orphan | smoke test (Pal5/Fjorm fail isochrone; Sylgr wraps) | ✓ |

## Real GD-1
| Claim | Value | Source | Status |
|---|---|---|---|
| p_impact (range over cuts) | 0.77–0.99 | StreamImpactDetector run (0.774); timeline run (0.985) | ✓ |
| OOD max σ (in-distribution) | 3.1σ (det.) / 2.7σ (timeline) | this-session detector + timeline runs | ✓ |
| OOD prior pipeline | 391σ raw / 13σ error-DR | `project memory` / 2026-05-30 changelog | ✓/📖 |
| Gap location, significance | φ₁≈50°, sig 37 | timeline run on STREAMFINDER | ✓ |
| Catalog | 811 members | `data/processed/GD1_streamfinder.h5` | ✓ |

## Completeness (test split)
| Claim | Value | Source | Status |
|---|---|---|---|
| Overall completeness | 0.57 | `detector_completeness.py` | ✓ |
| vs mass | 0.48→0.72 (10⁷·⁵→10⁸·⁵) | same (0.478→0.723) | ✓ |
| vs time | 0.75→0.39 (0.3→1.4 Gyr) | same (0.747→0.389) | ✓ |
| step in gap strength | 0.05→1.00 | same | ✓ |

## Characterization (embedding probe)
| Claim | Value | Source | Status |
|---|---|---|---|
| mass R² (detected) | ≈0.18 | `characterize_probe.py` (0.183) | ✓ |
| impact-parameter R² | ≈0.02 | same (0.024) | ✓ |
| time R² | ≈0.04 | same (0.043) | ✓ |
| dedicated mass_time recovery (§6) | — | `characterize_eval.py` on `detector_char_20260602` | ⏳ retrain finishing |

## DM-family distinguishability
| Claim | Value | Source | Status |
|---|---|---|---|
| N_det FDM 10⁻²² | ≈5 | `dm_family_distinguishability.py` | ✓ |
| N_det WDM 3/6 keV | ≈12 / ≈27 | same | ✓ |
| N_det FDM 10⁻²¹ | ≈135 | same | ✓ |
| SIDM via mass | ≈214k (hopeless) | same | ✓ |

## Forward model / population significance
| Claim | Value | Source | Status |
|---|---|---|---|
| Injection-recovery | rank 0/36, +90% over null | `run_injection_recovery.py` (90.4%) | ✓ |
| Real GD-1 single-subhalo fit | +3.2% over null | `run_timeline_forward_model.py` | ✓ |
| Multistream joint | Stouffer Z=1.40 (p=0.08), Fisher p=0.28 | `outputs/multistream/joint_significance_corrected.json` (1.396 / 0.277) | ✓ |
| Incoherent, frac positive | 71% | same (0.714) | ✓ |
| Naive z (collapse) | Sylgr 18.9, Pal5 11.8 → ≈±2 | same (per-stream z & le_z) | ✓ |

## Literature claims (cited)
| Claim | Reference | Status |
|---|---|---|
| `streamdf` action-angle stream model | Bovy (2014) | 📖 |
| `streamgapdf` single-gap model | Sanders, Bovy & Erkal (2016) | 📖 |
| `streamspraydf` particle spray | Fardal et al. (2015) | 📖 |
| Erkal–Belokurov Plummer impulse | Erkal & Belokurov (2015) | 📖 |
| concentration–mass relation | Ludlow et al. (2016) | 📖 |
| galpy / MWPotential2014 | Bovy (2015) | 📖 |
| galstreams tracks | Mateu (2023) | 📖 |
| GINEConv | Hu et al. (2020) | 📖 |
| STREAMFINDER GD-1 catalog | Ibata et al. (2021) | 📖 |
| GD-1 gap-and-spur | Bonaca et al. (2019); Price-Whelan & Bonaca (2018) | 📖 |
| subhalo mass-function constraints | Banik et al. (2021); Carlberg (2012) | 📖 |
| Gaia DR3 | Gaia Collaboration (2023) | 📖 |

## Outstanding (before submission)
- §6 dedicated-head recovery number (⏳ retrain `detector_char_20260602`).
- 391σ/13σ prior-OOD figures: confirm against the 2026-05-30 changelog wording.
- Regenerate the LaTeX from this markdown (done: `mnras_paper.tex`).
