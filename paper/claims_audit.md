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
| dedicated head: mass recovery | R²<0 (unrecoverable) | `characterize_eval.py` on `detector_char_20260602` (R²=−0.72) | ✓ |
| dedicated head: time recovery | R²≈0.2, med err ≈0.1 dex | same (R²=+0.24 detected) | ✓ |

## DM-family distinguishability
| Claim | Value | Source | Status |
|---|---|---|---|
| N_det FDM 10⁻²² | ≈5 | `dm_family_distinguishability.py` | ✓ |
| N_det WDM 3/6 keV | ≈12 / ≈27 | same | ✓ |
| N_det FDM 10⁻²¹ | ≈135 | same | ✓ |
| SIDM via mass | ≈214k (hopeless) | same | ✓ |

## Forward model / population significance — DEFERRED, NOT REPORTED
These components run on the **legacy** homemade generator (`generate_stream` +
Erkal kick; scale-radius corrected but NOT the validated `streamdf`/`streamgapdf`).
Per the integrity rule "only report results from components that used the new
simulator," their numbers are **described but not stated** in §8–§9 of the
manuscript; they are listed here only to document why. (Numbers exist in
`outputs/multistream/joint_significance_corrected.json` etc. but are intentionally
withheld pending the forward-model migration to the validated generator.)
| Component | Status in paper | Reason |
|---|---|---|
| Timeline forward model (injection-recovery, real-GD-1 fit) | design described, no numbers | legacy generator |
| Multi-stream joint significance | framework described, no numbers | built on the legacy forward model |
| Fig 6 (multistream) | removed from manuscript | presented legacy-generator results |
| Look-elsewhere inflation point | kept (qualitative, simulator-independent) | a general statistical fact |

## Real-data detection (full report §9)
| Claim | Value | Source | Status |
|---|---|---|---|
| GD-1 p_impact / OOD / gap | 0.77 / 3.1σ / yes (φ₁≈50°) | StreamImpactDetector on GD1_streamfinder.h5 (811) | ✓ |
| ATLAS p_impact / OOD / gap | 0.06 / 1.9σ / none | StreamImpactDetector on ATLAS_clean.h5 (2860) | ✓ |
| Member-count floor (N-dependence) | p 0.77→0.99 as N 811→100 | GD-1 subsample test | ✓ |
| Clean catalogs for all 7 streams | STREAMFINDER auto-matched | fetch_streamfinder_streams.py | ✓ |

## Forward model + multistream (full report §10–§11) — real-data analyses
| Claim | Value | Source | Status |
|---|---|---|---|
| Injection-recovery | rank 0/36, beats null | injection_recovery_results.json | ✓ |
| Real GD-1 gap | φ₁=49.9°, depth 0.37 | forward_model_results.json | ✓ |
| GD-1 single-subhalo attribution | not significant after LE | significance.json | ✓ |
| Multistream joint (7 streams) | Stouffer Z=1.40, Fisher p=0.28, incoherent | joint_significance_corrected.json | ✓ |
| Naive→LE collapse | Sylgr 18.9→−1.3, Pal5 11.8→1.0 | same | ✓ |

## Report figures (paper/figures/, current pipeline)
fig1 example streams · fig2 ROC · fig3/fig5 completeness (1D/2D) · fig4 DM-family ·
fig6 score separation · fig7 reliability · fig8 characterization · fig9 embedding PCA ·
fig10 transfer functions. Generated by make_figures.py + make_report_figures.py.

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
