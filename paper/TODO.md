# Path to publishable shape — working checklist

`[ ]` todo, `[~]` in progress, `[x]` done. Rebuilt 2026-06-02 after the
corrected-simulator pipeline superseded all v3 results.

## Integrity rule (2026-06-02)
The paper reports numbers ONLY from components that used the validated
`streamdf`/`streamgapdf` simulator (detector, completeness, characterization,
DM-family, real-GD-1 detection). The **timeline forward model** and **multi-stream
significance** still run on the legacy homemade generator, so their results
(injection-recovery, real-GD-1 fit, joint significance, Fig 6) are **described but
NOT stated** in §8–§9. *Principal remaining engineering step: migrate the forward
model's rewind/re-impact/re-evolve loop onto the validated generator, then report.*

## Status: manuscript reports CURRENT-STATE results only
No development history / before-after numbers appear in the paper.

## Core results (all on the validated simulator)
- [x] Simulator on validated DFs (streamdf/streamgapdf) + validation harness
- [x] Detector: **test AUC 0.982**, zero false-positive operating point
- [x] Real GD-1 in-distribution (3.1σ) + gap detected (φ1≈50)
- [x] Completeness map across impact type (mass/time/b/strength)
- [x] Characterization: mass unrecoverable (R²<0), time weakly (R²≈0.2)
- [x] DM-family population distinguishability (N_det per model)
- [~] Forward model + multistream: DESCRIBED only (legacy generator) — see integrity rule

## Manuscript (`manuscript.md`)
- [x] Full rewrite to current state, dual-register (plain-language boxes + Methods)
- [x] Abstract (plain-language summary + technical abstract)
- [x] All sections §1–§11 + Methods + Reproducibility
- [x] §6 finalized with dedicated-head result (mass R²<0, time R²≈0.2)
- [x] Tone pass (academic body) + citation audit (`claims_audit.md`); all cites resolve
- [x] Regenerated `mnras_paper.tex` from manuscript.md (ASCII-clean, Overleaf-ready)

## Figures (`paper/make_figures.py` → `paper/figures/`)
- [x] fig1 what an impact looks like (smooth vs gapped + density profile)
- [x] fig2 detector ROC (0.98)
- [x] fig3 completeness vs mass & time
- [x] fig4 DM-family distinguishability (distributions + N_det)
- [ ] Optional: pipeline schematic; characterization recovery curve (after retrain)

## Repo / reproducibility
- [x] All analysis scripts committed with driving code
- [ ] Verify `pytest -m "not slow"` passes before any release
- [ ] Regenerate the full PDF report to match the new manuscript (if kept)

## Decisions to surface to the human
- Venue/format (ApJ/MNRAS/arXiv).
- Whether to pursue kinematic (proper-motion) detector features to extend
  completeness to weak/old impacts (the §5/§10 frontier).
- Whether to obtain more clean external membership catalogs (needed for the
  population sample sizes of §7).
