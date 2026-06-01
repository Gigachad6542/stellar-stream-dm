# Path to publishable shape — working checklist

Updated each autonomous wake-up. `[ ]` todo, `[~]` in progress, `[x]` done.

## Pipeline (blocking the Results section)
- [~] Precompute error-DR profile cache on v3 (`profile_features_v3_errordr.h5`) — RUNNING
- [ ] Re-train detector: `train_v2 --error-dr --profile-features-path <cache>` on v3
- [ ] Calibrate: `calibrate_detector --error-dr` on the new checkpoint
- [ ] Evaluate detector: AUC, calibration (reliability), real-stream scores → §4.3
- [ ] Re-run `run_multistream_analysis` on v3 + new detector → §7 results
- [ ] Re-run injection–recovery on v3 → §5 numbers
- [ ] (Optional) SBC / TARP coverage refresh → §4.2

## Manuscript (can progress in parallel — doc only)
- [x] Scaffold + abstract/intro/methods/data/limitations draft (`manuscript.md`)
- [ ] Tighten Introduction with proper citations + figure callouts
- [ ] Methods: confirm all numbers against code/changelogs; remove any drift
- [ ] Results: fill `[PENDING v3]` as pipeline outputs land
- [ ] Limitations: ensure every systematic is honestly stated
- [ ] Conclusions
- [ ] Assemble bibliography (BibTeX)
- [ ] Decide venue/format (ApJ/MNRAS LaTeX vs arXiv) and convert from Markdown

## Figures (publishable quality)
- [ ] Inventory existing `_report_figures/` vs what the paper needs
- [ ] Fig: track-6D IC validation (sim vs track PMs) — Table 1 + figure
- [ ] Fig: sim-vs-real stream comparison (the `_simreal_gd1` figure, refreshed)
- [ ] Fig: detector calibration before/after error-DR
- [ ] Fig: timeline forward-model schematic + injection–recovery
- [ ] Fig: per-stream + joint significance (coherence-gated)
- [ ] Ensure all figures regenerate from a script (reproducibility)

## Repo / reproducibility (mostly done)
- [x] Scripts index, README refresh, gitignore, storage cleanup
- [x] Prune obsolete dev scripts
- [ ] Light `src/` dead-code pass — DO ONLY WHEN NO COMPUTE IS RUNNING
- [ ] Regenerate + update the full PDF report (`generate_pdf.py`) to match v3
- [ ] Verify full test suite passes (`pytest -m "not slow"`) before any release

## Decisions to surface to the human (don't block on these overnight)
- Venue/format choice.
- Whether to obtain an external GD-1 membership catalog before submission.
- Whether the static-cache error-DR trade is acceptable for the headline result
  or a fully-dynamic re-train is warranted.

## Wake-up log
- 2026-06-01 ~01:25 — Set up autonomous mode. Precompute running. Drafted
  manuscript scaffold + this checklist.
