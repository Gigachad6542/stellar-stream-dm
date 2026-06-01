# Path to publishable shape — working checklist

Updated each autonomous wake-up. `[ ]` todo, `[~]` in progress, `[x]` done.

## Pipeline (blocking the Results section)
- [x] Precompute error-DR profile cache on v3 → `data/processed/profile_features_v3_errordr.h5.npz` (100000×157, 43min)
- [~] Re-train detector: `train_v2 --error-dr --profile-features-path <cache>` on v3 — RUNNING (2.34 batches/s, ~4.5min/epoch, ~6h ETA, ckpt gnn_v2_timeline_errordr_v3_20260601)
- [ ] Calibrate: `calibrate_detector --error-dr` on the new checkpoint
- [ ] Evaluate detector: AUC, calibration (reliability), real-stream scores → §4.3
- [ ] Re-run `run_multistream_analysis` on v3 + new detector → §7 results
- [ ] Re-run injection–recovery on v3 → §5 numbers
- [ ] (Optional) SBC / TARP coverage refresh → §4.2

## Manuscript (can progress in parallel — doc only)
- [x] Scaffold + abstract/intro/methods/data/limitations draft (`manuscript.md`)
- [x] Tighten Introduction with proper citations (prose + \citep) 
- [ ] Methods: confirm all numbers against code/changelogs; remove any drift
- [ ] Results: fill `[PENDING v3]` as pipeline outputs land
- [ ] Limitations: ensure every systematic is honestly stated
- [ ] Conclusions
- [x] Assemble bibliography (BibTeX) → `paper/references.bib` (19 refs)
- [ ] Decide venue/format (ApJ/MNRAS LaTeX vs arXiv) and convert from Markdown

## Figures (publishable quality) — 22 exist in `_report_figures/`
REUSE (methodology, v3-independent): erkal_kick, pipeline/architecture,
timeline_pipeline, mass_function, transfer_function, multiepoch_rv, mc_time,
stream_properties.
REGENERATE after v3 (results-dependent):
- [ ] `_simreal_gd1` — sim-vs-real with the corrected track-6D generator
- [ ] `ood_fix` / detector calibration — after the v3 retrain
- [ ] `significance` — per-stream + joint (coherence-gated), v3 multistream
- [ ] `mock_recovery` — injection–recovery on v3
- [ ] `coverage` / `sbi_rounds` — if SBC/TARP refreshed
- [ ] Table 1 (track-6D IC validation) — already in manuscript; optional figure
- [ ] Confirm each regen has a driving script (reproducibility)

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
- 2026-06-01 ~02:10 — Precompute DONE (cache 100000×157). Found+fixed the 45h
  slowdown: error-DR was recomputing profiles on-the-fly every batch; with the
  precomputed cache + `--profile-features-path`, throughput went 0.29→2.34
  batches/s. Full 80-epoch training relaunched. Figure inventory done.
- 2026-06-01 ~02:51 — NOTE: ScheduleWakeup did NOT autonomously fire (user had to
  prod). Switching heartbeat to background-task-completion notifications, which
  are reliable. Training healthy at epoch 8/80, steady ~8-9 min/epoch (the 2.34
  batches/s was an epoch-1 transient; real ~1.1 b/s) → ~10h, ETA ~13:00. Paper:
  wrote references.bib (19 refs) + rewrote Introduction as cited prose.
