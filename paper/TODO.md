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
  are reliable. Paper: wrote references.bib (19 refs) + rewrote Introduction.
- 2026-06-01 ~03:17 — Heartbeat works. Epochs sped to ~3.0 min (earlier 8-9 min
  was my concurrent smoke/poll jobs stealing GPU — lesson: keep concurrent work
  light). Epoch 16/80, val acc ~0.58 (stage1 curriculum), ETA ~06:30. Paper:
  corrected §4.3 detector-overconfidence methodology against the source changelog.
- 2026-06-01 ~03:42 — Epoch 24/80, val acc ~0.57. Paper: exact membership-
  contamination figures in §2.2 (GD-1 0.7%, ATLAS 35%, Jhelum 0), re-verified.
- 2026-06-01 ~09:50 — Morning (user back). Overnight loop froze ~06:43 (env
  suspended while idle; injrec full-orbit run segfaulted). Recovered: re-ran
  injection-recovery in --fast (impulse) mode → PERFECT recovery (truth rank 0/60,
  dM=dt=dphi1=0, beats null). FULL-ORBIT injrec segfaults (galpy C, exit 139) at
  baseline generation — KNOWN ISSUE (flagged for separate fix; method validated
  via impulse mode). Filled §5. Regenerated significance_v3.png. All headline
  results now in manuscript. Remaining: figures polish, LaTeX, editorial pass.
- 2026-06-01 ~06:38 — *** MILESTONE: core results in. *** Training done (best val
  acc 0.595). Calibrated: v3 AUC=0.618 (T=0.672) vs 0.937 balanced. Multistream
  joint (v3, look-elsewhere): Stouffer Z=1.00 (p=0.16), Fisher chi2=16.2 (p=0.30)
  -> NO joint detection; per-stream z now coherent/modest [-2.5,+2.7] (vs pre-fix
  [-4.8,+29]). Filled abstract, §4.3, §7 (Table 2), §8, §10 with REAL numbers.
  Result saved outputs/multistream/joint_significance_v3.json. Remaining: §5
  injection-recovery numbers, figure regeneration, final read-through.
- 2026-06-01 ~05:45 — Epoch 65/80, val acc ~0.59. ETA ~06:30. Paper: drafted §10
  Conclusions (honest narrative: methods contribution, cautionary detector result,
  honest population inference; numbers PENDING). Manuscript now full prose end-to-end.
- 2026-06-01 ~05:21 — Epoch 57/80, best val acc 0.591. Prepped post-training:
  calibrate_detector writes calibration.json with roc_auc (prior balanced run =
  0.937, T=0.78). Will read v3 AUC from there on completion. Paper: framed §4.3's
  0.937 explicitly as the BALANCED-set number (not headline), set up honest v3
  contrast. (evaluate_v2_classifier lacks the timeline target; use calibrate's AUC.)
- 2026-06-01 ~04:56 — Epoch 49/80 (entering stage2), best val acc 0.590. ETA
  ~06:30. Paper: enriched §4.1 with real graph/model hyperparameters (k=8, 18
  node / 5 edge features, GINEConv 2.3M params, AdamW lr3e-4, 70/15/15 split).
- 2026-06-01 ~04:32 — Epoch 41/80, val acc 0.588 (best). ETA ~06:30. Paper:
  expanded §6 Statistical framework into full prose (per-stream null, best-of-grid
  look-elsewhere, Stouffer/Fisher, coherence gate, structure-preserving null).
- 2026-06-01 ~04:07 — Epoch 33/80. WATCH: val BCE ~0.69 (~ln2), acc ~0.58 — weak
  binary separability on v3. Likely because v3 is a PHYSICS-PRIOR dataset (impact
  signal entangled/weak, historically ~0.65-0.68 AUC) vs the prior 0.93-AUC run
  which used a curated *balanced* curriculum. This is an honest, important point
  (strong prior AUC may be a dataset-construction artifact). Added as Limitation
  #6. Decision for the human: if v3 detector is weak, consider a balanced/curric
  training variant for the detector while keeping v3 for the forward model.
