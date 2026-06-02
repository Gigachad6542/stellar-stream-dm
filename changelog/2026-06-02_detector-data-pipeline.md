# 2026-06-02 Detector data pipeline on streamdf/streamgapdf

scripts/generate_detector_data.py — parallel, chunked, RESUMABLE generator for a
balanced binary detector dataset, built on the validated streamdf/streamgapdf
engine (replaces the homemade rewind/kick).

## Design
- 50/50 no-impact (streamdf) / impact (streamgapdf), round-robin over the
  df-supported streams: GD1, Orphan, ATLAS, Jhelum.
  (Pal5/Fjorm fail the isochrone action-angle approx -- near-circular GC orbits;
   Sylgr wraps in phi1. Excluded by allow-list.)
- Each sim labelled by realised gap-depth (impact_strength) + Poisson-excess.
- Per-worker streamdf base cache (12s setup amortised); ~0.85 s/sim wall on -2 jobs.
- Reuses generate_training_data's pool/DLL-fix/write_simulation; resumes from chunks.

## Noise regime (important)
- --noise gaia  (DR3 pm/dist errors, NO foreground, RV->NaN): G6 AUC 0.96. <-- USE THIS
- --noise full  (5-30% foreground): G6 collapses to 0.56 -- foreground fills the
  phi1 gap, exactly the real-GD-1 false-negative mechanism. The detector must run
  on CLEAN membership-selected catalogs (STREAMFINDER lesson), so we train clean.
- RV is dropped by the Gaia model (RVS misses faint stars) -> consistent with the
  detector's ignore_rv=True mode.

## Validation (scripts/validate_generator.py on the written chunks)
  G1 smoothness-excess 0.017 PASS | G2 width 0.43 PASS | G6 AUC 0.96 PASS
  G4 (exact pm) below spec = accepted action-angle morphology limit (Path A).

## Next
- Full run: generate_detector_data.py --n-sims 20000 --noise gaia
  --output-dir data/simulations_detector_df  (~5h, resumable).
- Retrain detector on impact_detectable; expect real AUC gain (clean separable signal).
- Re-run v3 conclusions (timing, multistream) on corrected impacts.
