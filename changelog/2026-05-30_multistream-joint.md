# 2026-05-30 Multi-stream joint significance (and why it is an upper limit)

Subhalo abundance is a population property, so the scientifically meaningful
question is not "was this stream hit?" but "across the stream population, is
there perturbation beyond a no-impact null?". This adds the machinery to combine
per-stream forward-model significances, and -- importantly -- documents the
confounds that make a naive joint detection claim invalid.

## What was added
- `src/forward_model/multistream.py`: `combine_z_stouffer`, `combine_p_fisher`,
  and `combine_streams` (headlines Fisher's combination of EMPIRICAL p-values,
  with a coherence check; `MultiStreamResult`).
- `pipeline.build_data_null`: a data-driven best-of-grid null that removes only
  *localized* structure from the REAL stream (phi1 resampled from the smoothed
  density envelope; pm/vrad shuffled), valid despite the sim-to-real offset.
- `scripts/run_multistream_analysis.py`: per-stream detect -> localise -> grid
  -> null -> significance over the 7 streams, then combine.
- `tests/test_multistream.py` (11): combination math + the key tests that
  look-elsewhere correction and the coherence gate prevent false detections.

## Result (7 streams, fast mode, data-driven null)
Per-stream look-elsewhere z: GD1 -1.9, Pal5 +0.4, Orphan +29, ATLAS +1.5,
Jhelum +2.9, Fjorm -4.8, Sylgr +2.4. Fisher p = 0.004, Stouffer Z = 11.3.

**This is NOT a detection.** The per-stream significances are *incoherent*
(71% positive, range z = [-4.8, +29]); a genuine population signal would be
consistently positive. The apparent Fisher excess is driven by a few streams
pinned at the empirical-p floor, which we trace to confounds, not subhalos.

## Why naive significance over-claims here (the cautionary result)
Three independent effects each inflate a naive significance, and the analysis
controls for all of them:
1. **Look-elsewhere**: picking the best of many (mass, time, phi1) candidates
   fits noise; the corrected per-stream z collapses to ~0 (Section: improvement B).
2. **Sim-to-real gap**: the best candidate fits the *real* stream far worse than
   it fits a synthetic no-impact stream, so a synthetic null is invalid (it gave
   nonsensical z ~ -60). The null must be built from the real data.
3. **Null construction**: even a data-driven null over-removes structure --
   smearing phi1 broadens the stream; shuffling pm destroys the intrinsic smooth
   kinematic gradient -- inflating significance. Hence the coherence gate:
   incoherent per-stream signs => artifact, not signal.

## Honest conclusion
With realistic Erkal kicks, look-elsewhere correction, a data-driven null, and a
coherence requirement, neither single streams nor the 7-stream joint analysis
shows a coherent subhalo-impact detection in the current data. The result is an
**upper limit / consistency with CDM**, limited by the sim-to-real gap, the
degenerate GD-1 membership catalog, and sparse radial-velocity coverage. A
publication-grade joint constraint needs a structure-preserving null and the
domain-randomised (sim-to-real-robust) detector. This careful null result -- and
the demonstration of how each confound would otherwise manufacture a false
signal -- is itself the postable contribution.
