# 2026-05-31 Training-data regeneration with the fixed generator

The final piece of the "both" directive (Fardal-spray refactor + membership
cleaning + chunk regeneration). The old training chunks (`data/simulations_v2_plan2`)
were generated with the broken 5D-RMS progenitor IC optimiser, which produced
kinematically-wrong orbits (e.g. GD-1 at pm1 ~ -8.9 instead of -12.8). They must
be regenerated before any re-training.

## Small-batch validation (12 sims, n_jobs=1)
Ran a 12-sim test batch through the full `generate_training_data.py` pipeline to
confirm the corrected generator (track6d IC + calibrated `spray_age_gyr` +
structured spray) works end-to-end in the training path, not just in isolation.
The pipeline generates several stream types and each lands on its literature track:

| sim type | phi1 range | pm1 med | pm2 med | matches |
|---|---|---|---|---|
| GD-1-like | [-29, 90] | -11.3, -10.6 | -3.0, -2.7 | GD-1 (-12.8, -3.3), full extent |
| Pal 5-like | [-20, 15] | +3.9, +4.4 | +0.3, +0.6 | Pal 5 (+3.6) |
| Jhelum-like | [-10, 28] | -7.0 | +2.8 | Jhelum (-7.45, +3.37) |
| ATLAS-like | [-28, 30] | -0.2, -0.7 | -1.0 | ATLAS (pm2 -1.04) |
| Orphan-class | [-48, 50] | +3-4 | -1.0 | wide/long |

Kinematics are now correct by construction (vs the old chunks' wrong orbits) and
GD-1 reaches phi1 ~ 90 deg (real extent). Residual: phi2 width is still larger
than observed for the long streams (the documented Fardal width-vs-length tension).

## Full regeneration (running)
Launched the full 100k-sim run (25000/model x 4 models, chunk_size 250, n_jobs 8)
to a FRESH directory `data/simulations_v3_track6d` (v2_plan2 preserved for
comparison rather than clobbered). Benchmarked at 1.68 sims/s wall-clock -> ~16.5 h.
Progress in `logs_regen_v3.txt`. Detached background process.

- `config/training.yaml` `output_dir` repointed to `data/simulations_v3_track6d`
  so the eventual re-train (and `precompute_profile_features` / `train_v2`) use the
  corrected data. NOTE: wait for the run to finish before re-training.

## Next (after regen completes)
1. `precompute_profile_features.py --error-dr` on the v3 chunks.
2. Re-train detector (`train_v2.py --error-dr`) and re-calibrate.
3. Re-run multi-stream joint analysis with the corrected sims + retrained detector
   to see whether the sim-to-real gap (and the incoherent per-stream z) narrows.

## Speedup: BLAS thread-pinning (output-identical)
While the regen ran, profiling showed CPU pegged at 100% with only 8 workers on a
24-core/32-thread i9-14900KF: BLAS/OpenMP was auto-threading inside each worker, so
8 workers x ~4 threads oversubscribed all logical cores and burned cycles on
contention (this is why a naive high n_jobs had been *slower*, and 8 looked optimal).

Fix (in `generate_training_data.py`, before numpy import): pin
OMP/OPENBLAS/MKL/NUMEXPR/VECLIB threads to 1 via `os.environ.setdefault`, then run
one worker per physical core (`n_jobs: 22`, 2 reserved for the OS).

Verified OUTPUT-IDENTICAL: same seed gives a bit-for-bit identical particle hash
(`3c7c2804...94c4d`) with threads=1 vs auto (the physics is a seeded RNG + galpy C
dop853 integrator, not threaded BLAS). Confirmed the galpy C extension is active
(`integrateFullOrbit._ext_loaded = True`) — no Python-odeint fallback.

Measured: 1.68 -> 2.14 sims/s (~1.28x), full-run ETA ~16.5 h -> ~11.5 h, and the
machine stays responsive (CPU ~71% instead of 100%). The regen was restarted with
the new settings; it resumed from the 43 existing chunks (deterministic per-run_id
seeds -> identical sims), losing nothing.

## COMPLETE (2026-06-01 00:45)
Full regeneration finished: 400 chunks / 100,000 sims (25k x 4 DM models), 6.8 GB,
0 worker failures, all chunks valid. Integrity spot-check (first/mid/last) passed.
Resumed portion ran in 42,289 s (~2.11 sims/s), matching the thread-pinned estimate.
The corrected v3 dataset is ready for: precompute_profile_features --error-dr ->
train_v2 --error-dr -> calibrate_detector -> re-run multi-stream analysis.
