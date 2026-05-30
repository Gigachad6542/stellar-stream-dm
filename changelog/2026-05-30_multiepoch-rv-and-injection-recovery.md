# 2026-05-30 Multi-epoch real data, RV scoring, and injection-recovery

Two threads landed together: the injection-recovery validation (timeline
deliverable 2) and real multi-epoch / multi-survey data ingestion so the
"rewind" is constrained by the best available present-day phase space.

## Why multi-epoch data

Backward orbit integration diverges as `dx(t) ~ dv * t`, so velocity precision
sets how well we can reconstruct the un-impacted past stream and date the impact.
Two real, public levers:

- **Radial velocity** — the 6th phase-space dimension, *entirely empty* (`vrad`
  all NaN) in the base Gaia DR3 membership tables. Real RVs come from Gaia DR3
  RVS (bright members) and the S5 survey (southern streams).
- **Proper-motion epochs** — Gaia DR2 gives a second astrometric epoch; PM error
  scales `~baseline^-1.5`, so DR4 (~2.7x) and DR5 (~6.7x) will tighten PMs vs DR3.

## What changed

### New: `src/data/multi_epoch.py` (real fetchers + fusion)
- `fetch_gaia_dr3_rvs(source_ids)` — real Gaia DR3 RVS by source_id (batched async, cached).
- `fetch_gaia_dr2_proper_motions(...)` — DR2 second epoch via `gaiadr3.dr2_neighbourhood` (best neighbour).
- `fetch_s5_rvs()` — S5 survey RVs from VizieR `J/MNRAS/490/3508` (Li+2019), table `s5dr1`
  (Gaia id + `velcalib`/`vel50`, `good_star` filter, sentinel/speed cuts).
- `inverse_variance_combine`, `fuse_radial_velocities`, `fuse_proper_motions_icrs`,
  `pm_error_scaling_factor` (DR2/DR3/DR4/DR5 baselines).

### New: `scripts/fetch_multi_epoch.py`
Fetches + fuses real catalogs per stream and writes a drop-in augmented stream
file `data/processed/{stream}_multiepoch.h5` with `vrad`/`e_vrad`/`rv_mask`
populated from real measurements, plus a provenance JSON.

### New: radial-velocity scoring term
- `src/forward_model/scoring.py`: `radial_velocity_score()` (binned vrad-track
  residual, active only where observed RVs exist) + `ScoreResult.radial_velocity`
  + `ScoreWeights.radial_velocity` (0.8). `combined_score()` folds RV into the
  weighted sum **only when present**, so RV-less streams are unaffected.
- `src/forward_model/pipeline.py`: all five `combined_score` call sites
  (worker, null, candidate, model-comparison, multi-encounter) now pass
  `sim_vrad`/`obs_vrad`; `radial_velocity` is serialised through the parallel
  worker path and `CandidateResult.to_dict`.

### New: injection-recovery (timeline deliverable 2)
- `src/forward_model/injection.py`: `generate_injected_stream` +
  `run_injection_recovery` + `InjectionRecoveryResult`. Injects a known impact,
  runs the full pipeline, reports recovered params + errors + truth rank.
- `pipeline.prepare(observed_override=...)`: inject a synthetic stream as the
  observed data (no HDF5 needed).
- `scripts/run_injection_recovery.py`: CLI for full experiments.

### Tests
- `tests/test_multi_epoch.py` (12): inverse-variance fusion, PM-error scaling,
  RV fusion/alignment, RV score active/inactive.
- `tests/test_injection_recovery.py` (2, slow): strong-signal recovery (beats
  null, truth in top 3, mass+phi1 within one grid step) and offset-phi1 case.
- Full non-slow suite: **248 passed, 3 deselected**.

## Real-data results

- **ATLAS**: 281 members now have real RVs (Gaia RVS 15 + S5 296), median error
  2.4 km/s; `data/processed/ATLAS_multiepoch.h5` written.
- **Jhelum**: 206 members matched to S5.
- GD-1 / Pal 5 / Orphan: not in S5's southern footprint; GD-1 needs Gaia RVS
  (sparse) + APOGEE/dedicated catalogs (follow-up).
- Running the forward model on `ATLAS_multiepoch.h5` activates the RV term:
  top candidate `radial_velocity ≈ 15 km/s` RMS residual (dominant in the score).
- Injection-recovery (fast, sim-sim): RV residual ~0.7 km/s — confirming the
  ~15 km/s on real ATLAS is a genuine sim-vs-real offset.

## Known limitations / follow-ups

1. **RV zero-point / frame**: the ~15 km/s real-ATLAS residual likely includes a
   convention mismatch between the simulated heliocentric `vrad` and S5 `vlos`
   (heliocentric vs GSR). The RV term must be zero-point-calibrated before it is
   treated as physically tuned; its weight (0.8) may then need revisiting.
2. **DR2 PM fusion** is implemented in ICRS but not yet folded into the stored
   stream-frame `pm1/pm2` (needs the frame transform). Main PM gain is future (DR4/DR5).
3. **GD-1 RVs**: add APOGEE/DESI/dedicated catalogs (S5 does not cover GD-1).
4. Still open from the timeline plan: deliverable 3 (null distribution +
   multi-seed statistical rigor) and a Monte-Carlo uncertainty-aware impact-time
   estimate that propagates per-star errors (so multi-epoch precision maps to a
   tighter impact-time posterior).
