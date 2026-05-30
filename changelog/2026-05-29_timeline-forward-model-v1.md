# 2026-05-29 Timeline Forward Model v1 (with full orbit evolution)

## What changed

Created the timeline forward-modelling pipeline with FULL ORBIT INTEGRATION.
This implements the physically correct approach:
1. Generate stream particles (Fardal spray) from the progenitor
2. Integrate particles forward to the impact epoch (t = -t_impact)
3. Apply the subhalo velocity kick in 3D galactocentric coordinates
4. Integrate ALL kicked particles forward through the MW potential to present (t=0)
5. Convert to stream-frame observables and compare against Gaia observations

This captures the real non-linear phase-space evolution of gaps over Gyr timescales:
differential orbital precession, gap widening, caustic formation, and stream fanning.

### New files

- `src/forward_model/__init__.py`: Package init with module docstrings.
- `src/forward_model/evolve.py`: Full orbit-integrated timeline evolution.
  - `generate_perturbed_stream_evolved()`: The core physics function.
    Steps: progenitor orbit -> particle spray -> integrate to impact epoch
    -> 3D velocity kick -> integrate to present -> stream-frame conversion.
  - `_integrate_particles_to_epoch()`: Batched galpy orbit integration from
    varied start times to a common target epoch. Handles R<0.5kpc guard,
    speed cap, dop853_c -> leapfrog_c fallback.
  - `_apply_3d_velocity_kick()`: Hernquist impulse in galactocentric 3D.
    Kick direction is perpendicular to stream velocity in the orbital plane.
    Includes finite-extent correction and massive-subhalo correction.
- `src/forward_model/scoring.py`: Scoring functions for stream comparison.
  - `DensityProfile`, `GapFeature`, `ScoreResult` data containers
  - `compute_density_profile()`: 1D phi1 binning with optional membership weights
  - `detect_gaps()`: Gaussian-smoothed local-minimum detection with Poisson significance
  - `density_residual_score()`: Poisson-weighted L1 residual between profiles
  - `gap_agreement_score()`: Multi-criterion gap matching (location, depth, width)
  - `kinematic_perturbation_score()`: Binned median PM track comparison
  - `combined_score()`: Weighted combination with configurable `ScoreWeights`
- `src/forward_model/pipeline.py`: Three-stage orchestrator.
  - `ForwardModelConfig`: All hyperparameters in one dataclass, including
    `use_fast_mode` flag (False = full orbit, True = impulse approx)
  - `CandidateResult`: Per-candidate scores + metadata
  - `TimelineForwardModel`: Orchestrator class with:
    - `prepare()`: Load real data, compute observed profile + gaps, generate base stream
    - `build_parameter_grid()`: Cartesian product over (mass, time, phi1)
    - `evaluate_candidate()`: Full orbit evolution OR fast impulse + scoring
    - `run_grid()`: Evaluate all candidates, sort by combined score
    - `save_results()`: JSON output with top-k detail + full score list
- `scripts/run_timeline_forward_model.py`: CLI runner with argparse.
  - `--stream`, `--quick`, `--fast`, `--phi1`, `--log10-mass-range`, `--t-range`, etc.
  - Default mode: full orbit integration (correct physics)
  - `--fast`: impulse approximation mode for rapid grid scans
  - `--quick`: smoke test (small grid, implies --fast)
  - Auto-detects impact phi1 from config known_gaps or data percentiles

### Design decisions

1. **Full orbit integration by default**: Each candidate generates a fresh
   stream, integrates particles to the past impact epoch, applies the 3D kick,
   and integrates everything forward to present. This is the physically correct
   approach that captures non-linear gap evolution over Gyr timescales.

2. **Two-mode architecture**: Full mode (default) for physically correct results;
   fast mode (--fast) using the existing impulse approximation for rapid grid
   scans and coarse parameter space exploration.

3. **3D galactocentric velocity kick**: The kick is applied in full 3D (not
   stream-aligned) using the Hernquist impulse formula with finite-extent and
   massive-subhalo corrections. Kick direction is perpendicular to the stream
   velocity vector in the orbital plane.

4. **Data-driven phi1 range**: The pipeline uses the 2nd-98th percentile of
   observed phi1 rather than the config's simulation range, because the two
   may use different frame conventions.

5. **Modular scoring**: Each scorer returns a scalar (lower = better).
   Combined score is a weighted sum with physically motivated weights:
   gap morphology > density > kinematics > profile features.

6. **Batched orbit integration**: Particles sharing the same release time are
   integrated together using galpy's vectorised Orbit. This gives ~15x speedup
   over per-particle integration.

## Verification

- All files compile cleanly (`py_compile`)
- Import chain verified: `scoring`, `evolve`, `pipeline` all import without error
- Scoring module: 8 unit tests passing (density, gaps, kinematics, combined)
- Orbit integration verified: particles move 9+ kpc over 2 Gyr (correct physics)
- Full orbit mode: 6 candidates on GD-1 at 3.0 candidates/sec (800 stars each)
- Fast mode smoke test: 18 candidates at 544 candidates/sec
- Full test suite: 204 tests pass, 0 failures
- Output JSON written correctly with score breakdowns

## Known limitations

1. **Background contamination**: The observed GD-1 data has 137K stars with
   `membership_prob=1.0` (no background subtraction applied). Real gap
   detection needs proper membership filtering first.

2. **Frame convention**: Config `known_gaps` at phi1=-40 uses a different
   frame than the stored data (phi1 in [0, 78]). The pipeline handles this
   gracefully by falling back to data percentiles.

3. **Single encounter**: Currently evaluates one forced encounter per
   candidate. Multiple-encounter scenarios and encounter sequences are
   not yet supported.

4. **No profile feature scorer**: The `profile_distance` score (using the
   157-dim detector features) is stubbed but not connected yet. Needs the
   GNN encoder to compute embeddings.

5. **galpy C extension not loaded**: On this machine the C extension is missing,
   so orbit integration uses the Python odeint fallback. With the C extension,
   throughput would be ~5-10x higher.

6. **No parallelisation yet**: Candidates are evaluated sequentially. The grid
   evaluation is embarrassingly parallel and could use multiprocessing.

## Next steps

- Add background subtraction / membership probability threshold to data loading
- Fix galpy C extension (install Visual C++ runtime) for 5-10x speedup
- Add multiprocessing to evaluate grid candidates in parallel
- Connect profile_feature_distance scorer using the GNN encoder
- Add multi-encounter evaluation (sequential encounters)
- Implement finer grid refinement around top candidates (zoom-in strategy)
- Run production grid on GD-1: 11 masses x 20 times x 2 phi1s = 440 candidates
- Add proper stream membership filtering (phi2 < 0.5 deg, PM cuts)
- Compare unperturbed baseline against observed to establish null hypothesis
