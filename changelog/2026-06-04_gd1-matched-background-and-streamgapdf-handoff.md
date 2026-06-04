# 2026-06-04 GD-1 matched-background and streamgapdf handoff

## Scientific decision

The current real-GD1 candidate shortlist is not evidence for a subhalo and does
not support a perturber density-profile measurement.

- Smooth-background calibration improves held-out control-region
  density/kinematics by up to 64% and morphology by up to 32%. The smooth stream
  model is therefore a first-order systematic.
- None of the 24 fixed encounter/background combinations improves both local
  density/kinematics and morphology at the encounter's own target when averaged
  across three seeds.
- A matched two-arm `streamdf`/`streamgapdf` backend now spans the observed GD-1
  catalog and supports continuous `scale_radius_kpc`.
- The initial continuous-profile screens do not find a profile that improves
  both local diagnostic families.

Density-profile response remains physically informative in controlled
simulations. Real-data interpretation is blocked until the new backend passes
matched injection/recovery and no-impact false-positive validation.

## Implemented validation contract

New entry points:

- `scripts/run_gd1_streamgapdf_injection_recovery.py`
- `scripts/run_gd1_streamgapdf_null_fpr.py`
- shared helpers in `src/forward_model/profile_validation.py`

The fixed challenge contains 48 truth cells:

- 3 masses x 2 impact times x 2 impact parameters x 4 scale-radius families;
- 3 truth seeds per cell by default;
- known geometry with profile scale searched independently;
- recovered family, rank, per-diagnostic score shifts, log-scale bias/RMSE, and
  seed consistency recorded explicitly.

The nuisance and null challenges use the same frozen search and decision rule:

- candidate and null are same-seed matched two-arm realizations;
- the PWB18 footprint ratio and DESI along-stream targeting pattern are applied
  without copying real GD-1 density or morphology;
- a detection requires at least 5% improvement in both the local primary score
  and conditional cross-track morphology score;
- gates are profile-family accuracy >=70%, planted-geometry-neighborhood top-10%
  rate >=80%, recall >=60%, null FPR <=5%, and seed consistency >=80%.

No validation result is claimed yet. The scripts and frozen gates are ready for
the staged challenge.

## Source artifacts

- `outputs/profile_grid/GD1/gd1_background_calibration_expanded_multiseed.json`
- `outputs/profile_grid/GD1/gd1_background_sensitivity_single_shortlist.json`
- `outputs/profile_grid/GD1/gd1_streamgapdf_profile_screen.json`
- `outputs/profile_grid/GD1/gd1_streamgapdf_profile_spur_focused.json`
- `outputs/profile_grid/GD1/gd1_streamgapdf_profile_canonical_focused.json`
- `outputs/run_plans/work_note_20260604_gd1_background_and_streamgapdf.md`

Primary validation outputs, once run:

- `outputs/profile_validation/gd1_streamgapdf_injection_recovery.json`
- `outputs/profile_validation/gd1_streamgapdf_null_fpr.json`

## Reproduction commands

Run from the repository root.

```bash
python scripts/run_gd1_background_calibration.py \
  --ages 4.5 6 9 --progenitor-masses 5000 10000 20000 40000 \
  --velocity-factors 0.3 0.6 0.9 --seeds 42 43 44 \
  --out outputs/profile_grid/GD1/gd1_background_calibration_expanded_multiseed.json

python scripts/run_gd1_background_sensitivity_shortlist.py

python scripts/run_gd1_streamgapdf_profile_screen.py
python scripts/run_gd1_streamgapdf_profile_screen.py \
  --times 0.5 --impact-angles 0.65 --scale-factors 0.5 1 2 5 \
  --out outputs/profile_grid/GD1/gd1_streamgapdf_profile_spur_focused.json
python scripts/run_gd1_streamgapdf_profile_screen.py \
  --times 1.0 --impact-angles 0.65 --scale-factors 0.5 1 2 5 \
  --out outputs/profile_grid/GD1/gd1_streamgapdf_profile_canonical_focused.json

# Four-shard fixed profile-identifiability challenge
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode fixed --n-shards 4 --shard-index 0
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode fixed --n-shards 4 --shard-index 1
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode fixed --n-shards 4 --shard-index 2
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode fixed --n-shards 4 --shard-index 3
python scripts/run_gd1_streamgapdf_injection_recovery.py \
  --merge-shards "outputs/profile_validation/gd1_streamgapdf_injection_recovery.shard-*-of-04.json"

# Development nuisance and no-impact challenges
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode nuisance --n-cases 30
python scripts/run_gd1_streamgapdf_null_fpr.py --n-cases 30

# Publication-size challenges only after development gates pass
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode nuisance --n-cases 100
python scripts/run_gd1_streamgapdf_null_fpr.py --n-cases 100
```

## Verification

- Focused streamgapdf profile validation checks: 11 passed.
- Real backend plumbing smokes completed with 200 stars and one candidate:
  injection recovered the only tested profile family but did not pass the
  detection-recall gate; the no-impact smoke produced no false positive. These
  are execution checks only, not scientific validation results.
- The streamgapdf validation entry points now invoke the existing Windows/conda
  galpy DLL setup before backend import; without it, direct Python invocation
  can fall back to an unstable pure-Python integration path.
- Maintained suite before this synchronization: 322 passed, 9 skipped.
- Current maintained suite after synchronization: 330 passed, 9 skipped.
- The all-files pytest invocation also collects `scripts/test_jax_integrator.py`
  and hit an unrelated Windows/JAX access violation. Do not describe that run as
  a successful full-suite verification.

## Claude handoff

1. Run the fixed profile-identifiability challenge in four shards.
2. If the >=70% family-recovery gate fails, diagnose generator/scorer
   identifiability and do not proceed to real data.
3. If it passes, run the 30-case nuisance and 30-case no-impact challenges.
4. Proceed to the 100+100 publication challenge only if all development gates
   pass without changing the frozen scoring rule.
5. Run a broader real-GD1 grid only after recall, top-10 recovery, FPR, and seed
   consistency gates all pass. Report smooth-background sensitivity beside every
   real-data candidate.
