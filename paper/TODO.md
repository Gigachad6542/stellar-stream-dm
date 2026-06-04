# Path to publication-ready shape

`[ ]` todo, `[~]` in progress, `[x]` done. Rebuilt 2026-06-03 to match
`paper/full_report.md` / `paper/full_report.pdf`.

## Current integrity rule

The report now contains two explicitly separated result classes:

- **Validated DF pipeline:** detector training, simulator validation, completeness,
  characterization, real-stream detector checks, and the DM population forecasts use
  the validated `streamdf` / `streamgapdf` generator.
- **Corrected forward-model pipeline:** timeline replay, injection-recovery, real
  GD-1 single-subhalo attribution, and seven-stream significance use the corrected
  particle-spray + impulse/re-evolution backend. These numbers are reported as
  methods/upper-limit results with that backend caveat, not as claims that the
  forward model has been ported to `streamgapdf`.

The near-term dark-matter strategy is now also explicitly separated:

- **Population model selection:** current Gaia stream counts are underpowered for
  CDM/WDM/FDM model selection.
- **Perturber density-profile inference:** controlled simulations retain
  directional profile information, but current real-GD1 candidates fail matched
  smooth-background and joint morphology checks. The new matched two-arm
  `streamdf`/`streamgapdf` backend must pass injection/recovery and no-impact
  false-positive gates before any real profile constraint is attempted. ATLAS
  remains a high-N null/control stream.

## Core results in the current full report

- [x] Simulator on validated DFs (`streamdf`/`streamgapdf`) + validation harness.
- [x] Detector: test AUC 0.982, calibrated, zero false-positive operating point.
- [x] Real detector checks: GD-1 flagged at the known phi1~50 deg gap; ATLAS null.
- [x] Member-count floor documented: reliable clean-catalog verdicts need N >~ 500.
- [x] Completeness map across mass, impact time, impact parameter, and gap strength.
- [x] Characterization: single-gap mass recovery fails; impact time is weakly constrained.
- [x] DM-family forecast: abundance+mass is the main discriminator; favorable WDM/FDM
  alternatives need about two detected impacts. The stream count is normalization-sensitive:
  about 300-365 GD-1-like streams in the floor-normalized baseline, rising to about
  1,160-1,370 without the low-rate floor.
- [x] Seven-stream null-power check: CDM predicts only E[N_det]=0.18 and P0=0.83
  in the floor-normalized baseline, while favorable suppressed models also predict mostly
  null samples; the current seven-stream result is underpowered for DM model selection.
- [x] Forecast sensitivity grid: 180 cases over encounter-rate floor/normalization,
  completeness, detector-threshold proxy, mass band, and stream count. Across the grid,
  CDM P0(7 streams)=0.60-0.98 and the most favorable seven-stream WDM/FDM cases remain
  below 1 sigma.
- [x] Density-profile pivot artifact: local target readiness, analytic compact/cored
  impulse ladder, SIDM morphology summary, and next experiments written to
  `outputs/dm/density_profile_pivot.json`.
- [x] GD-1 smooth-background calibration and three-seed matched-control
  sensitivity test; no fixed candidate/background combination improves both
  local diagnostic families.
- [x] Matched two-arm `streamdf`/`streamgapdf` backend with continuous perturber
  scale radius and initial real-GD1 profile screens.
- [x] Frozen validation contract and shardable injection/recovery plus no-impact
  false-positive entry points.
- [x] SIDM morphology diagnostic: cored halos produce shallower gaps at fixed mass.
  Sharded publication grid shows core-strength sensitivity: mean depth-AUC about
  0.55, 0.63, 0.64, 0.68, 0.79, 0.82, 0.87, and 0.84 for 1.25x through 5x
  the NFW scale radius.
- [x] Timeline forward-model injection-recovery: planted GD-1-like impact recovered
  at rank 0/36.
- [x] Real seven-stream significance: no coherent detection after look-elsewhere
  correction (Stouffer Z=1.40, Fisher p=0.28).
- [x] Kinematic frontier diagnostic: proper-motion kink is real in noise-free sims
  but below current Gaia-like precision, so no retrain is justified yet.

## Publication-facing cleanup still needed

- [ ] Decide whether the submission target is a full technical report, ApJ/AJ,
  MNRAS, or arXiv-first manuscript.
- [ ] Run the 48-cell fixed-geometry matched-streamgapdf profile-identifiability
  challenge across three seeds.
- [ ] Run the 30+30 nuisance-geometry injection/recovery and no-impact development
  challenges; require the frozen recall, ranking, FPR, and seed-consistency gates.
- [ ] Run the 100+100 publication challenge only if the development gates pass.
- [ ] Run a broader real-GD1 density-profile grid only after validation passes;
  report smooth-background sensitivity beside every candidate.
- [ ] Replace hard CDM/WDM/FDM/SIDM labels in the single-gap analysis with
  continuous perturber-profile parameters: compactness, scale radius, core
  radius, and density within the impact radius.
- [ ] Add a power-spectrum/correlation baseline for GD-1 and ATLAS controls.
- [ ] Add threshold-specific ROC/completeness curves before optimizing detector
  threshold for profile inference.
- [ ] Regenerate the condensed manuscript from the current full report, or mark it
  clearly as a shorter draft derived from the report.
- [ ] Expand SIDM morphology beyond the small exploratory grid before making a
  strong SIDM claim.
- [x] Audit the abundance forecast normalization and completeness assumptions with a
  180-case sensitivity grid, including the encounter-rate floor.
- [ ] Replace simplified kinematic-noise diagnostics with per-star Gaia error
  distributions if kinematics become a paper claim.
- [ ] Verify `pytest -m "not slow"` passes before release.
- [ ] Regenerate all report figures and PDFs after figure-title cleanup.
