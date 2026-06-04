# Building the paper/report

The current source of truth is:

- `full_report.md` -> `full_report.pdf`: the 19-figure technical report
  with detector, DM-forecast, timeline, multistream, and kinematic-frontier results.
- `manuscript.md` -> `manuscript.pdf`: a condensed draft derived from the same
  project state. Keep it synchronized before using it for submission.
- `mnras_paper.tex`: a LaTeX draft generated from the condensed manuscript. Treat it
  as secondary until regenerated from the current manuscript.
- `references.bib`: shared bibliography.
- `figures/`: generated report figures.

## Regenerate analysis artifacts used by the full report

```bash
python scripts/dm_discrimination_forecast.py
python scripts/dm_forecast_sensitivity_grid.py
python scripts/dm_null_power_table.py
python scripts/dm_discriminants_table.py
python scripts/dm_density_profile_pivot.py
python scripts/dm_sidm_morphology.py
python scripts/kinematic_signal_test.py
python scripts/run_multistream_analysis.py --out outputs/multistream/joint_significance_corrected.json
```

The multistream command uses the corrected particle-spray/impulse forward-model
backend. It is not the validated `streamdf`/`streamgapdf` detector-training backend;
the report states this as a systematic caveat.

## Regenerate internal GD-1 matched-control/profile diagnostics

These artifacts support caveats and future validation decisions. They are not current
real-data density-profile constraints and should not be promoted into paper results
until the frozen validation gates pass.

```bash
python scripts/run_gd1_background_calibration.py \
  --ages 4.5 6 9 --progenitor-masses 5000 10000 20000 40000 \
  --velocity-factors 0.3 0.6 0.9 --seeds 42 43 44 \
  --out outputs/profile_grid/GD1/gd1_background_calibration_expanded_multiseed.json
python scripts/run_gd1_background_sensitivity_shortlist.py
python scripts/run_gd1_streamgapdf_profile_screen.py

# Inspect the frozen challenge sizes before launching long runs.
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode fixed --dry-run
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode nuisance --dry-run
python scripts/run_gd1_streamgapdf_null_fpr.py --dry-run
```

The detailed run order, four-shard commands, gates, and Claude handoff are in
`../changelog/2026-06-04_gd1-matched-background-and-streamgapdf-handoff.md`.

## Regenerate figures

```bash
python paper/make_figures.py
python paper/make_report_figures.py
python paper/make_report_figures2.py
```

Figure sources:

- `make_figures.py`: core detector/completeness/DM-family figures.
- `make_report_figures.py`: calibration, characterization, embedding, and transfer
  figures.
- `make_report_figures2.py`: real-data detector, multistream, forecast, SIDM, and
  methods figures.

## Build PDFs

```bash
python paper/build_pdf.py full_report.md full_report.pdf
python paper/build_pdf.py manuscript.md manuscript.pdf
```

The builder is pure Python (`markdown` + `xhtml2pdf`/`reportlab`) and resolves figure
paths relative to `paper/`.

## Release checklist

- [ ] Run `pytest -m "not slow"`.
- [ ] Rebuild all figures and both PDFs from a clean checkout.
- [ ] Confirm `paper/claims_audit.md` maps each headline number to an artifact.
- [ ] Confirm the report and condensed manuscript use the same current-state claims.
- [ ] Confirm no real-GD1 subhalo/profile inference is stated before matched
  injection/recovery and no-impact false-positive validation passes.
- [ ] Fill author, affiliation, acknowledgements, data availability, and code
  availability before submission.
