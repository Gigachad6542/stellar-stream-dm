# 2026-06-04 Profile-identifiability screen — result and paper update

## Scientific result (new)

The matched two-arm `streamgapdf` backend was put through the pre-declared
injection/recovery **profile-identifiability screen**, and it **fails the frozen
gates decisively**. This converts the previous "not yet validated, so no profile
claim" stance into an evidence-backed "validated as not identifiable on present
data" statement.

Screen design (cheap representative slice of the fixed challenge): 3 masses
(10^7.5, 10^8.0, 10^8.5 M_sun) x 4 perturber density-profile families
(compact_0p5x, nfw_like_1x, cored_3x, very_cored_10x), planted at the easiest
geometry (recent t=0.5 Gyr, close b=0.05 kpc, trailing arm), profile searched
freely under the frozen scoring rule and decision gates. Run both with the real
PWB18/DESI selection and on idealized, fully-sampled streams (`--no-real-selection`)
to separate fundamental degeneracy from data-quality limit.

| Metric | Real selection | Idealized (no selection) | Gate |
|---|---|---|---|
| Profile-family accuracy | 0.42 | 0.58 | >= 0.70 (fail) |
| Detection recall | 0.25 | 0.33 | >= 0.60 (fail) |
| Scale-radius recovery RMSE | 0.41 dex | 0.34 dex | (degenerate) |
| Seed-consistency | 1.0 | 1.0 | >= 0.80 (pass, trivial at 1 seed) |

Interpretation:

- **Both gates fail even in idealized clean data.** Removing the real selection
  lifts family accuracy 0.42 -> 0.58 and recall 0.25 -> 0.33, so selection
  sparsity *contributes* but is *not* the root cause.
- **The dominant limit is the intrinsic gap->profile degeneracy.** Two compounding
  effects: (1) the lowest-mass gaps (10^7.5 M_sun) are too shallow to detect at all
  (all four scale families undetected); (2) where a gap *is* detected, the joint
  density-and-morphology shape under-determines the perturber's internal scale
  radius (clean RMSE 0.34 dex ~ factor 2.2; detected cells scatter across families).
- **Decision (per pre-registered rule):** a failed identifiability screen blocks
  real-data profile inference. We make no subhalo-attribution, perturber-profile, or
  per-system DM-model claim for real GD-1. Density-profile DM typing remains
  physically informative in controlled simulations (report Section 8), not on
  present data.

The full 144-task challenge and the nuisance/null-FPR challenges were deliberately
**not** run: the gate fails decisively at the *easiest* geometry, and harder cells
(later impacts, larger impact parameters) can only lower recovery; per the handoff
staged rule, a failed family-recovery gate terminates the staged plan at "diagnose
and do not proceed to real data."

## Source artifacts

- `outputs/profile_validation/quickcheck_fixed.json` (real-selection screen, 12 cells)
- `outputs/profile_validation/quickcheck_noselection.json` (idealized screen, 12 cells)

## Paper / repo changes

- `paper/make_fig20_profile_validation.py` (new): builds the diagnostic figure from
  the two screen JSONs. Panel (a) gate metrics vs thresholds (both conditions);
  panel (b) recovered-vs-true scale-radius factor on the clean screen.
- `paper/figures/fig20_profile_validation.png` (new).
- `paper/full_report.md`:
  - Section 10: new "Profile-identifiability screen" paragraph + new Figure 18
    (caption + in-text refs 18a/18b); the matched-control paragraph now reports the
    screen instead of "not yet validated."
  - Downstream figures renumbered: multistream 18 -> 19, kinematic 19 -> 20
    (captions sequential 1-20, verified unique).
  - Abstract: added a closing sentence stating per-system DM typing is not warranted
    on present data (population forecast + simulation capability only).
  - Section 12 item 5: now reports the measured gap->profile degeneracy.
  - Plain-terms box in Section 10: added the "what kind of dark matter" explanation.
- `paper/manuscript.md`: parallel updates to the Section 8 matched-control paragraph,
  its plain-terms box, and the Section 10 limitation item 5 (no new figure; condensed).
- `paper/full_report.pdf` and `paper/manuscript.pdf` rebuilt.

## Reproduction

Run from the repository root with the conda env python.

```bash
# Real-selection screen (12 cells, seed 42)
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode fixed \
  --truth-seeds 42 --log10-masses 7.5 8.0 8.5 --times 0.5 --impact-params 0.05 \
  --scale-factors 0.5 1 3 10 \
  --out outputs/profile_validation/quickcheck_fixed.json

# Idealized no-selection screen (same cells)
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode fixed \
  --truth-seeds 42 --no-real-selection --log10-masses 7.5 8.0 8.5 --times 0.5 \
  --impact-params 0.05 --scale-factors 0.5 1 3 10 \
  --out outputs/profile_validation/quickcheck_noselection.json

# Figure + PDFs
python paper/make_fig20_profile_validation.py
cd paper && python build_pdf.py full_report.md full_report.pdf \
          && python build_pdf.py manuscript.md manuscript.pdf
```

## Optional refinement (not required to change the conclusion)

A 3-seed repeat of each screen (36 cells per condition) would put an error bar on
the central values; it will not flip the conclusion because 0.42-0.58 is far below
the 0.70 gate, not marginal.
