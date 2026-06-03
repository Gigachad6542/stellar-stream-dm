# Building the paper

The manuscript exists in two forms:
- `manuscript.md` — **the current, authoritative draft** (corrected-simulator
  pipeline, June 2026; dual-register with plain-language boxes + a Methods appendix).
- `mnras_paper.tex` — **SUPERSEDED** (encodes the old v3 numbers; regenerate from
  `manuscript.md` before submission — see the notice at its top).
- `references.bib` — shared BibTeX bibliography (20 entries; includes Bovy 2014
  `streamdf` and Sanders, Bovy & Erkal 2016 `streamgapdf`).
- `figures/` — generated figures (regenerate with `python paper/make_figures.py`).

## Figures
```bash
python paper/make_figures.py          # writes paper/figures/fig1..fig6 .png
```
fig1 example streams · fig2 simulator fix · fig3 detector ROC · fig4 completeness ·
fig5 DM-family distinguishability · fig6 multistream significance.

## Build the MNRAS PDF (after regenerating the .tex from manuscript.md)
**Overleaf (easiest):** new project from the MNRAS template; upload `mnras_paper.tex`,
`references.bib`, and `figures/`.
**Locally:** needs TeX Live/MiKTeX + `mnras.cls`/`mnras.bst`:
```bash
latexmk -pdf mnras_paper.tex
```

## Notes
- Target venue: MNRAS (or arXiv preprint).
- Author/affiliation/e-mail are placeholders — fill before submission.
- Headline numbers are sourced from the corrected pipeline: detector test
  AUC 0.982 (`checkpoints/detector_df_20260602`, `calibration.json`); multistream
  joint Stouffer Z=1.40 / Fisher p=0.28 (`outputs/multistream/joint_significance_corrected.json`);
  injection-recovery rank 0/36. See `../changelog/2026-06-02_*`.
