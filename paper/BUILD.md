# Building the paper

The manuscript exists in two forms:
- `manuscript.md` — the working draft (always current; results filled from v3).
- `mnras_paper.tex` — the MNRAS-formatted LaTeX for submission.
- `references.bib` — shared BibTeX bibliography (18 entries).
- `figures/` — generated figures (regenerate with `scripts/plot_paper_figures.py`).

## Build the MNRAS PDF

**Overleaf (easiest):** create a project from the "Monthly Notices of the Royal
Astronomical Society (MNRAS)" template, then upload `mnras_paper.tex`,
`references.bib`, and the `figures/` folder. It compiles as-is (the template
provides `mnras.cls` and `mnras.bst`).

**Locally:** requires a TeX distribution (TeX Live / MiKTeX) and the MNRAS class
files `mnras.cls` + `mnras.bst` (from the RAS or the Overleaf template) placed
next to the `.tex`:

```bash
latexmk -pdf mnras_paper.tex
# or:
pdflatex mnras_paper && bibtex mnras_paper && pdflatex mnras_paper && pdflatex mnras_paper
```

## Notes
- Target venue: MNRAS.
- Author/affiliation and the e-mail are placeholders — fill before submission.
- All headline numbers (AUC 0.618; joint Fisher p=0.30; injection-recovery rank
  0/60) are sourced from the v3 pipeline; see `../changelog/2026-06-01_*` and
  `outputs/multistream/joint_significance_v3.json`.
