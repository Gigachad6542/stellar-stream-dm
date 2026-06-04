# Stellar-stream dark-matter subhalo-impact detection

A reproducible pipeline that detects and interprets the density gaps left by
dark-matter subhalo flybys in thin Milky Way stellar streams (Gaia DR3), using a
validated stream simulator, a graph-neural-network (GNN) detector, and a
characterization/forward-model layer.

**Read the report** (in [`paper/`](paper/)):
- [`paper/full_report.pdf`](paper/full_report.pdf) — the comprehensive technical
  report (22 pp, 19 figures, real-data application + 7-stream significance);
  source [`full_report.md`](paper/full_report.md).
- [`paper/manuscript.pdf`](paper/manuscript.pdf) — a condensed paper draft;
  source [`manuscript.md`](paper/manuscript.md). The full report is the current
  source of truth.

## What it does, and what it finds

- **Simulator** — smooth streams from `streamdf` and single-gap streams from
  `streamgapdf` (Bovy 2014; Sanders, Bovy & Erkal 2016), gated by a pre-flight
  validation harness so only faithful batches are used for training.
- **Detector** — a GINEConv GNN reaches **test AUC 0.982** with a zero
  false-positive operating point, and is **in-distribution** on the real
  STREAMFINDER GD-1 catalog, where it recovers the known φ₁≈50° gap.
- **Completeness** — detectability rises with subhalo mass and falls with impact
  age (gaps phase-mix away); the detector is, in effect, a calibrated density-gap
  detector.
- **Characterization** — a single gap is **degeneracy-limited**: subhalo mass is
  essentially unrecoverable from one gap (R²<0); only recency is weakly constrained
  (R²≈0.2).
- **Dark-matter models** — distinguishing CDM/WDM/FDM/SIDM is a *population*
  measurement. A rate+mass Asimov forecast finds favorable suppressed models
  (WDM 3 keV / FDM 10^-22 eV) need only about two clean detections to separate
  from CDM, but CDM produces detectable impacts at only about 0.026 per
  GD-1-like stream in the floor-normalized baseline, so collecting those
  detections needs roughly 300-365 clean streams. Without that low-rate floor the
  same forecast rises to roughly 1,200-1,400 streams, making the normalization
  a first-order systematic. The present seven-stream sample is therefore
  underpowered: CDM predicts only E[N_det]=0.18 detectable impacts and
  P(0 detections)=0.83 in the floor-normalized baseline, while favorable
  suppressed models also mostly predict null samples. A 180-case sensitivity
  grid keeps the strongest seven-stream WDM/FDM alternatives below 1 sigma.
  SIDM is not separable by abundance; its handle is shallower gaps from
  sufficiently cored perturbers at fixed mass.
- **Density-profile pivot** — controlled simulations show physically directional
  compact/cored profile responses, but current real-GD1 candidates do not survive
  matched smooth-background and joint morphology checks. A matched two-arm
  `streamdf`/`streamgapdf` backend is implemented; real profile inference remains
  blocked on injection/recovery and no-impact false-positive validation. ATLAS
  remains the high-N null/control stream.
- **Timeline forward model + multi-stream significance** — implemented as a
  corrected particle-spray/impulse forward model with injection-recovery
  validation. It recovers a planted GD-1-like impact at rank 0/36, but the real
  seven-stream combination is null after look-elsewhere correction
  (Stouffer Z=1.40, Fisher p=0.28), giving a CDM-consistent upper-limit result.
- **Kinematic frontier** — a proper-motion kink is a real signal in noise-free
  simulations, but the diagnostic collapses to chance under current Gaia-like
  proper-motion noise; no expensive kinematic retrain is justified yet.

Simulator scope: detector training, completeness, characterization, and the
primary DM population forecasts use the validated `streamdf`/`streamgapdf`
pipeline. The timeline/multistream numbers use the corrected forward-model
backend (`generate_stream` + impulse/re-evolution), and are reported as
methods/upper-limit results with that systematic caveat.

## Target streams
The report studies seven streams: GD-1, Pal 5, Orphan-Chenab, ATLAS, Jhelum,
Fjorm, and Sylgr. The validated action-angle generator currently supports the
cleanest training batches for GD-1, ATLAS, Jhelum, and Orphan. The real-data
detector is reliable today only where clean member counts are high enough
(GD-1 and ATLAS); the population/forward-model analysis still records all seven
streams with catalog-quality caveats.

## Repository layout
```
src/            simulation/ (potential, streamdf/streamgapdf generation, subhalo physics),
                models/ (GINEConv GNN), data/ (Gaia/catalog processing, PyG datasets),
                forward_model/ (timeline detect→rewind→re-impact→score), analysis/
scripts/        runnable entry points (see scripts/README.md)
paper/          manuscript (.md + .tex), figures, references, claims audit, PDF builder
config/         streams.yaml, dm_models.yaml, training.yaml
tests/          pytest unit tests
changelog/      dated decision log
data/, checkpoints/, outputs/   gitignored — regenerate locally
```

## Setup
```bash
mamba env create -f environment.yml
conda activate stellar-stream-dm
# galstreams declares gala (no Windows wheels); install without deps:
pip install git+https://github.com/cmateu/galstreams.git@main --no-deps
python -c "import torch, galstreams, torch_geometric; print('imports OK', torch.cuda.is_available())"
```
> The simulation backend is **galpy** (conda-forge binary), not gala. On Windows,
> PyTorch is installed from `download.pytorch.org/whl/cu124`.

## Current pipeline (entry points)
```bash
# 1. Generate the detector dataset (validated DFs; parallel, resumable)
python scripts/generate_detector_data.py --n-sims 16000 --noise gaia \
    --output-dir data/simulations_detector_df

# 2. Validate a batch BEFORE training (critical gates: smoothness, length, separability)
python scripts/validate_generator.py --sim-dir data/simulations_detector_df

# 3. Train the detect+characterize GNN
python scripts/train_v2.py --sim-dir data/simulations_detector_df \
    --binary-target impact_strong --strength-threshold 0.5 --use-profile-branch

# 4. Calibrate + evaluate
python scripts/calibrate_detector.py  --checkpoint <ckpt> --sim-dir data/simulations_detector_df
python scripts/detector_completeness.py --checkpoint <ckpt> --sim-dir data/simulations_detector_df

# 5. Analyses
python scripts/characterize_probe.py --checkpoint <ckpt> --sim-dir data/simulations_detector_df
python scripts/dm_family_distinguishability.py
python scripts/dm_discrimination_forecast.py
python scripts/dm_sidm_morphology.py

# Forward-model / multistream (corrected particle-spray/impulse backend)
python scripts/run_injection_recovery.py --stream GD1 --fast
python scripts/run_timeline_forward_model.py --stream GD1 --auto-detect --detector-checkpoint <ckpt>
python scripts/run_multistream_analysis.py --out outputs/multistream/joint_significance_corrected.json

# Matched two-arm GD-1 profile-identifiability screen (fails its gates -> blocks real
# profile inference; see changelog/2026-06-04_profile-identifiability-screen-result.md)
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode fixed --dry-run
python scripts/run_gd1_streamgapdf_injection_recovery.py --mode nuisance --dry-run
python scripts/run_gd1_streamgapdf_null_fpr.py --dry-run

# Kinematic diagnostic and report figures
python scripts/kinematic_signal_test.py
python paper/make_figures.py
python paper/make_report_figures.py
python paper/make_report_figures2.py
python paper/build_pdf.py full_report.md full_report.pdf
```
See [`scripts/README.md`](scripts/README.md) for the full categorized index.

## Tests
```bash
pytest tests/ -v -m "not slow"
```

## Key design decisions
- **Validated distribution functions** (`streamdf`/`streamgapdf`) with a shared smooth
  track, so the only difference between training classes is the gap itself.
- **GINEConv** GNN: edge features carry the local kinematic perturbation signal.
- **Pre-flight validation harness** gates every generated batch before training.
- **Clean-catalog operating regime**: trained on Gaia-noise (not foreground-heavy)
  streams; applied to clean external membership catalogs (e.g. STREAMFINDER GD-1).
