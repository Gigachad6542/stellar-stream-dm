# Stellar-stream dark-matter subhalo-impact detection

A reproducible pipeline that detects and interprets the density gaps left by
dark-matter subhalo flybys in thin Milky Way stellar streams (Gaia DR3), using a
validated stream simulator, a graph-neural-network (GNN) detector, and a
characterization/forward-model layer.

**The paper** (current state, with figures) lives in [`paper/`](paper/):
read [`paper/manuscript.md`](paper/manuscript.md) or the rendered
[`paper/manuscript.pdf`](paper/manuscript.pdf).

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
  measurement; we quantify the required number of clean detections (≈5 for
  FDM 10⁻²² eV, ≈12–27 for WDM 3–6 keV, unreachable for SIDM via the mass spectrum).
- **Timeline forward model + multi-stream significance** — implemented and
  described, but currently on a separate (legacy) generator; their quantitative
  results are deferred until that generator is ported to the validated DFs.

All reported numbers derive from the validated `streamdf`/`streamgapdf` simulator.

## Target streams
GD-1, ATLAS, Jhelum, Orphan are supported by the action-angle stream model used for
training. (Pal 5 and Fjörm have near-circular orbits that break the isochrone
action-angle approximation and are excluded by an allow-list.)

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

# Forward-model / multistream (legacy generator — see paper §8–§9)
python scripts/run_injection_recovery.py --stream GD1 --fast
python scripts/run_timeline_forward_model.py --stream GD1 --auto-detect --detector-checkpoint <ckpt>
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
