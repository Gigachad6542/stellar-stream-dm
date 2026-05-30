# Stellar Stream Dark Matter Subhalo Detection

End-to-end pipeline using graph neural networks + simulation-based inference (SBI)
to detect dark matter subhalo perturbations in Milky Way stellar streams (Gaia DR3).

## Project structure

```
stellar-stream-dm/
├── config/                    # All configuration (streams, DM models, training)
│   ├── streams.yaml           #   Stream-specific parameters (phi1 ranges, distances, etc.)
│   ├── dm_models.yaml         #   Mass functions and inference priors per DM model
│   └── training.yaml          #   Hyperparameters, simulation budget, paths
├── src/                       # Source code (5 subpackages)
│   ├── data/                  #   Gaia queries, stream processing, PyG datasets
│   ├── simulation/            #   MW potential, stream generation, subhalo/baryonic perturbations
│   ├── models/                #   GNN encoder (GINEConv), baseline CNN, transformer
│   ├── inference/             #   SBI pipeline (SNPE-C), posteriors, calibration
│   └── analysis/              #   Gap catalog, model comparison, visualization
├── scripts/                   # Runnable entry points
│   ├── train.py               #   Train GNN or baseline CNN
│   ├── train_sbi.py           #   Train SBI posteriors for each DM model
│   ├── run_inference.py       #   Apply trained model to real streams
│   ├── combine_posteriors.py  #   Multi-stream posterior combination
│   ├── run_analysis.py        #   Generate gap catalog and figures
│   ├── demo_forward_pass.py   #   Pass real data through trained model (demo)
│   └── generate_training_data.py  # Parallel simulation generation
├── tests/                     # Unit tests (113 tests, pytest)
│   ├── test_simulation.py     #   Potential, mass functions, impulse, noise (55 tests)
│   ├── test_model.py          #   GNN, CNN, transformer, checkpoints (30 tests)
│   ├── test_inference.py      #   Priors, posteriors, gaps, Bayes factors (30 tests)
│   └── conftest.py            #   Pytest fixtures and --run-slow flag
├── notebooks/                 # Jupyter notebooks (exploration + results)
│   ├── 01_explore_streams.ipynb
│   ├── 02_simulation_validation.ipynb
│   ├── 03_training_diagnostics.ipynb
│   ├── 04_real_data_inference.ipynb
│   └── 05_results_summary.ipynb
├── data/                      # Data files (gitignored — generate locally)
│   ├── raw/                   #   Gaia FITS downloads
│   ├── processed/             #   Standardized HDF5 (streams.h5)
│   └── simulations/           #   Training simulations (82 HDF5 chunks)
├── checkpoints/               # Trained model weights (gitignored)
├── outputs/                   # Analysis outputs: CSV, PDF figures (gitignored)
├── logs/                      # Build logs, batch scripts, stale data (gitignored)
├── environment.yml            # Conda environment specification
├── .gitignore
└── README.md
```

## Science goal

Constrain the dark matter subhalo mass function and discriminate between:
- CDM (cold dark matter)
- WDM (warm dark matter, thermal relic)
- FDM (fuzzy/ultra-light axion dark matter)
- SIDM (self-interacting dark matter)

across 7 target stellar streams using combined multi-stream posterior inference.

## Target streams

GD-1, Pal 5, Orphan-Chenab, ATLAS, Jhelum, Fjorm, Sylgr

## Setup

### 1. Install Miniforge3

Download from https://github.com/conda-forge/miniforge/releases (Windows installer).

### 2. Create environment

```bash
mamba env create -f environment.yml
conda activate stellar-stream-dm
```

### 3. Install galstreams (manual step — required)

`galstreams` declares `gala` as a dependency; `gala` has no Windows binary wheels
and its C extensions fail with MSVC.  Since all runtime needs of galstreams
(astropy, numpy, scipy) are already in the env, install it without dependencies:

```bash
pip install git+https://github.com/cmateu/galstreams.git@main --no-deps
```

### 4. Verify GPU

```bash
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA!'; print(torch.version.cuda)"
python -c "import jax; print(jax.devices())"
python -c "import galstreams, sbi, torch_geometric; print('All imports OK')"
```

> **Note (Windows):** PyTorch is installed via pip from `download.pytorch.org/whl/cu124`
> (not the conda pytorch channel, which ships a CPU-only binary on Windows despite the
> CUDA build string).  The simulation backend uses **galpy** (not gala) for MW potentials
> and stream generation — galpy installs from conda-forge as a pre-built binary.

## Implementation sequence

1. **Data** — Query Gaia DR3 for each stream, process to HDF5
2. **Simulations** — Generate 40K training simulations (benchmark first)
3. **Baseline** — Train 1D CNN on density profiles (must reach >70% accuracy)
4. **GNN** — Train GINEConv encoder on phase-space graphs (must reach >80%)
5. **SBI** — Wire GNN embeddings to SNPE-C; run SBC calibration tests
6. **Inference** — Apply to real streams; combine posteriors

## Quickstart (exploration)

```bash
# Check stream data
jupyter lab notebooks/01_explore_streams.ipynb

# Benchmark simulations (100 jobs)
python scripts/generate_training_data.py --benchmark 100

# Full simulation run (~24 hours)
python scripts/generate_training_data.py --n-sims 10000

# Train GNN
python scripts/train.py --model gnn --epochs 200

# Run inference on GD-1
python scripts/run_inference.py --stream GD1 --all-models

# Combine posteriors
python scripts/combine_posteriors.py
```

## Tests

```bash
pytest tests/ -v -m "not slow"   # fast tests
pytest tests/ -v                  # including slow (requires gala)
```

## Key design decisions

- **GINEConv** (not GCN or GAT): edge features carry the kinematic perturbation signal
- **joblib loky backend**: avoids Windows fork() issues with gala/astropy imports
- **Two-level inference**: per-stream NPE → product-of-posteriors for DM model params
- **Stream-agnostic**: all stream parameters in `config/streams.yaml`; no stream names in Python

## Novel contribution

Multi-stream combined posterior inference + baryonic vs. dark-matter gap classifier.
The gap classifier assigns P(DM subhalo) / P(GMC) / P(noise) to each observed gap —
this analysis does not yet exist in the published literature as a systematic ML pipeline.
