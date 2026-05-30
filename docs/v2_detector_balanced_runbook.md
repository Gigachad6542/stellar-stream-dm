# V2 Detector-Balanced Runbook

This runbook tracks the current V2 path. It deliberately ignores the older
report/PDF work and focuses on simulation, detector training, and later SBI.

## What The Datasets Mean

- `data/simulations_v2_plan2`: physics-prior Plan 2 dataset. Useful for rate
  realism and later suppression inference, but not a clean first detector
  target because impact counts are entangled with DM family.
- `data/simulations_signal_ladder/*`: small controlled curriculum datasets.
  These proved that impact morphology is learnable when labels are balanced.
- `data/simulations_v2_detector_balanced`: current bridge dataset. It keeps
  realistic noise, foregrounds, baryonic perturbations, all seven streams, and
  varied stream sizes, while balancing impact/no-impact independently of DM
  family.

## Current Evidence

- Plan 2 `impact_strong` labels are weakly visible: cheap baselines only reach
  about 0.65-0.68 ROC-AUC, and positives are dominated by CDM/SIDM.
- The baryonic signal ladder is clearly learnable: cheap baselines reach about
  0.88-0.89 ROC-AUC, with positives balanced across CDM/WDM/FDM/SIDM.
- The next GNN should therefore train on `detector_balanced` first, not rerun
  another full Plan 2 training pass.

## After Generation Finishes

Run the handoff with the project conda Python, not plain `python`:

```powershell
C:\Users\Dwthe\miniforge3\envs\stellar-stream-dm\python.exe scripts\run_detector_balanced_handoff.py
```

The handoff does:

- environment check, including CUDA PyTorch
- chunk/simulation-count validation
- `impact_strength` computation
- threshold audit
- cheap-baseline signal audit
- writes exact train/evaluate commands to `outputs/run_plans/detector_balanced_next_steps.json`

## Training Target

Start with:

- `--binary-target impact_strong`
- `--strength-threshold 1.0` unless the new threshold audit says otherwise
- `--use-profile-branch`

Do not return to SBI until the detector-balanced GNN beats the cheap baselines
or clearly explains why it cannot.

## Environment Trap

Plain `python` in this shell has been observed to point at a CPU-only or
incomplete environment. Use:

```powershell
C:\Users\Dwthe\miniforge3\envs\stellar-stream-dm\python.exe
```

`train_v2.py` now logs the executable, PyTorch version, and CUDA availability
at startup. Treat a CPU-only training launch as a failed preflight, not a slow
run worth waiting out.
