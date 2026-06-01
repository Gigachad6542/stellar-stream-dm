# 2026-06-01 Making the GNN detector find the GD-1 impact

The clean STREAMFINDER GD-1 catalog let us use the known PWB18 gap as a positive
control. The model-free density detector finds it; the GNN did not. Two root causes
found and addressed.

## 1. OOD from radial velocity (fixed)
On real GD-1 the GNN was ~1200-sigma out-of-distribution. Per-feature diagnosis:
the entire OOD was `vrad` (real std ~595 km/s incl. a 16857 km/s fill value; even
clean, GD-1's +/-300 km/s RV gradient vs the sim `vrad` std ~14). Root cause: the
v3 simulations carry NO radial velocity (vrad=NaN), so the detector learned RV as
an imputed constant and never trained on it. Feeding real RV is a feature the net
never saw.
- Fix: `StreamImpactDetector(ignore_rv=True)` (default) NaNs vrad/e_vrad at
  inference so they impute to the training mean. OOD 1209 -> 3.2 sigma; detection
  now rests on phi1/phi2 + proper motions (in-distribution). 14/14 detection tests pass.
- Also clipped a garbage HRV fill value in the STREAMFINDER ingestion.

## 2. Detector hedges at p~0.47 (in progress)
Even in-distribution, p_impact(GD-1)=0.47 -- undecided. Diagnosis: the detector
responds to strength (med-mass impacts -> p~0.67) but its probabilities are
compressed around 0.47 because the `impact_timeline_detectable` target on the
realistic v3 population is dominated by subtle/near-undetectable impacts, so it
learned to hedge. AUC was 0.62.
- The Erkal gap-depth strength score (compute_impact_strength.py) shows v3 actually
  has 26695 strong impacts (26.7%) -- a healthy positive class.
- Action: written impact_strength into the v3 chunks (threshold 0.5) and retraining
  the detector on `--binary-target impact_strong` (pos_weight 2.75), error-DR +
  cached profiles, 80 epochs -> checkpoints/gnn_v2_impact_strong_v3_20260601.
- Success criterion: p_impact on the clean GD-1 gap rises decisively above 0.5
  (target >~0.8). Validate on intermediate checkpoints.

## Honest framing
This yields a detector for *detectable/strong* impacts (like GD-1's gap), distinct
from the realistic-population detector (AUC 0.62 -> population non-detection). Both
are reported, clearly labeled; they answer different questions.
