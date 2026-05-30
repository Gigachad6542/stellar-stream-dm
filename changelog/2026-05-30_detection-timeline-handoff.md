# 2026-05-30 Detection → timeline handoff (timeline deliverable 1/3)

## Why this exists

The timeline forward model (`src/forward_model/`) already simulated subhalo
encounters on the un-impacted past stream, evolved them to the present, and
scored them against real data. But it ran a **blind grid** over (mass, time,
phi1) — it never used the pipeline's detector to first decide *whether* there is
a probable impact, *where* it sits, or *when* it happened. This change wires that
front end in, realising the intended flow:

> find a probable-enough impact → estimate its time → simulate versions around
> that estimate → compare to the real stream.

## What changed

### New: `src/forward_model/detection.py`

- `detect_impacts_modelfree(...)` — always-available, data-only gap finder.
  Builds the membership-weighted density profile, runs `detect_gaps`, and returns
  gap longitudes + significances plus in-frame quartile fallback positions. This
  is the trustworthy backbone for **localisation**.
- `StreamImpactDetector` — optional learned detector. Loads a timeline
  `StreamGNNMultiTaskV2` checkpoint and reproduces the training inference recipe
  exactly (kNN graph with the saved `normalizer_v2.npz` at k=8 → profile
  features from raw node features → standardise → forward). Decodes:
    - `p_impact = sigmoid(binary_logit)`
    - `t_since_gyr = 10 ** regression_output[1]`  (regression targets are raw,
      not standardised — verified in `train_v2.py`: `loss_reg` compares the head
      output directly to `regression_targets_from_labels`, where
      `timeline_effective` → cols `[effective_n_impacts, log10_t_since_strongest]`)
    - `effective_n_impacts = regression_output[0]`
  Degrades gracefully (missing checkpoint / normalizer / NaN → marked
  unavailable, model-free result still returned).
- `seed_config_from_detection(...)` — turns a `DetectionResult` into a narrowed
  `ForwardModelConfig`: phi1 grid from detected gaps (else in-frame quartiles,
  never the out-of-frame default), and the time grid focused on the GNN estimate.

### Changed: `scripts/run_timeline_forward_model.py`

- `--auto-detect` runs the detection front-end after `prepare()` and seeds the
  grid in place before `run_grid()` (no recomputation — the base stream and null
  do not depend on grid params). Saves `detection_result.json`.
- `--detector-checkpoint`, `--t-window`, `--max-phi1-seeds`,
  `--detect-sig-threshold`, `--require-detection` control the handoff.
- `--use-gnn-scorer` (default **OFF**): the GNN embedding distance is excluded
  from the combined score by default because the encoder has a large sim-to-real
  gap (~40 vs ~1 sim-to-sim) and otherwise dominates/corrupts the score on real
  data. The combined score now uses density + gaps + kinematics (all functional).

### New: `tests/test_detection.py` (12 tests)

Model-free detection (clean gap, uniform null, in-frame fallbacks, JSON
serialisation), config seeding (gaps, fallback, time-window narrowing + clipping,
no-estimate passthrough, no mutation of base), and graceful degradation when the
checkpoint is missing.

## Verification

- All 12 detection tests pass; full suite **236 passed, 1 deselected** (no regressions).
- End-to-end on real GD-1 (`--quick --auto-detect` with the timeline checkpoint):
  detector loads, runs, seeds phi1 from in-frame quartiles and the time window
  from the GNN estimate, writes `detection_result.json` + `forward_model_results.json`.

## Findings (real GD-1, timeline checkpoint `gnn_v2_timeline_s3_20260529_123353`)

1. **GNN is over-confident on real data**: `p_impact = 1.00` every time —
   the documented sim-to-real gap. The GNN flag is therefore kept *separate*
   from the model-free gap flag (`gap_detected` vs `gnn_impact_flag`); the GNN
   time estimate is used to seed the grid but labelled advisory.
2. **No significant density gap** is found in the current GD-1 processing at the
   default thresholds (max significance ~0 at 1° bins). Likely the frame-
   convention / contamination issues already noted. Hence the in-frame quartile
   fallback for phi1 seeding rather than the out-of-frame config default (−40).
3. GNN time-since-impact estimate for GD-1 ≈ 7–8 Gyr (advisory).

## Next (timeline deliverables 2 and 3)

- **2. Injection-recovery test**: inject a known (mass, time, phi1) impact into a
  synthetic stream and verify the full pipeline recovers it — this quantifies how
  trustworthy the detection + scoring actually are before trusting real GD-1.
- **3. Statistical rigor**: null distribution over many no-impact realisations to
  express the best candidate as a p-value/σ, and multi-seed averaging so score
  differences reflect physics rather than sampling noise.
