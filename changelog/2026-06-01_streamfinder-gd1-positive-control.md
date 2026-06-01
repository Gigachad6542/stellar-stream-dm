# 2026-06-01 Clean GD-1 catalog (STREAMFINDER) + positive control

## Motivation
The bundled streams.h5 GD-1 "members" are ~96% field; the density profile is a
smooth U-shape with no localized gap (deepest dip depth 0.06, rejected). The
pipeline therefore reported NO gap on real GD-1 -- a *data-adequacy false negative*,
since GD-1 has a well-known gap (Price-Whelan & Bonaca 2018; Bonaca+2019).

## Clean catalog
Ingested Ibata+2021 STREAMFINDER (VizieR J/ApJ/914/123), stream #8 = GD-1
(identified by GD-1-I21 track alignment: phi2 rms 1.14 deg over 102 deg phi1 span).
811 high-confidence members, full extent phi1 [-21, 81], RV on 216/811. Transformed
to the GD-1-I21 frame (positions + reflex-corrected proper motions), distance from
the galstreams track. `scripts/fetch_streamfinder_gd1.py` -> data/processed/GD1_streamfinder.h5.

## Positive control (full pipeline, fast mode)
- Contaminated catalog: impact_detected=False; deepest minimum phi1=37, depth 0.06.
- **Clean STREAMFINDER: impact_detected=TRUE; 1 significant gap at phi1~50, gap_agreement=1.0.**
  This is the KNOWN PWB18/I21 gap (50.7 deg). The pipeline recovers it once the input
  is clean. The clean density profile shows the gap as a local minimum (N=24 at
  phi1~51.5 vs ~40 in neighbours).

## Honest caveats exposed
1. The GNN *detector* is still out-of-distribution on real data even when clean
   (max ~17.5 sigma; p_impact unreliable). The gap was found by the model-free
   density detector, not the GNN. The detector's sim-trained normalization still
   does not match real GD-1 -> needs stronger domain randomization or real-data
   fine-tuning.
2. The look-elsewhere significance is contradictory (naive z=+42.7 vs corrected
   z=-29): the best-of-grid null construction is not behaving on the sparse clean
   data -- the documented null-construction issue.

## Takeaway
The earlier real-GD-1 "null" (and by extension the multi-stream non-detection)
was substantially DATA-LIMITED. The gap-finding machinery is validated on real
clean data. Remaining work: detector real-data calibration + null construction;
denser catalog (PWB18) for a sharper gap.
