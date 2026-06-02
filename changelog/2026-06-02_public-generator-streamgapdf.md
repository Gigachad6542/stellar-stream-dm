# 2026-06-02 Replacing the homemade generator with galpy's validated DFs

Root-cause fix for the recurring sim problems: stop hand-rolling stream generation;
use galpy's published, validated tools end-to-end.

## Bugs found & fixed
1. **Clumpy streams** (no-impact gap-depth ~0.69, excess-over-Poisson ~0.4) ->
   `generate_stream_spray()` using `streamspraydf` (Fardal 2015): smooth (excess
   ~0.05), correct kinematics, RV present. [committed]
2. **scale_radius_from_mass 1000x units bug** (rho_crit Mpc^3 used as kpc^3) ->
   every impact in v3 AND all prior sims used point-like 0.3pc subhalos (8000+ km/s
   kicks, unrealistically sharp gaps). Fixed -> physical rs (1e8 -> 0.25 kpc).
   *** This invalidates the v3 detector/timing/multistream conclusions. ***

## The impact generator: streamdf + streamgapdf (Sanders, Bovy & Erkal 2016)
The canonical published tool for "a stream with a subhalo gap". Hard-won setup:
- `b` for actionAngleIsochroneApprox MUST come from `estimateBIsochrone(pot, R/ro, z/ro)`
  (~0.61 for GD-1). A wrong b (e.g. 0.8) -> "time not in integration domain".
- `impact_angle` sign must match the arm (leading=True -> positive angle), else
  "leading/trailing impact for trailing/leading arm".
- streamdf init ~12s, streamgapdf ~22s (incl. setup) per stream.

## Separability result (the thing that was blocking the detector)
- no-impact (streamdf): gap-depth ~0.28-0.40 (Poisson floor at 1200 stars)
- strong impact (low b, high mass): gap-depth ~0.86-1.0  -> cleanly separable
- G6 AUC over the FULL realistic impact range = 0.725 (vs homemade 0.57). Weak
  impacts (high b / low mass) are genuinely undetectable (correct physics). The
  crude max-gap-depth metric is Poisson-limited -> lower bound; the GNN profile
  branch + labeling by realized gap should push effective separability higher.

## Next
- Wire streamdf (no-impact) + streamgapdf (impact) into the generation pipeline,
  gated by scripts/validate_generator.py.
- Label by realized gap depth (not Erkal prediction).
- Regenerate, retrain detector -> expect a learnable signal at last.
- RE-RUN the v3 conclusions (detector AUC, timing, multistream) on the corrected sims.
