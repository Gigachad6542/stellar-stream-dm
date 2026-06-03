# 2026-06-02 Re-running the v3 conclusions on CORRECTED impacts

The rs-units bug + homemade spray invalidated the v3 forward-model science. Re-ran
the full chain with the corrected scale_radius_from_mass (physical NFW subhalos) and
the new detector. The qualitative conclusions HOLD and are physically sharper.

## 1. Injection-recovery (validation) -- PASS
fast mode, truth M=10^9 t=1.5 Gyr phi1=20: recovered EXACTLY (rank 0/36), best beats
null by 90%. Forward-model machinery is correct with the fixed rs.

## 2. Real GD-1 timeline (STREAMFINDER, new detector) -- detect strong, characterise marginal
- Detection: gap at phi1~50 (the PWB18 gap), model-free sig 37; GNN p_impact 0.985,
  OOD 2.7 sigma (IN-distribution), t~1.8 Gyr.
- Single-subhalo forward fit: improvement over null only 3.2% (2/30 beat null),
  best mass pinned at the 10^7.5 floor. The detected gap is real but a single
  realistic (softer) subhalo barely reproduces its depth.

## 3. Multistream joint significance (7 streams) -- NO joint detection, CDM-consistent
GD-1 now uses the clean STREAMFINDER catalog (was 96%-contaminated bundle).
- Per-stream look-elsewhere z: GD1 +0.41, ATLAS +2.22, Jhelum +2.67, Orphan +1.13,
  Pal5 +1.03, Fjorm -2.48, Sylgr -1.29  -> INCOHERENT (71% positive, mixed signs).
- Joint: Stouffer z=1.40 (p=0.081), Fisher p=0.277, coherence gate = NOT a detection.
- Cautionary methods result intact: naive z up to +18.9 (Sylgr) / +11.8 (Pal5) ->
  look-elsewhere-corrected ~+/-2. Naive significance inflated ~10-40x by
  null-construction + look-elsewhere, as before.
- Verdict: upper limit / CDM-consistency given current data. outputs/multistream/
  joint_significance_corrected.json.

## Bottom line
The simulator fix UPGRADED the detector (AUC 0.62->0.982, real GD-1 in-distribution)
without changing the honest science verdict: gaps are detectable, but attributing them
to specific subhalo impacts is marginal in single streams and shows no coherent
multi-stream signal -- consistent with CDM. The corrected (softer) subhalo physics
makes the "hard to attribute a shallow gap to one subhalo" point stronger.
