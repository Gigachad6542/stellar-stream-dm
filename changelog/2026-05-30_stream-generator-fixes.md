# 2026-05-30 Why simulations did not match real data, and the fixes

A full diagnosis of the sim-vs-real mismatch (the root blocker behind the
underpowered detection results), plus the first iteration of fixes.

## Diagnosis

Comparing a freshly generated GD-1 to real GD-1 exposed several severe problems:

1. **The progenitor IC optimiser produced kinematically-wrong orbits.** The 5D
   phi2-RMS optimiser (`set_progenitor_ic`) matched the track's phi2 *geometry*
   but, with its velocity-angle/scale freedom, landed on the wrong orbit: the
   generated GD-1 had pm1 = -8.9 (and the chunk sims even further off) versus the
   literature value -12.8. Wrong velocity -> wrong orbit -> wrong stream.

2. **Streams were far too short.** The generated GD-1 spanned only phi1 ~ [11,48]
   versus the observed [-21,81] (~100 deg). The simple Fardal spray under-produces
   angular length per unit disruption time.

3. **The real GD-1 "members" catalog is ~96% contamination.** Its pm1 histogram
   peaks at -3 to -5 (field stars); only 3.6% have GD-1's actual pm1 < -9. With
   uniform membership_prob = 1.0 and 137k stars, `streams.h5` is a broad field
   selection, not a clean GD-1 membership catalog. (This is why earlier gap and
   significance tests were confounded.)

## Fixes (this iteration)

- **`set_progenitor_ic_track6d`** (new, now the default for `generate_stream`
  and the evolve path): derive the progenitor IC DIRECTLY from the galstreams 6D
  track point nearest the centre of the observed phi1 range. The track carries
  literature kinematics, so the orbit is correct by construction. Validated on 5
  streams -- proper motions now match the tracks essentially exactly:

  | stream | sim pm1 / track pm1 | sim pm2 / track pm2 |
  |---|---|---|
  | GD-1 | -13.14 / -13.13 | -3.25 / -3.26 |
  | Pal 5 | 3.67 / 3.64 | 0.64 / 0.63 |
  | Jhelum | -7.45 / -7.45 | 3.37 / 3.37 |
  | ATLAS | 0.15 / 0.23 | -1.07 / -1.04 |
  | Orphan | 1.06 / 1.12 | 1.44 / 1.46 |

  phi2 tracks are centred (~0) and tight. The old optimiser is kept only as a
  fallback for tracks lacking 6D velocities.

- **`spray_age_gyr` length calibration** (new config field, separate from the
  physical `disruption_age_gyr`): the spray timescale tuned so the generated
  extent matches the observed track extent. Calibrated per stream via
  `scripts/calibrate_spray_age.py`: GD-1 9, Pal 5 12, Orphan 14, ATLAS 4,
  Jhelum 2, Fjorm 10, Sylgr 2. GD-1 now spans [-21,81] with pm1 = -12.2.

## Verification
- 51 simulation tests pass.
- GD-1 end-to-end (config-driven): phi1 [-21,81] (matches real), pm1 -12.2 (real -12.8).

## Known remaining issues (next iterations)
1. **Width vs length tension.** Matching the long extent (high spray age) widens
   the stream (GD-1 phi2 std ~0.85 at age 9 vs observed ~0.5). The simple Fardal
   spray cannot simultaneously match length and thinness -- a proper Fardal/
   streakline spray (correct release offsets) is the real fix.
2. **Very long streams** (Orphan ~107 deg, Fjorm ~65 deg) still fall short of
   full extent even at high spray age -- the spray's ceiling.
3. **Real membership catalog**: replace the contaminated `streams.h5` GD-1 with a
   proper STREAMFINDER/PWB18 membership catalog (with probabilities) so sim-vs-real
   comparison uses clean members.
4. Old simulation chunks were generated with the broken IC -> regenerate training
   data with the track-6D IC + calibrated spray ages before re-training.
