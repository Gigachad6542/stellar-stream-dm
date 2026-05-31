# 2026-05-30 Erkal+2015 kick physics + full report update

## Erkal & Belokurov (2015) velocity kick (accuracy improvement A)

Replaced the hand-rolled subhalo fly-by kick with the closed-form Plummer
impulse from Erkal & Belokurov 2015:

    dv = -(2 G M / w) * p / (|p|^2 + r_s^2)

where ``p`` is each star's true 3D perpendicular offset to the subhalo's
straight-line trajectory, ``w`` the relative speed, ``r_s`` the Plummer scale
radius (minus sign: the impulse points toward the trajectory). This removes the
previous heuristic's Gaussian-in-phi1 localisation, fixed kick axis, 30%
along-stream fudge, and 50 km/s cap.

### Changes
- `src/simulation/subhalo.py`: new `erkal_plummer_kick()` (vectorised 3D kick)
  and `build_encounter_geometry()` (gap-forming default geometry, or isotropic
  sampling). `_hernquist_impulse_kick` (fast-mode magnitude) replaced by the
  bounded Plummer form ``2GM/w * d/(d^2+r_s^2)`` — no cap, no correction fudges.
- `src/forward_model/evolve.py`: `_apply_3d_velocity_kick` now calls
  `erkal_plummer_kick` with geometry built from the local stream tangent and
  galactocentric radial direction (full-orbit path, both single + multi-encounter).
- Tests: `tests/test_erkal_kick.py` (9: point-mass limit, mass/velocity scaling,
  boundedness, peak at d=r_s, transverse-to-w, geometry); `test_simulation.py`
  cap test replaced by a boundedness test.

### Key physical consequence
The bounded kick is **far smaller and more realistic** than the old
cap-saturated values (e.g. a 10^8.5 Msun perturber at b=0.1 kpc gives ~5 km/s,
not the old clipped 50 km/s). Detectable gaps in the short test streams now
require ~10^9 Msun; injection-recovery tests retuned accordingly and pass
(10^9 recovers exactly, beats null 0.37 vs 1.01). Full suite: 269 passed.

A sign bug (kick pointing away from, not toward, the perturber) was caught by
the point-mass-limit unit test and fixed.

## Report update

`generate_pdf.py` + `Stellar_Stream_DM_Full_Report.pdf` regenerated with a new
**Section 13 "Timeline Forward Model and Sim-to-Real Robustness"** (6
subsections, 5 new figures): the rewind/replay workflow, the Erkal kick (Fig 16),
the detector over-confidence diagnosis + error-DR fix (Fig 17, 391->13 sigma),
GD-1 gap localisation + frame transform, multi-epoch/S5 radial-velocity fusion
(Fig 18), and injection-recovery + significance validation (Fig 19). Abstract,
table of contents, and references (Erkal & Belokurov 2015; Price-Whelan &
Bonaca 2018; Li+2019 S5) updated. (reportlab added to the environment.)

## Remaining (of the four requested improvements)
B: look-elsewhere-corrected significance; C: GD-1 RVs from APOGEE/DESI;
D: full-orbit injection-recovery validation.
