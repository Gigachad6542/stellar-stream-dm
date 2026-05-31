# 2026-05-30 GD-1 PWB18 <-> I21 frame transform (localization accuracy)

## Why

GD-1's famous density gap is documented at phi1 ~ -40 in the **Koposov-2010 /
Price-Whelan & Bonaca 2018 (PWB18)** frame, but the pipeline works in the
**Ibata-2021 (I21)** frame (galstreams `GD-1-I21`, phi1 in [0.6, 72.8]). The two
are different rotations of the sky, so the config `known_gaps` (PWB18) were
unusable and fell out of range. galstreams ships only the I21 GD-1 frame, and
`gala` (which has `GD1Koposov10`) is not installed.

The 2 GB PWB18 region catalog (Zenodo 1295543, `gd1-with-masks.fits`) is
impractical to download, so we solve the localization with a frame transform
rather than a new catalog.

## What changed

- `src/data/gd1_frames.py` (new): the Koposov-2010 ICRS->GD1 rotation matrix
  (matching `gala.coordinates.GD1Koposov10`) plus:
  - `koposov_phi_to_icrs(phi1, phi2)` — PWB18 frame -> ICRS.
  - `pwb18_phi1_to_i21(phi1)` — PWB18 phi1 -> I21 phi1 (via ICRS + galstreams I21 frame).
  - `transform_known_gaps_to_i21(known_gaps)` — convert config gaps, keeping the
    PWB18 value as `phi1_center_pwb18`.
- `config/streams.yaml`: GD-1 `known_gaps` now carry **I21-frame** `phi1_center`
  (30.7, 50.7) with the original PWB18 values retained (`phi1_center_pwb18`).
- `tests/test_gd1_frames.py`: rotation is orthonormal/det=+1, sky position sane,
  and (slow) the PWB18->I21 mapping is monotonic and in-range.

## Key validation

The PWB18->I21 map is a clean ~70.7-deg longitude shift:

| phi1 PWB18 | phi1 I21 |
|---|---|
| -60 | 10.7 |
| **-40 (gap_1)** | **30.7** |
| -20 (spur) | 50.7 |
| 0 | 70.7 |

GD-1's documented gap at **-40 (PWB18) -> 30.7 (I21)** lands within ~5 deg of the
**data-driven deepest density minimum at I21~36** found by `find_density_minima`.
This independently confirms that the shallow minimum the pipeline localizes IS
GD-1's real gap — it only looks shallow because the current I21 member catalog
has degenerate membership (all 1.0) and contamination dilutes the contrast.

Suite: **254 passed, 6 deselected**.

## Note

The remaining limitation is the *catalog* (degenerate membership), not the
localization: a proper PWB18 membership catalog with probabilities would deepen
the gap contrast, but the gap's location is now pinned down two independent ways.
