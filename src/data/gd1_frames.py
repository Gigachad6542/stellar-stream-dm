"""
GD-1 stream-frame transforms.

GD-1's morphology (the famous density gap at phi1 ~ -40 and the spur) is
documented in the **Koposov et al. 2010 / Price-Whelan & Bonaca 2018 (PWB18)**
stream frame. The simulation/forward-model pipeline, however, works in the
**Ibata et al. 2021 (I21)** frame supplied by galstreams (key ``GD-1-I21``).
These are two different rotations of the sky, so a phi1 value in one frame is
meaningless in the other — which is why the config ``known_gaps`` (phi1=-40,
PWB18) do not apply to the I21-frame data (phi1 in [0.6, 72.8]).

This module provides the Koposov-2010 rotation (gala's ``GD1Koposov10``) so a
location on the GD-1 track can be moved between the PWB18 frame and the I21
frame via ICRS. galstreams does not ship the Koposov frame and ``gala`` is not
installed here, so the rotation matrix is defined explicitly.

Reference: Koposov, Rix & Hogg 2010, ApJ 712, 260; Price-Whelan & Bonaca 2018,
ApJ 863, L20.
"""

from __future__ import annotations

import numpy as np

# Rotation matrix mapping ICRS unit vectors to the GD-1 (Koposov 2010) frame:
#   v_gd1 = R_ICRS_TO_GD1 @ v_icrs
# Values match gala.coordinates.GD1Koposov10.
R_ICRS_TO_GD1 = np.array([
    [-0.4776303088, -0.1738432154, 0.8611897727],
    [0.510844589, -0.8524449229, 0.111245042],
    [0.7147776536, 0.4930681392, 0.4959603976],
])


def koposov_phi_to_icrs(phi1_deg, phi2_deg=0.0):
    """Convert GD-1 (Koposov/PWB18) phi1/phi2 [deg] to ICRS (ra, dec) [deg]."""
    phi1 = np.radians(np.asarray(phi1_deg, dtype=float))
    phi2 = np.radians(np.asarray(phi2_deg, dtype=float))
    v_gd1 = np.stack([
        np.cos(phi1) * np.cos(phi2),
        np.sin(phi1) * np.cos(phi2),
        np.sin(phi2) * np.ones_like(phi1),
    ], axis=-1)                                   # [..., 3]
    v_icrs = v_gd1 @ R_ICRS_TO_GD1                # (R^T applied) GD1 -> ICRS
    ra = np.degrees(np.arctan2(v_icrs[..., 1], v_icrs[..., 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(v_icrs[..., 2], -1.0, 1.0)))
    return ra, dec


def pwb18_phi1_to_i21(phi1_pwb_deg, phi2_pwb_deg=0.0, mws=None):
    """Map a PWB18/Koposov phi1 [deg] on the GD-1 track to the I21-frame phi1.

    Args:
        phi1_pwb_deg: scalar or array of phi1 in the Koposov/PWB18 frame.
        phi2_pwb_deg: corresponding phi2 (default on-track, 0).
        mws: optional preloaded ``galstreams.MWStreams`` (avoids re-init).

    Returns:
        phi1 in the GD-1-I21 frame (same shape as input).
    """
    import astropy.units as u
    from astropy.coordinates import ICRS, SkyCoord

    if mws is None:
        import galstreams
        mws = galstreams.MWStreams(verbose=False)
    i21_frame = mws["GD-1-I21"].stream_frame

    ra, dec = koposov_phi_to_icrs(phi1_pwb_deg, phi2_pwb_deg)
    c = SkyCoord(ra=np.atleast_1d(ra) * u.deg, dec=np.atleast_1d(dec) * u.deg, frame=ICRS)
    ci = c.transform_to(i21_frame)
    phi1_i21 = np.asarray(ci.phi1.to_value(u.deg))
    return float(phi1_i21[0]) if np.ndim(phi1_pwb_deg) == 0 else phi1_i21


def transform_known_gaps_to_i21(known_gaps, mws=None):
    """Transform a list of config ``known_gaps`` (PWB18 frame) into I21 phi1.

    Each input gap is a dict with at least ``phi1_center`` (PWB18). Returns a new
    list with ``phi1_center`` replaced by the I21-frame value and the original
    kept as ``phi1_center_pwb18``.
    """
    out = []
    for g in known_gaps:
        g2 = dict(g)
        g2["phi1_center_pwb18"] = g["phi1_center"]
        g2["phi1_center"] = pwb18_phi1_to_i21(g["phi1_center"], mws=mws)
        out.append(g2)
    return out
