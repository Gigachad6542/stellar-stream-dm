#!/usr/bin/env python
"""
Fetch a CLEAN external GD-1 membership catalog (Ibata+2021 STREAMFINDER,
VizieR J/ApJ/914/123) and write it to our standardized HDF5 schema.

The bundled data/processed/streams.h5 GD-1 "members" are a broad, ~96%-field
selection that washes out the known Price-Whelan & Bonaca (2018) gap. STREAMFINDER
stream #8 is GD-1: 811 high-confidence members spanning the full ~100 deg extent
(incl. the gap region), with proper motions and radial velocities.

Identification: stream #8 is the numbered STREAMFINDER stream whose members lie
along the GD-1-I21 track (phi2 rms ~1.1 deg over a 102 deg phi1 span).

Output: data/processed/GD1_streamfinder.h5  (same layout as streams.h5)

Usage:
    python scripts/fetch_streamfinder_gd1.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d


def main() -> int:
    from astroquery.vizier import Vizier
    import galstreams

    v = Vizier(catalog="J/ApJ/914/123",
               columns=["Stream", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "plx",
                        "HRV", "e_HRV", "Gmag0"])
    v.ROW_LIMIT = -1
    t = v.get_catalogs("J/ApJ/914/123/table1")[0]
    s = np.array([str(x).strip() for x in t["Stream"]])
    m = s == "8"  # GD-1 (identified by GD-1-I21 track alignment)
    ra = np.array(t["RAJ2000"], float)[m]
    dec = np.array(t["DEJ2000"], float)[m]
    pmra = np.array(t["pmRA"], float)[m]
    pmdec = np.array(t["pmDE"], float)[m]
    hrv = np.array(t["HRV"], float)[m]
    e_hrv = np.array(t["e_HRV"], float)[m]
    n = len(ra)
    print(f"STREAMFINDER GD-1 (#8): {n} members")

    mws = galstreams.MWStreams(verbose=False)
    track = mws["GD-1-I21"]
    fr = track.stream_frame

    # Positions + proper motions into the GD-1 stream frame.
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg,
                 pm_ra_cosdec=pmra * u.mas / u.yr,
                 pm_dec=pmdec * u.mas / u.yr).transform_to(fr)
    phi1 = (np.array(c.phi1.deg) + 180) % 360 - 180
    phi2 = np.array(c.phi2.deg)
    pm1 = np.array(c.pm_phi1_cosphi2.value)
    pm2 = np.array(c.pm_phi2.value)

    # Distance from the galstreams GD-1 distance track (interp vs phi1); STREAMFINDER
    # parallaxes are too noisy at ~8 kpc to use per-star.
    trc = track.track.transform_to(fr)
    tphi1 = (np.array(trc.phi1.deg) + 180) % 360 - 180
    o = np.argsort(tphi1)
    f_dist = interp1d(tphi1[o], np.array(trc.distance.to(u.kpc).value)[o],
                      kind="linear", bounds_error=False, fill_value="extrapolate")
    dist = f_dist(phi1).astype(float)

    # HRV has occasional garbage fill values (e.g. one star at 16857 km/s);
    # reject anything outside a generous physical line-of-sight window.
    hrv_ok = np.isfinite(hrv) & (np.abs(hrv) < 500.0)
    vrad = np.where(hrv_ok, hrv, np.nan).astype(float)
    e_vrad = np.where(hrv_ok & np.isfinite(e_hrv) & (e_hrv > 0), e_hrv, np.nan).astype(float)

    # Per-star uncertainties: STREAMFINDER members are bright Gaia EDR3 sources;
    # use representative EDR3 proper-motion errors and a stream-distance error.
    e_pm1 = np.full(n, 0.2, float)
    e_pm2 = np.full(n, 0.2, float)
    e_dist = np.full(n, 0.5, float)
    membership_prob = np.ones(n, np.float32)

    out = "data/processed/GD1_streamfinder.h5"
    cols = dict(phi1=phi1, phi2=phi2, dist=dist, pm1=pm1, pm2=pm2, vrad=vrad,
                e_pm1=e_pm1, e_pm2=e_pm2, e_dist=e_dist, e_vrad=e_vrad,
                membership_prob=membership_prob)
    with h5py.File(out, "w") as fw:
        g = fw.create_group("streams/GD1/members")
        for k, vv in cols.items():
            g.create_dataset(k, data=np.asarray(vv, np.float32))
        meta = fw.create_group("streams/GD1/meta")
        meta.attrs["source"] = "Ibata+2021 STREAMFINDER (VizieR J/ApJ/914/123), stream #8"
        meta.attrs["frame"] = "GD-1-I21"
        meta.attrs["n_members"] = n
    print(f"phi1 [{phi1.min():.1f}, {phi1.max():.1f}], phi2 rms {np.sqrt(np.mean(phi2**2)):.2f}, "
          f"RV on {np.isfinite(vrad).sum()}/{n}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
