#!/usr/bin/env python
"""
Fetch CLEAN external membership catalogs for ALL target streams from the
Ibata+2021 STREAMFINDER catalog (VizieR J/ApJ/914/123), by auto-identifying which
numbered STREAMFINDER stream aligns with each target's galstreams track.

Writes data/processed/{STREAM}_streamfinder.h5 (standardized schema) for every
target with a confident track match. Generalizes fetch_streamfinder_gd1.py.

Usage:
    python scripts/fetch_streamfinder_streams.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
import yaml

TARGETS = {  # detector-vocab name -> galstreams key
    "GD1": "GD-1-I21", "ATLAS": "ATLAS-I21", "Jhelum": "Jhelum-I21",
    "Orphan": "Orphan-I21", "Pal5": "Pal5-I21", "Fjorm": "Fjorm-I21",
    "Sylgr": "Sylgr-I21",
}


def _frame_for(mws, key):
    # tolerate naming variants in galstreams
    for k in (key, key.replace("-I21", ""), key.split("-")[0]):
        try:
            return mws[k]
        except Exception:
            continue
    return None


def main() -> int:
    from astroquery.vizier import Vizier
    import galstreams

    v = Vizier(catalog="J/ApJ/914/123",
               columns=["Stream", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "HRV", "e_HRV"])
    v.ROW_LIMIT = -1
    t = v.get_catalogs("J/ApJ/914/123/table1")[0]
    snum = np.array([str(x).strip() for x in t["Stream"]])
    RA = np.array(t["RAJ2000"], float); DE = np.array(t["DEJ2000"], float)
    PMRA = np.array(t["pmRA"], float); PMDE = np.array(t["pmDE"], float)
    HRV = np.array(t["HRV"], float); EHRV = np.array(t["e_HRV"], float)
    uniq = sorted(set(snum), key=lambda x: (len(x), x))
    print(f"STREAMFINDER table1: {len(RA)} members across {len(uniq)} numbered streams")

    mws = galstreams.MWStreams(verbose=False)
    cfg = yaml.safe_load(open("config/streams.yaml"))["streams"]
    allc = SkyCoord(ra=RA * u.deg, dec=DE * u.deg,
                    pm_ra_cosdec=PMRA * u.mas / u.yr, pm_dec=PMDE * u.mas / u.yr)

    written = []
    for name, key in TARGETS.items():
        gk = cfg.get(name, {}).get("galstreams_key", key)
        track = _frame_for(mws, gk)
        if track is None:
            print(f"  {name}: no galstreams track ({gk}); skip"); continue
        fr = track.stream_frame
        cc = allc.transform_to(fr)
        p1 = (np.array(cc.phi1.deg) + 180) % 360 - 180
        p2 = np.array(cc.phi2.deg)
        # score each SF stream number by alignment to this target track
        best = None
        for s in uniq:
            m = snum == s
            if m.sum() < 40:
                continue
            rms = float(np.sqrt(np.nanmean(p2[m] ** 2)))
            span = float(np.nanpercentile(p1[m], 98) - np.nanpercentile(p1[m], 2))
            if rms < 3.0 and span > 10.0:
                score = span / (1.0 + rms)
                if best is None or score > best[1]:
                    best = (s, score, int(m.sum()), rms, span)
        if best is None:
            print(f"  {name}: no STREAMFINDER match"); continue
        s, score, n, rms, span = best
        m = snum == s
        phi1 = p1[m]; phi2 = p2[m]
        pm1 = np.array(cc.pm_phi1_cosphi2.value)[m]; pm2 = np.array(cc.pm_phi2.value)[m]
        trc = track.track.transform_to(fr)
        tp1 = (np.array(trc.phi1.deg) + 180) % 360 - 180; o = np.argsort(tp1)
        dist = interp1d(tp1[o], np.array(trc.distance.to(u.kpc).value)[o], kind="linear",
                        bounds_error=False, fill_value="extrapolate")(phi1).astype(float)
        hrv = HRV[m]; ehrv = EHRV[m]
        ok = np.isfinite(hrv) & (np.abs(hrv) < 500.0)
        vrad = np.where(ok, hrv, np.nan).astype(float)
        e_vrad = np.where(ok & np.isfinite(ehrv) & (ehrv > 0), ehrv, np.nan).astype(float)
        out = f"data/processed/{name}_streamfinder.h5"
        cols = dict(phi1=phi1, phi2=phi2, dist=dist, pm1=pm1, pm2=pm2, vrad=vrad,
                    e_pm1=np.full(n, 0.2), e_pm2=np.full(n, 0.2), e_dist=np.full(n, 0.5),
                    e_vrad=e_vrad, membership_prob=np.ones(n))
        with h5py.File(out, "w") as fw:
            g = fw.create_group(f"streams/{name}/members")
            for k, vv in cols.items():
                g.create_dataset(k, data=np.asarray(vv, np.float32))
            meta = fw.create_group(f"streams/{name}/meta")
            meta.attrs["source"] = f"Ibata+2021 STREAMFINDER stream #{s}"
            meta.attrs["frame"] = gk; meta.attrs["n_members"] = n
        print(f"  {name}: SF#{s}  N={n}  phi2_rms={rms:.2f}  span={span:.0f}deg  -> {out}")
        written.append(name)
    print(f"\nwrote clean catalogs for: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
