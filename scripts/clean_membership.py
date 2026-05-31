#!/usr/bin/env python
"""
Extract clean stream members from the (contaminated) processed catalog by
selecting stars consistent with the galstreams track in proper-motion and phi2
space. The base streams.h5 catalogs are broad field selections with uniform
membership_prob (e.g. GD-1 is ~96% field contamination); this recovers the
actual stream stars that trace the track.

For each member, the track phi2/pm1/pm2 are interpolated at the star's phi1 and
the star is kept if it lies within the configured tolerances. Writes a sidecar
``data/processed/{stream}_clean.h5`` (same layout as streams.h5).

Usage:
    python scripts/clean_membership.py --stream GD1
    python scripts/clean_membership.py --stream GD1 --dpm 2.0 --dphi2 1.0
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--config", default="config/streams.yaml")
    ap.add_argument("--h5", default="data/processed/streams.h5")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--dphi2", type=float, default=1.0, help="|phi2 - track| tolerance [deg]")
    ap.add_argument("--dpm", type=float, default=2.0, help="|pm - track| tolerance [mas/yr] (each component)")
    args = ap.parse_args()

    import yaml, astropy.units as u
    import galstreams
    from scipy.interpolate import interp1d

    with open(args.config) as f:
        sc = yaml.safe_load(f)["streams"][args.stream]
    with h5py.File(args.h5, "r") as fr:
        grp = fr[f"streams/{args.stream}/members"]
        cols = {k: grp[k][:] for k in grp.keys()}
    phi1 = cols["phi1"].astype(float); phi2 = cols["phi2"].astype(float)
    pm1 = cols["pm1"].astype(float); pm2 = cols["pm2"].astype(float)
    n0 = len(phi1)

    # galstreams track interpolators (phi2/pm1/pm2 vs phi1)
    mws = galstreams.MWStreams(verbose=False)
    trsf = mws[sc["galstreams_key"]].track.transform_to(mws[sc["galstreams_key"]].stream_frame)
    tphi1 = (np.array(trsf.phi1.deg) + 180) % 360 - 180
    o = np.argsort(tphi1)
    def _interp(arr):
        return interp1d(tphi1[o], np.array(arr)[o], kind="linear",
                        bounds_error=False, fill_value="extrapolate")
    f_phi2 = _interp(trsf.phi2.deg)
    try:
        f_pm1 = _interp(trsf.pm_phi1_cosphi2); f_pm2 = _interp(trsf.pm_phi2); has_pm = True
    except Exception:
        has_pm = False

    keep = np.abs(phi2 - f_phi2(phi1)) < args.dphi2
    if has_pm:
        keep &= np.abs(pm1 - f_pm1(phi1)) < args.dpm
        keep &= np.abs(pm2 - f_pm2(phi1)) < args.dpm
    n1 = int(keep.sum())
    print(f"{args.stream}: {n0} raw -> {n1} clean members "
          f"({100*n1/max(n0,1):.1f}%) within |dphi2|<{args.dphi2}, |dpm|<{args.dpm}")
    if n1:
        print(f"  clean pm1 median={np.median(pm1[keep]):.2f} (track {np.median(f_pm1(phi1[keep])):.2f})"
              if has_pm else "")

    out = os.path.join(args.out_dir, f"{args.stream}_clean.h5")
    with h5py.File(out, "w") as fw:
        mg = fw.create_group(f"streams/{args.stream}/members")
        for k, v in cols.items():
            if k == "membership_prob":
                continue  # replaced below with a real track-consistency flag
            mg.create_dataset(k, data=v[keep])
        # set a real membership flag (1 for kept track-consistent stars)
        mg.create_dataset("membership_prob", data=np.ones(n1, dtype=np.float32))
        with h5py.File(args.h5, "r") as src:
            if f"streams/{args.stream}/meta" in src:
                src.copy(src[f"streams/{args.stream}/meta"], fw[f"streams/{args.stream}"], name="meta")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
