#!/usr/bin/env python
"""
SIDM discrimination via gap SHAPE (the one handle SIDM has).

SIDM subhalos are cored / lower-concentration: at fixed mass their effective
scale radius is larger than a cuspy NFW halo, so the Plummer impulse is softer and
the resulting gap is shallower (and broader). CDM/SIDM share the subhalo mass
function, so they are indistinguishable by abundance or detected-mass spectrum
(dm_discrimination_forecast.py) -- the only signal is the gap morphology at fixed mass.

This test generates matched-mass impacts with NFW (CDM) vs enlarged (SIDM-cored)
scale radii using streamgapdf and measures the gap-depth separability.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_training_data import _fix_galpy_dll_path
_fix_galpy_dll_path()
import numpy as np
import astropy.units as u
from galpy.df import streamdf, streamgapdf
from galpy.actionAngle import actionAngleIsochroneApprox, estimateBIsochrone
from src.simulation.potentials import get_mw_potential, _RO, _VO
from src.simulation.stream_gen import set_progenitor_ic_track6d, _pos_vel_to_orbit, _galactocentric_to_stream_coords
from src.simulation.subhalo import scale_radius_from_mass
from scripts.validate_generator import max_gap_depth, auc
import galstreams


def main() -> int:
    mws = galstreams.MWStreams(verbose=False)
    pot = get_mw_potential("config/streams.yaml"); frame = mws["GD-1-I21"].stream_frame
    ic = set_progenitor_ic_track6d("GD1", "config/streams.yaml", mws=mws)
    prog = _pos_vel_to_orbit(np.array(ic["pos_kpc"]), np.array(ic["vel_kms"]))
    b = float(estimateBIsochrone(pot, prog.R(use_physical=True)/_RO, prog.z(use_physical=True)/_RO))
    aA = actionAngleIsochroneApprox(pot=pot, b=b)
    common = dict(progenitor=prog, pot=pot, aA=aA, leading=True, nTrackChunks=5,
                  tdisrupt=3.0*u.Gyr, ro=_RO, vo=_VO)

    def ph(xv):
        xv = np.asarray(xv, float); R, vR, vT, z, vz, phi = xv
        if np.median(np.abs(R)) < 3.0: R, z = R*_RO, z*_RO
        return _galactocentric_to_stream_coords(np.array([R*np.cos(phi), R*np.sin(phi), z]),
                                                np.zeros((3, len(R))), frame)[0]

    CORE = 3.0  # SIDM effective scale radius / NFW scale radius (strong-SIDM representative)
    rng = np.random.default_rng(0)
    print(f"SIDM core factor = {CORE}x NFW r_s  (cored subhalo softens the kick)")
    print(f"{'logM':>5} {'r_s NFW':>8} {'gapdepth CDM':>13} {'gapdepth SIDM':>14} {'AUC(CDM>SIDM)':>14}")
    rows = []
    for logM in (8.0, 8.3, 8.6):
        M = 10**logM; rs = scale_radius_from_mass(M)
        gd_cdm, gd_sidm = [], []
        for i in range(6):
            ang = rng.uniform(0.3, 0.6); bb = rng.uniform(0.0, 0.2); tt = rng.uniform(0.4, 1.0)
            for rs_use, store in ((rs, gd_cdm), (rs*CORE, gd_sidm)):
                sg = streamgapdf(0.5*u.km/u.s, **common, impactb=bb*u.kpc,
                                 subhalovel=np.array([0., 150., 0.])*u.km/u.s,
                                 timpact=tt*u.Gyr, impact_angle=ang*u.rad,
                                 GM=M*u.Msun, rs=rs_use*u.kpc)
                store.append(max_gap_depth(ph(sg.sample(n=1200))))
        gd_cdm, gd_sidm = np.array(gd_cdm), np.array(gd_sidm)
        a = auc(gd_cdm, gd_sidm)  # P(CDM gap deeper than SIDM gap)
        print(f"{logM:5.1f} {rs:8.2f} {np.median(gd_cdm):13.2f} {np.median(gd_sidm):14.2f} {a:14.2f}")
        rows.append((logM, float(np.median(gd_cdm)), float(np.median(gd_sidm)), float(a)))
    aucs = [r[3] for r in rows]
    print(f"\nMean gap-depth separability CDM vs SIDM (cored): AUC = {np.mean(aucs):.2f}")
    print("AUC ~0.5 -> indistinguishable; >0.7 -> cored gaps are systematically shallower.")
    import json; os.makedirs("outputs/dm", exist_ok=True)
    json.dump({"core_factor": CORE, "rows": rows, "mean_auc": float(np.mean(aucs))},
              open("outputs/dm/sidm_morphology.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
