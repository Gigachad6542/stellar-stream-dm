#!/usr/bin/env python
"""
Calibrate per-stream spray_age_gyr so the generated stream matches the observed
angular (phi1) extent of its galstreams track (clipped to the config phi1 range).

The simple Fardal spray under-produces length per unit time, so the spray age
needed to reproduce the observed extent is larger than the literature disruption
timescale; this is a length-calibration knob (kept separate from the physical
disruption_age_gyr). Prints a recommended spray_age_gyr per stream.

Usage:
    python scripts/calibrate_spray_age.py --streams GD1 Pal5 ATLAS Jhelum Orphan Fjorm Sylgr
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL = ["GD1", "Pal5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=ALL)
    ap.add_argument("--config", default="config/streams.yaml")
    ap.add_argument("--n-stars", type=int, default=2000)
    ap.add_argument("--ages", type=float, nargs="+", default=[2, 3, 4, 6, 8, 10, 12, 14])
    args = ap.parse_args()

    import galstreams
    from src.simulation.potentials import get_mw_potential
    from src.simulation.stream_gen import generate_stream, _load_stream_config

    mws = galstreams.MWStreams(verbose=False)
    pot = get_mw_potential(args.config)

    print(f"{'stream':8s} {'target_span':>11s} {'best_age':>9s} {'sim_span':>9s} {'phi2_std':>9s}")
    recs = {}
    for name in args.streams:
        try:
            sc = _load_stream_config(name, args.config)
            track = mws[sc["galstreams_key"]]; frame = track.stream_frame
            tphi1 = (np.array(track.track.transform_to(frame).phi1.deg) + 180) % 360 - 180
            lo_c, hi_c = sc["phi1_range_deg"]
            t_in = tphi1[(tphi1 >= lo_c) & (tphi1 <= hi_c)]
            target = float(np.percentile(t_in, 98) - np.percentile(t_in, 2)) if len(t_in) > 5 else (hi_c - lo_c)
            best = None
            for age in args.ages:
                sp = generate_stream(name, pot, n_stars=args.n_stars, seed=1,
                                     stream_age_gyr=age, config_path=args.config, mws=mws)
                if len(sp.phi1) < 50:
                    continue
                span = float(np.percentile(sp.phi1, 99) - np.percentile(sp.phi1, 1))
                err = abs(span - target)
                if best is None or err < best[0]:
                    best = (err, age, span, float(sp.phi2.std()))
            if best:
                recs[name] = best[1]
                print(f"{name:8s} {target:11.1f} {best[1]:9.1f} {best[2]:9.1f} {best[3]:9.2f}")
        except Exception as e:
            print(f"{name:8s} FAILED: {type(e).__name__}: {str(e)[:60]}")
    print("\nRecommended spray_age_gyr:", {k: v for k, v in recs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
