#!/usr/bin/env python
"""
Validate the streamdf/streamgapdf generator against the pre-flight gates.

Generates a small batch of no-impact (streamdf) + impact (streamgapdf) streams
directly and runs scripts/validate_generator.evaluate_gates on them. This is the
GATE that must pass before any full regeneration.

Usage:
  python scripts/validate_df_generator.py --streams GD1 --n-per 10
  python scripts/validate_df_generator.py --streams GD1 Pal5 ATLAS --n-per 8
"""
from __future__ import annotations
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# galpy C-extension DLL fix (same as generate_training_data) -----------------
from scripts.generate_training_data import _fix_galpy_dll_path  # noqa: E402
_fix_galpy_dll_path()

import numpy as np  # noqa: E402
import astropy.units as u  # noqa: E402

from src.simulation.potentials import get_mw_potential, _RO  # noqa: E402
from src.simulation.stream_gen import (  # noqa: E402
    generate_stream_df,
    sample_impact_params,
    _load_stream_config,
)
from scripts.validate_generator import evaluate_gates, print_report  # noqa: E402


def track_benchmarks(stream_name: str, mws, config_path: str) -> dict | None:
    """Extent + median pm of the galstreams track in the observed window."""
    try:
        sc = _load_stream_config(stream_name, config_path)
        tr = mws[sc["galstreams_key"]]
        trsf = tr.track.transform_to(tr.stream_frame)
        phi1 = (np.asarray(trsf.phi1.deg) + 180.0) % 360.0 - 180.0
        lo, hi = sc["phi1_range_deg"]
        win = (phi1 >= lo) & (phi1 <= hi)
        if win.sum() < 5:
            win = np.ones_like(phi1, bool)
        bm = {"track_extent": float(np.ptp(phi1[win]))}
        try:
            bm["pm1"] = float(np.nanmedian(np.asarray(trsf.pm_phi1_cosphi2.value)[win]))
            bm["pm2"] = float(np.nanmedian(np.asarray(trsf.pm_phi2.value)[win]))
        except Exception:
            pass
        return bm
    except Exception as e:
        print(f"  (no benchmarks for {stream_name}: {e!r})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="+", default=["GD1"])
    ap.add_argument("--n-per", type=int, default=10, help="no-impact + impact sims per stream")
    ap.add_argument("--n-stars", type=int, default=1200)
    ap.add_argument("--config-streams", default="config/streams.yaml")
    args = ap.parse_args()

    import galstreams
    mws = galstreams.MWStreams(verbose=False)
    pot = get_mw_potential(args.config_streams)

    no_impact: list[dict] = []
    strong_impact: list[dict] = []
    bm_first = None
    rng = np.random.default_rng(0)

    for sname in args.streams:
        t0 = time.time()
        bm = track_benchmarks(sname, mws, args.config_streams)
        if bm_first is None:
            bm_first = bm
        # no-impact (streamdf): base setup paid once, samples are fast
        for i in range(args.n_per):
            p = generate_stream_df(sname, pot, n_stars=args.n_stars,
                                   seed=1000 + i, impact=False,
                                   config_path=args.config_streams, mws=mws)
            no_impact.append({k: getattr(p, k) for k in ("phi1", "phi2", "pm1", "pm2", "vrad")})
        # impact (streamgapdf): each ~15s
        for i in range(args.n_per):
            ip = sample_impact_params(rng)
            p = generate_stream_df(sname, pot, n_stars=args.n_stars,
                                   seed=2000 + i, impact=True, impact_params=ip,
                                   config_path=args.config_streams, mws=mws)
            strong_impact.append({k: getattr(p, k) for k in ("phi1", "phi2", "pm1", "pm2", "vrad")})
        print(f"  {sname}: {args.n_per} no-impact + {args.n_per} impact in {time.time()-t0:.0f}s")

    print(f"\nGenerated {len(no_impact)} no-impact + {len(strong_impact)} impact streams")
    gates = evaluate_gates(no_impact, strong_impact, benchmarks=bm_first)
    ok = print_report(f"streamdf/streamgapdf GENERATOR VALIDATION ({','.join(args.streams)})", gates)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
