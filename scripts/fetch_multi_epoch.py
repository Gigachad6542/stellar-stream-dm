#!/usr/bin/env python
"""
Fetch real multi-epoch / multi-survey data for a stream and fuse it.

Pulls public radial velocities (Gaia DR3 RVS + S5 survey) and, optionally, a
Gaia DR2 second proper-motion epoch, then inverse-variance combines them and
writes an augmented, drop-in stream file with the previously-empty ``vrad``
column populated from real measurements.

Usage:
    python scripts/fetch_multi_epoch.py --stream ATLAS
    python scripts/fetch_multi_epoch.py --stream GD1 --sources gaia_rvs
    python scripts/fetch_multi_epoch.py --stream ATLAS --max-stars 4000

Output:
    data/processed/{stream}_multiepoch.h5   (same layout as streams.h5; drop-in)
    data/raw/multi_epoch/...                (cached source catalogs)
    outputs/multi_epoch/{stream}_provenance.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch + fuse real multi-epoch data for a stream.")
    p.add_argument("--stream", type=str, required=True)
    p.add_argument("--h5", type=str, default="data/processed/streams.h5")
    p.add_argument("--sources", type=str, nargs="+",
                   default=["gaia_rvs", "s5"],
                   choices=["gaia_rvs", "s5", "gaia_dr2"],
                   help="Which real catalogs to fetch and fuse.")
    p.add_argument("--max-stars", type=int, default=None,
                   help="Subsample members before fetching (for quick tests).")
    p.add_argument("--cache-dir", type=str, default="data/raw/multi_epoch")
    p.add_argument("--out-dir", type=str, default="data/processed")
    p.add_argument("--prov-dir", type=str, default="outputs/multi_epoch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("fetch_multi_epoch")

    from src.data.multi_epoch import (
        FusionReport, fetch_gaia_dr2_proper_motions, fetch_gaia_dr3_rvs,
        fetch_s5_rvs, fuse_radial_velocities, pm_error_scaling_factor,
    )

    stream = args.stream
    cache_dir = Path(args.cache_dir)
    src_h5 = Path(args.h5)

    # --- Load all member fields ---
    with h5py.File(src_h5, "r") as f:
        grp = f[f"streams/{stream}/members"]
        members = {k: grp[k][:] for k in grp.keys()}
    n_all = len(members["source_id"])
    if args.max_stars and args.max_stars < n_all:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n_all, args.max_stars, replace=False))
        members = {k: v[idx] for k, v in members.items()}
        log.info("Subsampled %d -> %d members for fetching", n_all, len(members["source_id"]))
    sids = members["source_id"].astype(np.int64)
    log.info("Stream %s: %d members", stream, len(sids))

    # --- Fetch requested real catalogs ---
    rv_cats = []
    if "gaia_rvs" in args.sources:
        rv_cats.append(fetch_gaia_dr3_rvs(
            sids, cache_path=cache_dir / f"{stream}_gaia_rvs.npz"))
    if "s5" in args.sources:
        s5 = fetch_s5_rvs(cache_path=cache_dir / "s5_dr1.npz")
        if s5 is not None:
            rv_cats.append(s5)

    # --- Fuse radial velocities ---
    rv, e_rv, rv_mask, per_survey = fuse_radial_velocities(sids, rv_cats)
    report = FusionReport(
        n_members=len(sids),
        rv_n_measured=int(rv_mask.sum()),
        rv_sources=per_survey,
        rv_median_error=float(np.nanmedian(e_rv)) if np.isfinite(e_rv).any() else float("nan"),
    )

    # --- Optional Gaia DR2 second PM epoch (reported; ICRS) ---
    dr2_info = {}
    if "gaia_dr2" in args.sources:
        dr2 = fetch_gaia_dr2_proper_motions(
            sids, cache_path=cache_dir / f"{stream}_gaia_dr2_pm.npz")
        report.pm2epoch_n_matched = len(dr2.get("dr3_source_id", []))
        dr2_info = {
            "n_matched": report.pm2epoch_n_matched,
            "note": "DR2 is ICRS pmra/pmdec; stream-frame fusion needs a frame "
                    "transform. Main multi-epoch PM gain is future (DR4/DR5).",
        }

    # --- Write augmented drop-in stream file ---
    members["vrad"] = np.where(rv_mask > 0, rv, np.nan).astype(np.float32)
    members["e_vrad"] = np.where(rv_mask > 0, e_rv, np.nan).astype(np.float32)
    members["rv_mask"] = rv_mask.astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stream}_multiepoch.h5"
    with h5py.File(out_path, "w") as f:
        mg = f.create_group(f"streams/{stream}/members")
        for k, v in members.items():
            mg.create_dataset(k, data=v)
        # Copy meta if present
        with h5py.File(src_h5, "r") as src:
            if f"streams/{stream}/meta" in src:
                src.copy(src[f"streams/{stream}/meta"], f[f"streams/{stream}"], name="meta")

    # --- Provenance ---
    prov_dir = Path(args.prov_dir)
    prov_dir.mkdir(parents=True, exist_ok=True)
    prov = report.to_dict()
    prov["stream"] = stream
    prov["sources_requested"] = args.sources
    prov["gaia_dr2"] = dr2_info
    prov["pm_error_projection_vs_DR3"] = {
        dr: pm_error_scaling_factor(dr) for dr in ("DR4", "DR5")
    }
    prov["output_file"] = str(out_path)
    prov_path = prov_dir / f"{stream}_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)

    # --- Report ---
    log.info("=" * 64)
    log.info("MULTI-EPOCH FUSION SUMMARY for %s", stream)
    log.info("  members            : %d", report.n_members)
    log.info("  RV measured (real) : %d (%.1f%%)  sources=%s",
             report.rv_n_measured, 100.0 * report.rv_n_measured / max(report.n_members, 1),
             per_survey)
    log.info("  RV median error    : %.2f km/s", report.rv_median_error)
    if dr2_info:
        log.info("  DR2 2nd PM epoch   : %d matched", report.pm2epoch_n_matched)
    log.info("  PM-error projection: DR4 ~%.2fx, DR5 ~%.2fx of DR3",
             pm_error_scaling_factor("DR4"), pm_error_scaling_factor("DR5"))
    log.info("  augmented stream   : %s", out_path)
    log.info("  provenance         : %s", prov_path)
    log.info("=" * 64)


if __name__ == "__main__":
    main()
