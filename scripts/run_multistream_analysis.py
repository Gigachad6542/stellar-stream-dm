#!/usr/bin/env python
"""
Multi-stream joint subhalo-impact significance.

For each target stream, run the timeline forward model (detect -> localise ->
grid -> null distribution -> significance), then combine the per-stream
look-elsewhere-corrected significances across the population (Stouffer + Fisher).
Subhalo abundance is a population property, so the joint result is the
scientifically meaningful one; a single short stream is only marginal.

Uses fast (impulse) mode by default -- a population-level estimate over 7 streams
x grid x null realisations is impractical in full-orbit mode. Prefers a stream's
{stream}_multiepoch.h5 (real radial velocities) when present.

Usage:
    python scripts/run_multistream_analysis.py
    python scripts/run_multistream_analysis.py --streams GD1 ATLAS Jhelum --n-null 15 --look-elsewhere 12
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

ALL_STREAMS = ["GD1", "Pal5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-stream joint impact significance.")
    p.add_argument("--streams", nargs="+", default=ALL_STREAMS)
    p.add_argument("--config", default="config/streams.yaml")
    p.add_argument("--h5", default="data/processed/streams.h5")
    p.add_argument("--n-null", type=int, default=12, help="No-impact realizations for the per-stream null.")
    p.add_argument("--look-elsewhere", type=int, default=10,
                   help="Best-of-grid null realizations per stream (0 to skip; then only naive z).")
    p.add_argument("--n-stars", type=int, default=1200)
    p.add_argument("--mass-range", type=float, nargs=2, default=[7.5, 9.0])
    p.add_argument("--mass-step", type=float, default=0.5)
    p.add_argument("--t-step", type=float, default=0.5)
    p.add_argument("--max-phi1", type=int, default=3, help="Number of phi1 seeds from detected minima.")
    p.add_argument("--out", default="outputs/multistream/joint_significance.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("multistream")
    log.setLevel(logging.INFO)

    import yaml
    import galstreams
    from src.forward_model.pipeline import ForwardModelConfig, TimelineForwardModel
    from src.forward_model.scoring import ScoreWeights
    from src.forward_model.detection import detect_impacts_modelfree, seed_config_from_detection
    from src.forward_model.significance import compute_significance
    from src.forward_model.multistream import combine_streams

    with open(args.config) as f:
        streams_cfg = yaml.safe_load(f)["streams"]
    mws = galstreams.MWStreams(verbose=False)
    per_stream = []

    for name in args.streams:
        t0 = time.time()
        scfg = streams_cfg.get(name, {})
        age = float(scfg.get("disruption_age_gyr", scfg.get("isochrone_age_gyr", 5.0)))
        # Catalog choice for the forward model (model-free gap + data-driven null,
        # which is robust to contamination but benefits from member count): prefer a
        # CLEAN external STREAMFINDER catalog only when it is dense enough (>500
        # members, e.g. GD-1); otherwise fall back to the denser multi-epoch / bundle
        # so sparse clean catalogs don't starve the gap finder.
        sf = Path("data/processed") / f"{name}_streamfinder.h5"
        me = Path("data/processed") / f"{name}_multiepoch.h5"
        sf_n = 0
        if sf.exists():
            try:
                import h5py as _h5
                with _h5.File(sf, "r") as _f:
                    sf_n = int(_f[f"streams/{name}/members/phi1"].shape[0])
            except Exception:
                sf_n = 0
        if sf.exists() and sf_n >= 500:
            h5 = str(sf)
        elif me.exists():
            h5 = str(me)
        else:
            h5 = args.h5
        try:
            cfg = ForwardModelConfig(
                stream_name=name, config_path=args.config, processed_h5_path=h5,
                log10_mass_range=tuple(args.mass_range), log10_mass_step=args.mass_step,
                t_since_range=(0.5, max(1.0, age)), t_since_step=args.t_step,
                impact_phi1_values=[0.0], n_stars_sim=args.n_stars, base_seed=42,
                use_gnn_scorer=False, use_fast_mode=True, n_workers=1,
                score_weights=ScoreWeights(),
            )
            model = TimelineForwardModel(cfg)
            model._mws = mws
            model.prepare()

            # Localise from the data, seed the grid (capped at the stream age).
            det = detect_impacts_modelfree(model.obs_particles, model.phi1_range, name)
            model.cfg = seed_config_from_detection(
                det, model.cfg, max_phi1_seeds=args.max_phi1,
                min_t_since_gyr=0.5, max_t_since_gyr=max(1.0, age), stream_age_gyr=age)

            results = model.run_grid()
            best = results[0].score.combined
            naive = compute_significance(best, model.build_null_distribution(args.n_null, seed0=2000))
            entry = {
                "stream": name, "h5": Path(h5).name, "rv": me.exists(),
                "n_obs": int(len(model.obs_particles["phi1"])),
                "best_score": best, "best_log10_mass": results[0].log10_mass,
                "best_t_since": results[0].t_since_gyr, "best_phi1": results[0].impact_phi1,
                "z": naive.z_score, "p": naive.p_value,
                "null_mean": naive.null_mean, "null_std": naive.null_std,
            }
            if args.look_elsewhere > 0:
                # Data-driven null (gap-removed real data) -- valid for real
                # observations despite the sim-to-real offset, unlike a synthetic null.
                le = compute_significance(best, model.build_data_null(args.look_elsewhere, seed0=3000))
                entry["le_z"] = le.z_score; entry["le_p"] = le.p_value
                entry["le_null_mean"] = le.null_mean
            log.info("  %-7s: naive z=%.2f%s  (best=%.3f, %d stars, RV=%s) [%.0fs]",
                     name, naive.z_score,
                     f"  LE z={entry.get('le_z', float('nan')):.2f}" if "le_z" in entry else "",
                     best, entry["n_obs"], entry["rv"], time.time() - t0)
            per_stream.append(entry)
        except Exception as e:
            log.warning("  %-7s: FAILED (%s)", name, e)

    if not per_stream:
        log.error("No streams succeeded.")
        return 1

    use = "look_elsewhere" if args.look_elsewhere > 0 else "naive"
    joint = combine_streams(per_stream, use=use)

    log.info("=" * 70)
    log.info("MULTI-STREAM JOINT SIGNIFICANCE (%d streams, combining %s)", joint.n_streams, use)
    log.info("=" * 70)
    for s in per_stream:
        log.info("  %-7s z=%+.2f p=%.3f%s", s["stream"], s.get("le_z", s["z"]),
                 s.get("le_p", s["p"]), "  [RV]" if s["rv"] else "")
    log.info("  Stouffer Z = %.2f  (p = %.3g)", joint.stouffer_z, joint.stouffer_p)
    log.info("  Fisher chi2 = %.1f (p = %.3g)", joint.fisher_chi2, joint.fisher_p)
    log.info("  -> %s", joint.interpretation)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(joint.to_dict(), indent=2))
    log.info("Saved %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
