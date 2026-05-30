"""
Pre-compute JAX-optimised progenitor ICs for all 7 target streams.

This must be run ONCE before the JAX-accelerated 40K production run.
Each stream takes ~3-8 minutes (differential_evolution global search +
Nelder-Mead polish using JAX's leapfrog force table).

The resulting cache files are written to data/processed/ with suffix
``_progenitor_ic_jax.json``.  The production runner (generate_training_data.py
with USE_JAX_INTEGRATOR=1) picks them up automatically.

Usage
-----
    PYTHONNOUSERSITE=1 python -u scripts/optimize_jax_ics.py
    PYTHONNOUSERSITE=1 python -u scripts/optimize_jax_ics.py --streams GD1 Pal5

Expected runtime:
    All 7 streams:  ~30-60 min  (3-8 min each)
    Force table:    ~20 s once (built before first stream, then cached)
    JIT compile:    ~30 s once (first orbit evaluation, then cached)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "optimize_jax_ics.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

STREAMS = ["GD1", "Pal5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]


def main(args: argparse.Namespace) -> None:
    import yaml
    from src.simulation.potentials import get_mw_potential
    from src.simulation.stream_gen import set_progenitor_ic

    cfg_path = ROOT / "config" / "training.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cache_dir = ROOT / cfg["paths"].get("processed_data", "data/processed")
    config_streams = str(ROOT / "config" / "streams.yaml")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use the BASE potential — the JAX force table is built from this once
    # and cached for all streams.  vary_potential() would require rebuilding
    # the table for each stream (expensive).
    log.info("Loading base MW potential (MWPotential2014) ...")
    potential = get_mw_potential()

    # Pre-load galstreams ONCE for all streams (avoids 14-s reload per stream)
    log.info("Loading galstreams catalog (~14 s) ...")
    import galstreams  # noqa: PLC0415
    mws = galstreams.MWStreams(verbose=False)
    log.info("galstreams loaded (%d tracks)", len(mws))

    streams = args.streams
    log.info("Optimising JAX ICs for %d streams: %s", len(streams), streams)
    log.info("Force table will be built on the first stream call (~20 s), "
             "then JIT-compiled on the first orbit evaluation (~30 s).")

    results = {}
    t_total = time.time()

    for stream_name in streams:
        log.info("=" * 60)
        log.info("STREAM: %s", stream_name)
        log.info("=" * 60)
        t0 = time.time()

        try:
            ic = set_progenitor_ic(
                stream_name=stream_name,
                potential=potential,
                config_path=config_streams,
                cache_dir=cache_dir,
                mws=mws,
                use_jax=True,
            )
            elapsed = time.time() - t0
            rms = ic.get("_rms_deg")
            rms_str = f"{rms:.3f}" if rms is not None else "N/A"
            log.info("DONE %s: phi2 RMS = %s deg  (%.1f min)",
                     stream_name, rms_str, elapsed / 60)
            results[stream_name] = {"rms_deg": rms, "elapsed_s": elapsed, "ok": True}

            if rms is not None and rms > 1.0:
                log.warning("  WARNING: %s phi2 RMS = %.3f > 1.0 deg target",
                            stream_name, rms)
            elif rms is not None:
                log.info("  PASS: %s phi2 RMS = %.3f < 1.0 deg", stream_name, rms)

        except Exception as exc:
            elapsed = time.time() - t0
            log.error("FAILED %s after %.1f min: %s", stream_name, elapsed / 60, exc,
                      exc_info=True)
            results[stream_name] = {"rms_deg": None, "elapsed_s": elapsed, "ok": False}

    # Summary table
    log.info("")
    log.info("=" * 60)
    log.info("IC OPTIMISATION SUMMARY")
    log.info("=" * 60)
    log.info("%-12s  %10s  %8s  %s", "Stream", "phi2 RMS", "Time", "Status")
    log.info("-" * 50)
    n_pass = 0
    for sname, r in results.items():
        rms_str = f"{r['rms_deg']:.3f} deg" if r["rms_deg"] is not None else "   N/A   "
        status = "PASS" if r["ok"] and (r["rms_deg"] or 999) < 1.0 else (
                  "WARN" if r["ok"] else "FAIL")
        if status == "PASS":
            n_pass += 1
        log.info("%-12s  %10s  %6.1f min  %s",
                 sname, rms_str, r["elapsed_s"] / 60, status)

    total_elapsed = time.time() - t_total
    log.info("-" * 50)
    log.info("Total: %.1f min  |  %d/%d streams PASS", total_elapsed / 60, n_pass, len(streams))

    if n_pass == len(streams):
        log.info("")
        log.info("All streams optimised successfully.")
        log.info("You can now run the 40K production run:")
        log.info("  set USE_JAX_INTEGRATOR=1")
        log.info("  python -u scripts/generate_training_data.py --n-sims 10000 --n-jobs -2")
    else:
        n_fail = len(streams) - n_pass
        log.warning("%d streams did not meet the 1.0 deg target — "
                    "inspect logs and consider re-running with a wider search budget.", n_fail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute JAX-optimised progenitor ICs for all streams."
    )
    parser.add_argument(
        "--streams", nargs="+", default=STREAMS,
        metavar="STREAM",
        help="Streams to optimise (default: all 7)",
    )
    args = parser.parse_args()
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    main(args)
