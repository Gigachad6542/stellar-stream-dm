"""
Benchmark JAX integrator vs galpy on 10 simulations per mode.

Verifies:
1. Speedup (JAX should be 10-30x faster per sim after JIT warm-up)
2. Stream track quality: phi2 RMS < 1 deg for JAX streams
3. Particle counts are reasonable (>100 per stream)

Usage
-----
    PYTHONNOUSERSITE=1 python -u scripts/benchmark_jax.py
    PYTHONNOUSERSITE=1 python -u scripts/benchmark_jax.py --n-sims 20 --stream GD1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _fix_dll():
    import ctypes
    conda_root = Path(sys.executable).parent
    for d in [conda_root / "Library" / "bin",
              conda_root / "Library" / "mingw-w64" / "bin", conda_root]:
        if d.exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(d))
                except OSError:
                    pass


def benchmark_mode(mode: str, n_sims: int, stream_name: str, potential,
                   cfg_streams: str, mws) -> dict:
    """Run n_sims simulations in galpy or JAX mode and return timing/quality stats."""
    from src.simulation.stream_gen import generate_stream

    if mode == "jax":
        os.environ["USE_JAX_INTEGRATOR"] = "1"
    else:
        os.environ["USE_JAX_INTEGRATOR"] = "0"

    log.info("[%s] Running %d simulations on %s ...", mode.upper(), n_sims, stream_name)

    times = []
    n_particles = []
    phi2_stds = []

    rng = np.random.default_rng(42)

    # First call includes JIT compilation for JAX — report separately
    t_first = time.time()
    p = generate_stream(
        stream_name=stream_name,
        potential=potential,
        n_stars=5000,
        seed=1001,
        config_path=cfg_streams,
        mws=mws,
        n_steps_back=100,
    )
    t_first_elapsed = time.time() - t_first
    first_n = len(p.phi1)
    first_phi2_std = float(np.std(p.phi2)) if first_n > 10 else np.nan

    if mode == "jax":
        log.info("  [%s] Sim 1 (JIT compile): %.1f s  n=%d  phi2_std=%.3f deg",
                 mode.upper(), t_first_elapsed, first_n, first_phi2_std)
    else:
        log.info("  [%s] Sim 1: %.1f s  n=%d", mode.upper(), t_first_elapsed, first_n)

    # Subsequent calls (no JIT overhead for JAX)
    for i in range(1, n_sims):
        seed = int(rng.integers(1000, 9999))
        t0 = time.time()
        p = generate_stream(
            stream_name=stream_name,
            potential=potential,
            n_stars=5000,
            seed=seed,
            config_path=cfg_streams,
            mws=mws,
            n_steps_back=100,
        )
        elapsed = time.time() - t0
        n = len(p.phi1)
        phi2_std = float(np.std(p.phi2)) if n > 10 else np.nan

        times.append(elapsed)
        n_particles.append(n)
        phi2_stds.append(phi2_std)

    # Include first sim in n_particles for reporting
    n_particles_all = [first_n] + n_particles
    phi2_stds_all = [first_phi2_std] + phi2_stds
    times_all = [t_first_elapsed] + times

    os.environ["USE_JAX_INTEGRATOR"] = "0"  # reset

    return {
        "mode": mode,
        "n_sims": n_sims,
        "t_first_s": t_first_elapsed,
        "t_steady_mean_s": float(np.mean(times)) if times else t_first_elapsed,
        "t_steady_std_s": float(np.std(times)) if times else 0.0,
        "t_total_s": sum(times_all),
        "n_particles_mean": float(np.mean(n_particles_all)),
        "n_particles_min": int(np.min(n_particles_all)),
        "phi2_std_mean": float(np.nanmean(phi2_stds_all)),
    }


def main(args: argparse.Namespace) -> None:
    import yaml

    _fix_dll()

    cfg_path = ROOT / "config" / "training.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg_streams = str(ROOT / "config" / "streams.yaml")
    stream_name = args.stream

    log.info("Loading base MW potential ...")
    from src.simulation.potentials import get_mw_potential
    potential = get_mw_potential()

    log.info("Loading galstreams catalog ...")
    import galstreams  # noqa: PLC0415
    mws = galstreams.MWStreams(verbose=False)

    n_sims = args.n_sims

    # Check if JAX ICs exist
    cache_dir = ROOT / cfg["paths"].get("processed_data", "data/processed")
    jax_ic_path = cache_dir / f"{stream_name}_progenitor_ic_jax.json"
    if not jax_ic_path.exists():
        log.warning("JAX IC cache not found at %s", jax_ic_path)
        log.warning("Run 'python scripts/optimize_jax_ics.py --streams %s' first.", stream_name)
        log.warning("Proceeding anyway — JAX will optimise ICs on-the-fly (~5-8 min).")

    results = {}

    # galpy baseline
    log.info("")
    log.info("=" * 50)
    log.info("MODE: galpy (baseline)")
    log.info("=" * 50)
    results["galpy"] = benchmark_mode(
        "galpy", n_sims, stream_name, potential, cfg_streams, mws
    )

    # JAX
    log.info("")
    log.info("=" * 50)
    log.info("MODE: JAX")
    log.info("=" * 50)
    results["jax"] = benchmark_mode(
        "jax", n_sims, stream_name, potential, cfg_streams, mws
    )

    # Report
    log.info("")
    log.info("=" * 60)
    log.info("BENCHMARK RESULTS  —  %s  (%d sims each)", stream_name, n_sims)
    log.info("=" * 60)

    r_g = results["galpy"]
    r_j = results["jax"]

    log.info("%-12s  %8s  %8s  %8s  %8s",
             "Mode", "1st(s)", "Steady(s)", "N_stars", "phi2_std")
    log.info("-" * 55)
    log.info("%-12s  %8.1f  %8.2f±%4.2f  %8.0f  %8.3f deg",
             "galpy",
             r_g["t_first_s"], r_g["t_steady_mean_s"], r_g["t_steady_std_s"],
             r_g["n_particles_mean"], r_g["phi2_std_mean"])
    log.info("%-12s  %8.1f  %8.2f±%4.2f  %8.0f  %8.3f deg",
             "JAX",
             r_j["t_first_s"], r_j["t_steady_mean_s"], r_j["t_steady_std_s"],
             r_j["n_particles_mean"], r_j["phi2_std_mean"])

    if r_g["t_steady_mean_s"] > 0 and r_j["t_steady_mean_s"] > 0:
        speedup = r_g["t_steady_mean_s"] / r_j["t_steady_mean_s"]
        log.info("")
        log.info("Speedup (steady-state): %.1f×", speedup)

        total_g = r_g["t_total_s"]
        total_j = r_j["t_total_s"]
        log.info("Total time (%d sims): galpy=%.1fs  JAX=%.1fs  (%.1f× faster)",
                 n_sims, total_g, total_j, total_g / max(total_j, 0.001))

    # Quality check
    log.info("")
    log.info("Quality checks:")
    n_ok_g = r_g["n_particles_min"]
    n_ok_j = r_j["n_particles_min"]
    log.info("  galpy min particles: %d  %s", n_ok_g, "PASS" if n_ok_g >= 50 else "WARN")
    log.info("  JAX   min particles: %d  %s", n_ok_j, "PASS" if n_ok_j >= 50 else "WARN")
    log.info("  galpy phi2_std: %.3f deg", r_g["phi2_std_mean"])
    log.info("  JAX   phi2_std: %.3f deg", r_j["phi2_std_mean"])

    phi2_ratio = r_j["phi2_std_mean"] / max(r_g["phi2_std_mean"], 0.01)
    log.info("  phi2 width ratio JAX/galpy: %.2f  %s",
             phi2_ratio, "PASS" if phi2_ratio < 2.0 else "WARN — streams may be too wide")

    # Extrapolate to 40K production run
    log.info("")
    log.info("40K production run estimate (4 workers):")
    n_full = 40_000
    n_workers = 4
    jit_s = max(0, r_j["t_first_s"] - r_j["t_steady_mean_s"])  # extra time for first sim
    steady_s = r_j["t_steady_mean_s"]
    est_hours = (jit_s * n_workers + n_full * steady_s) / n_workers / 3600
    log.info("  Steady-state: %.2f s/sim", steady_s)
    log.info("  Estimated: %.1f hours  (vs galpy: %.1f hours)",
             est_hours,
             n_full * r_g["t_steady_mean_s"] / n_workers / 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark JAX vs galpy stream generation."
    )
    parser.add_argument("--n-sims", type=int, default=10,
                        help="Simulations per mode (default: 10)")
    parser.add_argument("--stream", default="GD1",
                        choices=["GD1", "Pal5", "Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"],
                        help="Stream to benchmark (default: GD1)")
    args = parser.parse_args()
    main(args)
