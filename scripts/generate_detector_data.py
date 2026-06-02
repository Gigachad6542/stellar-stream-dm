#!/usr/bin/env python
"""
Detector training-data generation with the validated streamdf/streamgapdf engine.

This replaces the homemade rewind/kick impact generator. No-impact streams come
from streamdf and impacts from streamgapdf (a streamdf subclass) so the two
classes share an identical smooth action-angle track -- the only difference is
the subhalo gap. Validated 2026-06-02 (scripts/validate_df_generator.py): smooth
no-impact (gap-depth excess ~0.01), detectable impacts (G6 AUC 0.94).

Balanced binary dataset: 50% no-impact (label 0) / 50% impact (label 1). Each
sim is labelled by its REALISED gap-depth excess (impact_strength), so the
detector can be trained / evaluated as a function of detectability.

Parallel + chunked + resumable (same machinery as generate_training_data): a
suspended run resumes from existing chunk_*.h5 files.

Usage:
  # validate a small batch end-to-end (writes chunks, then check the harness):
  python scripts/generate_detector_data.py --n-sims 200 --n-jobs -2 \
      --output-dir data/simulations_detector_df_test
  python scripts/validate_generator.py --sim-dir data/simulations_detector_df_test --max-sims 200
  # full run:
  python scripts/generate_detector_data.py --n-sims 20000 --n-jobs -2 \
      --output-dir data/simulations_detector_df
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the heavy, battle-tested setup from the main pipeline (thread pinning,
# galpy DLL fix, galstreams cache, chunk writer, worker init).
from scripts.generate_training_data import (  # noqa: E402
    _fix_galpy_dll_path,
    _get_mws,
    _load_yaml_cached,
    _pool_worker_init,
    write_simulation,
)

_fix_galpy_dll_path()

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from src.simulation.potentials import get_mw_potential  # noqa: E402
from src.simulation.stream_gen import (  # noqa: E402
    StreamParticles,
    generate_stream_df,
    sample_impact_params,
)
from src.simulation.noise import (  # noqa: E402
    add_foreground_contamination,
    add_gaia_noise_randomized,
    build_membership_probabilities,
)
from scripts.validate_generator import gap_depth_excess, max_gap_depth  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Streams where the streamgapdf action-angle model is valid (eccentric, 6D track).
# Pal5/Fjorm fail the isochrone approximation (near-circular GC orbits) and Sylgr
# wraps in phi1 -> excluded. Confirmed by the 2026-06-02 smoke test.
DF_SUPPORTED_STREAMS = ["GD1", "Orphan", "ATLAS", "Jhelum"]


def generate_one_detector_sim(args_tuple) -> dict | None:
    """Generate one detector training example (no-impact or impact)."""
    (run_id, stream_name, seed, is_impact, cfg_streams, cfg_training, noise_mode) = args_tuple
    try:
        rng = np.random.default_rng(seed)
        training_cfg = _load_yaml_cached(cfg_training)
        stream_cfg = _load_yaml_cached(cfg_streams)["streams"][stream_name]
        sim_cfg = training_cfg.get("simulation", {})
        prep = training_cfg.get("preprocessing", {})
        target_n = int(prep.get("max_stars_per_sim", 1200))
        target_n = int(round(target_n * rng.uniform(0.6, 1.0)))  # vary star count
        source_n = max(int(target_n * 1.4), 400)

        potential = get_mw_potential(cfg_streams)
        impact_params = sample_impact_params(rng) if is_impact else None
        particles = generate_stream_df(
            stream_name, potential, n_stars=source_n, seed=seed,
            impact=is_impact, impact_params=impact_params,
            config_path=cfg_streams, mws=_get_mws(),
        )
        if len(particles.phi1) < 50:
            return None

        # Realised detectability. impact_strength = raw gap-depth (0.5-1.0 for a
        # real gap) so the harness's S>0.5 strong-impact selection works; the
        # Poisson-excess is kept separately as the shot-noise-corrected measure.
        if is_impact:
            impact_strength = float(max_gap_depth(particles.phi1))
            impact_strength_excess = float(gap_depth_excess(particles.phi1))
        else:
            impact_strength = 0.0
            impact_strength_excess = 0.0

        # ── observational realism ────────────────────────────────────────────
        noise_scale = 0.0
        contamination = 0.0
        if noise_mode in ("gaia", "full"):
            noise_scale = float(rng.uniform(0.5, 2.0))
        if noise_mode == "full":
            contamination = float(rng.uniform(0.05, 0.30))

        n_pre = len(particles.phi1)
        if noise_scale <= 0.0:
            sig_pm1 = np.zeros(n_pre, np.float32); sig_pm2 = np.zeros(n_pre, np.float32)
            sig_dist = np.zeros(n_pre, np.float32); sig_vrad = np.zeros(n_pre, np.float32)
        else:
            particles, sig_pm1, sig_pm2, sig_dist, sig_vrad = add_gaia_noise_randomized(
                particles, stream_cfg, noise_scale_factor=noise_scale, seed=seed + 400)
        particles = add_foreground_contamination(
            particles, stream_cfg, contamination_fraction=contamination, seed=seed + 500)

        n_cont = len(particles.phi1) - n_pre
        membership_prob = build_membership_probabilities(
            n_stream=n_pre, n_foreground=n_cont,
            corruption_fraction=(0.05 if noise_mode == "full" else 0.0),
            noise_sigma=(0.03 if noise_mode == "full" else 0.0), seed=seed + 501)
        if n_cont > 0:
            sig_pm1 = np.concatenate([sig_pm1, np.full(n_cont, 0.5, np.float32)])
            sig_pm2 = np.concatenate([sig_pm2, np.full(n_cont, 0.5, np.float32)])
            sig_dist = np.concatenate([sig_dist, np.full(n_cont, 5.0, np.float32)])
            sig_vrad = np.concatenate([sig_vrad, np.full(n_cont, np.nan, np.float32)])

        # Resample to the observed star count.
        if len(particles.phi1) > target_n:
            ridx = np.random.default_rng(seed + 600).choice(
                len(particles.phi1), target_n, replace=False)
            particles = StreamParticles(
                phi1=particles.phi1[ridx], phi2=particles.phi2[ridx],
                dist=particles.dist[ridx], pm1=particles.pm1[ridx],
                pm2=particles.pm2[ridx], vrad=particles.vrad[ridx],
                xyz_kpc=particles.xyz_kpc[:, ridx], vxyz_kms=particles.vxyz_kms[:, ridx])
            sig_pm1, sig_pm2 = sig_pm1[ridx], sig_pm2[ridx]
            sig_dist, sig_vrad = sig_dist[ridx], sig_vrad[ridx]
            membership_prob = membership_prob[ridx]

        if is_impact:
            enc_masses = np.array([impact_params["mass"]], np.float32)
            enc_bkpc = np.array([impact_params["impactb_kpc"]], np.float32)
            enc_vkms = np.array([impact_params["vsub_kms"]], np.float32)
            enc_t_since = np.array([impact_params["timpact_gyr"]], np.float32)
            log_m_mean = float(np.log10(impact_params["mass"]))
        else:
            enc_masses = np.array([], np.float32); enc_bkpc = np.array([], np.float32)
            enc_vkms = np.array([], np.float32); enc_t_since = np.array([], np.float32)
            log_m_mean = float(rng.uniform(7.5, 8.7))  # prior draw (no info)

        labels = {
            "dm_model_idx": 0, "dm_model": "CDM",
            "log_m_sub_mean": log_m_mean, "log10_M_hm": 4.0,
            "n_subhalos": int(is_impact),
            "impact_detectable": int(is_impact),
            "impact_strength": impact_strength,
            "impact_strength_excess": impact_strength_excess,
            "log_t_since_last_impact_gyr": float(
                np.log10(impact_params["timpact_gyr"]) if is_impact else 1.0),
            "noise_scale_factor": noise_scale,
            "contamination_fraction": contamination,
            "generator": "streamgapdf" if is_impact else "streamdf",
        }
        return {
            "run_id": run_id, "stream_name": stream_name, "particles": particles,
            "labels": labels, "enc_masses": enc_masses, "enc_bkpc": enc_bkpc,
            "enc_vkms": enc_vkms, "enc_t_since": enc_t_since,
            "sigma_pm1": sig_pm1, "sigma_pm2": sig_pm2, "sigma_dist": sig_dist,
            "sigma_vrad": sig_vrad, "membership_prob": membership_prob,
        }
    except Exception as exc:
        log.error("detector sim failed run_id=%s stream=%s impact=%s seed=%d: %s",
                  run_id, stream_name, is_impact, seed, exc)
        return None


def build_jobs(n_sims: int, streams: list[str], cfg_streams: str, cfg_training: str,
               noise_mode: str, seed: int = 42) -> list:
    """Balanced impact/no-impact jobs, round-robin over streams."""
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_sims):
        stream = streams[i % len(streams)]
        is_impact = bool(i % 2)            # exactly balanced
        s = int(rng.integers(0, 2**31))
        jobs.append((str(uuid.uuid4())[:8], stream, s, is_impact,
                     cfg_streams, cfg_training, noise_mode))
    return jobs


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    streams = args.streams or DF_SUPPORTED_STREAMS
    jobs = build_jobs(args.n_sims, streams, args.config_streams,
                      args.config_training, args.noise, seed=args.seed)

    chunk_size = args.chunk_size
    existing = sorted(output_dir.glob("chunk_*.h5"))
    chunk_id = len(existing)
    if existing:
        skip = min(len(jobs), len(existing) * chunk_size)
        log.info("Resume: %d existing chunks -> skipping %d jobs", len(existing), skip)
        jobs = jobs[skip:]
    if not jobs:
        log.info("No remaining jobs.")
        return
    log.info("Streams: %s | noise=%s | remaining jobs: %d", streams, args.noise, len(jobs))

    import multiprocessing as mp
    n_jobs = args.n_jobs
    if n_jobs < 0:
        n_jobs = max(1, mp.cpu_count() + n_jobs + 1)
    log.info("Pool n_jobs=%d, maxtasksperchild=200", n_jobs)

    try:
        from tqdm import tqdm
        pbar = tqdm(desc="Detector sims", total=len(jobs))
    except ImportError:
        pbar = None

    def flush(results, cid):
        path = output_dir / f"chunk_{cid:05d}.h5"
        valid = [r for r in results if r is not None]
        if valid:
            with h5py.File(str(path), "w") as f:
                for r in valid:
                    write_simulation(f, r)
        n_imp = sum(int(r["labels"]["impact_detectable"]) for r in valid)
        log.info("chunk %d -> %s (%d/%d valid, %d impact)", cid, path.name,
                 len(valid), len(results), n_imp)
        return len(valid)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_jobs, initializer=_pool_worker_init, maxtasksperchild=200)
    start = time.time()
    n_done = 0
    buf: list = []
    try:
        it = pool.imap_unordered(generate_one_detector_sim, jobs, chunksize=1)
        completed = 0
        while completed < len(jobs):
            try:
                r = next(it)
            except StopIteration:
                break
            except Exception as exc:
                log.error("sim crashed (pool continues): %s", type(exc).__name__)
                r = None
            completed += 1
            buf.append(r)
            if pbar:
                pbar.update(1)
            if len(buf) >= chunk_size:
                n_done += flush(buf, chunk_id); chunk_id += 1; buf = []
    finally:
        pool.close(); pool.join()
    if buf:
        n_done += flush(buf, chunk_id)
    if pbar:
        pbar.close()
    el = time.time() - start
    log.info("Done: %d sims in %.0fs (%.1f s/sim wall)", n_done, el, el / max(n_done, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate detector training data (streamdf/streamgapdf).")
    ap.add_argument("--n-sims", type=int, required=True)
    ap.add_argument("--n-jobs", type=int, default=-2)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--streams", nargs="+", default=None,
                    help=f"default: {DF_SUPPORTED_STREAMS}")
    ap.add_argument("--noise", choices=["none", "gaia", "full"], default="full")
    ap.add_argument("--config-streams", default="config/streams.yaml")
    ap.add_argument("--config-training", default="config/training.yaml")
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())
