"""
Parallel Gaia download + processing for all remaining stellar streams.

All 5 Gaia TAP async jobs are fired simultaneously — they run independently
on the ESA servers. Each thread polls its own job, downloads, and transforms.
HDF5 writes are serialized with a threading.Lock() to avoid corruption.

Usage:
    python scripts/process_streams_parallel.py
    python scripts/process_streams_parallel.py --streams Orphan ATLAS Jhelum
    python scripts/process_streams_parallel.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gaia_query import query_stream_members
from src.data.stream_process import (
    _load_stream_config,
    _to_stream_frame_no_mc,
    compute_dust_correction,
    estimate_distances,
    to_stream_frame,
    write_to_hdf5,
)

# Shared lock — HDF5 does not support concurrent writes to the same file
_HDF5_LOCK = threading.Lock()

PROCESSED_HDF5 = "data/processed/streams.h5"
RAW_DIR = Path("data/raw")
CONFIG_PATH = "config/streams.yaml"
DEFAULT_STREAMS = ["Orphan", "ATLAS", "Jhelum", "Fjorm", "Sylgr"]


def setup_logging(stream_name: str) -> logging.Logger:
    """Per-stream logger that prefixes every line with the stream name."""
    log = logging.getLogger(stream_name)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s [{stream_name:8s}] %(levelname)s %(message)s"
        ))
        log.addHandler(handler)
        fh = logging.FileHandler(f"parallel_{stream_name}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            f"%(asctime)s [{stream_name:8s}] %(levelname)s %(message)s"
        ))
        log.addHandler(fh)
        log.setLevel(logging.INFO)
    return log


def already_in_hdf5(stream_name: str) -> bool:
    p = Path(PROCESSED_HDF5)
    if not p.exists():
        return False
    try:
        with h5py.File(str(p), "r") as f:
            return f"streams/{stream_name}" in f
    except OSError:
        return False


def _download_and_transform(stream_name: str, force: bool) -> tuple[str, object] | None:
    """
    Worker function: runs entirely in a thread.
    Returns (stream_name, processed_astropy_Table) or None on failure.
    Does NOT touch the HDF5 file.
    """
    log = setup_logging(stream_name)
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 1. Gaia download (thread-safe: each job has its own ID on the server)
    # ------------------------------------------------------------------ #
    raw_path = RAW_DIR / f"{stream_name}_gaia_raw.fits"
    try:
        raw_table = query_stream_members(
            stream_name,
            output_path=raw_path,
            config_path=CONFIG_PATH,
            force=force,
        )
        log.info("Download complete: %d rows in %.1f s", len(raw_table), time.time() - t0)
    except Exception as exc:
        log.error("Gaia download FAILED: %s", exc, exc_info=True)
        return None

    if len(raw_table) == 0:
        log.warning("Empty Gaia result — skipping.")
        return None

    # ------------------------------------------------------------------ #
    # 2. Transform in memory (no shared state — safe to run in parallel)
    # ------------------------------------------------------------------ #
    t1 = time.time()
    sc = _load_stream_config(stream_name, CONFIG_PATH)

    try:
        if len(raw_table) > 50_000:
            log.info("Large catalog (%d rows): applying pre-filter.", len(raw_table))
            import astropy.coordinates as coord  # noqa
            import astropy.units as u            # noqa
            import numpy as np

            # _to_stream_frame_no_mc internally calls _get_stream_frame(), which uses
            # either the custom GreatCircleICRSFrame (if custom_frame_pole_* set in config)
            # or galstreams.  Do NOT duplicate that frame lookup here.
            table_light = _to_stream_frame_no_mc(raw_table, stream_name, CONFIG_PATH)

            phi1_min, phi1_max = sc["phi1_range_deg"]
            phi2_hw = sc.get("phi2_selection_deg", 10.0) * 2.0
            pm1_min, pm1_max = sc.get("pm1_range_masyr", [-100, 100])
            pm2_min, pm2_max = sc.get("pm2_range_masyr", [-100, 100])
            pm1_pad = (pm1_max - pm1_min) * 0.5
            pm2_pad = (pm2_max - pm2_min) * 0.5

            import numpy as np
            phi1 = np.asarray(table_light["phi1"])
            phi2 = np.asarray(table_light["phi2"])

            # galstreams returns phi1 in [0, 360].  When phi1_min < 0, the stream
            # straddles the 0/360 boundary (e.g. Sylgr at phi1=[345,11]).
            # Convert the negative lower bound to the equivalent positive angle.
            if phi1_min < 0:
                phi1_mask = (phi1 >= (360 + phi1_min)) | (phi1 <= phi1_max)
            else:
                phi1_mask = (phi1 >= phi1_min) & (phi1 <= phi1_max)

            mask = phi1_mask & (np.abs(phi2) <= phi2_hw)
            # Only apply stream-frame PM cut if the no-MC transform produced PM columns.
            # Frames without differential support (e.g. NGC3201-P21, Sylgr-I21) omit
            # these columns; in that case the ADQL PM pre-cut is sufficient.
            if "pm_phi1" in table_light.colnames and "pm_phi2" in table_light.colnames:
                pm1 = np.asarray(table_light["pm_phi1"])
                pm2 = np.asarray(table_light["pm_phi2"])
                mask = (
                    mask
                    & (pm1 >= pm1_min - pm1_pad) & (pm1 <= pm1_max + pm1_pad)
                    & (pm2 >= pm2_min - pm2_pad) & (pm2 <= pm2_max + pm2_pad)
                )
            raw_table = raw_table[mask]
            log.info("Pre-filter -> %d rows (%.2f%%)", len(raw_table),
                     100 * len(raw_table) / max(len(table_light), 1))

        table = to_stream_frame(raw_table, stream_name, CONFIG_PATH)
        table = estimate_distances(table, stream_name, CONFIG_PATH)
        table = compute_dust_correction(table)
        log.info("Transform complete in %.1f s — %d stars", time.time() - t1, len(table))
    except Exception as exc:
        log.error("Transform FAILED: %s", exc, exc_info=True)
        return None

    log.info("Ready to write. Total elapsed: %.1f s", time.time() - t0)
    return stream_name, table


def process_parallel(streams: list[str], force: bool = False, max_workers: int = 5) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    root_log = logging.getLogger("parallel_main")

    to_run = []
    for s in streams:
        if already_in_hdf5(s) and not force:
            root_log.info("%-10s already in HDF5 — skipping (--force to reprocess).", s)
        else:
            to_run.append(s)

    if not to_run:
        root_log.info("All streams already processed.")
        return {}

    root_log.info("Launching %d parallel Gaia queries: %s", len(to_run), to_run)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_download_and_transform, s, force): s
            for s in to_run
        }

        for future in as_completed(future_map):
            stream_name = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                root_log.error("%s worker raised: %s", stream_name, exc, exc_info=True)
                results[stream_name] = "FAILED"
                continue

            if result is None:
                results[stream_name] = "FAILED"
                continue

            name, table = result
            # Serialize HDF5 writes — only one thread writes at a time
            with _HDF5_LOCK:
                try:
                    write_to_hdf5(name, table, PROCESSED_HDF5)
                    root_log.info("%-10s written to HDF5 (%d stars).", name, len(table))
                    results[name] = "OK"
                except Exception as exc:
                    root_log.error("HDF5 write FAILED for %s: %s", name, exc, exc_info=True)
                    results[name] = "FAILED"

    return results


def print_hdf5_summary() -> None:
    if not Path(PROCESSED_HDF5).exists():
        return
    with h5py.File(PROCESSED_HDF5, "r") as f:
        streams = sorted(f.get("streams", {}).keys())
        print("\n--- HDF5 Summary ---")
        for s in streams:
            n = f[f"streams/{s}"].attrs.get("n_members", "?")
            print(f"  {s:<12} {n} members")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MAIN    ] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("process_parallel.log"),
        ],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", nargs="+", default=DEFAULT_STREAMS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        valid = set(yaml.safe_load(f)["streams"].keys())
    unknown = [s for s in args.streams if s not in valid]
    if unknown:
        print(f"Unknown streams: {unknown}. Valid: {sorted(valid)}")
        sys.exit(1)

    print_hdf5_summary()
    t0 = time.time()
    results = process_parallel(args.streams, force=args.force)

    print(f"\n--- Results ({time.time()-t0:.0f}s total) ---")
    for s, status in results.items():
        print(f"  {s:<12} {status}")
    print_hdf5_summary()

    if any(v == "FAILED" for v in results.values()):
        sys.exit(1)
